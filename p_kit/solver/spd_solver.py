"""
SPDSolver: continuous correlated solver for p-bit / Ising models.

The solver maps the discrete Ising parameters (J, h) to a symmetric
positive-definite (SPD) representation before sampling. The SPD structure
provides a valid continuous correlated state space, allowing proposals to
be generated collectively rather than by updating p-bits independently.

A continuous solver may provide better results, but the problem must first be
mapped successfully to a continuous space. This is the first and main obstacle.
Therefore, the J,h -> SPD mapping is validated before use. If the mapping quality
is below the configured threshold, SPDSolver can fall back to another solver.
Use fall_back_solver= in this case.

- mode="fast" uses a fixed SPD geometry for efficient continuous correlated proposals
- mode="analog" evolves the SPD state dynamically with Riemannian operations to better
  reflect a future analog hardware implementation.

SPDSolver is particularly useful for correlated sampling, optimization,
and future analog implementations where continuous SPD dynamics may be
implemented directly into hardware.

For problems requiring very accurate sampling of the target Boltzmann
distribution, especially as problem size grows, GibbsSolver may still
provide better distribution fidelity.
"""

from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np

EPS=1e-9

@dataclass
class SPDMapping:
    K: np.ndarray
    A: np.ndarray
    shift: float
    min_eig: float

@dataclass
class MappingQuality:
    score: float
    fidelity: float
    nrmse: float
    rank: float
    condition: float
    condition_score: float
    n_states: int

def sanitize_jh(J,h):
    J=np.asarray(J,float); h=np.asarray(h,float).reshape(-1)
    if J.shape!=(h.size,h.size): raise ValueError(f"J {J.shape}, h {h.shape}")
    J=.5*(J+J.T); J=J.copy(); np.fill_diagonal(J,0.)
    return J,h

def dense_jh(obj):
    if hasattr(obj,"J") and hasattr(obj,"h"):
        J=np.asarray(obj.J,float); h=np.asarray(obj.h,float).reshape(-1)
        if J.ndim==2 and J.shape==(h.size,h.size): return sanitize_jh(J,h)
    if hasattr(obj,"synthesize"):
        x=obj.synthesize(format="dense")
        if isinstance(x,tuple) and len(x)>=2: return sanitize_jh(x[0],x[1])
        if isinstance(x,dict) and "J" in x and "h" in x: return sanitize_jh(x["J"],x["h"])
    if hasattr(obj,"circuit"): return dense_jh(obj.circuit)
    raise TypeError("Cannot extract dense J,h")

def states(n): return np.asarray(list(itertools.product((-1.,1.),repeat=n)))

def energy(S,J,h):
    S=np.asarray(S,float); S=S[None,:] if S.ndim==1 else S
    return -.5*np.einsum("bi,ij,bj->b",S,J,S)-S@h

def ising_to_spd(J,h,i0=.8,margin=.05):
    J,h=sanitize_jh(J,h); n=h.size
    A=np.zeros((n+1,n+1)); A[:n,:n]=J; A[:n,n]=h; A[n,:n]=h; A*=i0
    shift=float(np.linalg.eigvalsh(A)[-1]+margin); K=shift*np.eye(n+1)-A; me=float(np.linalg.eigvalsh(K)[0])
    if me<=0: raise RuntimeError("SPD mapping failed")
    return SPDMapping(K,A,shift,me)

def _ranks(x):
    x=np.asarray(x); order=np.argsort(x,kind="mergesort"); r=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and np.isclose(x[order[j]],x[order[i]],rtol=1e-12,atol=1e-12): j+=1
        r[order[i:j]]=(i+j-1)/2.; i=j
    return r

def mapping_quality(J,h,m,max_condition=1e6,max_states=1<<16,random_states=20000,seed=12345):
    J,h=sanitize_jh(J,h); n=h.size; total=(1<<n) if n<63 else max_states+1
    S=states(n) if total<=max_states else np.random.default_rng(seed).choice((-1.,1.),size=(random_states,n))
    E=energy(S,J,h); Q=np.column_stack((S,np.ones(len(S)))); Z=np.einsum("bi,ij,bj->b",Q,m.K,Q)
    a,b=np.linalg.lstsq(np.column_stack((E,np.ones_like(E))),Z,rcond=None)[0]; Zh=a*E+b
    rmse=float(np.sqrt(np.mean((Z-Zh)**2))); scale=float(np.ptp(Zh)); nrmse=0. if scale<=EPS and rmse<=1e-12 else (float("inf") if scale<=EPS else rmse/scale)
    rE,rZ=_ranks(E),_ranks(Z); rank=float(np.corrcoef(rE,rZ)[0,1]) if len(E)>1 and np.std(rE)>0 and np.std(rZ)>0 else 1.
    fidelity=0. if a<=0 or not np.isfinite(nrmse) else float(np.clip(1.-nrmse,0.,1.)); cond=float(np.linalg.cond(m.K)); cscore=float(min(1.,max_condition/max(cond,1.)))
    return MappingQuality(fidelity*cscore,fidelity,float(nrmse),rank,cond,cscore,len(S))

def mapping_identity_error(J,h,m,i0=.8,max_states=1<<16):
    n=len(h)
    if (1<<n)>max_states: return float("nan")
    S=states(n); Q=np.column_stack((S,np.ones(len(S))))
    return float(np.max(np.abs(np.einsum("bi,ij,bj->b",Q,m.K,Q)-(m.shift*(n+1)+2.*i0*energy(S,J,h)))))

def spd_eigh(X):
    e,V=np.linalg.eigh(.5*(X+X.T)); return np.maximum(e,EPS),V

def spd_pow(X,p):
    e,V=spd_eigh(X); return (V*(e**p))@V.T

def riem_log(X,Y):
    H=spd_pow(X,.5); Hi=spd_pow(X,-.5); e,V=spd_eigh(Hi@Y@Hi)
    return H@((V*np.log(e))@V.T)@H

def riem_exp(X,Vt):
    H=spd_pow(X,.5); Hi=spd_pow(X,-.5); Z=.5*(Hi@Vt@Hi+(Hi@Vt@Hi).T); e,V=np.linalg.eigh(Z)
    return H@((V*np.exp(np.clip(e,-30.,30.)))@V.T)@H

def riem_dist(X,Y):
    e,_=spd_eigh(spd_pow(X,-.5)@Y@spd_pow(X,-.5)); return float(np.linalg.norm(np.log(e)))

def to_corr(X):
    e,V=spd_eigh(X); X=(V*e)@V.T; d=np.sqrt(np.maximum(np.diag(X),EPS)); C=X/np.outer(d,d)
    return .5*(C+C.T)+EPS*np.eye(len(C))

def target_corr(m,n): return to_corr(np.linalg.inv(m.K)[:n,:n])

def gaussian(rng,C):
    e,V=spd_eigh(C); return V@(np.sqrt(e)*rng.normal(size=len(e)))

class SPDSolver:
    
    def __init__(self,Nt=10000,dt=.1667,i0=.8,seed=None,mode="fast",margin=.05,min_mapping_score=.999,max_condition=1e6,rate=.08,noise=.025,flip_prob=.18,fall_back_solver=None,verbose=True):
        if Nt<=0 or i0<=0 or margin<=0 or rate<=0 or noise<0 or not 0<flip_prob<.5: raise ValueError("invalid solver parameter")
        if mode not in ("fast","analog"): raise ValueError("mode must be 'fast' or 'analog'")
        if not 0<=min_mapping_score<=1 or max_condition<=0: raise ValueError("invalid mapping threshold")
        self.Nt=Nt; self.dt=dt; self.i0=i0; self.seed=seed; self.mode=mode; self.margin=margin; self.min_mapping_score=min_mapping_score; self.max_condition=max_condition
        self.rate=rate; self.noise=noise; self.flip_prob=flip_prob; self.fall_back_solver=fall_back_solver; self.verbose=verbose
        self.mapping_=self.mapping_quality_=self.mapping_error_=self.manifold_state_=self.energies_=None
        self.acceptance_rate_=self.final_distance_=None; self.used_fallback_=False; self.fallback_reason_=None; self.fallback_solver_=None
    def _make_fallback(self):
        fb=self.fall_back_solver
        if fb is None: return None
        if not isinstance(fb,type) and hasattr(fb,"solve"): return fb
        if not callable(fb): raise TypeError("fall_back_solver must be a solver instance or callable")
        for kw in ({"Nt":self.Nt,"dt":self.dt,"i0":self.i0,"seed":self.seed},{"Nt":self.Nt,"dt":self.dt,"i0":self.i0},{"Nt":self.Nt,"i0":self.i0},{}):
            try: return fb(**kw)
            except TypeError: pass
        raise TypeError("could not instantiate fall_back_solver")
    def _run_fallback(self,obj,reason):
        fb=self._make_fallback()
        if fb is None: raise RuntimeError(reason)
        self.used_fallback_=True; self.fallback_reason_=reason; self.fallback_solver_=fb
        name=fb.__class__.__name__; warnings.warn(f"SPDSolver: {reason}; falling back to {name}.",RuntimeWarning,stacklevel=2)
        try: out=fb.solve(obj)
        except (TypeError,AttributeError):
            if not hasattr(obj,"circuit"): raise
            out=fb.solve(obj.circuit)
        if not isinstance(out,tuple) or len(out)<2: raise TypeError("fallback solver must return (I, m)")
        return out[0],out[1]
    def solve(self,circuit):
        self.used_fallback_=False; self.fallback_reason_=None; self.fallback_solver_=None
        J,h=dense_jh(circuit); n=h.size; m=ising_to_spd(J,h,self.i0,self.margin); q=mapping_quality(J,h,m,self.max_condition)
        self.mapping_=m; self.mapping_quality_=q; self.mapping_error_=mapping_identity_error(J,h,m,self.i0)
        if self.verbose: print(f"SPD map score={q.score:.6f} fidelity={q.fidelity:.6f} rank={q.rank:.6f} nrmse={q.nrmse:.2e} cond={q.condition:.2e}")
        if q.score<self.min_mapping_score:
            return self._run_fallback(circuit,f"mapping score {q.score:.6f} below minimum {self.min_mapping_score:.6f}")
        rng=np.random.default_rng(self.seed); T=target_corr(m,n); X=T.copy() if self.mode=="fast" else np.eye(n); s=rng.choice((-1.,1.),size=n); E=float(energy(s,J,h)[0]); th=NormalDist().inv_cdf(1.-self.flip_prob)
        all_I=np.empty((self.Nt,n)); all_m=np.empty((self.Nt,n)); all_E=np.empty(self.Nt); accepted=0
        for t in range(self.Nt):
            if self.mode=="analog":
                drift=riem_log(X,T); G=rng.normal(size=(n,n)); G=.5*(G+G.T); H=spd_pow(X,.5)
                X=to_corr(riem_exp(X,self.rate*drift+self.noise*np.sqrt(self.rate)*(H@G@H)))
            flip=gaussian(rng,X)>th; sp=s.copy(); sp[flip]*=-1.; Ep=float(energy(sp,J,h)[0]); dE=Ep-E
            if dE<=0 or rng.random()<np.exp(-self.i0*dE): s=sp; E=Ep; accepted+=1
            all_m[t]=s; all_I[t]=self.i0*(J@s+h); all_E[t]=E
        self.manifold_state_=X; self.energies_=all_E; self.acceptance_rate_=accepted/self.Nt; self.final_distance_=riem_dist(X,T)
        return all_I,all_m
