import itertools
from functools import lru_cache
import numpy as np
import pytest
p_kit=pytest.importorskip("p_kit")
from p_kit import psl
from p_kit.psl import Port
from p_kit.solver.csd_solver import CaSuDaSolver
try:
    from p_kit.solver.gibbs_solver import GibbsSolver
except ImportError:
    from p_kit.solver.gibbs import GibbsSolver
try:
    from p_kit.solver.spd_solver import SPDSolver,dense_jh,energy
except ImportError:
    from spd_solver import SPDSolver,dense_jh,energy

NT=5000; TAIL=2000; I0=.8; DT=.1667; SEEDS=(7,17,27)

@psl.pcircuit(n_pbits=3)
class ANDCircuit:
    A=Port("A"); B=Port("B"); C=Port("C")
    J=np.array([[0.,-1.,2.],[-1.,0.,2.],[2.,2.,0.]])
    h=np.array([[1.],[1.],[-2.]])

@psl.pcircuit(n_pbits=3)
class ORCircuit:
    A=Port("A"); B=Port("B"); C=Port("C")
    J=np.array([[0.,-1.,2.],[-1.,0.,2.],[2.,2.,0.]])
    h=np.array([[-1.],[-1.],[2.]])

@psl.pcircuit(n_pbits=4)
class FrustratedRing4:
    p0=Port("p0"); p1=Port("p1"); p2=Port("p2"); p3=Port("p3")
    J=np.array([[0.,1.4,0.,-1.],[1.4,0.,1.1,0.],[0.,1.1,0.,1.3],[-1.,0.,1.3,0.]])
    h=np.array([[.25],[-.15],[.10],[-.20]])

CASES=(("AND",ANDCircuit()),("OR",ORCircuit()),("Ring",FrustratedRing4()))
CASE_MAP=dict(CASES)

def states(n): return np.asarray(list(itertools.product((-1.,1.),repeat=n)))

def exact_distribution(J,h):
    S=states(len(h)); E=energy(S,J,h); z=-I0*E; p=np.exp(z-z.max()); return S,E,p/p.sum()

def empirical(samples,S):
    X=np.where(np.asarray(samples)>=0,1.,-1.); idx={tuple(s):i for i,s in enumerate(S)}; c=np.zeros(len(S))
    for s in X: c[idx[tuple(s)]]+=1
    return c/c.sum()

def metrics(samples,J,h):
    X=np.where(np.asarray(samples)>=0,1.,-1.)[-TAIL:]; S,E0,p=exact_distribution(J,h); pe=empirical(X,S); E=energy(X,J,h); g=E0.min()
    return {"tv":.5*float(np.abs(pe-p).sum()),"ground":float(np.mean(np.isclose(E,g))),"tail_e":float(E.mean()),"states":len({tuple(x) for x in X})}

def make_solver(cls,seed):
    for kw in ({"Nt":NT,"dt":DT,"i0":I0,"seed":seed},{"Nt":NT,"dt":DT,"i0":I0},{"Nt":NT,"i0":I0}):
        try: return cls(**kw)
        except TypeError: pass
    return cls()

def run(solver,circuit,J,h,seed):
    np.random.seed(seed)
    try: out=solver.solve(circuit)
    except (TypeError,AttributeError,ValueError):
        out=solver.solve(circuit.circuit) if hasattr(circuit,"circuit") else solver.solve(circuit)
    M=np.asarray(out[1],float)
    if M.ndim==1: M=M.reshape(-1,len(h))
    if M.shape[-1]!=len(h) and M.shape[0]==len(h): M=M.T
    return metrics(M,J,h)

@lru_cache(None)
def averages(name):
    circuit=CASE_MAP[name]; J,h=dense_jh(circuit); out={"SPD":[],"Gibbs":[],"CaSuDa":[]}
    for seed in SEEDS:
        out["SPD"].append(run(SPDSolver(Nt=NT,dt=DT,i0=I0,seed=seed,mode="fast",verbose=False),circuit,J,h,seed))
        out["Gibbs"].append(run(make_solver(GibbsSolver,seed),circuit,J,h,seed))
        out["CaSuDa"].append(run(make_solver(CaSuDaSolver,seed),circuit,J,h,seed))
    return {k:{m:float(np.mean([x[m] for x in v])) for m in v[0]} for k,v in out.items()},J,h

@pytest.mark.parametrize("name,circuit",CASES)
def test_spd_matches_exact_distribution_reasonably(name,circuit):
    out,J,h=averages(name); S,E,p=exact_distribution(J,h); target_ground=float(p[np.isclose(E,E.min())].sum()); spd=out["SPD"]
    assert spd["tv"]<.12,f"{name}: SPD TV={spd['tv']:.4f}"
    assert abs(spd["ground"]-target_ground)<.10,f"{name}: SPD ground={spd['ground']:.4f}, exact={target_ground:.4f}"
    assert spd["states"]>=min(6,len(S)-3),f"{name}: poor state coverage {spd['states']:.1f}/{len(S)}"

@pytest.mark.parametrize("name,circuit",CASES)
def test_spd_is_competitive_with_pkit_solvers(name,circuit):
    out,_,_=averages(name); s,g,c=out["SPD"],out["Gibbs"],out["CaSuDa"]
    assert s["tv"]<=g["tv"]+.12,f"{name}: SPD TV {s['tv']:.4f} vs Gibbs {g['tv']:.4f}"
    assert s["tv"]<=c["tv"]+.10,f"{name}: SPD TV {s['tv']:.4f} vs CaSuDa {c['tv']:.4f}"
    assert s["ground"]>=g["ground"]-.10,f"{name}: SPD ground {s['ground']:.4f} vs Gibbs {g['ground']:.4f}"
    assert s["ground"]>=c["ground"]-.10,f"{name}: SPD ground {s['ground']:.4f} vs CaSuDa {c['ground']:.4f}"
