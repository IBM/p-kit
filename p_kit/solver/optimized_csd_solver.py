"""
A heavily optimized version of the CaSuDaSolver.
"""
import numpy as np
from scipy.sparse import csr_matrix

from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.solver.annealing import constant
from p_kit.backends import NumpyBackend

try:
    from numba import njit
except ImportError:
    njit=None


if njit:
    @njit(cache=True)
    def _final_dense_numba(m,J,h,anneal,rnd,dt,threshold,tmp):
        ns,n=m.shape
        for run in range(len(anneal)):
            for s in range(ns):
                for i in range(n):
                    field=h[i]
                    for j in range(n):
                        field+=m[s,j]*J[j,i]
                    I=anneal[run]*field
                    p=np.exp(-dt*np.exp(-m[s,i]*(I+threshold)))
                    tmp[s,i]=m[s,i]*(1. if p-rnd[run,s,i]>=0 else -1.)
            m[:]=tmp
        return m

    @njit(cache=True)
    def _final_sparse_numba(m,indptr,indices,data,h,anneal,rnd,dt,threshold,tmp):
        ns,n=m.shape
        for run in range(len(anneal)):
            for s in range(ns):
                for i in range(n):
                    field=h[i]
                    for k in range(indptr[i],indptr[i+1]):
                        field+=data[k]*m[s,indices[k]]
                    I=anneal[run]*field
                    p=np.exp(-dt*np.exp(-m[s,i]*(I+threshold)))
                    tmp[s,i]=m[s,i]*(1. if p-rnd[run,s,i]>=0 else -1.)
            m[:]=tmp
        return m


class CaSuDaOptimized(CaSuDaSolver):
    """
    CaSuDaSolver with optional optimizations for repeated/recurrent execution.

    return_final=True avoids trajectory/current/energy storage in solove()

    Options:
      use_sparse     - sparse J representation
      use_numba      - JIT compiled final-state path
      reuse_buffers  - reuse working arrays
      cache_static   - cache fixed J and annealing data

    use_sparse/use_numba currently apply to CPU only.
    With cache_static=True, call clear_cache() after modifying c.J.
    """

    def __init__(
        self,Nt,dt,i0,expected_mean=0,seed=None,backend=None,tau=.1,
        use_sparse=False,use_numba=False,reuse_buffers=False,
        cache_static=False
    ):
        super().__init__(Nt,dt,i0,expected_mean,seed,backend,tau)
        self.use_sparse=use_sparse
        self.use_numba=use_numba
        self.reuse_buffers=reuse_buffers
        self.cache_static=cache_static

        self._J=None
        self._Jsparse=None
        self._anneal=None
        self._anneal_key=None
        self._buffers={}

        if use_numba and njit is None:
            raise ImportError("use_numba=True requires numba")
        if not isinstance(self.backend,NumpyBackend) and (use_numba or use_sparse):
            raise NotImplementedError(
                "use_numba/use_sparse currently support the NumPy backend only"
            )

    def clear_cache(self):
        self._J=self._Jsparse=self._anneal=self._anneal_key=None
        self._buffers.clear()

    def _buffer(self,name,shape):
        if not self.reuse_buffers:
            return np.empty(shape)

        a=self._buffers.get(name)
        if a is None or a.shape!=shape:
            a=np.empty(shape)
            self._buffers[name]=a
        return a

    def _get_J(self,c):
        # This method's caching/sparse-conversion is NumPy-only (see
        # __init__'s use_numba/use_sparse restriction). For any other
        # backend, defer to CaSuDaSolver._get_J so J is converted through
        # self.backend like everything else solve() touches - returning a
        # bare NumPy array here would silently upcast/misbehave when mixed
        # into e.g. Torch ops (torch_tensor @ numpy_array "works" but
        # upcasts to float64, breaking dtype consistency with the rest of
        # the backend's float32 state).
        if not isinstance(self.backend,NumpyBackend):
            return super()._get_J(c)

        if not self.cache_static:
            J=np.asarray(c.J)
        elif self._J is None:
            self._J=np.asarray(c.J).copy()
            J=self._J
        else:
            J=self._J

        if not self.use_sparse:
            return J

        if not self.cache_static:
            return csr_matrix(J.T)

        if self._Jsparse is None:
            self._Jsparse=csr_matrix(J.T)
        return self._Jsparse

    def _get_anneal(self,func):
        key=(id(func),self.Nt,float(self.i0))
        if self.cache_static and self._anneal is not None \
                and key==self._anneal_key:
            return self._anneal

        a=np.asarray([func(self,r) for r in range(self.Nt)],dtype=float)
        if self.cache_static:
            self._anneal=a
            self._anneal_key=key
        return a

    def _initial(self,n,n_shots,initial_state):
        if initial_state is None:
            return np.sign(.5-self.random((n_shots,n)))

        state=np.asarray(initial_state)
        if state.size!=n:
            raise ValueError(f"initial_state must contain {n} values")

        state=state.reshape(n)
        if not np.all((state==-1)|(state==1)):
            raise ValueError("initial_state must contain only -1 or +1")

        return np.tile(state,(n_shots,1))

    def _solve_final_cpu(
        self,c,annealing_func,n_shots,initial_state,bias_func
    ):
        n=c.n_pbits
        h=np.asarray(c.h).reshape(-1)
        J=self._get_J(c)
        anneal=self._get_anneal(annealing_func)
        threshold=float(np.arctanh(self.expected_mean))
        m=self._initial(n,n_shots,initial_state)

        if self.use_numba and bias_func is None:
            rnd=np.asarray(self.random((self.Nt,n_shots,n)))
            tmp=self._buffer("tmp",(n_shots,n))

            if self.use_sparse:
                m=_final_sparse_numba(
                    m,J.indptr,J.indices,J.data,h,anneal,rnd,
                    self.dt,threshold,tmp
                )
            else:
                m=_final_dense_numba(
                    m,J,h,anneal,rnd,self.dt,threshold,tmp
                )

            return m[0] if n_shots==1 else m

        m_filt=self._buffer("m_filt",(n_shots,n))
        m_filt.fill(0)

        for run in range(self.Nt):
            h_eff=h if bias_func is None else np.asarray(
                bias_func(run,m,m_filt)
            )

            field=J.dot(m.T).T if self.use_sparse else m@J
            I=anneal[run]*(field+h_eff)
            s=np.exp(-self.dt*np.exp(-m*(I+threshold)))
            m*=np.sign(s-self.random((n_shots,n)))
            m_filt=(1-self.tau)*m_filt+self.tau*m

        return m[0] if n_shots==1 else m

    def solve(
        self,c,annealing_func=constant,n_shots=1,
        bias_func=None,return_filtered=False,initial_state=None,
        return_final=False
    ):
        if not return_final:
            return super().solve(
                c,annealing_func,n_shots,bias_func,
                return_filtered,initial_state
            )

        if return_filtered:
            raise ValueError(
                "return_final and return_filtered cannot both be True"
            )

        if isinstance(self.backend,NumpyBackend):
            return self._solve_final_cpu(
                c,annealing_func,n_shots,initial_state,bias_func
            )

        # Non-NumPy backend fallback: preserve return_final semantics using CaSuDaSolver.
        result=super().solve(
            c,annealing_func,n_shots,bias_func,False,initial_state
        )
        if n_shots==1:
            return result[1][-1]
        return result[-1]

    def copy(self):
        return CaSuDaOptimized(
            Nt=self.Nt,
            dt=self.dt,
            i0=self.i0,
            expected_mean=self.expected_mean,
            seed=self.seed,
            backend=self.backend,
            tau=self.tau,
            use_sparse=self.use_sparse,
            use_numba=self.use_numba,
            reuse_buffers=self.reuse_buffers,
            cache_static=self.cache_static
        )