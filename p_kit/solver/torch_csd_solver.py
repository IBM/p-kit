"""

This is a version of CaSuDaSolver using PyTorch. 

TorchCaSuDaSolver can allow an easy switch between CPU, GPU and for example
IBM Spyre AI accelerator chip.

From one hand it is good to have an alternative solver, but it also adds the
heavy dependency of PyTorch. And it some respects the idea was not to use 
PyTorch.

NOTES:
  - The solver keeps the same PCircuit, J/h model and CaSuDa dynamics.
  - It is intended mainly as an accelerator-oriented alternative backend.
  - For small p-bit models, PyTorch CPU can be slower than the NumPy solver.
  - GPU benefits are expected mainly for larger workloads.
  - cache_J is currently disabled because PCircuit.J may be modified in place.
  - compile=True may improve accelerator performance but adds compilation overhead.
  - Spyre support is currently disabled pending integration with the common
    annealing backend.
  - PyTorch and NumPy solvers use different random generators, so stochastic
    trajectories and final results are not expected to be identical.
    
  - TorchCaSuDaSolver is not fully optimized for CUDA yet.
    Although the solver kernels run on the selected Torch device, solve()
    returns NumPy arrays for compatibility with the existing p-kit API.
    This requires copying results from GPU to CPU after every solve.
    For recurrent workloads such as the language-model reservoir, repeated
    CPU<->GPU transfers can significantly reduce the benefit of CUDA.
    A future device-native solve_torch() path could keep J, h, state and
    intermediate results as Torch tensors on the accelerator.
"""

import numpy as np
import torch

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


def _single_kernel(m,J,h,anneal,rnd,dt,threshold,i0):
    Nt,n=rnd.shape
    all_m=torch.empty((Nt,n),dtype=m.dtype,device=m.device)
    all_I=torch.empty_like(all_m)
    E=torch.empty(Nt,dtype=m.dtype,device=m.device)

    for run in range(Nt):
        field=m@J
        if run:
            E[run-1]=i0*(torch.dot(m,h)+.5*torch.dot(field,m))
        I=anneal[run]*(field+h)
        s=torch.exp(-dt*torch.exp(-m*(I+threshold)))
        m=m*torch.sign(s-rnd[run])
        all_m[run]=m; all_I[run]=I

    field=m@J
    E[-1]=i0*(torch.dot(m,h)+.5*torch.dot(field,m))
    return all_I,all_m,E


def _multi_kernel(m,J,h,anneal,rnd,dt,threshold):
    Nt,n_shots,n=rnd.shape
    all_m=torch.empty((Nt,n_shots,n),dtype=m.dtype,device=m.device)
    for run in range(Nt):
        I=anneal[run]*(m@J+h)
        s=torch.exp(-dt*torch.exp(-m*(I+threshold)))
        m=m*torch.sign(s-rnd[run])
        all_m[run]=m
    return all_m


class TorchCaSuDaSolver(Solver):
    def __init__(
        self,Nt,dt,i0,expected_mean=0,seed=None,
        device="cpu",dtype=torch.float32,compile=False,cache_J=False,
    ):
        if device=="spyre":
            raise NotImplementedError("Spyre support is not enabled yet")
        if cache_J:
            raise NotImplementedError("cache_J is disabled because J may change in place")
        if device not in ("cpu","cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        super().__init__(Nt,dt,i0,expected_mean,seed,device="cpu")

        self.device=device
        self.dtype=dtype
        self.compile=compile
        self.cache_J=False
        self.torch_device=torch.device(device)

        self.generator=torch.Generator(device=self.torch_device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self._anneal=None
        self._anneal_key=None

        if compile:
            self._single=torch.compile(_single_kernel,mode="reduce-overhead")
            self._multi=torch.compile(_multi_kernel,mode="reduce-overhead")
        else:
            self._single=_single_kernel
            self._multi=_multi_kernel

    def random(self,shape):
        return self._random(shape)

    def clear_cache(self):
        self._anneal=self._anneal_key=None

    def _get_J(self,c):
        return torch.as_tensor(c.J,dtype=self.dtype,device=self.torch_device)

    def _annealing(self,func):
        key=(id(func),self.Nt,float(self.i0))
        if func is constant and self._anneal is not None and key==self._anneal_key:
            return self._anneal

        a=torch.tensor(
            [float(func(self,run)) for run in range(self.Nt)],
            dtype=self.dtype,device=self.torch_device
        )
        if func is constant:
            self._anneal=a; self._anneal_key=key
        return a

    def _random(self,shape):
        return torch.rand(
            shape,dtype=self.dtype,device=self.torch_device,
            generator=self.generator
        )

    @torch.inference_mode()
    def solve(
        self,c:PCircuit,annealing_func=constant,n_shots=1,
        bias_func=None,return_filtered=False,initial_state=None
    ):
        n=c.n_pbits
        J=self._get_J(c)
        h=torch.as_tensor(
            np.asarray(c.h).reshape(-1),
            dtype=self.dtype,device=self.torch_device
        )
        anneal=self._annealing(annealing_func)
        threshold=float(np.arctanh(self.expected_mean))
        shape=(n,) if n_shots==1 else (n_shots,n)

        if initial_state is None:
            m=torch.sign(.5-self._random(shape))
        else:
            if torch.is_tensor(initial_state):
                state=initial_state.detach().to(self.torch_device,self.dtype)
                if state.numel()!=n:
                    raise ValueError(f"initial_state must contain {n} values")
                state=state.reshape(n)
                valid=bool(torch.all((state==-1)|(state==1)).item())
            else:
                state=np.asarray(initial_state)
                if state.size!=n:
                    raise ValueError(f"initial_state must contain {n} values")
                state=state.reshape(n)
                valid=np.all((state==-1)|(state==1))
                state=torch.as_tensor(state,dtype=self.dtype,device=self.torch_device)

            if not valid:
                raise ValueError("initial_state must contain only -1 or +1")
            m=state.repeat(n_shots,1) if n_shots>1 else state

        # Fast existing path
        if bias_func is None and not return_filtered:
            rnd=self._random((self.Nt,)+shape)
            if n_shots==1:
                I,m,E=self._single(m,J,h,anneal,rnd,self.dt,threshold,self.i0)
                return I.cpu().numpy(),m.cpu().numpy(),E.cpu().numpy()
            return self._multi(
                m,J,h,anneal,rnd,self.dt,threshold
            ).cpu().numpy()

        # Full CaSuDaSolver-compatible path
        if n_shots==1:
            m=m.unsqueeze(0)

        m_filt=torch.zeros_like(m)
        all_m=torch.empty((self.Nt,n_shots,n),dtype=self.dtype,device=self.torch_device)
        all_I=torch.empty((self.Nt,n),dtype=self.dtype,device=self.torch_device)
        all_mfilt=torch.empty_like(all_m)
        E=torch.empty(self.Nt,dtype=self.dtype,device=self.torch_device)

        for run in range(self.Nt):
            h_eff=h if bias_func is None else torch.as_tensor(
                bias_func(run,m,m_filt),dtype=self.dtype,device=self.torch_device
            )
            I=anneal[run]*(m@J+h_eff)
            s=torch.exp(-self.dt*torch.exp(-m*(I+threshold)))
            m=m*torch.sign(s-self._random((n_shots,n)))
            m_filt=(1-self.tau)*m_filt+self.tau*m

            all_m[run]=m
            all_I[run]=I[0]
            all_mfilt[run]=m_filt

            h0=h if bias_func is None else torch.broadcast_to(h_eff,(n_shots,n))[0]
            E[run]=self.i0*(torch.dot(m[0],h0)+.5*torch.dot(m[0]@J,m[0]))

        all_I=all_I.cpu().numpy()
        all_m=all_m.cpu().numpy()
        all_mfilt=all_mfilt.cpu().numpy()
        E=E.cpu().numpy()

        if n_shots==1:
            if return_filtered:
                return all_I,all_m[:,0,:],E,all_mfilt[:,0,:]
            return all_I,all_m[:,0,:],E
        if return_filtered:
            return all_m,all_mfilt
        return all_m

    def copy(self):
        return TorchCaSuDaSolver(
            self.Nt,self.dt,self.i0,self.expected_mean,self.seed,
            self.device,self.dtype,self.compile,False
        )