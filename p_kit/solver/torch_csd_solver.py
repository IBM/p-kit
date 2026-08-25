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
  - GPU/accelerator benefits are expected mainly for larger workloads.
  - cache_J=True improves performance when J remains fixed.
  - If c.J is modified while cache_J=True, call solver.clear_cache().
  - compile=True may improve accelerator performance but adds compilation overhead.
  - device="spyre" requires torch-spyre and access to compatible Spyre hardware/runtime.
  - PyTorch and NumPy solvers use different random generators, so stochastic
    trajectories and final results are not expected to be identical.

"""

import numpy as np
import torch

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


def _single_kernel(m, J, h, anneal, rnd, dt, threshold, i0):
    Nt, n_pbits = rnd.shape
    all_m = torch.empty((Nt, n_pbits), dtype=m.dtype, device=m.device)
    all_I = torch.empty_like(all_m)
    E = torch.empty(Nt, dtype=m.dtype, device=m.device)

    for run in range(Nt):
        field = m @ J

        if run:
            E[run-1] = i0 * (
                torch.dot(m, h) + 0.5 * torch.dot(field, m)
            )

        I = anneal[run] * (field + h)
        s = torch.exp(-dt * torch.exp(-m * (I + threshold)))
        m = m * torch.sign(s - rnd[run])

        all_m[run] = m
        all_I[run] = I

    field = m @ J
    E[-1] = i0 * (
        torch.dot(m, h) + 0.5 * torch.dot(field, m)
    )

    return all_I, all_m, E


def _multi_kernel(m, J, h, anneal, rnd, dt, threshold):
    Nt, n_shots, n_pbits = rnd.shape
    all_m = torch.empty(
        (Nt, n_shots, n_pbits),
        dtype=m.dtype, device=m.device
    )

    for run in range(Nt):
        I = anneal[run] * (m @ J + h)
        s = torch.exp(-dt * torch.exp(-m * (I + threshold)))
        m = m * torch.sign(s - rnd[run])
        all_m[run] = m

    return all_m


class TorchCaSuDaSolver(Solver):

    def __init__(
        self, Nt, dt, i0, expected_mean=0, seed=None,
        device="cpu", dtype=torch.float32,
        compile=False, cache_J=False,
    ):
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.cache_J = cache_J

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device == "spyre":
            try:
                import torch_spyre
            except ImportError as e:
                raise ImportError(
                    "torch-spyre is required for device='spyre'"
                ) from e

        self.torch_device = torch.device(device)
        self.rng_device = (
            self.torch_device
            if self.torch_device.type in ("cpu", "cuda")
            else torch.device("cpu")
        )

        self.generator = torch.Generator(device=self.rng_device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self._J = None
        self._J_key = None
        self._anneal = None
        self._anneal_key = None

        if compile:
            self._single = torch.compile(
                _single_kernel, mode="reduce-overhead"
            )
            self._multi = torch.compile(
                _multi_kernel, mode="reduce-overhead"
            )
        else:
            self._single = _single_kernel
            self._multi = _multi_kernel

    def clear_cache(self):
        self._J = self._J_key = None

    def _get_J(self, c):
        if not self.cache_J:
            return torch.as_tensor(
                c.J, dtype=self.dtype, device=self.torch_device
            )

        key = (id(c), id(c.J), c.J.shape)

        if self._J is None or key != self._J_key:
            self._J = torch.as_tensor(
                c.J, dtype=self.dtype, device=self.torch_device
            ).contiguous()
            self._J_key = key

        return self._J

    def _annealing(self, annealing_func):
        key = (id(annealing_func), self.Nt, float(self.i0))

        if (
            annealing_func is constant
            and self._anneal is not None
            and key == self._anneal_key
        ):
            return self._anneal

        values = [
            float(annealing_func(self, run))
            for run in range(self.Nt)
        ]

        anneal = torch.tensor(
            values, dtype=self.dtype, device=self.torch_device
        )

        if annealing_func is constant:
            self._anneal = anneal
            self._anneal_key = key

        return anneal

    def _random(self, shape):
        x = torch.rand(
            shape,
            dtype=self.dtype,
            device=self.rng_device,
            generator=self.generator,
        )

        if self.rng_device != self.torch_device:
            x = x.to(self.torch_device)

        return x

    @torch.inference_mode()
    def solve(
        self, c: PCircuit, annealing_func=constant,
        n_shots=1, initial_state=None,
    ):
        n_pbits = c.n_pbits
        J = self._get_J(c)
        h = torch.as_tensor(
            np.asarray(c.h).reshape(-1),
            dtype=self.dtype,
            device=self.torch_device,
        )

        anneal = self._annealing(annealing_func)
        threshold = float(np.arctanh(self.expected_mean))
        shape = (n_pbits,) if n_shots == 1 else (n_shots, n_pbits)

        if initial_state is None:
            m = torch.sign(0.5 - self._random(shape))
        else:
            state = np.asarray(initial_state)

            if state.size != n_pbits:
                raise ValueError(
                    f"initial_state must contain {n_pbits} values"
                )

            state = state.reshape(n_pbits)

            if not np.all((state == -1) | (state == 1)):
                raise ValueError(
                    "initial_state must contain only -1 or +1"
                )

            m = torch.as_tensor(
                state, dtype=self.dtype, device=self.torch_device
            )

            if n_shots > 1:
                m = m.repeat(n_shots, 1)

        rnd = self._random((self.Nt,) + shape)

        if n_shots == 1:
            all_I, all_m, E = self._single(
                m, J, h, anneal, rnd,
                self.dt, threshold, self.i0,
            )

            return (
                all_I.cpu().numpy(),
                all_m.cpu().numpy(),
                E.cpu().numpy(),
            )

        return self._multi(
            m, J, h, anneal, rnd,
            self.dt, threshold,
        ).cpu().numpy()

    def copy(self):
        return TorchCaSuDaSolver(
            self.Nt, self.dt, self.i0,
            self.expected_mean, self.seed,
            self.device, self.dtype,
            self.compile, self.cache_J,
        )