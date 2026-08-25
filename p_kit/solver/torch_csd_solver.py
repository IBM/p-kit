"""

This is a version of CaSuDaSolver using PyTorch. 

TorchCaSuDaSolver can allow an easy switch between CPU, GPU and for example
IBM Spyre AI accelerator chip.

From one hand it is good to have an alternative solver, but it also adds the
heavy dependency of PyTorch. And it some respects the idea was not to use 
PyTorch.


"""

import numpy as np
import torch

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


class TorchCaSuDaSolver(Solver):
    """PyTorch implementation of CaSuDaSolver."""

    def __init__(
        self, Nt, dt, i0, expected_mean=0, seed=None,
        device="cpu", dtype=torch.float32, compile=False
    ):
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self.seed = seed
        self.device = device
        self.dtype = dtype

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device == "spyre":
            try:
                import torch_spyre
            except ImportError as e:
                raise ImportError("torch-spyre is required for device='spyre'") from e

        self.torch_device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)

        self._step = torch.compile(self._torch_step) if compile else self._torch_step

    @staticmethod
    def _torch_step(m, J, h, anneal, dt, threshold, rnd):
        I = anneal * (m @ J + h)
        s = torch.exp(-dt * torch.exp(-m * (I + threshold)))
        m = torch.where(rnd < s, m, -m)
        return m, I

    @torch.no_grad()
    def solve(
        self, c: PCircuit, annealing_func=constant,
        n_shots=1, initial_state=None
    ):
        n_pbits = c.n_pbits
        device = self.torch_device

        J = torch.as_tensor(c.J, dtype=self.dtype, device=device)
        h = torch.as_tensor(np.asarray(c.h).reshape(-1),
                            dtype=self.dtype, device=device)
        threshold = float(np.arctanh(self.expected_mean))

        # Generate randomness on CPU once, then transfer once.
        rnd = torch.rand(
            (self.Nt + 1, n_shots, n_pbits),
            generator=self.generator, dtype=self.dtype
        ).to(device)

        if initial_state is None:
            m = torch.where(
                rnd[0] < 0.5,
                torch.ones((), dtype=self.dtype, device=device),
                -torch.ones((), dtype=self.dtype, device=device),
            )
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
            state = torch.as_tensor(state, dtype=self.dtype, device=device)
            m = state.unsqueeze(0).repeat(n_shots, 1)

        all_m = torch.empty(
            (self.Nt, n_shots, n_pbits),
            dtype=self.dtype, device=device
        )
        all_I = torch.empty(
            (self.Nt, n_pbits),
            dtype=self.dtype, device=device
        )
        E = torch.empty(self.Nt, dtype=self.dtype, device=device)

        for run in range(self.Nt):
            anneal = float(annealing_func(self, run))

            m, I = self._step(
                m, J, h, anneal,
                self.dt, threshold, rnd[run + 1]
            )

            all_m[run] = m
            all_I[run] = I[0]
            E[run] = self.i0 * (
                torch.dot(m[0], h)
                + 0.5 * torch.dot(m[0] @ J, m[0])
            )

        all_I = all_I.cpu().numpy()
        all_m = all_m.cpu().numpy()
        E = E.cpu().numpy()

        if n_shots == 1:
            return all_I, all_m[:, 0, :], E
        return all_m

    def copy(self):
        return TorchCaSuDaSolver(
            self.Nt, self.dt, self.i0,
            self.expected_mean, self.seed,
            self.device, self.dtype
        )