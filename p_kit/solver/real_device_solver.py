"""
Solver for physical p-bit devices.

For example, it can be used with the physical probabilistic computer Probana:
https://github.com/toncho11/probana

RealDeviceSolver connects p-kit's PCircuit interface to a physical-device
backend. The solver passes J and h to the backend and returns sampled states
in the standard p-kit solver format.

The Probana backend handles USB communication, while Probana performs the
stochastic p-bit updates, calibration correction, annealing, and sampling
locally on the board.

Multiple shots are executed sequentially on the physical device, avoiding
parallel access to a single USB connection.
"""

import numpy as np

from .base_solver import Solver
from .annealing import constant, linear


class RealDeviceSolver(Solver):
    def __init__(self, Nt, backend, i0=1.0, burn_in=100, thin=1):
        if not getattr(backend, "supports_native_pbits", False):
            raise TypeError("Backend does not support physical p-bits")
        if burn_in < 0 or thin < 1:
            raise ValueError("burn_in must be >= 0 and thin must be >= 1")
        
        super().__init__(Nt=Nt, dt=1.0, i0=i0, backend=backend)
        self.burn_in = int(burn_in)
        self.thin = int(thin)

    def _annealing(self, func):
        if func is None or func is constant:
            return ("constant", self.i0)

        if func is linear:
            start = float(func(self, 0))
            end = float(func(self, self.Nt - 1))
            steps = self.burn_in + self.Nt * self.thin
            return ("linear", start, end, max(1, steps))

        raise NotImplementedError(
            "RealDeviceSolver currently supports constant and linear annealing"
        )

    def _sample_scales(self, annealing):
        if annealing[0] == "constant":
            return np.full(self.Nt, annealing[1], dtype=float)

        _, start, end, steps = annealing
        idx = self.burn_in + (np.arange(self.Nt) + 1) * self.thin - 1
        x = np.minimum(1.0, idx / max(1, steps - 1))
        return start + x * (end - start)

    def solve(self, c, annealing_func=constant, n_shots=1,
              bias_func=None, return_filtered=False,
              initial_state=None, return_final=False):

        if n_shots < 1:
            raise ValueError("n_shots must be >= 1")

        if bias_func is not None or return_filtered or initial_state is not None:
            raise NotImplementedError(
                "Dynamic bias, filtering and initial_state are not supported"
            )
            
        if (annealing_func not in (None, constant) and not getattr(self.backend, "supports_pkit_annealing_func", True)):
            raise NotImplementedError(
                "This p-kit annealing schedule cannot be mapped directly to this "
                "hardware backend. Support for a custom annealing_func is disabled."
            )

        J = np.asarray(c.J, dtype=float)
        h = np.asarray(c.h, dtype=float).reshape(-1)
        annealing = self._annealing(annealing_func)

        shots = [
            self.backend.run_circuit(
                J, h, samples=self.Nt,
                burn_in=self.burn_in, thin=self.thin,
                annealing=annealing
            )
            for _ in range(n_shots)
        ]

        all_m = np.stack(shots, axis=1)  # (Nt, n_shots, n_pbits)

        if return_final:
            return all_m[-1, 0] if n_shots == 1 else all_m[-1]

        if n_shots > 1:
            return all_m

        m = all_m[:, 0]
        scale = self._sample_scales(annealing)
        I = scale[:, None] * (m @ J + h)

        E = self.i0 * (
            m @ h + 0.5 * np.einsum("bi,ij,bj->b", m, J, m)
        )

        return I, m, E

    def copy(self):
        return RealDeviceSolver(
            Nt=self.Nt, backend=self.backend, i0=self.i0,
            burn_in=self.burn_in, thin=self.thin
        )