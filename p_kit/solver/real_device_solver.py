"""
Solver for physical p-bit devices.

For example, it can be used with the physical probabilistic computer Probana:
https://github.com/toncho11/probana

`RealDeviceSolver` connects p-kit's `PCircuit` interface to a physical-device backend. 
The solver passes `J` and `h` to the backend and returns the sampled states in the 
standard p-kit solver format.

The Probana backend handles USB communication, while Probana performs the stochastic 
p-bit updates, calibration correction, and sampling locally on the board.

"""

import numpy as np

from .base_solver import Solver
from .annealing import constant


class RealDeviceSolver(Solver):
    def __init__(self, Nt, backend, i0=1.0, burn_in=100, thin=1):
        if not getattr(backend, "supports_native_pbits", False):
            raise TypeError("Backend does not support physical p-bits")
        super().__init__(Nt=Nt, dt=1.0, i0=i0, backend=backend)
        self.burn_in, self.thin = int(burn_in), int(thin)

    def solve(self, c, annealing_func=constant, n_shots=1,
              bias_func=None, return_filtered=False,
              initial_state=None, return_final=False):

        if n_shots != 1:
            raise NotImplementedError("Physical backend currently supports n_shots=1")
        if annealing_func is not constant:
            raise NotImplementedError("Dynamic annealing is not supported")
        if bias_func is not None or return_filtered or initial_state is not None:
            raise NotImplementedError("Dynamic bias, filtering and initial_state are not supported")

        J = np.asarray(c.J, dtype=float)
        h = np.asarray(c.h, dtype=float).reshape(-1)

        # i0 is applied before upload because the update loop runs on the board.
        m = self.backend.run_circuit(
            self.i0 * J, self.i0 * h,
            samples=self.Nt, burn_in=self.burn_in, thin=self.thin
        )

        if return_final:
            return m[-1]

        I = self.i0 * (m @ J + h)
        E = self.i0 * (
            m @ h + 0.5 * np.einsum("bi,ij,bj->b", m, J, m)
        )
        return I, m, E

    def copy(self):
        return RealDeviceSolver(
            Nt=self.Nt, backend=self.backend, i0=self.i0,
            burn_in=self.burn_in, thin=self.thin
        )