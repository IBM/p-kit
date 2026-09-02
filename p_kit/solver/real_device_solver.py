"""
Solver for physical p-bit devices.

Connects p-kit's PCircuit interface to physical-device backends such as:
https://github.com/toncho11/probana

The backend executes calibration, p-bit updates, annealing and sampling
locally. Multiple shots run sequentially to avoid concurrent USB access.

Backends should support these attributes: 
  * `supports_native_pbits` indicates a backend designed for physical p-bits
  * `supports_pkit_annealing_func` indicates support for arbitrary p-kit annealing schedules.

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
        self.burn_in, self.thin = int(burn_in), int(thin)

    def _annealing(self, func):
        if func is None or func is constant:
            return "constant", self.i0

        if getattr(self.backend, "supports_pkit_annealing_func", False):
            values = np.asarray(
                [func(self, run) for run in range(self.Nt)], dtype=float
            )
            if values.shape != (self.Nt,):
                raise ValueError("annealing_func must return one scalar per timestep")
            if not np.all(np.isfinite(values)):
                raise ValueError("annealing_func returned a non-finite value")
            return "table", values

        if func is linear:
            start, end = float(func(self, 0)), float(func(self, self.Nt - 1))
            steps = max(1, self.burn_in + self.Nt * self.thin)
            return "linear", start, end, steps

        raise NotImplementedError(
            f"{type(self.backend).__name__} does not support arbitrary "
            "p-kit annealing functions"
        )

    def _sample_scales(self, annealing):
        if annealing[0] == "constant":
            return np.full(self.Nt, annealing[1], dtype=float)
        if annealing[0] == "table":
            return np.asarray(annealing[1], dtype=float)

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

        J = np.asarray(c.J, dtype=float)
        h = np.asarray(c.h, dtype=float).reshape(-1)
        annealing = self._annealing(annealing_func)

        shots = [
            self.backend.run_circuit(
                J, h, samples=self.Nt, burn_in=self.burn_in,
                thin=self.thin, annealing=annealing
            )
            for _ in range(n_shots)
        ]

        all_m = np.stack(shots, axis=1)

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
