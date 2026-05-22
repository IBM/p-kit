from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver
import numpy as np


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, 'p-bits for probabilistic spin logic', Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1):

        # credit: https://www.purdue.edu/p-bit/blog.html
        xp = self.xp
        n_pbits = c.n_pbits

        J = xp.asarray(c.J)
        h = xp.asarray(c.h)
        threshold = float(np.arctanh(self.expected_mean))

        # m is (n_shots, n_pbits) — works for n_shots=1 too
        all_m = xp.zeros((self.Nt, n_shots, n_pbits))
        all_I = xp.zeros((self.Nt, n_pbits))
        E = xp.zeros(self.Nt)
        m = xp.sign(0.5 - self.random((n_shots, n_pbits)))

        for run in range(self.Nt):
            I = annealing_func(self, run) * (m @ J + h)
            s = xp.exp(-self.dt * xp.exp(-m * (I + threshold)))
            m = m * xp.sign(s - self.random((n_shots, n_pbits)))
            all_m[run] = m
            all_I[run] = I[0]
            E[run] = self.i0 * (xp.dot(m[0], h) + 0.5 * xp.dot(xp.dot(m[0], J), m[0]))

        if self.device == 'cuda':
            all_I, all_m, E = all_I.get(), all_m.get(), E.get()

        if n_shots == 1:
            return all_I, all_m[:, 0, :], E
        return all_m

    def copy(self):
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean, self.seed, self.device)
