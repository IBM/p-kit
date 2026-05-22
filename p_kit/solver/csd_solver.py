from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver
import numpy as np


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, 'p-bits for probabilistic spin logic', Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit, annealing_func=constant):

        # credit: https://www.purdue.edu/p-bit/blog.html
        xp = self.xp
        n_pbits = c.n_pbits

        J = xp.asarray(c.J)
        h = xp.asarray(c.h)

        all_I = xp.zeros((self.Nt, n_pbits))
        all_m = xp.zeros((self.Nt, n_pbits))
        E = xp.zeros(self.Nt)

        m = xp.sign(0.5 - self.random(n_pbits))

        threshold = float(np.arctanh(self.expected_mean))

        for run in range(self.Nt):

            # compute input biases
            I = annealing_func(self, run) * (xp.dot(m, J) + h)

            # apply S(input)
            s = xp.exp(-self.dt * xp.exp(-m * (I + threshold)))

            # compute new output
            m = m * xp.sign(s - self.random(n_pbits))

            all_I[run] = I
            all_m[run] = m

            E[run] = self.i0 * (xp.dot(m, h) + 0.5 * xp.dot(xp.dot(m, J), m))

        if self.device == 'cuda':
            return all_I.get(), all_m.get(), E.get()
        return all_I, all_m, E

    def copy(self):
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean, self.seed, self.device)
