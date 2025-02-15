from p_kit.core.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver
import numpy as np


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, ‘p-bits for probabilistic spin logic’, Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit, annealing_func=constant):
    
        # credit: https://www.purdue.edu/p-bit/blog.html
        n_pbits = c.n_pbits

        all_I = np.zeros((self.Nt, n_pbits))
        all_m = np.zeros((self.Nt, n_pbits))
        E = np.zeros(self.Nt)

        m = np.sign(0.5 - self.random(n_pbits))

        threshold = np.arctanh(self.expected_mean)

        for run in range(self.Nt):

            # compute input biases
            I = annealing_func(self, run) * (np.dot(m, c.J) + c.h)

            # apply S(input)
            s = np.exp(-self.dt * np.exp(-m * (I + threshold)))

            # compute new output
            m = m * np.sign(s - self.random(n_pbits))
            
            all_I[run] = I
            all_m[run] = m

            E[run] = self.i0 * (np.dot(m, c.h) + 0.5 * np.dot(np.dot(m, c.J), m))

        return all_I, all_m, E

    def copy(self):
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean, self.seed)
