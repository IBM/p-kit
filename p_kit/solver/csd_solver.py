from p_kit.core.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver
import numpy as np

class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, ‘p-bits for probabilistic spin logic’, Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit, annealing_func=constant):
    
        # credit: https://www.purdue.edu/p-bit/blog.html
        n_pbits = c.n_pbits
        indices = range(n_pbits)

        all_I = [[]] * self.Nt
        all_m = [[]] * self.Nt
        E = [0] * self.Nt

        m = [np.sign(0.5 - self.random()) for _ in indices]

        for run in range(self.Nt):

            # compute input biases
            I = [annealing_func(self, run) * (np.dot(m, c.J[i]) + c.h[i]) for i in indices]
            
            # apply S(input)
            threshold = np.arctanh(self.expected_mean)
            s = [np.exp(-1 * self.dt * np.exp(-1 * m[i] * (I[i] + threshold))) for i in indices]

            # compute new output
            m = [m[i] * np.sign(s[i] - self.random()) for i in indices]
            
            all_I[run] = [_ for _ in I]
            all_m[run] = [_ for _ in m]

            E[run] = self.i0 * (np.dot(m, c.h) + np.multiply(0.5, np.dot(np.dot(m, c.J), m)))

        return np.array(all_I), np.array(all_m), E

    def copy(self):
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean, self.seed)