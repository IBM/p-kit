from p_kit.core.p_circuit import PCircuit
from .base_solver import Solver
from random import random
import numpy as np

class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, ‘p-bits for probabilistic spin logic’, Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit):
        # credit: https://www.purdue.edu/p-bit/blog.html
        n_pbits = c.n_pbits
        indices = range(n_pbits)

        all_I = [[]] * self.Nt
        all_m = [[]] * self.Nt

        I = [0] * n_pbits
        s = [0] * n_pbits
        m = [np.sign(0.5 - random()) for _ in indices]
        
        
        for run in range(self.Nt):

            # compute input biases
            I = [-1 * self.i0 * (np.dot(m, c.J[i]) + c.h[i]) for i in indices]
            
            # apply S(input)
            s = [np.exp(-1 * self.dt * np.exp(-1 * m[i] * I[i])) for i in indices]

            # compute new output
            m = [m[i] * np.sign(s[i] - random()) for i in indices]
            
            all_I[run] = [_ for _ in I]
            all_m[run] = [_ for _ in m]

        return np.array(all_I), np.array(all_m)