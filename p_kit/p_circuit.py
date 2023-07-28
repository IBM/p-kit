"""Module for pipelines."""
import numpy as np
from random import random

class PCircuit():

    """Create and holds J and h parameters.

    Parameters
    ----------
    n_pbits: string
        Identifier of the pipeline (for log purposes).

    Attributes
    ----------
    h : np.array((n_pbits, 1))
        biases
    J : np.array((n_pbits, n_pbits))
        weights

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def __init__(self, n_pbits, i0):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits, 1))
        self.J = np.zeros((n_pbits, n_pbits))
        self.i0 = i0
    
    def solve(self, Nt, dt):
        # credit: https://www.purdue.edu/p-bit/blog.html
        n_pbits = self.n_pbits
        indices = range(n_pbits)

        all_m = [[]] * Nt

        I = [0] * n_pbits
        s = [0] * n_pbits
        m = [np.sign(0.5 - random()) for _ in indices]
        
        
        for run in range(Nt):

            # compute input biases
            I = [-1 * self.i0 * (np.dot(m, self.J[i]) + self.h[i]) for i in indices]
            
            # apply S(input)
            s = [np.exp(-1 * dt * np.exp(-1 * m[i] * I[i])) for i in indices]

            # compute new output
            m = [m[i] * np.sign(s[i] - random()) for i in indices]
            
            all_m[run] = [_ for _ in m]

        return all_m
                    