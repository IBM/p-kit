"""Module for pipelines."""
import numpy as np
from random import random
from math import tanh, exp

def rand(min, max):
    return random() * (max - min) + min

def S(X):
    return tanh(X)

def sign(X):
    return X / abs(X)

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

    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits, 1))
        self.J = np.zeros((n_pbits, n_pbits))
        self.i0 = 0
        self.I = np.zeros((n_pbits, 1)) + self.i0
        self.M = np.zeros((n_pbits, 1))
    
    def rf(self, t):
        return self.i0 * exp(-t)

    def update_I(self, t):
        for i in range(self.n_pbits):
            self.I[i] = self.rf(t) * (
                            self.h[i] +
                            sum(self.J[i, j] * self.M[j] for j in range(self.n_pbits))
                        )

    def update_M(self): # t?
        for i in range(self.n_pbits):
            self.M[i] = sign(rand(-1, 1) + S(self.I[i]))

    def solve(self, t):
        self.update_I(t)
        self.update_M()
