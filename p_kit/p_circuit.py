"""Module for pipelines."""
import numpy as np
from random import random
from math import tanh

def rand(min, max):
    return random() * (max - min) - min

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
    .. versionadded:: 0.1.0

    """

    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits,1))
        self.J = np.zeros((n_pbits, n_pbits))
        self.i0 = 0
    
    def i(self, pbit, t):
        return self.i0 * (self.h[pbit] + sum(self.J[pbit, j] * self.m(pbit, t) for j in range(self.n_pbits) if not j == pbit))

    def m(self, pbit, t):
        return sign(rand(-1, 1) + S(self.i(pbit, t)))
