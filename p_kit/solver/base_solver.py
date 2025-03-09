import random
import time
import numpy as np

class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self.seed = seed
        self._random_gen = np.random.default_rng(self.seed)
    
    def random(self, n_pbits):
        return self._random_gen.random(n_pbits)

    def solve(self, annealing_func):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
