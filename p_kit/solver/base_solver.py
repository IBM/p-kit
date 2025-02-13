import random
import time

class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        # self.seed = time.time() if seed is None else seed
        self.seed = seed
        self._random_gen = random.Random(seed)
    
    def random(self):
        return self._random_gen.random()

    def solve(self, annealing_func):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
