import random

class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self._random_gen = random.Random(seed)
    
    def random(self):
        return self._random_gen.random()

    def solve(self, annealing_func):
        raise NotImplementedError()
