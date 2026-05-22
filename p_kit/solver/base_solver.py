import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, device='cpu') -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self.seed = seed
        self.device = device
        if device == 'cuda':
            if cp is None:
                raise ImportError("cupy is required for device='cuda'. Install with: pip install cupy-cuda13x")
            self.xp = cp
        else:
            self.xp = np
        self._random_gen = self.xp.random.default_rng(self.seed)

    def random(self, n_pbits):
        return self._random_gen.random(n_pbits)

    def solve(self, annealing_func):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()
