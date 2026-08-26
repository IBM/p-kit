import numpy as np

from .base import Backend


class NumpyBackend(Backend):
    """Default backend. Runs the CSD algorithm on the CPU via NumPy."""

    def __init__(self, dtype=None):
        self.dtype = dtype

    @property
    def xp(self):
        return np

    def asarray(self, array, dtype=None):
        return np.asarray(array, dtype=dtype if dtype is not None else self.dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype if dtype is not None else self.dtype)

    def make_generator(self, seed):
        return np.random.default_rng(seed)

    def random(self, generator, shape, dtype=None):
        return generator.random(shape)

    def to_numpy(self, array):
        return array
