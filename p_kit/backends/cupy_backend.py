from .base import Backend

try:
    import cupy as cp
except ImportError:
    cp = None


class CupyBackend(Backend):
    """Runs the CSD algorithm on a CUDA GPU via CuPy."""

    prefers_vectorized_shots = True

    def __init__(self, dtype=None):
        if cp is None:
            raise ImportError(
                "cupy is required for CupyBackend. "
                "Install with: pip install cupy-cuda13x"
            )
        self.dtype = dtype

    @property
    def xp(self):
        return cp

    def asarray(self, array, dtype=None):
        return cp.asarray(array, dtype=dtype if dtype is not None else self.dtype)

    def zeros(self, shape, dtype=None):
        return cp.zeros(shape, dtype=dtype if dtype is not None else self.dtype)

    def make_generator(self, seed):
        return cp.random.default_rng(seed)

    def random(self, generator, shape, dtype=None):
        return generator.random(shape)

    def to_numpy(self, array):
        return array.get()
