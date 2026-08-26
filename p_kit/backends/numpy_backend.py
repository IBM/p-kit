import numpy as np
from scipy.sparse import csr_matrix

from .base import Backend

try:
    import numba
except ImportError:
    numba = None


class NumpyBackend(Backend):
    """Default backend. Runs the CSD algorithm on the CPU via NumPy.

    compile=True JIT-compiles whatever pure-NumPy function it's handed via
    numba (used by CaSuDaSolver for its per-timestep update, and - since
    compile_enabled is readable by any caller - also to opt CaSuDaSolver
    into its whole-loop return_final fast path). Like TorchBackend's
    compile=True, this only compiles what it's given - it's the caller's
    job to decide how much of its own loop to hand over.
    """

    supports_sparse = True

    def __init__(self, dtype=None, compile=False):
        self.dtype = dtype
        if compile and numba is None:
            raise ImportError(
                "numba is required for NumpyBackend(compile=True). "
                "Install with: pip install numba"
            )
        self.compile_enabled = compile
        self._compiled_fns = {}

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

    def compile(self, fn):
        if not self.compile_enabled:
            return fn
        # fn is typically a fresh closure per caller (e.g. one built per
        # CaSuDaSolver instance), but closures built from the same source
        # share the same __code__ object - key on that so solvers sharing
        # this backend share one compiled kernel instead of recompiling.
        key = fn.__code__
        if key not in self._compiled_fns:
            self._compiled_fns[key] = numba.njit(cache=True)(fn)
        return self._compiled_fns[key]

    def sparse(self, array):
        return csr_matrix(array)
