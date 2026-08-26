import numpy as np

from .base import Backend

try:
    import numba
except ImportError:
    numba = None


def _csd_step_numba_impl(m, J, h_eff, I_scale, threshold, dt, rnd):
    """Same update as csd_solver._csd_step, but numba-compilable.

    numba's nopython mode can't type a module (like `xp`) passed as a
    runtime argument, so this hardcodes `np` as a global instead of taking
    an array-library namespace parameter - unlike the generic `_csd_step`
    it mirrors, which every backend's eager path shares.
    """
    field = m @ J
    I = I_scale * (field + h_eff)
    s = np.exp(-dt * np.exp(-m * (I + threshold)))
    m = m * np.sign(s - rnd)
    return m, I


class NumpyBackend(Backend):
    """Default backend. Runs the CSD algorithm on the CPU via NumPy.

    compile=True JIT-compiles the per-timestep update (see
    CaSuDaSolver._step) via numba, instead of running it as plain NumPy.
    Like TorchBackend's compile=True, this only compiles one step, not the
    whole Nt-step Python loop, so it won't fuse away per-iteration Python
    overhead - it mainly pays off when the elementwise math dominates
    (larger n_pbits/n_shots).
    """

    def __init__(self, dtype=None, compile=False):
        self.dtype = dtype
        if compile and numba is None:
            raise ImportError(
                "numba is required for NumpyBackend(compile=True). "
                "Install with: pip install numba"
            )
        self._compile = compile
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
        if not self._compile:
            return fn
        if fn not in self._compiled_fns:
            jitted = numba.njit(cache=True)(_csd_step_numba_impl)

            def wrapper(xp, m, J, h_eff, I_scale, threshold, dt, rnd):
                return jitted(m, J, h_eff, I_scale, threshold, dt, rnd)

            self._compiled_fns[fn] = wrapper
        return self._compiled_fns[fn]
