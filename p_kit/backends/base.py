from abc import ABC, abstractmethod
from contextlib import nullcontext


class Backend(ABC):
    """Adapter between the CSD solver algorithm and a specific array
    library/device (NumPy, CuPy, PyTorch, ...).

    A Backend owns everything that varies across array libraries: array
    creation, RNG, host transfer, and any device-specific execution hooks
    (autograd disabling, JIT compilation). The CSD dynamics themselves live
    once in CaSuDaSolver and never branch on backend type - they only call
    `backend.xp.<fn>` (matmul via `@`, `exp`, `sign`, `tile`, `broadcast_to`,
    which NumPy/CuPy/Torch all agree on) and the methods below.
    """

    # Whether a single batched (n_shots, n_pbits) call is faster than
    # dispatching n_shots independent solves across worker processes.
    # True for backends that are already device-parallel (CuPy/Torch on any
    # device) - joblib process-based parallelism would just add pickling
    # overhead (and can't pickle a CUDA context or a compiled kernel at all).
    prefers_vectorized_shots = False

    @property
    @abstractmethod
    def xp(self):
        """The array-library namespace (numpy, cupy, torch, ...)."""

    @abstractmethod
    def asarray(self, array, dtype=None):
        """Host array/circuit data -> backend-native array."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        ...

    @abstractmethod
    def make_generator(self, seed):
        """Create a backend-native RNG generator seeded for one solver
        instance. RNG state belongs to the Solver, not the Backend, so a
        Backend instance can be shared across solver copies."""

    @abstractmethod
    def random(self, generator, shape, dtype=None):
        """Draw uniform [0, 1) samples of `shape` from `generator`."""

    @abstractmethod
    def to_numpy(self, array):
        """Backend-native array -> numpy array (identity for NumPy)."""

    def no_grad(self):
        """Context manager to disable autograd tracking, if applicable."""
        return nullcontext()

    def compile(self, fn):
        """Optionally JIT-compile a hot-path function. Identity by default."""
        return fn
