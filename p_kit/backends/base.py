from abc import ABC, abstractmethod
from contextlib import nullcontext


class Backend(ABC):
    """Adapter between the CSD solver algorithm and a specific array
    library/device (NumPy, CuPy, PyTorch, ...).

    A Backend owns everything that varies across array libraries: array
    creation, RNG, host transfer, and any device-specific execution hooks
    (autograd disabling, JIT compilation, sparse representations, scratch
    memory). It knows nothing about the CSD dynamics themselves - those live
    once in CaSuDaSolver, which only calls `backend.xp.<fn>` (matmul via
    `@`, `exp`, `sign`, `tile`, `broadcast_to`, which NumPy/CuPy/Torch all
    agree on) and the methods below.
    """

    # Whether a single batched (n_shots, n_pbits) call is faster than
    # dispatching n_shots independent solves across worker processes.
    # True for backends that are already device-parallel (CuPy/Torch on any
    # device) - joblib process-based parallelism would just add pickling
    # overhead (and can't pickle a CUDA context or a compiled kernel at all).
    prefers_vectorized_shots = False

    # Whether sparse(...) is implemented. NumPy/SciPy is the only backend
    # that currently supports it - CuPy and Torch have their own, different
    # sparse APIs this doesn't attempt to unify.
    supports_sparse = False

    # Whether this backend's compile() was asked to JIT-compile things
    # (each backend that supports compile= sets this in its own __init__).
    # A readable capability flag, not just an internal switch, so a caller
    # like CaSuDaSolver can key its own optional fast paths off of "did the
    # user ask this backend to accelerate things" without needing to know
    # which specific JIT library a given backend uses under the hood.
    compile_enabled = False

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
        """Optionally JIT-compile fn (a pure array-in-array-out callable
        this backend's `xp` namespace, closed over by the caller so it
        needs no per-call `xp` argument). Identity by default."""
        return fn

    def sparse(self, array):
        """Convert `array` to this backend's sparse representation, if it
        has one - see `supports_sparse`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support sparse arrays"
        )

    def buffer(self, name, shape, dtype=None):
        """Get (and cache) a scratch array of `shape`, keyed by `name`. A
        plain memory pool with no algorithm awareness - callers decide
        whether/when reuse is safe (e.g. only across calls that don't alias
        the buffer across overlapping lifetimes); this just avoids
        reallocating when they do."""
        cache = self.__dict__.setdefault("_buffer_cache", {})
        a = cache.get(name)
        if a is None or a.shape != shape:
            a = self.zeros(shape, dtype)
            cache[name] = a
        return a
