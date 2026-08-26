from .base import Backend

import torch


class TorchBackend(Backend):
    """Runs the CSD algorithm through PyTorch - CPU, CUDA, or another
    accelerator PyTorch supports (e.g. device="spyre" via torch-spyre).

    NOTES:
      - cache_J on the solver improves performance when J stays fixed; this
        backend just supplies the asarray() it's built on.
      - compile=True JIT-compiles the per-timestep update (see
        CaSuDaSolver._step) via torch.compile. It does not compile the whole
        Nt-step Python loop, so it won't fuse away per-iteration Python
        overhead the way a hand-written whole-loop kernel would - it mainly
        pays off for larger n_pbits/n_shots where the elementwise math
        dominates.
      - PyTorch and NumPy/CuPy use different random generators, so
        trajectories and final results are not expected to be identical
        across backends even with the same seed.
    """

    prefers_vectorized_shots = True

    def __init__(self, device="cpu", dtype=torch.float32, compile=False):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device == "spyre":
            try:
                import torch_spyre  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "torch-spyre is required for device='spyre'"
                ) from e

        self.torch_device = torch.device(device)
        self.dtype = dtype
        self._compile = compile
        self._compiled_fns = {}

        # Some accelerator devices don't support torch.Generator directly;
        # draw randomness on CPU/CUDA and move it over.
        self.rng_device = (
            self.torch_device
            if self.torch_device.type in ("cpu", "cuda")
            else torch.device("cpu")
        )

    @property
    def xp(self):
        return torch

    def asarray(self, array, dtype=None):
        return torch.as_tensor(
            array, dtype=dtype if dtype is not None else self.dtype,
            device=self.torch_device,
        )

    def zeros(self, shape, dtype=None):
        return torch.zeros(
            shape, dtype=dtype if dtype is not None else self.dtype,
            device=self.torch_device,
        )

    def make_generator(self, seed):
        generator = torch.Generator(device=self.rng_device)
        if seed is not None:
            generator.manual_seed(seed)
        return generator

    def random(self, generator, shape, dtype=None):
        x = torch.rand(
            shape,
            dtype=dtype if dtype is not None else self.dtype,
            device=self.rng_device,
            generator=generator,
        )
        if self.rng_device != self.torch_device:
            x = x.to(self.torch_device)
        return x

    def to_numpy(self, array):
        return array.cpu().numpy()

    def no_grad(self):
        return torch.inference_mode()

    def compile(self, fn):
        if not self._compile:
            return fn
        # fn is typically a fresh closure per caller (e.g. one built per
        # CaSuDaSolver instance), but closures built from the same source
        # share the same __code__ object - key on that so solvers sharing
        # this backend share one compiled kernel instead of recompiling.
        key = fn.__code__
        if key not in self._compiled_fns:
            self._compiled_fns[key] = torch.compile(fn, mode="reduce-overhead")
        return self._compiled_fns[key]
