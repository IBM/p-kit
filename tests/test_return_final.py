"""Tests for CaSuDaSolver.solve(return_final=True).

Covers the return-final fast path: skip trajectory/current/energy storage
and return only the final +/-1 state. This used to live only on the
separate CaSuDaOptimized class; it's now a first-class, backend-agnostic
solve() option.

use_sparse/reuse_buffers/cache_static (also formerly CaSuDaOptimized-only)
are CaSuDaSolver constructor options. use_numba is not a separate option -
it's inferred from the backend: `solver.use_numba` is True exactly when the
backend is NumpyBackend(compile=True), since numba only understands plain
NumPy arrays and that's the same "please accelerate things" signal
compile=True already means for the per-step path. Any other backend (or
compile=False) leaves it False and solve(return_final=True) falls back to
the generic storage-skipping loop, or to the eager (non-JIT) path if
use_sparse=True was set without triggering use_numba.
"""
import numpy as np
import pytest
from p_kit.psl.gates import ANDGate
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.backends import NumpyBackend


# ── Default (NumpyBackend, generic storage-skip) ────────────────────────────

def test_return_final_default_backend_single_shot():
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    out = solver.solve(gate, return_final=True)

    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


def test_return_final_default_backend_multi_shot():
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    out = solver.solve(gate, n_shots=4, return_final=True)

    assert out.shape == (4, 3)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


def test_return_final_matches_last_trajectory_state():
    """return_final's final state matches the last row of a full-trajectory
    solve() with the same seed - it's a storage optimization, not a
    different algorithm."""
    gate = ANDGate()
    _, all_m, _ = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=7).solve(gate.copy())
    final = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=7).solve(
        gate.copy(), return_final=True
    )
    assert np.array_equal(all_m[-1], final)


def test_return_final_and_return_filtered_conflict():
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=10, dt=0.1667, i0=0.9, seed=1)
    with pytest.raises(ValueError):
        solver.solve(gate, return_final=True, return_filtered=True)


def test_return_final_with_bias_func():
    """The generic storage-skip path still honors bias_func."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.9, seed=3)

    def force_negative(run, m, m_filt):
        return np.full(m.shape, -10.0)

    out = solver.solve(gate, bias_func=force_negative, return_final=True)
    assert np.all(out < 0)


def test_use_numba_defaults_false():
    """No compile=True on the backend -> no whole-loop fast path, even
    though NumpyBackend is otherwise capable of it."""
    solver = CaSuDaSolver(Nt=10, dt=0.1667, i0=0.9, backend=NumpyBackend())
    assert solver.use_numba is False


# ── use_numba (inferred from NumpyBackend(compile=True)) / use_sparse ───────

numba = pytest.importorskip("numba")


def test_use_numba_inferred_from_compile_enabled_backend():
    solver = CaSuDaSolver(Nt=10, dt=0.1667, i0=0.9, backend=NumpyBackend(compile=True))
    assert solver.use_numba is True


@pytest.mark.parametrize("use_sparse", [False, True])
def test_return_final_numba(use_sparse):
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=200, dt=0.1667, i0=0.9, seed=42,
        backend=NumpyBackend(compile=True), use_sparse=use_sparse,
    )
    out = solver.solve(gate, return_final=True)

    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


def test_return_final_sparse_only_eager_path():
    """use_sparse without a compile-enabled backend exercises
    _solve_final_fast's eager (non-JIT) sparse branch."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.9, seed=5, use_sparse=True)
    assert solver.use_numba is False
    out = solver.solve(gate, return_final=True)

    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


def test_return_final_numba_bias_func_falls_back_to_eager():
    """bias_func disables the numba whole-loop kernel (it can't call back
    into Python), so this exercises _solve_final_fast's eager branch even
    with use_numba=True."""
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=100, dt=0.1667, i0=0.9, seed=6,
        backend=NumpyBackend(compile=True), reuse_buffers=True, cache_static=True,
    )

    def force_positive(run, m, m_filt):
        return np.full(m.shape, 10.0)

    out = solver.solve(gate, bias_func=force_positive, return_final=True)
    assert np.all(out > 0)


def test_return_final_reuse_buffers_and_cache_static():
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=50, dt=0.1667, i0=0.9, seed=9,
        backend=NumpyBackend(compile=True), use_sparse=True,
        reuse_buffers=True, cache_static=True,
    )
    out1 = solver.solve(gate, return_final=True)
    out2 = solver.solve(gate, return_final=True)

    assert out1.shape == out2.shape == (3,)


def test_copy_preserves_fast_path_flags():
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=50, dt=0.1667, i0=0.9, seed=10,
        backend=NumpyBackend(compile=True), use_sparse=True,
        reuse_buffers=True, cache_static=True,
    )
    copied = solver.copy()
    assert copied.use_numba is True
    out = copied.solve(gate, return_final=True)
    assert out.shape == (3,)


# ── use_sparse is still rejected by backends that can't offer it ────────────

def test_use_sparse_rejected_by_backend_without_support():
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    with pytest.raises(NotImplementedError):
        CaSuDaSolver(
            Nt=10, dt=0.1667, i0=0.9,
            backend=TorchBackend(device="cpu"), use_sparse=True,
        )


# ── Non-NumPy backend: use_numba stays False regardless of compile= ─────────

def test_compile_enabled_non_numpy_backend_does_not_infer_use_numba():
    """TorchBackend(compile=True) is a legitimate, different thing (torch.compile
    for the per-step path) - it must not accidentally enable the
    NumPy/numba-only whole-loop fast path. Only checks the inferred flag,
    not an actual solve(): torch.compile's own compilation is a separate,
    environment-dependent concern (needs a C++ toolchain) this test isn't
    about."""
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    solver = CaSuDaSolver(
        Nt=100, dt=0.1667, i0=0.9, seed=13,
        backend=TorchBackend(device="cpu", compile=True),
    )
    assert solver.use_numba is False


def test_return_final_non_numpy_backend_falls_back_generically():
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=100, dt=0.1667, i0=0.9, seed=14,
        backend=TorchBackend(device="cpu"),
    )
    out = solver.solve(gate, return_final=True)

    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})
