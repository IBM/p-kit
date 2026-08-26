"""Tests for CaSuDaSolver.solve(return_final=True).

Covers the return-final fast path: skip trajectory/current/energy storage
and return only the final +/-1 state. This used to live only on the
separate CaSuDaOptimized class; it's now a first-class, backend-agnostic
solve() option. use_numba/use_sparse/reuse_buffers/cache_static (also
formerly CaSuDaOptimized-only) are CaSuDaSolver options too - they pick a
whole-loop-JIT fast path that only a plain-NumPy backend can run, since
numba only understands NumPy arrays and the sparse representation is
NumPy/SciPy-specific. Any other backend falls back to the generic
storage-skipping loop.
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


# ── use_numba/use_sparse whole-loop fast path (NumPy backend only) ──────────

numba = pytest.importorskip("numba")


@pytest.mark.parametrize("use_sparse", [False, True])
def test_return_final_numba(use_sparse):
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=200, dt=0.1667, i0=0.9, seed=42,
        use_numba=True, use_sparse=use_sparse,
    )
    out = solver.solve(gate, return_final=True)

    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


def test_return_final_sparse_only_eager_path():
    """use_sparse without use_numba exercises _solve_final_fast's eager
    (non-JIT) sparse branch."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.9, seed=5, use_sparse=True)
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
        use_numba=True, reuse_buffers=True, cache_static=True,
    )

    def force_positive(run, m, m_filt):
        return np.full(m.shape, 10.0)

    out = solver.solve(gate, bias_func=force_positive, return_final=True)
    assert np.all(out > 0)


def test_return_final_reuse_buffers_and_cache_static():
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=50, dt=0.1667, i0=0.9, seed=9,
        use_numba=True, use_sparse=True,
        reuse_buffers=True, cache_static=True,
    )
    out1 = solver.solve(gate, return_final=True)
    out2 = solver.solve(gate, return_final=True)

    assert out1.shape == out2.shape == (3,)


def test_copy_preserves_fast_path_flags():
    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=50, dt=0.1667, i0=0.9, seed=10,
        use_numba=True, use_sparse=True,
        reuse_buffers=True, cache_static=True,
    )
    out = solver.copy().solve(gate, return_final=True)
    assert out.shape == (3,)


# ── Backend/flag mismatches are rejected at construction time ───────────────

def test_use_sparse_rejected_by_backend_without_support():
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    with pytest.raises(NotImplementedError):
        CaSuDaSolver(
            Nt=10, dt=0.1667, i0=0.9,
            backend=TorchBackend(device="cpu"), use_sparse=True,
        )


def test_use_numba_rejected_by_non_numpy_backend():
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    with pytest.raises(NotImplementedError):
        CaSuDaSolver(
            Nt=10, dt=0.1667, i0=0.9,
            backend=TorchBackend(device="cpu"), use_numba=True,
        )


# ── Non-NumPy backend without the fast-path flags: generic path applies ─────

def test_return_final_non_numpy_backend_falls_back_generically():
    torch = pytest.importorskip("torch")
    from p_kit.backends import TorchBackend

    gate = ANDGate()
    solver = CaSuDaSolver(
        Nt=100, dt=0.1667, i0=0.9, seed=13,
        backend=TorchBackend(device="cpu"),
    )
    out = solver.solve(gate, return_final=True)

    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert set(np.unique(out)).issubset({-1.0, 1.0})
