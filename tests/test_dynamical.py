"""Tests for the dynamical p-bit extensions to CaSuDaSolver.

Covers the step-1 scaffolding for the "dynamical / RTRBM" direction:
  - a per-timestep ``bias_func`` hook (recurrence / adaptive biasing),
  - a filtered companion state ``m_filt`` (graded [-1, 1] primitive),
  - the opt-in ``return_filtered`` output.

All of these must leave the existing return contract and sampler behavior
untouched when unused.
"""
import numpy as np
from p_kit.psl.gates import ANDGate
from p_kit.solver.csd_solver import CaSuDaSolver


# ── Backward compatibility ────────────────────────────────────────────────────

def test_default_return_contract_unchanged():
    """Default solve() still returns the (all_I, all_m, E) 3-tuple of +/-1."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    out = solver.solve(gate)

    assert isinstance(out, tuple) and len(out) == 3
    _, all_m, _ = out
    assert all_m.shape == (200, 3)
    # The hard sampler is untouched: states remain exactly +/-1.
    assert set(np.unique(all_m)).issubset({-1.0, 1.0})


def test_bias_func_none_is_identical_to_static_h():
    """bias_func=None reproduces the static-h run bit-for-bit (seeded)."""
    gate = ANDGate()
    a = CaSuDaSolver(Nt=300, dt=0.1667, i0=0.9, seed=7).solve(gate.copy())
    b = CaSuDaSolver(Nt=300, dt=0.1667, i0=0.9, seed=7).solve(
        gate.copy(), bias_func=None
    )
    assert np.array_equal(a[1], b[1])


# ── Filtered companion state ──────────────────────────────────────────────────

def test_return_filtered_shape_and_bounds():
    """return_filtered appends m_filt, graded and bounded in [-1, 1]."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=200, dt=0.1667, i0=0.9, seed=1, tau=0.05)
    out = solver.solve(gate, return_filtered=True)

    assert len(out) == 4
    _, _, _, all_mfilt = out
    assert all_mfilt.shape == (200, 3)
    assert all_mfilt.min() >= -1.0 - 1e-9
    assert all_mfilt.max() <= 1.0 + 1e-9


# ── Dynamical bias hook ───────────────────────────────────────────────────────

def test_bias_func_overrides_static_h():
    """A strong bias_func drives the filtered activity regardless of gate h."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=1000, dt=0.1667, i0=0.9, seed=3, tau=0.02)

    def force_negative(run, m, m_filt):
        return np.full(m.shape, -10.0)

    _, _, _, all_mfilt = solver.solve(
        gate, bias_func=force_negative, return_filtered=True
    )
    # After the filter settles, every p-bit should sit near -1.
    assert np.all(all_mfilt[-1] < -0.8)


def test_bias_func_sees_zero_initialized_memory_first():
    """On step 0 the hook receives zero memory — no leakage from the future."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=5, dt=0.1667, i0=0.9, seed=0, tau=0.3)
    seen = []

    def spy(run, m, m_filt):
        seen.append((run, np.array(m_filt)))
        return gate.h.flatten()

    solver.solve(gate, bias_func=spy)

    assert seen[0][0] == 0
    assert np.allclose(seen[0][1], 0.0)
    # And memory becomes non-trivial once the loop advances.
    assert not np.allclose(seen[-1][1], 0.0)


def test_energy_reflects_effective_bias():
    """E is computed in the biased landscape (h_eff), not the static h."""
    gate = ANDGate()
    delta = 3.0
    solver = CaSuDaSolver(Nt=80, dt=0.1667, i0=0.9, seed=11)
    h_eff = gate.h.flatten() + delta

    def biased(run, m, m_filt):
        return h_eff

    _, all_m, E, _ = solver.solve(gate, bias_func=biased, return_filtered=True)
    # Reconstruct shot-0 energy from the recorded states using h_eff. With the
    # old static-h energy this would not match (delta would be dropped).
    expected = solver.i0 * (
        all_m @ h_eff
        + 0.5 * np.einsum('ti,ij,tj->t', all_m, gate.J, all_m)
    )
    assert np.allclose(E, expected)


# ── Multi-shot (n_shots > 1) ──────────────────────────────────────────────────

def test_return_filtered_multi_shot_shapes():
    """n_shots>1 returns (all_m, all_mfilt), both (Nt, n_shots, n_pbits)."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=150, dt=0.1667, i0=0.9, seed=2, tau=0.05)
    out = solver.solve(gate, n_shots=4, return_filtered=True)

    assert isinstance(out, tuple) and len(out) == 2
    all_m, all_mfilt = out
    assert all_m.shape == (150, 4, 3)
    assert all_mfilt.shape == (150, 4, 3)
    assert set(np.unique(all_m)).issubset({-1.0, 1.0})
    assert all_mfilt.min() >= -1.0 - 1e-9 and all_mfilt.max() <= 1.0 + 1e-9


def test_return_filtered_multi_shot_default_contract_unchanged():
    """Without return_filtered, n_shots>1 still returns just all_m."""
    gate = ANDGate()
    out = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.9, seed=2).solve(gate, n_shots=3)
    assert not isinstance(out, tuple)
    assert out.shape == (100, 3, 3)


def test_bias_func_per_shot_broadcasting():
    """A per-shot bias_func output ((n_shots, n_pbits)) drives each shot."""
    gate = ANDGate()
    n_shots = 3
    solver = CaSuDaSolver(Nt=800, dt=0.1667, i0=0.9, seed=5, tau=0.02)

    # Row s pushes every p-bit toward sign(s - 1): shot 0 -> -1, shots 1,2 -> +1.
    signs = np.array([-1.0, 1.0, 1.0])

    def per_shot(run, m, m_filt):
        return np.repeat((10.0 * signs)[:, None], gate.n_pbits, axis=1)

    all_m, all_mfilt = solver.solve(
        gate, n_shots=n_shots, bias_func=per_shot, return_filtered=True
    )
    final = all_mfilt[-1]  # (n_shots, n_pbits)
    assert np.all(final[0] < -0.8)   # shot 0 driven negative
    assert np.all(final[1] > 0.8)    # shot 1 driven positive
    assert np.all(final[2] > 0.8)    # shot 2 driven positive


# ── Filter (tau) edge invariants ──────────────────────────────────────────────

def test_tau_one_filter_tracks_state_exactly():
    """tau=1 makes the leaky integrator a pass-through: m_filt == all_m."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=120, dt=0.1667, i0=0.9, seed=9, tau=1.0)
    _, all_m, _, all_mfilt = solver.solve(gate, return_filtered=True)
    assert np.array_equal(all_mfilt, all_m)


def test_tau_zero_freezes_filter_at_zero():
    """tau=0 disables integration: m_filt stays at its zero initialization."""
    gate = ANDGate()
    solver = CaSuDaSolver(Nt=120, dt=0.1667, i0=0.9, seed=9, tau=0.0)
    _, _, _, all_mfilt = solver.solve(gate, return_filtered=True)
    assert np.all(all_mfilt == 0.0)
