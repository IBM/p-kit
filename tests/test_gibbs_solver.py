"""Tests for GibbsSolver, the asynchronous single-p-bit-per-step Gibbs
sampler.

Contrasts with CaSuDaSolver (synchronous, vectorized update of all p-bits
at once): GibbsSolver resamples exactly one randomly chosen p-bit per
timestep, conditioned on the current state of all the others. These tests
cover the return contract (mirrors CaSuDaSolver's default, no-bias_func
path), the asynchronous-update invariant, seeded reproducibility,
energy-formula consistency, and functional correctness against AND's
truth table.
"""
import numpy as np
from p_kit.psl.gates import ANDGate
from p_kit.solver import GibbsSolver


# ── Return contract ──────────────────────────────────────────────────────

def test_return_contract_single_shot():
    """n_shots=1 returns the (all_I, all_m, E) 3-tuple; all_m is +/-1."""
    gate = ANDGate()
    solver = GibbsSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    out = solver.solve(gate)

    assert isinstance(out, tuple) and len(out) == 3
    all_I, all_m, E = out
    assert all_m.shape == (200, 3)
    assert set(np.unique(all_m)).issubset({-1.0, 1.0})


def test_return_contract_multi_shot():
    """n_shots>1 returns a bare array (no tuple), shape (Nt, n_shots, n)."""
    gate = ANDGate()
    solver = GibbsSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    out = solver.solve(gate, n_shots=4)

    assert not isinstance(out, tuple)
    assert out.shape == (200, 4, 3)
    assert set(np.unique(out)).issubset({-1.0, 1.0})


# ── Sweep semantics ───────────────────────────────────────────────────────

def test_nt_counts_full_sweeps_not_single_flips():
    """Each Nt step is a full sweep (every p-bit resampled once, in a
    random order), not a single-p-bit update — so consecutive recorded
    states may differ in up to n_pbits coordinates. This also rules out a
    regression to a single-flip-per-step design, which would make this
    solver's Nt incomparable to CaSuDaSolver's (where one Nt tick also
    updates every p-bit once).
    """
    gate = ANDGate()
    solver = GibbsSolver(Nt=200, dt=0.1667, i0=0.9, seed=42)
    _, all_m, _ = solver.solve(gate)

    diffs = np.count_nonzero(all_m[:-1] != all_m[1:], axis=1)
    assert np.all(diffs <= gate.n_pbits)
    # With n_pbits=3 and 200 sweeps, some sweeps should flip more than one
    # p-bit — proof this is sweeping all p-bits, not resampling just one.
    assert np.any(diffs > 1)


# ── Seeded reproducibility ────────────────────────────────────────────────

def test_seeded_reproducibility():
    """Two solvers with the same seed produce bit-identical trajectories."""
    gate = ANDGate()
    a = GibbsSolver(Nt=300, dt=0.1667, i0=0.9, seed=7).solve(gate.copy())
    b = GibbsSolver(Nt=300, dt=0.1667, i0=0.9, seed=7).solve(gate.copy())

    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[2], b[2])


# ── Energy formula consistency ────────────────────────────────────────────

def test_energy_matches_documented_formula():
    """E is reconstructible from all_m via i0*(m@h + 0.5*m@J@m) per step."""
    gate = ANDGate()
    solver = GibbsSolver(Nt=300, dt=0.1667, i0=0.9, seed=11)
    _, all_m, E = solver.solve(gate)

    h = gate.h.flatten()
    expected = solver.i0 * (
        all_m @ h + 0.5 * np.einsum('ti,ij,tj->t', all_m, gate.J, all_m)
    )
    assert np.allclose(E, expected)


# ── Functional correctness (truth table) ──────────────────────────────────

def test_and_gate_truth_table():
    """AND gate produces the correct truth table under the Gibbs sampler."""
    gate = ANDGate()
    solver = GibbsSolver(Nt=20000, dt=0.1667, i0=0.9, seed=42)

    test_cases = [
        ([-1, -1], -1),  # 0 AND 0 = 0
        ([-1, 1], -1),   # 0 AND 1 = 0
        ([1, -1], -1),   # 1 AND 0 = 0
        ([1, 1], 1),     # 1 AND 1 = 1
    ]

    for inputs, expected_output in test_cases:
        gate.h[0] = inputs[0] * 10
        gate.h[1] = inputs[1] * 10
        _, output, _ = solver.solve(gate)

        # Output is at index 2 (order: input1, input2, output).
        output_states = output[:, 2]
        most_common = 1 if np.mean(output_states) > 0 else -1
        assert most_common == expected_output, (
            f"AND({inputs[0]}, {inputs[1]}) expected {expected_output}, "
            f"got {most_common}"
        )
