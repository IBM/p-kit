import itertools
from functools import lru_cache

import numpy as np
import pytest

p_kit = pytest.importorskip("p_kit")

from p_kit import psl
from p_kit.psl import Port
from p_kit.solver.csd_solver import CaSuDaSolver

try:
    from p_kit.solver.gibbs_solver import GibbsSolver
except ImportError:
    from p_kit.solver.gibbs import GibbsSolver

try:
    from p_kit.solver.spd_solver import SPDSolver, dense_jh, energy
except ImportError:
    from spd_solver import SPDSolver, dense_jh, energy


NT = 5000
TAIL = 2000
I0 = 0.8
DT = 0.1667
SEEDS = (7, 17, 27)


@psl.pcircuit(n_pbits=3)
class ANDCircuit:
    A = Port("A")
    B = Port("B")
    C = Port("C")

    J = np.array(
        [
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    h = np.array([[1.0], [1.0], [-2.0]])


@psl.pcircuit(n_pbits=3)
class ORCircuit:
    A = Port("A")
    B = Port("B")
    C = Port("C")

    J = np.array(
        [
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    h = np.array([[-1.0], [-1.0], [2.0]])


@psl.pcircuit(n_pbits=4)
class FrustratedRing4:
    p0 = Port("p0")
    p1 = Port("p1")
    p2 = Port("p2")
    p3 = Port("p3")

    J = np.array(
        [
            [0.0, 1.4, 0.0, -1.0],
            [1.4, 0.0, 1.1, 0.0],
            [0.0, 1.1, 0.0, 1.3],
            [-1.0, 0.0, 1.3, 0.0],
        ]
    )
    h = np.array([[0.25], [-0.15], [0.10], [-0.20]])


CASES = (
    ("AND", ANDCircuit()),
    ("OR", ORCircuit()),
    ("Ring", FrustratedRing4()),
)
CASE_MAP = dict(CASES)


def states(n):
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))


def exact_distribution(J, h):
    """Exact Boltzmann distribution for the raw Ising Hamiltonian."""
    S = states(len(h))
    E = energy(S, J, h)
    z = -I0 * E
    p = np.exp(z - z.max())
    return S, E, p / p.sum()


def empirical(samples, S):
    X = np.where(np.asarray(samples) >= 0, 1.0, -1.0)
    index = {tuple(s): i for i, s in enumerate(S)}
    counts = np.zeros(len(S))

    for state in X:
        counts[index[tuple(state)]] += 1

    return counts / counts.sum()


def metrics(samples, J, h):
    X = np.where(np.asarray(samples) >= 0, 1.0, -1.0)[-TAIL:]
    S, E0, p = exact_distribution(J, h)
    pe = empirical(X, S)
    E = energy(X, J, h)
    ground_energy = E0.min()

    return {
        "tv": 0.5 * float(np.abs(pe - p).sum()),
        "ground": float(np.mean(np.isclose(E, ground_energy))),
        "tail_e": float(E.mean()),
        "states": len({tuple(x) for x in X}),
    }


def make_solver(cls, seed):
    """Instantiate existing p-kit solvers across minor API differences."""
    candidates = (
        {"Nt": NT, "dt": DT, "i0": I0, "seed": seed},
        {"Nt": NT, "dt": DT, "i0": I0},
        {"Nt": NT, "i0": I0},
    )

    for kwargs in candidates:
        try:
            return cls(**kwargs)
        except TypeError:
            pass

    return cls()


def run(solver, circuit, J, h, seed):
    """Run one single-shot solver and return distribution metrics.

    All current p-kit single-shot solvers are expected to return ``(I, m, E)``.
    Keeping this assertion here makes the comparison test also protect the
    common Solver API introduced for SPDSolver.
    """
    np.random.seed(seed)

    try:
        out = solver.solve(circuit)
    except (TypeError, AttributeError, ValueError):
        if hasattr(circuit, "circuit"):
            out = solver.solve(circuit.circuit)
        else:
            raise

    assert isinstance(out, tuple), (
        f"{solver.__class__.__name__}.solve() must return (I, m, E) "
        "for a single shot"
    )
    assert len(out) == 3, (
        f"{solver.__class__.__name__}.solve() returned {len(out)} values; "
        "expected (I, m, E)"
    )

    _, M, E = out
    M = np.asarray(M, dtype=float)
    E = np.asarray(E, dtype=float)

    if M.ndim == 1:
        M = M.reshape(-1, len(h))

    if M.shape[-1] != len(h) and M.shape[0] == len(h):
        M = M.T

    assert M.ndim == 2
    assert M.shape[-1] == len(h)
    assert E.ndim == 1
    assert E.shape[0] == M.shape[0]
    assert np.isfinite(E).all()

    return metrics(M, J, h)


@lru_cache(None)
def averages(name):
    circuit = CASE_MAP[name]
    J, h = dense_jh(circuit)
    results = {"SPD": [], "Gibbs": [], "CaSuDa": []}

    for seed in SEEDS:
        # Diagnostics are deliberately disabled in the comparison benchmark:
        # they must not affect sampling quality and should not add benchmark
        # overhead. Mandatory SPD mapping validation still runs internally.
        spd = SPDSolver(
            Nt=NT,
            dt=DT,
            i0=I0,
            seed=seed,
            mode="fast",
            diagnostics=False,
            verbose=False,
        )

        results["SPD"].append(run(spd, circuit, J, h, seed))
        results["Gibbs"].append(
            run(make_solver(GibbsSolver, seed), circuit, J, h, seed)
        )
        results["CaSuDa"].append(
            run(make_solver(CaSuDaSolver, seed), circuit, J, h, seed)
        )

    means = {
        solver_name: {
            metric: float(np.mean([result[metric] for result in solver_results]))
            for metric in solver_results[0]
        }
        for solver_name, solver_results in results.items()
    }

    return means, J, h


@pytest.mark.parametrize("name,circuit", CASES)
def test_spd_matches_exact_distribution_reasonably(name, circuit):
    out, J, h = averages(name)
    S, E, p = exact_distribution(J, h)
    target_ground = float(p[np.isclose(E, E.min())].sum())
    spd = out["SPD"]

    assert spd["tv"] < 0.12, f"{name}: SPD TV={spd['tv']:.4f}"
    assert abs(spd["ground"] - target_ground) < 0.10, (
        f"{name}: SPD ground={spd['ground']:.4f}, "
        f"exact={target_ground:.4f}"
    )
    assert spd["states"] >= min(6, len(S) - 3), (
        f"{name}: poor state coverage {spd['states']:.1f}/{len(S)}"
    )


@pytest.mark.parametrize("name,circuit", CASES)
def test_spd_is_competitive_with_pkit_solvers(name, circuit):
    out, _, _ = averages(name)
    spd = out["SPD"]
    gibbs = out["Gibbs"]
    casuda = out["CaSuDa"]

    assert spd["tv"] <= gibbs["tv"] + 0.12, (
        f"{name}: SPD TV {spd['tv']:.4f} vs Gibbs {gibbs['tv']:.4f}"
    )
    assert spd["tv"] <= casuda["tv"] + 0.10, (
        f"{name}: SPD TV {spd['tv']:.4f} vs CaSuDa {casuda['tv']:.4f}"
    )
    assert spd["ground"] >= gibbs["ground"] - 0.10, (
        f"{name}: SPD ground {spd['ground']:.4f} "
        f"vs Gibbs {gibbs['ground']:.4f}"
    )
    assert spd["ground"] >= casuda["ground"] - 0.10, (
        f"{name}: SPD ground {spd['ground']:.4f} "
        f"vs CaSuDa {casuda['ground']:.4f}"
    )
