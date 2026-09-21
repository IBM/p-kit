import numpy as np
import pytest

from p_kit.backends import NumpyBackend
from p_kit.solver.base_solver import Solver
from p_kit.solver.spd_solver import (
    SPDSolver,
    dense_jh,
    energy,
    ising_to_spd,
    mapping_identity_error,
    mapping_quality,
    target_corr,
)


class ANDCircuit:
    J = np.array(
        [
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    h = np.array([[1.0], [1.0], [-2.0]])


class DummyFallback:
    """Small solver-like fallback following the current p-kit API."""

    def __init__(self, Nt=8, **kwargs):
        self.Nt = Nt
        self.called = False
        self.annealing_func = None
        self.n_shots = None

    def solve(self, circuit, annealing_func=None, n_shots=1):
        self.called = True
        self.annealing_func = annealing_func
        self.n_shots = n_shots

        _, h = dense_jh(circuit)
        n = len(h)

        if n_shots == 1:
            I = np.zeros((self.Nt, n))
            m = np.ones((self.Nt, n))
            E = np.zeros(self.Nt)
            return I, m, E

        return np.ones((self.Nt, n_shots, n))


class OldTwoValueFallback:
    """Used to verify that an obsolete fallback API is rejected clearly."""

    def __init__(self, Nt=8, **kwargs):
        self.Nt = Nt

    def solve(self, circuit):
        _, h = dense_jh(circuit)
        n = len(h)
        return np.zeros((self.Nt, n)), np.ones((self.Nt, n))


def fixed_annealing(_solver, _run):
    return 0.25


def test_mapping_is_spd_and_exact():
    J, h = dense_jh(ANDCircuit())
    mapping = ising_to_spd(J, h)
    quality = mapping_quality(J, h, mapping)

    assert mapping.K.shape == (4, 4)
    assert np.linalg.eigvalsh(mapping.K).min() > 0
    assert quality.score > 0.999999
    assert quality.fidelity > 0.999999
    assert mapping_identity_error(J, h, mapping) < 1e-10


def test_mapping_preserves_energy_ordering():
    J, h = dense_jh(ANDCircuit())
    mapping = ising_to_spd(J, h)
    quality = mapping_quality(J, h, mapping)

    assert quality.rank > 0.999999
    assert quality.nrmse < 1e-12


def test_target_correlation_is_spd():
    J, h = dense_jh(ANDCircuit())
    corr = target_corr(ising_to_spd(J, h), len(h))

    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-7)
    assert np.linalg.eigvalsh(corr).min() > 0


def test_solver_inherits_standard_solver_api():
    solver = SPDSolver(Nt=10, seed=1, verbose=False)

    assert isinstance(solver, Solver)
    assert isinstance(solver.backend, NumpyBackend)


def test_fast_solver_api_state_and_energy():
    solver = SPDSolver(
        Nt=300,
        seed=1,
        mode="fast",
        diagnostics=True,
        verbose=False,
    )
    I, M, E = solver.solve(ANDCircuit())

    assert I.shape == M.shape == (300, 3)
    assert E.shape == (300,)
    assert set(np.unique(M)) <= {-1.0, 1.0}

    assert solver.mapping_quality_.score >= solver.min_mapping_score
    assert not solver.used_fallback_
    assert 0 <= solver.acceptance_rate_ <= 1
    assert np.isfinite(solver.energies_).all()
    assert np.isfinite(solver.ising_energies_).all()

    # Public E follows the same convention as Gibbs/CaSuDa:
    # i0 * (m @ h + 0.5 * m @ J @ m).
    J, h = dense_jh(ANDCircuit())
    expected_E = solver.i0 * (
        M @ h + 0.5 * np.einsum("bi,ij,bj->b", M, J, M)
    )
    assert np.allclose(E, expected_E)
    assert np.allclose(solver.energies_, E)


def test_analog_solver_api_and_spd_state():
    solver = SPDSolver(
        Nt=80,
        seed=2,
        mode="analog",
        diagnostics=True,
        verbose=False,
    )
    I, M, E = solver.solve(ANDCircuit())

    assert I.shape == M.shape == (80, 3)
    assert E.shape == (80,)
    assert np.linalg.eigvalsh(solver.manifold_state_).min() > 0
    assert np.isfinite(solver.final_distance_)


def test_custom_annealing_controls_current():
    solver = SPDSolver(
        Nt=40,
        i0=0.8,
        seed=3,
        diagnostics=False,
        verbose=False,
    )
    I, M, _ = solver.solve(
        ANDCircuit(),
        annealing_func=fixed_annealing,
    )

    J, h = dense_jh(ANDCircuit())
    expected_I = 0.25 * (M @ J + h)

    assert np.allclose(I, expected_I)


def test_diagnostics_can_be_disabled():
    solver = SPDSolver(
        Nt=50,
        seed=4,
        diagnostics=False,
        verbose=True,
    )
    I, M, E = solver.solve(ANDCircuit())

    assert I.shape == M.shape == (50, 3)
    assert E.shape == (50,)

    # Mapping validation still happens internally, but optional diagnostic
    # objects/histories are not retained or computed.
    assert solver.mapping_ is not None
    assert solver.mapping_quality_ is None
    assert solver.mapping_error_ is None
    assert solver.ising_energies_ is None
    assert solver.acceptance_rate_ is None
    assert solver.acceptance_rates_ is None
    assert solver.final_distance_ is None
    assert solver.final_distances_ is None

    # Standard solver state remains available.
    assert solver.manifold_state_ is not None
    assert np.isfinite(solver.energies_).all()


def test_diagnostics_can_be_reenabled_between_runs():
    solver = SPDSolver(
        Nt=30,
        seed=5,
        diagnostics=False,
        verbose=False,
    )
    solver.solve(ANDCircuit())
    assert solver.mapping_quality_ is None
    assert solver.acceptance_rate_ is None

    solver.diagnostics = True
    solver.solve(ANDCircuit())

    assert solver.mapping_quality_ is not None
    assert solver.mapping_error_ is not None
    assert solver.ising_energies_ is not None
    assert 0 <= solver.acceptance_rate_ <= 1
    assert np.isfinite(solver.final_distance_)


def test_multishot_shape_and_diagnostics():
    solver = SPDSolver(
        Nt=25,
        seed=6,
        diagnostics=True,
        verbose=False,
    )
    M = solver.solve(ANDCircuit(), n_shots=4)

    assert M.shape == (25, 4, 3)
    assert solver.energies_.shape == (25, 4)
    assert solver.ising_energies_.shape == (25, 4)
    assert solver.manifold_state_.shape == (4, 3, 3)
    assert solver.acceptance_rates_.shape == (4,)
    assert solver.final_distances_.shape == (4,)


def test_copy_preserves_configuration_and_is_independent_object():
    solver = SPDSolver(
        Nt=123,
        dt=0.2,
        i0=0.7,
        expected_mean=0.1,
        seed=7,
        tau=0.3,
        mode="analog",
        margin=0.08,
        min_mapping_score=0.95,
        max_condition=2e5,
        rate=0.04,
        noise=0.01,
        flip_prob=0.2,
        diagnostics=False,
        verbose=False,
    )

    clone = solver.copy()

    assert clone is not solver
    assert isinstance(clone, SPDSolver)
    assert clone.Nt == solver.Nt
    assert clone.dt == solver.dt
    assert clone.i0 == solver.i0
    assert clone.expected_mean == solver.expected_mean
    assert clone.seed == solver.seed
    assert clone.tau == solver.tau
    assert clone.mode == solver.mode
    assert clone.margin == solver.margin
    assert clone.min_mapping_score == solver.min_mapping_score
    assert clone.max_condition == solver.max_condition
    assert clone.rate == solver.rate
    assert clone.noise == solver.noise
    assert clone.flip_prob == solver.flip_prob
    assert clone.diagnostics == solver.diagnostics
    assert clone.verbose == solver.verbose


def test_unsupported_backend_has_clear_error():
    class UnsupportedBackend:
        pass

    with pytest.raises(
        NotImplementedError,
        match="currently supports only NumpyBackend",
    ):
        SPDSolver(backend=UnsupportedBackend())


def test_explicit_numpy_backend_is_supported():
    backend = NumpyBackend()
    solver = SPDSolver(
        Nt=10,
        backend=backend,
        diagnostics=False,
        verbose=False,
    )

    assert solver.backend is backend


def test_fallback_instance_warns_and_runs():
    fallback = DummyFallback(Nt=20)
    solver = SPDSolver(
        Nt=20,
        max_condition=1.0,
        min_mapping_score=0.999,
        fall_back_solver=fallback,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back to DummyFallback"):
        I, M, E = solver.solve(ANDCircuit())

    assert solver.used_fallback_
    assert fallback.called
    assert I.shape == M.shape == (20, 3)
    assert E.shape == (20,)
    assert "mapping score" in solver.fallback_reason_


def test_fallback_receives_annealing_and_n_shots_api():
    fallback = DummyFallback(Nt=12)
    solver = SPDSolver(
        Nt=12,
        max_condition=1.0,
        fall_back_solver=fallback,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back"):
        M = solver.solve(
            ANDCircuit(),
            annealing_func=fixed_annealing,
            n_shots=3,
        )

    assert fallback.called
    assert fallback.annealing_func is fixed_annealing
    assert fallback.n_shots == 3
    assert M.shape == (12, 3, 3)


def test_fallback_class_is_instantiated():
    solver = SPDSolver(
        Nt=12,
        max_condition=1.0,
        fall_back_solver=DummyFallback,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back"):
        I, M, E = solver.solve(ANDCircuit())

    assert I.shape == M.shape == (12, 3)
    assert E.shape == (12,)
    assert isinstance(solver.fallback_solver_, DummyFallback)


def test_obsolete_two_value_fallback_is_rejected():
    solver = SPDSolver(
        Nt=10,
        max_condition=1.0,
        fall_back_solver=OldTwoValueFallback,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back"):
        with pytest.raises(TypeError, match=r"return \(I, m, E\)"):
            solver.solve(ANDCircuit())


def test_low_score_without_fallback_raises():
    solver = SPDSolver(
        Nt=10,
        max_condition=1.0,
        min_mapping_score=0.999,
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="mapping score"):
        solver.solve(ANDCircuit())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"Nt": 0},
        {"i0": 0},
        {"margin": 0},
        {"flip_prob": 0},
        {"flip_prob": 0.5},
        {"min_mapping_score": 1.1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SPDSolver(**kwargs)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_n_shots(value):
    solver = SPDSolver(Nt=10, diagnostics=False, verbose=False)

    with pytest.raises(ValueError, match="n_shots"):
        solver.solve(ANDCircuit(), n_shots=value)


def test_non_callable_annealing_is_rejected():
    solver = SPDSolver(Nt=10, diagnostics=False, verbose=False)

    with pytest.raises(TypeError, match="annealing_func must be callable"):
        solver.solve(ANDCircuit(), annealing_func=0.8)


def test_invalid_annealing_value_is_rejected():
    def bad_annealing(_solver, _run):
        return -1.0

    solver = SPDSolver(Nt=10, diagnostics=False, verbose=False)

    with pytest.raises(ValueError, match="finite, non-negative"):
        solver.solve(ANDCircuit(), annealing_func=bad_annealing)


def test_and_truth_table_are_ground_states():
    J, h = dense_jh(ANDCircuit())
    S = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    E = energy(S, J, h)
    ground = {
        tuple(x.astype(int))
        for x in S[np.isclose(E, E.min())]
    }
    expected = {
        (-1, -1, -1),
        (-1, 1, -1),
        (1, -1, -1),
        (1, 1, 1),
    }

    assert ground == expected
