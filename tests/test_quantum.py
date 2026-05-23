import numpy as np
import pytest
from p_kit.library.quantum import TransverseFieldIsing
from p_kit.solver import CaSuDaSolver


def _make(n=3, R=4, gamma=0.5, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    J_q = rng.uniform(-1, 1, (n, n))
    J_q = (J_q + J_q.T) / 2
    np.fill_diagonal(J_q, 0)
    h_q = rng.uniform(-1, 1, n)
    return TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R), n, R


def test_shape():
    c, n, R = _make()
    assert c.n_pbits == n * R
    assert c.J.shape == (n * R, n * R)
    assert c.h.shape == (n * R,)


def test_symmetry():
    c, _, _ = _make()
    np.testing.assert_array_almost_equal(c.J, c.J.T)


def test_zero_diagonal():
    c, n, R = _make()
    np.testing.assert_array_equal(np.diag(c.J), np.zeros(n * R))


def test_intra_replica_scaling():
    n, R, beta = 2, 3, 2.0
    J_q = np.array([[0.0, 1.0], [1.0, 0.0]])
    h_q = np.array([0.5, -0.5])
    c = TransverseFieldIsing(J_q, h_q, gamma=0.0, beta=beta, n_replicas=R)
    scale = beta / R
    for tau in range(R):
        s = tau * n
        np.testing.assert_array_almost_equal(c.J[s:s + n, s:s + n], scale * J_q)
        np.testing.assert_array_almost_equal(c.h[s:s + n], scale * h_q)


def test_no_inter_replica_when_gamma_zero():
    n, R = 2, 3
    J_q = np.array([[0.0, 1.0], [1.0, 0.0]])
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=0.0, n_replicas=R)
    for tau in range(R):
        tau_next = (tau + 1) % R
        assert c.J[tau * n, tau_next * n] == 0.0


def test_inter_replica_coupling_value():
    n, R = 2, 4
    beta, gamma = 1.0, 0.5
    J_q = np.zeros((n, n))
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R)
    K = -0.5 * np.log(np.tanh(beta * gamma / R))
    # qubit 0: replica 0 connected to replica 1 at index [0, n]
    assert c.J[0, n] == pytest.approx(K)
    assert c.J[n, 0] == pytest.approx(K)


def test_r2_no_double_counting():
    """For R=2 the ring has one unique temporal bond per qubit — not two."""
    n, R = 2, 2
    beta, gamma = 1.0, 0.5
    J_q = np.zeros((n, n))
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R)
    K = -0.5 * np.log(np.tanh(beta * gamma / R))
    assert c.J[0, n] == pytest.approx(K)   # not 2*K


def test_solve():
    c, _, _ = _make(n=2, R=4)
    solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.8, seed=42)
    _, all_m, _ = solver.solve(c)
    assert all_m.shape[-1] == c.n_pbits
