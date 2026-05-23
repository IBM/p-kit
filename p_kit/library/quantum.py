import numpy as np
from p_kit.psl.p_circuit import PCircuit


class TransverseFieldIsing(PCircuit):
    """Maps a stoquastic transverse-field Ising model to a p-bit circuit
    via the Suzuki-Trotter (path integral) decomposition.

    An n-qubit quantum system becomes an n * n_replicas p-bit circuit.
    P-bit (i, tau) maps to index tau * n + i (replica-major ordering).

    This is the probabilistic analog of QAOA (PAOA): intra-replica couplings
    encode the QAOA cost Hamiltonian H_C = -sum_ij J_ij sz_i sz_j, while
    inter-replica Trotter couplings encode the QAOA mixer H_B = -gamma sum_i sx_i.
    ``gamma`` is the mixer strength and ``n_replicas`` the number of QAOA layers.
    Use ``PolyOptimizer`` instead for the classical baseline (gamma=0 equivalent).

    Parameters
    ----------
    j_q : np.ndarray, shape (n, n)
        Symmetric Ising coupling matrix of the quantum system (QAOA cost H_C).
    h_q : np.ndarray, shape (n,)
        Longitudinal field (bias) of the quantum system.
    gamma : float
        Transverse field / QAOA mixer strength. Use 0 for a purely classical Ising model.
    beta : float, default 1.0
        Inverse temperature (1/T).
    n_replicas : int, default 10
        Number of Trotter replicas (QAOA layers). Higher values give a better approximation.

    Notes
    -----
    Ref: Camsari et al., arXiv:1809.04028, Fig. 9

    .. versionadded:: 0.0.1
    """

    def __init__(self, j_q, h_q, gamma, beta=1.0, n_replicas=10):
        n = len(h_q)
        R = n_replicas
        super().__init__(n * R)

        J = np.zeros((n * R, n * R))
        h_vec = np.zeros(n * R)

        # Intra-replica: spatial couplings and bias scaled by beta/R
        scale = beta / R
        for tau in range(R):
            s = tau * n
            J[s:s + n, s:s + n] += scale * j_q
            h_vec[s:s + n] += scale * h_q

        # Inter-replica: Trotter coupling K = -(1/2) ln(tanh(beta*gamma/R))
        # K > 0 ferromagnetically couples same qubit across adjacent replicas.
        # gamma=0 means classical limit (no inter-replica coupling).
        if gamma > 0:
            K = -0.5 * np.log(np.tanh(beta * gamma / R))
            seen = set()
            for tau in range(R):
                tau_next = (tau + 1) % R
                pair = (min(tau, tau_next), max(tau, tau_next))
                if pair in seen:
                    continue
                seen.add(pair)
                idx = np.arange(n)
                J[tau * n + idx, tau_next * n + idx] += K
                J[tau_next * n + idx, tau * n + idx] += K

        np.fill_diagonal(J, 0)
        self.J = J
        self.h = h_vec

    @classmethod
    def from_qubo(cls, qubo, gamma, beta=1.0, n_replicas=10):
        """Build a TransverseFieldIsing circuit from a QUBO matrix.

        Minimises x^T Q x with x ∈ {0, 1} by mapping to the Ising
        spin variables s ∈ {-1, +1} via x_i = (1 + s_i) / 2.

        Parameters
        ----------
        qubo : np.ndarray, shape (n, n)
            QUBO matrix (upper or full triangle).
        gamma : float
            Transverse field strength.
        beta : float, default 1.0
            Inverse temperature (1/T).
        n_replicas : int, default 10
            Number of Trotter replicas.
        """
        qubo = np.asarray(qubo, dtype=float)
        qubo_sym = qubo + qubo.T - np.diag(np.diag(qubo))   # symmetrise, count diagonal once
        j_q = -qubo_sym / 4
        np.fill_diagonal(j_q, 0)
        h_q = -np.sum(qubo_sym, axis=1) / 4
        return cls(j_q, h_q, gamma, beta, n_replicas)
