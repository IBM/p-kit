# Transverse-Field Ising Model (TFIM) emulated with p-bits
# via the Suzuki-Trotter path integral decomposition.
#
# Ref: Camsari et al., arXiv:1809.04028, Fig. 9
#
# A stoquastic n-qubit TFIM is mapped to a p-circuit with
# n * n_replicas p-bits. Intra-replica couplings reproduce
# the Ising Hamiltonian; inter-replica couplings encode the
# transverse field via the Trotter coupling K.

import matplotlib.pyplot as plt
import numpy as np

from p_kit.library.quantum import TransverseFieldIsing
from p_kit.solver.annealing import linear, execute
from p_kit.solver.csd_solver import CaSuDaSolver

# --- Problem definition --------------------------------------------------
# 3-qubit ferromagnetic TFIM: J_ij = 1 (spins want to align)
n = 3
J_q = np.array([
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.],
])
h_q = np.zeros(n)

# Transverse field strength (quantum fluctuations).
# gamma=0  → classical Ising (all replicas lock together)
# gamma>>0 → strong quantum fluctuations
gamma = 0.5
beta = 2.0    # inverse temperature
n_replicas = 20

# --- Build p-circuit from Trotter decomposition --------------------------
circuit = TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=n_replicas)
print(f"Qubits: {n}  |  Replicas: {n_replicas}  |  Total p-bits: {circuit.n_pbits}")
print(f"Trotter coupling K = {-0.5 * np.log(np.tanh(beta * gamma / n_replicas)):.4f}")

# --- Solve with annealing ------------------------------------------------
solver = CaSuDaSolver(Nt=int(5e4), dt=0.1, i0=0, seed=42)
n_shots = 100
samples = execute(solver, circuit, linear, n_shots=n_shots, n_last_samples=50, n_jobs=-1)

# Recover qubit magnetizations by averaging over replicas
# samples shape: (n_shots * n_last_samples, n * n_replicas)
qubit_m = samples.reshape(-1, n_replicas, n)  # (samples, replicas, qubits)
qubit_m_avg = qubit_m.mean(axis=1)            # (samples, qubits): per-qubit mean over replicas

# --- Plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"TFIM emulated via Suzuki-Trotter  (n={n}, R={n_replicas}, γ={gamma}, β={beta})")

# Left: histogram of qubit-0 averaged magnetization
axes[0].hist(qubit_m_avg[:, 0], bins=30, density=True, color="steelblue", edgecolor="white")
axes[0].set_xlabel("⟨m₀⟩ (averaged over replicas)")
axes[0].set_ylabel("Probability density")
axes[0].set_title("Qubit-0 magnetization distribution")

# Right: inter-qubit correlation <m_i * m_j>
corr = np.mean(qubit_m_avg[:, :, None] * qubit_m_avg[:, None, :], axis=0)
im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
axes[1].set_title("Qubit-qubit correlations ⟨mᵢ mⱼ⟩")
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
