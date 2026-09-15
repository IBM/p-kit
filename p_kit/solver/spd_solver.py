"""
SPDSolver: continuous correlated solver for p-bit / Ising models.

The solver maps the discrete Ising parameters (J, h) to a symmetric
positive-definite (SPD) representation before sampling. The SPD structure
provides a valid continuous correlated state space, allowing proposals to
be generated collectively rather than by updating p-bits independently.

A continuous solver may provide better results, but the problem must first be
mapped successfully to a continuous space. This is the first and main obstacle.
Therefore, the J,h -> SPD mapping is validated before use. If the mapping quality
is below the configured threshold, SPDSolver can fall back to another solver.
Use fall_back_solver= in this case.

- mode="fast" uses a fixed SPD geometry for efficient continuous correlated proposals
- mode="analog" evolves the SPD state dynamically with Riemannian operations to better
  reflect a future analog hardware implementation.

SPDSolver is particularly useful for correlated sampling, optimization,
and future analog implementations where continuous SPD dynamics may be
implemented directly into hardware.

For problems requiring very accurate sampling of the target Boltzmann
distribution, especially as problem size grows, GibbsSolver may still
provide better distribution fidelity.
"""
from __future__ import annotations

import inspect
import itertools
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EPS = 1e-9


@dataclass
class SPDMapping:
    K: np.ndarray
    A: np.ndarray
    shift: float
    min_eig: float


@dataclass
class MappingQuality:
    score: float
    fidelity: float
    nrmse: float
    rank: float
    condition: float
    condition_score: float
    n_states: int


def sanitize_jh(J, h):
    """Symmetrize J, zero its diagonal, and flatten h."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    if J.shape != (h.size, h.size):
        raise ValueError(
            f"J has shape {J.shape}, expected {(h.size, h.size)} to match h"
        )
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J, h


def dense_jh(obj):
    """Extract dense (J, h) from a p-kit circuit-like object."""
    if hasattr(obj, "J") and hasattr(obj, "h"):
        J = np.asarray(obj.J, dtype=float)
        h = np.asarray(obj.h, dtype=float).reshape(-1)
        if J.ndim == 2 and J.shape == (h.size, h.size):
            return sanitize_jh(J, h)

    if hasattr(obj, "synthesize"):
        synthesized = obj.synthesize(format="dense")
        if isinstance(synthesized, tuple) and len(synthesized) >= 2:
            return sanitize_jh(synthesized[0], synthesized[1])
        if isinstance(synthesized, dict) and "J" in synthesized and "h" in synthesized:
            return sanitize_jh(synthesized["J"], synthesized["h"])

    if hasattr(obj, "circuit"):
        return dense_jh(obj.circuit)

    raise TypeError("Cannot extract dense J,h from object")


def states(n):
    """Enumerate all {-1,+1} states."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))


def energy(S, J, h):
    """Return Ising energies for one state or a batch of states."""
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[None, :]
    return -0.5 * np.einsum("bi,ij,bj->b", S, J, S) - S @ h


def ising_to_spd(J, h, i0=0.8, margin=0.05):
    """Map Ising parameters to an SPD precision matrix."""
    J, h = sanitize_jh(J, h)
    n = h.size

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = J
    A[:n, n] = h
    A[n, :n] = h
    A *= i0

    top_eig = float(np.linalg.eigvalsh(A)[-1])
    shift = top_eig + margin
    K = shift * np.eye(n + 1) - A
    min_eig = float(np.linalg.eigvalsh(K)[0])

    if min_eig <= 0:
        raise RuntimeError(
            "SPD mapping failed: "
            f"min eigenvalue {min_eig:.3e} <= 0 "
            f"(top_eig={top_eig:.3e}, margin={margin}, i0={i0})"
        )
    return SPDMapping(K, A, shift, min_eig)


def _ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0

    while i < len(x):
        j = i + 1
        while j < len(x) and np.isclose(
            x[order[j]], x[order[i]], rtol=1e-12, atol=1e-12
        ):
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def mapping_quality(
    J,
    h,
    mapping,
    max_condition=1e6,
    max_states=1 << 16,
    random_states=20000,
    seed=12345,
):
    """Measure energy-landscape fidelity and numerical conditioning."""
    J, h = sanitize_jh(J, h)
    n = h.size
    total = (1 << n) if n < 63 else max_states + 1

    if total <= max_states:
        S = states(n)
    else:
        rng = np.random.default_rng(seed)
        S = rng.choice((-1.0, 1.0), size=(random_states, n))

    E = energy(S, J, h)
    Q = np.column_stack((S, np.ones(len(S))))
    Z = np.einsum("bi,ij,bj->b", Q, mapping.K, Q)

    design = np.column_stack((E, np.ones_like(E)))
    a, b = np.linalg.lstsq(design, Z, rcond=None)[0]
    fitted = a * E + b

    rmse = float(np.sqrt(np.mean((Z - fitted) ** 2)))
    scale = float(np.ptp(fitted))
    if scale <= EPS:
        nrmse = 0.0 if rmse <= 1e-12 else float("inf")
    else:
        nrmse = rmse / scale

    rank_E = _ranks(E)
    rank_Z = _ranks(Z)
    if len(E) > 1 and np.std(rank_E) > 0 and np.std(rank_Z) > 0:
        rank = float(np.corrcoef(rank_E, rank_Z)[0, 1])
    else:
        rank = 1.0

    if a <= 0 or not np.isfinite(nrmse):
        fidelity = 0.0
    else:
        fidelity = float(np.clip(1.0 - nrmse, 0.0, 1.0))

    condition = float(np.linalg.cond(mapping.K))
    condition_score = float(min(1.0, max_condition / max(condition, 1.0)))
    score = fidelity * condition_score

    return MappingQuality(
        score=score,
        fidelity=fidelity,
        nrmse=float(nrmse),
        rank=rank,
        condition=condition,
        condition_score=condition_score,
        n_states=len(S),
    )


def mapping_identity_error(J, h, mapping, i0=0.8, max_states=1 << 16):
    """Check the exact quadratic identity when full enumeration is feasible."""
    n = len(h)
    if (1 << n) > max_states:
        return float("nan")

    S = states(n)
    Q = np.column_stack((S, np.ones(len(S))))
    mapped_energy = np.einsum("bi,ij,bj->b", Q, mapping.K, Q)
    expected = mapping.shift * (n + 1) + 2.0 * i0 * energy(S, J, h)
    return float(np.max(np.abs(mapped_energy - expected)))


def spd_eigh(X):
    """Symmetric eigendecomposition with a positive eigenvalue floor."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (X + X.T))
    return np.maximum(eigenvalues, EPS), eigenvectors


def spd_pow(X, power):
    eigenvalues, eigenvectors = spd_eigh(X)
    return (eigenvectors * (eigenvalues**power)) @ eigenvectors.T


def riem_log(X, Y):
    """Affine-invariant Riemannian logarithm Log_X(Y)."""
    sqrt_X = spd_pow(X, 0.5)
    inv_sqrt_X = spd_pow(X, -0.5)
    eigenvalues, eigenvectors = spd_eigh(inv_sqrt_X @ Y @ inv_sqrt_X)
    inner = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    return sqrt_X @ inner @ sqrt_X


def riem_exp(X, tangent):
    """Affine-invariant Riemannian exponential Exp_X(tangent)."""
    sqrt_X = spd_pow(X, 0.5)
    inv_sqrt_X = spd_pow(X, -0.5)
    whitened = inv_sqrt_X @ tangent @ inv_sqrt_X
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    exp_values = np.exp(np.clip(eigenvalues, -30.0, 30.0))
    inner = (eigenvectors * exp_values) @ eigenvectors.T
    return sqrt_X @ inner @ sqrt_X


def riem_dist(X, Y):
    """Affine-invariant Riemannian distance."""
    inv_sqrt_X = spd_pow(X, -0.5)
    eigenvalues, _ = spd_eigh(inv_sqrt_X @ Y @ inv_sqrt_X)
    return float(np.linalg.norm(np.log(eigenvalues)))


def to_corr(X):
    """Project an SPD matrix to an SPD correlation matrix."""
    eigenvalues, eigenvectors = spd_eigh(X)
    X = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(X), EPS))
    corr = X / np.outer(scale, scale)
    return 0.5 * (corr + corr.T) + EPS * np.eye(len(corr))


def target_corr(mapping, n):
    """Correlation matrix induced by the mapped SPD precision."""
    covariance = np.linalg.solve(mapping.K, np.eye(mapping.K.shape[0]))
    return to_corr(covariance[:n, :n])


def gaussian(rng, covariance):
    """Sample a zero-mean Gaussian with the requested covariance."""
    eigenvalues, eigenvectors = spd_eigh(covariance)
    z = rng.normal(size=len(eigenvalues))
    return eigenvectors @ (np.sqrt(eigenvalues) * z)


class SPDSolver:
    def __init__(
        self,
        Nt=10000,
        dt=0.1667,
        i0=0.8,
        seed=None,
        mode="fast",
        margin=0.05,
        min_mapping_score=0.999,
        max_condition=1e6,
        rate=0.08,
        noise=0.025,
        flip_prob=0.18,
        fall_back_solver=None,
        verbose=True,
    ):
        if (
            Nt <= 0
            or i0 <= 0
            or margin <= 0
            or rate <= 0
            or noise < 0
            or not 0 < flip_prob < 0.5
        ):
            raise ValueError("invalid solver parameter")
        if mode not in ("fast", "analog"):
            raise ValueError("mode must be 'fast' or 'analog'")
        if not 0 <= min_mapping_score <= 1 or max_condition <= 0:
            raise ValueError("invalid mapping threshold")

        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.seed = seed
        self.mode = mode
        self.margin = margin
        self.min_mapping_score = min_mapping_score
        self.max_condition = max_condition
        self.rate = rate
        self.noise = noise
        self.flip_prob = flip_prob
        self.fall_back_solver = fall_back_solver
        self.verbose = verbose

        self.mapping_ = None
        self.mapping_quality_ = None
        self.mapping_error_ = None
        self.manifold_state_ = None
        self.energies_ = None
        self.acceptance_rate_ = None
        self.final_distance_ = None
        self.used_fallback_ = False
        self.fallback_reason_ = None
        self.fallback_solver_ = None

    def _make_fallback(self):
        """Return a fallback solver without exception-driven API probing."""
        fallback = self.fall_back_solver
        if fallback is None:
            return None

        if not isinstance(fallback, type):
            if not callable(getattr(fallback, "solve", None)):
                raise TypeError(
                    "fall_back_solver must expose solve(circuit)"
                )
            return fallback

        available = {
            "Nt": self.Nt,
            "dt": self.dt,
            "i0": self.i0,
            "seed": self.seed,
        }
        signature = inspect.signature(fallback)
        parameters = signature.parameters.values()
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

        if accepts_kwargs:
            kwargs = available
        else:
            accepted_names = {
                parameter.name
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            kwargs = {
                name: value
                for name, value in available.items()
                if name in accepted_names
            }

        missing = [
            parameter.name
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and parameter.name not in kwargs
        ]
        if missing:
            raise TypeError(
                f"cannot instantiate fallback solver {fallback.__name__}: "
                f"missing required constructor arguments {missing}; "
                "pass a configured solver instance instead"
            )

        instance = fallback(**kwargs)
        if not callable(getattr(instance, "solve", None)):
            raise TypeError(
                f"fallback solver {fallback.__name__} must expose solve(circuit)"
            )
        return instance

    def _run_fallback(self, circuit, reason):
        fallback = self._make_fallback()
        if fallback is None:
            raise RuntimeError(reason)

        self.used_fallback_ = True
        self.fallback_reason_ = reason
        self.fallback_solver_ = fallback

        name = fallback.__class__.__name__
        warnings.warn(
            f"SPDSolver: {reason}; falling back to {name}.",
            RuntimeWarning,
            stacklevel=2,
        )

        # p-kit solver contract: solve(circuit) -> (I, m).
        out = fallback.solve(circuit)
        if not isinstance(out, tuple) or len(out) < 2:
            raise TypeError(f"{name}.solve(circuit) must return an (I, m) tuple")
        return out[0], out[1]

    def solve(self, circuit):
        self.used_fallback_ = False
        self.fallback_reason_ = None
        self.fallback_solver_ = None

        J, h = dense_jh(circuit)
        n = h.size
        mapping = ising_to_spd(J, h, self.i0, self.margin)
        quality = mapping_quality(J, h, mapping, self.max_condition)

        self.mapping_ = mapping
        self.mapping_quality_ = quality
        self.mapping_error_ = mapping_identity_error(J, h, mapping, self.i0)

        if self.verbose:
            print(
                f"SPD map score={quality.score:.6f} "
                f"fidelity={quality.fidelity:.6f} "
                f"rank={quality.rank:.6f} "
                f"nrmse={quality.nrmse:.2e} "
                f"cond={quality.condition:.2e}"
            )

        if quality.score < self.min_mapping_score:
            reason = (
                f"mapping score {quality.score:.6f} below minimum "
                f"{self.min_mapping_score:.6f}"
            )
            return self._run_fallback(circuit, reason)

        rng = np.random.default_rng(self.seed)
        target = target_corr(mapping, n)
        X = target.copy() if self.mode == "fast" else np.eye(n)
        state = rng.choice((-1.0, 1.0), size=n)
        current_energy = float(energy(state, J, h)[0])
        threshold = NormalDist().inv_cdf(1.0 - self.flip_prob)

        all_I = np.empty((self.Nt, n))
        all_m = np.empty((self.Nt, n))
        all_E = np.empty(self.Nt)
        accepted = 0

        for t in range(self.Nt):
            if self.mode == "analog":
                drift = riem_log(X, target)
                noise = rng.normal(size=(n, n))
                noise = 0.5 * (noise + noise.T)
                sqrt_X = spd_pow(X, 0.5)
                tangent_noise = sqrt_X @ noise @ sqrt_X
                tangent = (
                    self.rate * drift
                    + self.noise * np.sqrt(self.rate) * tangent_noise
                )
                X = to_corr(riem_exp(X, tangent))

            flip = gaussian(rng, X) > threshold
            proposal = state.copy()
            proposal[flip] *= -1.0
            proposal_energy = float(energy(proposal, J, h)[0])
            delta_energy = proposal_energy - current_energy

            # For delta_energy > 0 the exponent is <= 0; large values
            # safely underflow to zero, corresponding to rejection.
            accept = (
                delta_energy <= 0
                or rng.random() < np.exp(-self.i0 * delta_energy)
            )
            if accept:
                state = proposal
                current_energy = proposal_energy
                accepted += 1

            all_m[t] = state
            all_I[t] = self.i0 * (J @ state + h)
            all_E[t] = current_energy

        self.manifold_state_ = X
        self.energies_ = all_E
        self.acceptance_rate_ = accepted / self.Nt
        self.final_distance_ = riem_dist(X, target)
        return all_I, all_m
