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
from p_kit.backends import NumpyBackend
from p_kit.solver.annealing import constant
from .base_solver import Solver

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
        if (
            isinstance(synthesized, dict)
            and "J" in synthesized
            and "h" in synthesized
        ):
            return sanitize_jh(synthesized["J"], synthesized["h"])

    if hasattr(obj, "circuit"):
        return dense_jh(obj.circuit)

    raise TypeError("Cannot extract dense J,h from object")

def states(n):
    """Enumerate all {-1,+1} states."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))

def energy(S, J, h):
    """Return raw Ising energies for one state or a batch of states.

    The convention is

        H(m) = -0.5 * m.T @ J @ m - h.T @ m

    This raw Hamiltonian is used internally by the Metropolis-Hastings
    acceptance rule. The public p-kit energy returned by ``solve`` follows the
    existing solver convention and is ``-i0 * H(m)``.
    """
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
    """Measure energy-landscape fidelity and numerical conditioning.

    This validation is part of solver safety/selection, not merely an optional
    diagnostic: SPDSolver uses ``score`` to decide whether the SPD mapping is
    acceptable or whether the configured fallback solver should be used.
    """
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
    """Raise an SPD matrix to a scalar power."""
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

class SPDSolver(Solver):
    """Continuous correlated solver based on an SPD representation.

    Parameters
    ----------
    Nt : int
        Number of solver timesteps.
    dt : float
        Kept for p-kit Solver API compatibility. The current SPD sampler does
        not use ``dt`` directly.
    i0 : float
        Nominal inverse-temperature/current scale.
    expected_mean : float, default=0
        Kept for p-kit Solver API compatibility. Not currently used by the SPD
        update rule.
    seed : int or None
        Random seed.
    backend : Backend or None
        Currently only ``NumpyBackend`` is supported. ``None`` selects the
        default ``NumpyBackend`` through the base ``Solver``.
    tau : float, default=0.1
        Kept for p-kit Solver API compatibility. Not currently used by the SPD
        update rule.
    mode : {"fast", "analog"}
        ``fast`` uses a fixed target correlation geometry. ``analog`` evolves
        the SPD/correlation state with Riemannian drift and noise.
    margin : float
        Positive spectral margin used to make the mapped precision SPD.
    min_mapping_score : float
        Minimum mapping-quality score required to use the SPD sampler.
    max_condition : float
        Conditioning scale used by ``mapping_quality``.
    rate : float
        Riemannian drift rate in ``analog`` mode.
    noise : float
        Riemannian tangent noise scale in ``analog`` mode.
    flip_prob : float
        Marginal threshold probability controlling which correlated Gaussian
        components propose spin flips.
    fall_back_solver : solver instance, solver class, or None
        Optional fallback used when the SPD mapping fails validation.
    diagnostics : bool, default=True
        Enable optional diagnostic calculations and retained diagnostic state.
        Mandatory mapping validation still runs when False because it controls
        whether the solver is safe to use.
    verbose : bool, default=True
        Print mapping diagnostics when ``diagnostics=True``. Ignored when
        ``diagnostics=False``.
    """

    def __init__(
        self,
        Nt=10000,
        dt=0.1667,
        i0=0.8,
        expected_mean=0,
        seed=None,
        backend=None,
        tau=0.1,
        mode="fast",
        margin=0.05,
        min_mapping_score=0.999,
        max_condition=1e6,
        rate=0.08,
        noise=0.025,
        flip_prob=0.18,
        fall_back_solver=None,
        diagnostics=True,
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

        if not isinstance(diagnostics, (bool, np.bool_)):
            raise TypeError("diagnostics must be a boolean")

        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean")

        # Fail explicitly rather than silently moving data back to NumPy.
        # This keeps backend semantics honest while support is added gradually.
        if backend is not None and not isinstance(backend, NumpyBackend):
            backend_name = type(backend).__name__
            raise NotImplementedError(
                "SPDSolver currently supports only NumpyBackend. "
                f"Received backend={backend_name}. "
                "The SPD/Riemannian kernels are currently implemented with "
                "NumPy. Support for additional p-kit backends will be added "
                "progressively. For now, omit 'backend' or use "
                "backend=NumpyBackend()."
            )

        super().__init__(
            Nt=Nt,
            dt=dt,
            i0=i0,
            expected_mean=expected_mean,
            seed=seed,
            backend=backend,
            tau=tau,
        )

        # Defensive check in case the base Solver default backend changes later.
        if not isinstance(self.backend, NumpyBackend):
            backend_name = type(self.backend).__name__
            raise NotImplementedError(
                "SPDSolver currently supports only NumpyBackend. "
                f"Solver selected backend={backend_name}. "
                "Support for additional p-kit backends will be added "
                "progressively."
            )

        self.mode = mode
        self.margin = margin
        self.min_mapping_score = min_mapping_score
        self.max_condition = max_condition
        self.rate = rate
        self.noise = noise
        self.flip_prob = flip_prob
        self.fall_back_solver = fall_back_solver
        self.diagnostics = bool(diagnostics)
        self.verbose = bool(verbose)

        # Mapping state. The mapping itself is retained because it is part of
        # the solver state used to construct the proposal geometry.
        self.mapping_ = None

        # Optional diagnostics. These remain None when diagnostics=False.
        self.mapping_quality_ = None
        self.mapping_error_ = None
        self.ising_energies_ = None
        self.acceptance_rate_ = None
        self.acceptance_rates_ = None
        self.final_distance_ = None
        self.final_distances_ = None

        # Standard/public run state. energies_ follows p-kit's public energy
        # convention and is available regardless of diagnostics because E is
        # already required by the single-shot solver return contract.
        self.manifold_state_ = None
        self.energies_ = None

        self.used_fallback_ = False
        self.fallback_reason_ = None
        self.fallback_solver_ = None

    def _clear_run_state(self):
        """Clear state from a previous solve call."""
        self.mapping_ = None
        self.mapping_quality_ = None
        self.mapping_error_ = None
        self.manifold_state_ = None
        self.energies_ = None
        self.ising_energies_ = None
        self.acceptance_rate_ = None
        self.acceptance_rates_ = None
        self.final_distance_ = None
        self.final_distances_ = None
        self.used_fallback_ = False
        self.fallback_reason_ = None
        self.fallback_solver_ = None

    def _make_fallback(self):
        """Create or return the configured fallback solver."""
        fallback = self.fall_back_solver

        if fallback is None:
            return None

        if not isinstance(fallback, type):
            if not callable(getattr(fallback, "solve", None)):
                raise TypeError("fall_back_solver must expose solve(circuit)")
            return fallback

        available = {
            "Nt": self.Nt,
            "dt": self.dt,
            "i0": self.i0,
            "expected_mean": self.expected_mean,
            "seed": self.seed,
            "backend": self.backend,
            "tau": self.tau,
        }

        signature = inspect.signature(fallback)
        parameters = list(signature.parameters.values())
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

        if accepts_kwargs:
            kwargs = available.copy()
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

    def _call_fallback(self, fallback, circuit, annealing_func, n_shots):
        """Call a fallback while respecting the p-kit solver interface."""
        solve_signature = inspect.signature(fallback.solve)
        parameters = list(solve_signature.parameters.values())
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        parameter_names = {parameter.name for parameter in parameters}

        kwargs = {}
        if accepts_kwargs or "annealing_func" in parameter_names:
            kwargs["annealing_func"] = annealing_func
        if accepts_kwargs or "n_shots" in parameter_names:
            kwargs["n_shots"] = n_shots

        out = fallback.solve(circuit, **kwargs)

        if n_shots == 1:
            if not isinstance(out, tuple) or len(out) < 3:
                name = fallback.__class__.__name__
                raise TypeError(
                    f"{name}.solve(circuit) must follow the p-kit single-shot "
                    "solver contract and return (I, m, E)"
                )
            return out[0], out[1], out[2]

        return out

    def _run_fallback(
        self,
        circuit,
        reason,
        annealing_func,
        n_shots,
    ):
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

        return self._call_fallback(
            fallback,
            circuit,
            annealing_func,
            n_shots,
        )

    def _solve_single(self, J, h, target, annealing_func):
        """Run one independent SPD trajectory."""
        n = h.size
        rng = self._generator

        X = target.copy() if self.mode == "fast" else np.eye(n)
        state = rng.choice((-1.0, 1.0), size=n)

        # Raw Ising Hamiltonian used by the MH acceptance rule.
        current_energy = float(energy(state, J, h)[0])
        threshold = NormalDist().inv_cdf(1.0 - self.flip_prob)

        all_I = np.empty((self.Nt, n))
        all_m = np.empty((self.Nt, n))
        all_E = np.empty(self.Nt)

        # Allocate/compute these only when diagnostics are enabled.
        all_ising_E = np.empty(self.Nt) if self.diagnostics else None
        accepted = 0

        for t in range(self.Nt):
            a = float(annealing_func(self, t))
            if not np.isfinite(a) or a < 0:
                raise ValueError(
                    "annealing_func must return a finite, non-negative value; "
                    f"got {a!r} at timestep {t}"
                )

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

            # Correlated continuous proposal.
            flip = gaussian(rng, X) > threshold
            proposal = state.copy()
            proposal[flip] *= -1.0

            proposal_energy = float(energy(proposal, J, h)[0])
            delta_energy = proposal_energy - current_energy

            # ``a`` is the standard p-kit annealing/current scale. For uphill
            # moves delta_energy > 0, so the exponent is <= 0 and large values
            # safely underflow to zero (rejection).
            accept = (
                delta_energy <= 0
                or rng.random() < np.exp(-a * delta_energy)
            )

            if accept:
                state = proposal
                current_energy = proposal_energy
                if self.diagnostics:
                    accepted += 1

            all_m[t] = state
            all_I[t] = a * (J @ state + h)

            # Match the existing p-kit public energy convention:
            # E = i0 * (m@h + 0.5*m@J@m) = -i0 * H_raw(m)
            all_E[t] = -self.i0 * current_energy

            if self.diagnostics:
                all_ising_E[t] = current_energy

        if self.diagnostics:
            acceptance_rate = accepted / self.Nt
            final_distance = riem_dist(X, target)
        else:
            acceptance_rate = None
            final_distance = None

        return (
            all_I,
            all_m,
            all_E,
            all_ising_E,
            X,
            acceptance_rate,
            final_distance,
        )

    def solve(self, circuit, annealing_func=constant, n_shots=1):
        """Run the SPD correlated sampler.

        Parameters
        ----------
        circuit
            p-kit circuit or circuit-like object exposing dense J and h, or a
            ``synthesize(format="dense")`` method.
        annealing_func : callable, default=constant
            Standard p-kit annealing function with signature
            ``annealing_func(solver, run)``.
        n_shots : int, default=1
            Number of independent trajectories.

        Returns
        -------
        n_shots == 1
            ``(all_I, all_m, E)``
        n_shots > 1
            ``all_m`` with shape ``(Nt, n_shots, n_pbits)``.
        """
        if (
            not isinstance(n_shots, (int, np.integer))
            or isinstance(n_shots, (bool, np.bool_))
            or n_shots < 1
        ):
            raise ValueError("n_shots must be a positive integer")

        if not callable(annealing_func):
            raise TypeError("annealing_func must be callable")

        self._clear_run_state()

        J, h = dense_jh(circuit)
        n = h.size

        try:
            mapping = ising_to_spd(J, h, self.i0, self.margin)
            quality = mapping_quality(
                J,
                h,
                mapping,
                self.max_condition,
            )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            reason = f"SPD mapping failed: {exc}"
            return self._run_fallback(
                circuit,
                reason,
                annealing_func,
                n_shots,
            )

        self.mapping_ = mapping

        # The score must always be computed because it determines solver
        # validity/fallback. Retaining and extending the diagnostic state is
        # optional.
        if self.diagnostics:
            self.mapping_quality_ = quality
            self.mapping_error_ = mapping_identity_error(
                J,
                h,
                mapping,
                self.i0,
            )

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
            return self._run_fallback(
                circuit,
                reason,
                annealing_func,
                n_shots,
            )

        try:
            target = target_corr(mapping, n)
        except np.linalg.LinAlgError as exc:
            reason = f"failed to construct SPD target correlation: {exc}"
            return self._run_fallback(
                circuit,
                reason,
                annealing_func,
                n_shots,
            )

        if n_shots == 1:
            (
                all_I,
                all_m,
                all_E,
                all_ising_E,
                X,
                acceptance_rate,
                final_distance,
            ) = self._solve_single(
                J,
                h,
                target,
                annealing_func,
            )

            self.manifold_state_ = X
            self.energies_ = all_E

            if self.diagnostics:
                self.ising_energies_ = all_ising_E
                self.acceptance_rate_ = acceptance_rate
                self.acceptance_rates_ = np.asarray([acceptance_rate])
                self.final_distance_ = final_distance
                self.final_distances_ = np.asarray([final_distance])

            return all_I, all_m, all_E

        # NumPy backend: independent shots are run sequentially here. p-kit's
        # annealing.execute() will normally parallelize solver copies externally
        # because NumpyBackend does not prefer vectorized shots, but direct
        # solve(..., n_shots=N) remains supported and follows the common shape.
        all_m = np.empty((self.Nt, n_shots, n))
        energies = np.empty((self.Nt, n_shots))
        manifold_states = np.empty((n_shots, n, n))

        if self.diagnostics:
            ising_energies = np.empty((self.Nt, n_shots))
            acceptance_rates = np.empty(n_shots)
            final_distances = np.empty(n_shots)
        else:
            ising_energies = None
            acceptance_rates = None
            final_distances = None

        for shot in range(n_shots):
            (
                _all_I,
                shot_m,
                shot_E,
                shot_ising_E,
                shot_X,
                shot_acceptance,
                shot_distance,
            ) = self._solve_single(
                J,
                h,
                target,
                annealing_func,
            )

            all_m[:, shot, :] = shot_m
            energies[:, shot] = shot_E
            manifold_states[shot] = shot_X

            if self.diagnostics:
                ising_energies[:, shot] = shot_ising_E
                acceptance_rates[shot] = shot_acceptance
                final_distances[shot] = shot_distance

        self.manifold_state_ = manifold_states
        self.energies_ = energies

        if self.diagnostics:
            self.ising_energies_ = ising_energies
            self.acceptance_rates_ = acceptance_rates
            self.acceptance_rate_ = float(np.mean(acceptance_rates))
            self.final_distances_ = final_distances
            self.final_distance_ = float(np.mean(final_distances))

        return all_m

    def copy(self):
        """Return a new solver with the same configuration."""
        return SPDSolver(
            Nt=self.Nt,
            dt=self.dt,
            i0=self.i0,
            expected_mean=self.expected_mean,
            seed=self.seed,
            backend=self.backend,
            tau=self.tau,
            mode=self.mode,
            margin=self.margin,
            min_mapping_score=self.min_mapping_score,
            max_condition=self.max_condition,
            rate=self.rate,
            noise=self.noise,
            flip_prob=self.flip_prob,
            fall_back_solver=self.fall_back_solver,
            diagnostics=self.diagnostics,
            verbose=self.verbose,
        )
