import numpy as np

from p_kit.solver.annealing import constant
from p_kit.solver.base_solver import Solver


def hold(value=1.0):
    """Return a component schedule that holds one constant scale."""
    value = float(value)

    def schedule(_solver, _run):
        return value

    return schedule


def linear_ramp(start_value=0.0, end_value=1.0,
                start_fraction=0.0, end_fraction=1.0):
    """Ramp a correlation component over a fraction of the solver run."""
    start_value = float(start_value)
    end_value = float(end_value)
    start_fraction = float(start_fraction)
    end_fraction = float(end_fraction)

    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be in [0, 1]")
    if not 0.0 <= end_fraction <= 1.0:
        raise ValueError("end_fraction must be in [0, 1]")
    if end_fraction <= start_fraction:
        raise ValueError("end_fraction must be greater than start_fraction")

    def schedule(solver, run):
        if solver.Nt <= 1:
            x = 1.0
        else:
            x = run / (solver.Nt - 1)

        if x <= start_fraction:
            return start_value
        if x >= end_fraction:
            return end_value

        t = (x - start_fraction) / (end_fraction - start_fraction)
        return start_value + t * (end_value - start_value)

    return schedule


def staged_ramp(start_fraction=0.20, end_fraction=0.80):
    """0 -> 1 schedule: evidence-only, ramp correlations, full relaxation."""
    return linear_ramp(
        0.0, 1.0,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )


class CorrelationAnnealingSolver(Solver):
    """Gibbs solver with dynamically controlled named J/h components.

    A ``PCircuit`` may contain a static base ``J``/``h`` plus named
    correlation components created with ``set_correlation_component()`` or
    the correlation primitives on ``PCircuit``. Each component receives an
    independent time-dependent scale during a solve:

        J(t) = J_base + sum_r alpha_r(t) J_r
        h(t) = h_base + sum_r alpha_r(t) h_r

    The same component scale is applied to its J and h parts, so a Potts/Ising
    model can be annealed without breaking its exact J/h conversion.

    With ``block_size=None`` the solver performs ordinary sequential binary
    Gibbs p-bit updates. With ``block_size=K`` it performs exact categorical
    block updates: each contiguous K-p-bit group always contains exactly one
    +1 state. The latter is useful for one-hot feature variables while still
    allowing the inter-block correlations to be controlled through J.

    This first implementation deliberately uses the NumPy backend only. It is
    intended to establish the primitive and algorithm cleanly before adding
    optimized Torch/CuPy kernels.
    """

    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, backend=None,
                 tau=0.1, component_schedules=None, block_size=None):
        super().__init__(Nt, dt, i0, expected_mean, seed, backend, tau)

        if self.backend.xp is not np:
            raise NotImplementedError(
                "CorrelationAnnealingSolver currently supports NumpyBackend only"
            )

        if block_size is not None and int(block_size) < 2:
            raise ValueError("block_size must be >= 2")

        self.block_size = None if block_size is None else int(block_size)
        self.component_schedules = dict(component_schedules or {})

    def _components(self, c):
        return {
            name: c.get_correlation_component(name)
            for name in c.correlation_components()
        }

    def _schedule_value(self, name, run):
        schedule = self.component_schedules.get(name, 1.0)
        if callable(schedule):
            return float(schedule(self, run))
        return float(schedule)

    def _compose(self, c, run):
        J = np.asarray(c.J, dtype=float).copy()
        h = np.asarray(c.h, dtype=float).reshape(-1).copy()
        scales = {}

        for name, comp in self._components(c).items():
            scale = self._schedule_value(name, run)
            scales[name] = scale
            J += scale * np.asarray(comp["J"], dtype=float)
            h += scale * np.asarray(comp["h"], dtype=float).reshape(-1)

        return J, h, scales

    def _compose_target(self, c, target_scales):
        components = self._components(c)
        scales = {} if target_scales is None else dict(target_scales)

        J = np.asarray(c.J, dtype=float).copy()
        h = np.asarray(c.h, dtype=float).reshape(-1).copy()

        for name, comp in components.items():
            scale = float(scales.get(name, 1.0))
            J += scale * np.asarray(comp["J"], dtype=float)
            h += scale * np.asarray(comp["h"], dtype=float).reshape(-1)

        return J, h

    @staticmethod
    def _energy(m, J, h):
        return (
            -0.5 * np.einsum("bi,ij,bj->b", m, J, m)
            - m @ h
        )

    def _validate_binary_initial(self, initial_state, n_shots, n_pbits):
        if initial_state is None:
            return np.where(
                self.random((n_shots, n_pbits)) < 0.5,
                -1.0,
                1.0,
            )

        state = np.asarray(initial_state, dtype=float)
        if state.ndim == 1:
            if state.size != n_pbits:
                raise ValueError(f"initial_state must contain {n_pbits} values")
            state = np.tile(state.reshape(1, -1), (n_shots, 1))
        elif state.shape != (n_shots, n_pbits):
            raise ValueError(
                f"initial_state shape {state.shape}, expected {(n_shots, n_pbits)}"
            )

        if not np.all((state == -1) | (state == 1)):
            raise ValueError("initial_state must contain only -1 or +1")
        return state.copy()

    def _validate_block_initial(self, initial_state, n_shots, n_pbits):
        K = self.block_size
        if n_pbits % K:
            raise ValueError(
                f"n_pbits ({n_pbits}) must be divisible by block_size ({K})"
            )

        n_blocks = n_pbits // K

        if initial_state is None:
            m = -np.ones((n_shots, n_pbits), dtype=float)
            winners = self._generator.integers(
                0, K, size=(n_shots, n_blocks)
            )
            for s in range(n_shots):
                for blk in range(n_blocks):
                    m[s, blk * K + winners[s, blk]] = 1.0
            return m

        m = self._validate_binary_initial(initial_state, n_shots, n_pbits)
        blocks = m.reshape(n_shots, n_blocks, K)
        active = (blocks > 0).sum(axis=2)
        if not np.all(active == 1):
            raise ValueError(
                "block initial_state must contain exactly one +1 per block"
            )
        return m

    def _validate_J_components(self, c, tol=1e-12):
        matrices = [("base", np.asarray(c.J, dtype=float))]
        matrices.extend(
            (name, np.asarray(comp["J"], dtype=float))
            for name, comp in self._components(c).items()
        )

        for name, J in matrices:
            if not np.allclose(J, J.T, atol=tol, rtol=0):
                raise ValueError(
                    f"component '{name}' must be symmetric for Gibbs energy sampling"
                )

            if self.block_size is None:
                if np.max(np.abs(np.diag(J))) > tol:
                    raise ValueError(
                        f"component '{name}' must have zero diagonal for binary Gibbs updates"
                    )
                continue

            K = self.block_size
            n = c.n_pbits
            for start in range(0, n, K):
                block = J[start:start + K, start:start + K]
                if np.max(np.abs(block)) > tol:
                    raise ValueError(
                        f"component '{name}' has intra-block J couplings; "
                        "exact categorical block updates require zero "
                        "coupling within each block"
                    )

    def _binary_sweep(self, m, J, h, beta):
        n_shots, n_pbits = m.shape
        field = m @ J + h

        for i in self._generator.permutation(n_pbits):
            i = int(i)
            logits = 2.0 * beta * field[:, i]
            logits = np.clip(logits, -60.0, 60.0)
            p_plus = 1.0 / (1.0 + np.exp(-logits))
            new = np.where(self.random((n_shots,)) < p_plus, 1.0, -1.0)
            delta = new - m[:, i]
            changed = delta != 0
            if np.any(changed):
                m[:, i] = new
                field += delta[:, None] * J[i, :][None, :]

        return m

    def _block_sweep(self, m, J, h, beta):
        K = self.block_size
        n_shots, n_pbits = m.shape
        n_blocks = n_pbits // K
        field = m @ J + h

        for blk in self._generator.permutation(n_blocks):
            blk = int(blk)
            start = blk * K
            idx = slice(start, start + K)

            # For one +1 and K-1 -1 states, changing the winner changes
            # the Ising energy by -2*field_k up to a block-independent
            # constant. Hence the exact categorical conditional is
            # softmax(2*beta*field_k).
            logits = 2.0 * beta * field[:, idx]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)

            J_block = J[start:start + K, :]

            for s in range(n_shots):
                k_new = self._generator.choice(K, p=p[s])
                old = m[s, idx].copy()
                if old[k_new] > 0:
                    continue

                new = -np.ones(K, dtype=float)
                new[k_new] = 1.0
                delta = new - old
                m[s, idx] = new
                field[s] += delta @ J_block

        return m

    def solve(self, c, annealing_func=constant, n_shots=1,
              initial_state=None, return_final=False, return_best=False,
              target_scales=None):
        """Run controlled-correlation Gibbs annealing.

        ``component_schedules`` control how each named J/h component is
        introduced. ``annealing_func`` still controls the ordinary p-kit
        inverse-temperature/current scale (``constant`` means ``i0``).

        ``return_best=True`` tracks, for every shot, the lowest energy state
        seen under the FULL target model. The initial state is included, so
        the stochastic refinement never loses the best state it started from.
        """
        if return_final and return_best:
            raise ValueError("return_final and return_best are mutually exclusive")

        n_pbits = c.n_pbits
        self._validate_J_components(c)
        if self.block_size is None:
            m = self._validate_binary_initial(initial_state, n_shots, n_pbits)
        else:
            m = self._validate_block_initial(initial_state, n_shots, n_pbits)

        target_J, target_h = self._compose_target(c, target_scales)
        best_m = m.copy()
        best_E = self._energy(m, target_J, target_h)

        if not return_final and not return_best:
            all_m = np.zeros((self.Nt, n_shots, n_pbits), dtype=float)
            all_E = np.zeros((self.Nt, n_shots), dtype=float)
            all_scales = []

        for run in range(self.Nt):
            J, h, scales = self._compose(c, run)
            beta = float(annealing_func(self, run))

            if self.block_size is None:
                m = self._binary_sweep(m, J, h, beta)
            else:
                m = self._block_sweep(m, J, h, beta)

            target_E = self._energy(m, target_J, target_h)
            improved = target_E < best_E
            if np.any(improved):
                best_E[improved] = target_E[improved]
                best_m[improved] = m[improved]

            if not return_final and not return_best:
                all_m[run] = m
                all_E[run] = target_E
                all_scales.append(scales)

        if return_best:
            return best_m, best_E
        if return_final:
            return m[0] if n_shots == 1 else m

        if n_shots == 1:
            return all_m[:, 0, :], all_E[:, 0], all_scales
        return all_m, all_E, all_scales

    def copy(self):
        return CorrelationAnnealingSolver(
            Nt=self.Nt,
            dt=self.dt,
            i0=self.i0,
            expected_mean=self.expected_mean,
            seed=self.seed,
            backend=self.backend,
            tau=self.tau,
            component_schedules=self.component_schedules,
            block_size=self.block_size,
        )

