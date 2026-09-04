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
    start_value, end_value = float(start_value), float(end_value)
    start_fraction, end_fraction = float(start_fraction), float(end_fraction)
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be in [0, 1]")
    if not 0.0 <= end_fraction <= 1.0:
        raise ValueError("end_fraction must be in [0, 1]")
    if end_fraction <= start_fraction:
        raise ValueError("end_fraction must be greater than start_fraction")

    def schedule(solver, run):
        x = 1.0 if solver.Nt <= 1 else run / (solver.Nt - 1)
        if x <= start_fraction:
            return start_value
        if x >= end_fraction:
            return end_value
        t = (x - start_fraction) / (end_fraction - start_fraction)
        return start_value + t * (end_value - start_value)
    return schedule


def staged_ramp(start_fraction=0.20, end_fraction=0.80):
    """0 -> 1 schedule: evidence-only, ramp correlations, full relaxation."""
    return linear_ramp(0.0, 1.0, start_fraction, end_fraction)


class CorrelationAnnealingSolver(Solver):
    """Gibbs solver with dynamically controlled named J/h components.

    J(t) = J_base + sum_r alpha_r(t) J_r
    h(t) = h_base + sum_r alpha_r(t) h_r

    block_size=None uses binary Gibbs updates; block_size=K uses exact
    categorical K-p-bit updates. Validation and block-sparse J layouts are
    cached across solves; component fields are updated incrementally.
    """

    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, backend=None,
                 tau=0.1, component_schedules=None, block_size=None):
        super().__init__(Nt, dt, i0, expected_mean, seed, backend, tau)
        if self.backend.xp is not np:
            raise NotImplementedError("CorrelationAnnealingSolver currently supports NumpyBackend only")
        if block_size is not None and int(block_size) < 2:
            raise ValueError("block_size must be >= 2")
        self.block_size = None if block_size is None else int(block_size)
        self.component_schedules = dict(component_schedules or {})
        self._structure_key = None
        self._layouts = None

    def _components(self, c):
        return {name: c.get_correlation_component(name)
                for name in c.correlation_components()}

    def _schedule_value(self, name, run):
        schedule = self.component_schedules.get(name, 1.0)
        return float(schedule(self, run) if callable(schedule) else schedule)

    def _cache_key(self, c, components):
        return (id(c), id(c.J), getattr(c, "_j_version", 0), self.block_size,
                tuple((name, id(comp["J"])) for name, comp in components.items()))

    def _validate_J_components(self, c, components, tol=1e-12):
        matrices = [("base", np.asarray(c.J, dtype=float))]
        matrices += [(name, np.asarray(comp["J"], dtype=float))
                     for name, comp in components.items()]
        for name, J in matrices:
            if not np.allclose(J, J.T, atol=tol, rtol=0):
                raise ValueError(f"component '{name}' must be symmetric for Gibbs energy sampling")
            if self.block_size is None:
                if np.max(np.abs(np.diag(J))) > tol:
                    raise ValueError(f"component '{name}' must have zero diagonal for binary Gibbs updates")
            else:
                K = self.block_size
                for start in range(0, c.n_pbits, K):
                    if np.max(np.abs(J[start:start+K, start:start+K])) > tol:
                        raise ValueError(
                            f"component '{name}' has intra-block J couplings; "
                            "exact categorical block updates require zero coupling within each block"
                        )

    def _block_layout(self, J):
        K, n = self.block_size, len(J)
        nb = n // K
        rows, count = [], 0
        for a in range(nb):
            row = []
            for b in range(nb):
                W = J[a*K:(a+1)*K, b*K:(b+1)*K]
                if np.any(W):
                    row.append((b, W))
                    count += 1
            rows.append(row)
        return rows if count < nb * nb / 2 else None

    def _prepare(self, c, components):
        key = self._cache_key(c, components)
        if key == self._structure_key:
            return self._layouts
        self._validate_J_components(c, components)
        if self.block_size is None:
            layouts = None
        else:
            layouts = {"base": self._block_layout(np.asarray(c.J, dtype=float))}
            layouts.update({name: self._block_layout(np.asarray(comp["J"], dtype=float))
                            for name, comp in components.items()})
        self._structure_key, self._layouts = key, layouts
        return layouts

    def _validate_binary_initial(self, initial_state, n_shots, n_pbits):
        if initial_state is None:
            return np.where(self.random((n_shots, n_pbits)) < 0.5, -1.0, 1.0)
        m = np.asarray(initial_state, dtype=float)
        if m.ndim == 1:
            if m.size != n_pbits:
                raise ValueError(f"initial_state must contain {n_pbits} values")
            m = np.tile(m, (n_shots, 1))
        elif m.shape != (n_shots, n_pbits):
            raise ValueError(f"initial_state shape {m.shape}, expected {(n_shots, n_pbits)}")
        if not np.all((m == -1) | (m == 1)):
            raise ValueError("initial_state must contain only -1 or +1")
        return m.copy()

    def _validate_block_initial(self, initial_state, n_shots, n_pbits):
        K = self.block_size
        if n_pbits % K:
            raise ValueError(f"n_pbits ({n_pbits}) must be divisible by block_size ({K})")
        nb = n_pbits // K
        if initial_state is None:
            m = -np.ones((n_shots, nb, K))
            winners = self._generator.integers(0, K, size=(n_shots, nb))
            m[np.arange(n_shots)[:, None], np.arange(nb), winners] = 1.0
            return m.reshape(n_shots, n_pbits)
        m = self._validate_binary_initial(initial_state, n_shots, n_pbits)
        if not np.all((m.reshape(n_shots, nb, K) > 0).sum(axis=2) == 1):
            raise ValueError("block initial_state must contain exactly one +1 per block")
        return m

    @staticmethod
    def _target_energy(m, fields, base_h, comp_h, scales):
        field = fields["base"].copy()
        h = base_h.copy()
        for name, scale in scales.items():
            field += scale * fields[name]
            h += scale * comp_h[name]
        return -0.5 * np.sum(m * field, axis=1) - m @ h

    def _apply_block_delta(self, field, live, delta, blk, J, layout, scale=1.0):
        K = self.block_size
        if layout is None:
            d = delta @ J[blk*K:(blk+1)*K, :]
            field += d
            live += scale * d
            return
        for dst, W in layout[blk]:
            idx = slice(dst*K, (dst+1)*K)
            d = delta @ W
            field[:, idx] += d
            live[:, idx] += scale * d

    def _binary_sweep(self, m, base_J, comp_J, base_h, comp_h, fields, scales, beta):
        live = fields["base"] + base_h
        for name, scale in scales.items():
            live += scale * (fields[name] + comp_h[name])

        for i in self._generator.permutation(m.shape[1]):
            logits = np.clip(2.0 * beta * live[:, i], -60.0, 60.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            new = np.where(self.random((len(m),)) < p, 1.0, -1.0)
            delta = new - m[:, i]
            m[:, i] = new

            d = delta[:, None] * base_J[i][None, :]
            fields["base"] += d
            live += d
            for name, scale in scales.items():
                d = delta[:, None] * comp_J[name][i][None, :]
                fields[name] += d
                live += scale * d
        return m

    def _block_sweep(self, m, base_J, comp_J, base_h, comp_h,
                     fields, scales, layouts, beta):
        K, nb = self.block_size, m.shape[1] // self.block_size
        live = fields["base"] + base_h
        for name, scale in scales.items():
            live += scale * (fields[name] + comp_h[name])

        for blk in self._generator.permutation(nb):
            idx = slice(blk*K, (blk+1)*K)
            logits = 2.0 * beta * live[:, idx]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)

            cdf = np.cumsum(p, axis=1)
            cdf[:, -1] = 1.0
            winner = (cdf < self.random((len(m), 1))).sum(axis=1)

            new = -np.ones((len(m), K))
            new[np.arange(len(m)), winner] = 1.0
            delta = new - m[:, idx]
            m[:, idx] = new

            self._apply_block_delta(fields["base"], live, delta, blk,
                                    base_J, layouts["base"])
            for name, scale in scales.items():
                self._apply_block_delta(fields[name], live, delta, blk,
                                        comp_J[name], layouts[name], scale)
        return m

    def solve(self, c, annealing_func=constant, n_shots=1,
              initial_state=None, return_final=False, return_best=False,
              target_scales=None):
        """Run controlled-correlation Gibbs annealing."""
        if return_final and return_best:
            raise ValueError("return_final and return_best are mutually exclusive")

        components = self._components(c)
        layouts = self._prepare(c, components)
        base_J = np.asarray(c.J, dtype=float)
        base_h = np.asarray(c.h, dtype=float).reshape(-1)
        comp_J = {name: np.asarray(comp["J"], dtype=float) for name, comp in components.items()}
        comp_h = {name: np.asarray(comp["h"], dtype=float).reshape(-1)
                  for name, comp in components.items()}

        if self.block_size is None:
            m = self._validate_binary_initial(initial_state, n_shots, c.n_pbits)
        else:
            m = self._validate_block_initial(initial_state, n_shots, c.n_pbits)

        fields = {"base": m @ base_J}
        fields.update({name: m @ J for name, J in comp_J.items()})
        target = {name: float((target_scales or {}).get(name, 1.0))
                  for name in components}

        best_m = m.copy()
        best_E = self._target_energy(m, fields, base_h, comp_h, target)

        if not return_final and not return_best:
            all_m = np.zeros((self.Nt, n_shots, c.n_pbits))
            all_E = np.zeros((self.Nt, n_shots))
            all_scales = []

        for run in range(self.Nt):
            scales = {name: self._schedule_value(name, run) for name in components}
            beta = float(annealing_func(self, run))

            if self.block_size is None:
                m = self._binary_sweep(m, base_J, comp_J, base_h, comp_h,
                                       fields, scales, beta)
            else:
                m = self._block_sweep(m, base_J, comp_J, base_h, comp_h,
                                      fields, scales, layouts, beta)

            target_E = self._target_energy(m, fields, base_h, comp_h, target)
            improved = target_E < best_E
            best_E[improved] = target_E[improved]
            best_m[improved] = m[improved]

            if not return_final and not return_best:
                all_m[run], all_E[run] = m, target_E
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
            Nt=self.Nt, dt=self.dt, i0=self.i0,
            expected_mean=self.expected_mean, seed=self.seed,
            backend=self.backend, tau=self.tau,
            component_schedules=self.component_schedules,
            block_size=self.block_size,
        )