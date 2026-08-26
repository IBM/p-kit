import numpy as np

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver

try:
    import numba
except ImportError:
    numba = None


def _validate_initial_state(initial_state, n_pbits):
    state = np.asarray(initial_state)

    if state.size != n_pbits:
        raise ValueError(f"initial_state must contain {n_pbits} values")

    state = state.reshape(n_pbits)

    if not np.all((state == -1) | (state == 1)):
        raise ValueError("initial_state must contain only -1 or +1")

    return state


def _make_step(xp):
    """Build one CSD update (I(m) -> stochastic flip probability -> new m)
    closed over a specific array-library namespace.

    A closure, not a module-level function taking `xp` as an argument, so a
    backend's compile() can JIT it with no per-call indirection: numba's
    nopython mode can't type a module passed as a runtime argument, but it
    can JIT a function that closes over one (resolved once, at compile
    time). CaSuDaSolver builds one of these per backend and hands it to
    backend.compile() - the backend never needs to know what the function
    computes, only that it's a callable worth accelerating.
    """
    def _csd_step(m, J, h_eff, I_scale, threshold, dt, rnd):
        field = m @ J
        I = I_scale * (field + h_eff)
        s = xp.exp(-dt * xp.exp(-m * (I + threshold)))
        m = m * xp.sign(s - rnd)
        return m, I
    return _csd_step


if numba is not None:
    @numba.njit(cache=True)
    def _final_dense_numba(m, J, h, anneal, rnd, dt, threshold, tmp):
        ns, n = m.shape
        for run in range(len(anneal)):
            for s in range(ns):
                for i in range(n):
                    field = h[i]
                    for j in range(n):
                        field += m[s, j] * J[j, i]
                    I = anneal[run] * field
                    p = np.exp(-dt * np.exp(-m[s, i] * (I + threshold)))
                    tmp[s, i] = m[s, i] * (1.0 if p - rnd[run, s, i] >= 0 else -1.0)
            m[:] = tmp
        return m

    @numba.njit(cache=True)
    def _final_sparse_numba(m, indptr, indices, data, h, anneal, rnd, dt, threshold, tmp):
        ns, n = m.shape
        for run in range(len(anneal)):
            for s in range(ns):
                for i in range(n):
                    field = h[i]
                    for k in range(indptr[i], indptr[i + 1]):
                        field += data[k] * m[s, indices[k]]
                    I = anneal[run] * field
                    p = np.exp(-dt * np.exp(-m[s, i] * (I + threshold)))
                    tmp[s, i] = m[s, i] * (1.0 if p - rnd[run, s, i] >= 0 else -1.0)
            m[:] = tmp
        return m


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, 'p-bits for probabilistic spin logic', Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, backend=None,
                 tau=0.1, cache_J=False,
                 use_sparse=False, use_numba=False,
                 reuse_buffers=False, cache_static=False):
        super().__init__(Nt, dt, i0, expected_mean, seed, backend, tau)
        # cache_J avoids re-converting c.J to the backend's array type on
        # every solve() call when J is unchanged (matters for CuPy/Torch,
        # where that conversion is a host<->device transfer; a no-op copy
        # for NumPy). PCircuit.set_weight() bumps c._j_version, so this
        # detects weight updates made that way automatically - if c.J is
        # replaced or mutated through some other means, call clear_cache().
        self.cache_J = cache_J
        self._J_cache = None
        self._J_cache_key = None
        self._step = self.backend.compile(_make_step(self.backend.xp))

        # use_sparse/use_numba/reuse_buffers/cache_static power
        # solve(return_final=True)'s whole-loop fast path (_solve_final_fast
        # below): JIT-compile the *entire* Nt-step loop and skip trajectory
        # storage entirely, instead of just skipping storage (which the
        # generic loop in solve() already does for any backend). Only
        # NumPy backs this today - numba only understands plain NumPy
        # arrays, and use_sparse needs a backend that offers one (only
        # NumpyBackend does, via scipy.sparse).
        if use_numba and numba is None:
            raise ImportError(
                "numba is required for CaSuDaSolver(use_numba=True). "
                "Install with: pip install numba"
            )
        if use_sparse and not self.backend.supports_sparse:
            raise NotImplementedError(
                f"use_sparse=True is not supported by "
                f"{type(self.backend).__name__}"
            )
        if use_numba and self.backend.xp is not np:
            raise NotImplementedError(
                "use_numba=True currently requires a NumPy-array backend "
                "(numba only understands plain NumPy arrays)"
            )
        self.use_sparse = use_sparse
        self.use_numba = use_numba
        self.reuse_buffers = reuse_buffers
        self.cache_static = cache_static
        self._static_J = None
        self._static_Jsparse = None
        self._static_anneal = None
        self._static_anneal_key = None

    def clear_cache(self):
        self._J_cache = self._J_cache_key = None
        self._static_J = self._static_Jsparse = None
        self._static_anneal = self._static_anneal_key = None

    def _get_J(self, c):
        if not self.cache_J:
            return self.backend.asarray(c.J)

        key = (id(c), id(c.J), c.J.shape, getattr(c, '_j_version', 0))

        if self._J_cache is None or key != self._J_cache_key:
            self._J_cache = self.backend.asarray(c.J)
            self._J_cache_key = key

        return self._J_cache

    def _get_static_J(self, c):
        if not self.cache_static:
            J = np.asarray(c.J)
        elif self._static_J is None:
            self._static_J = np.asarray(c.J).copy()
            J = self._static_J
        else:
            J = self._static_J

        if not self.use_sparse:
            return J

        if not self.cache_static:
            return self.backend.sparse(J.T)

        if self._static_Jsparse is None:
            self._static_Jsparse = self.backend.sparse(J.T)
        return self._static_Jsparse

    def _get_static_anneal(self, func):
        key = (id(func), self.Nt, float(self.i0))
        if self.cache_static and self._static_anneal is not None \
                and key == self._static_anneal_key:
            return self._static_anneal

        a = np.asarray([func(self, r) for r in range(self.Nt)], dtype=float)
        if self.cache_static:
            self._static_anneal = a
            self._static_anneal_key = key
        return a

    def _solve_final_fast(self, c, annealing_func, n_shots, bias_func,
                           initial_state):
        """solve(return_final=True)'s use_numba/use_sparse whole-loop fast
        path: skips trajectory storage *and* the per-run Python loop
        overhead the generic path still has (via numba), or just storage
        (the eager sparse-only path, when bias_func rules numba out)."""
        n = c.n_pbits
        h = np.asarray(c.h).reshape(-1)
        J = self._get_static_J(c)
        anneal = self._get_static_anneal(annealing_func)
        threshold = float(np.arctanh(self.expected_mean))

        if initial_state is None:
            m = np.sign(0.5 - self.random((n_shots, n)))
        else:
            state = _validate_initial_state(initial_state, n)
            m = np.tile(state, (n_shots, 1))

        if self.use_numba and bias_func is None:
            rnd = np.asarray(self.random((self.Nt, n_shots, n)))
            tmp = self.backend.buffer("tmp", (n_shots, n)) \
                if self.reuse_buffers else np.empty((n_shots, n))

            if self.use_sparse:
                m = _final_sparse_numba(
                    m, J.indptr, J.indices, J.data, h, anneal, rnd,
                    self.dt, threshold, tmp
                )
            else:
                m = _final_dense_numba(
                    m, J, h, anneal, rnd, self.dt, threshold, tmp
                )

            return m[0] if n_shots == 1 else m

        m_filt = self.backend.buffer("m_filt", (n_shots, n)) \
            if self.reuse_buffers else np.empty((n_shots, n))
        m_filt.fill(0)

        for run in range(self.Nt):
            h_eff = h if bias_func is None else np.asarray(
                bias_func(run, m, m_filt)
            )

            field = J.dot(m.T).T if self.use_sparse else m @ J
            I = anneal[run] * (field + h_eff)
            s = np.exp(-self.dt * np.exp(-m * (I + threshold)))
            m = m * np.sign(s - self.random((n_shots, n)))
            m_filt = (1 - self.tau) * m_filt + self.tau * m

        return m[0] if n_shots == 1 else m

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1,
              bias_func=None, return_filtered=False, initial_state=None,
              return_final=False):
        """Run the CSD p-bit sampler.

        Parameters
        ----------
        bias_func : callable, optional
            Dynamical bias hook ``f(run, m, m_filt) -> array``. Called every
            timestep; its return value replaces the circuit's static ``h`` for
            that step and must broadcast against ``(n_shots, n_pbits)``. ``m`` is
            the current hard +/-1 state and ``m_filt`` is the filtered companion
            state *from the previous step* (temporal memory). When ``None``
            (default) the static ``c.h`` is used and behavior is identical to
            before — this is the recurrence/adaptive-biasing entry point.
        return_filtered : bool, optional
            When True, append the full ``m_filt`` trajectory to the return so
            callers can read the graded state / higher-moment statistics. The
            default return contract is unchanged. Cannot be combined with
            ``return_final``.
        initial_state : array-like, optional
            Initial +/-1 state for ``m``, tiled across shots. When ``None``
            (default) ``m`` is initialized randomly, as before.
        return_final : bool, optional
            When True, skip trajectory/current/energy storage and return only
            the final +/-1 state - useful for repeated/recurrent execution
            (e.g. a reservoir) where only the last state is ever read. If
            ``use_numba`` or ``use_sparse`` was set on this solver, uses the
            whole-loop fast path; otherwise the same loop with
            history-recording skipped.

        Returns
        -------
        Default (``return_filtered=False``, ``return_final=False``):
            ``n_shots == 1`` -> ``(all_I, all_m, E)``
            ``n_shots > 1``  -> ``all_m``
        With ``return_filtered=True``, ``all_mfilt`` is appended:
            ``n_shots == 1`` -> ``(all_I, all_m, E, all_mfilt)``
            ``n_shots > 1``  -> ``(all_m, all_mfilt)``
        With ``return_final=True``:
            ``n_shots == 1`` -> final state, shape ``(n_pbits,)``
            ``n_shots > 1``  -> final state, shape ``(n_shots, n_pbits)``
        """

        if return_final and return_filtered:
            raise ValueError(
                "return_final and return_filtered cannot both be True"
            )

        if return_final and (self.use_numba or self.use_sparse):
            return self._solve_final_fast(
                c, annealing_func, n_shots, bias_func, initial_state
            )

        # credit: https://www.purdue.edu/p-bit/blog.html
        backend = self.backend
        xp = backend.xp
        n_pbits = c.n_pbits

        J = self._get_J(c)
        h = backend.asarray(c.h).reshape(-1)  # Ensure h is 1D for proper broadcasting
        threshold = float(np.arctanh(self.expected_mean))
        tau = self.tau

        with backend.no_grad():
            # m is (n_shots, n_pbits) — works for n_shots=1 too
            if not return_final:
                all_m = backend.zeros((self.Nt, n_shots, n_pbits))
                all_I = backend.zeros((self.Nt, n_pbits))
                all_mfilt = backend.zeros((self.Nt, n_shots, n_pbits))
                E = backend.zeros((self.Nt,))

            if initial_state is None:
                m = xp.sign(0.5 - self.random((n_shots, n_pbits)))
            else:
                state = _validate_initial_state(initial_state, n_pbits)
                if n_shots > 1:
                    state = np.tile(state, (n_shots, 1))
                m = backend.asarray(state)

            # Filtered companion state (leaky integrator). Holds step (run-1) when
            # bias_func is evaluated, giving the hook temporal memory of the past.
            m_filt = backend.zeros((n_shots, n_pbits))

            for run in range(self.Nt):
                h_eff = h if bias_func is None else backend.asarray(bias_func(run, m, m_filt))
                I_scale = backend.asarray(annealing_func(self, run))
                rnd = self.random((n_shots, n_pbits))

                m, I = self._step(m, J, h_eff, I_scale, threshold, self.dt, rnd)

                if return_final:
                    continue

                m_filt = (1 - tau) * m_filt + tau * m
                all_m[run] = m
                all_I[run] = I[0]
                all_mfilt[run] = m_filt
                # Energy of shot 0 in the landscape actually sampled this step. Use
                # the shot-0 slice of h_eff so E stays consistent with all_I when a
                # bias_func is active (otherwise E would describe the static-h
                # landscape the sampler is no longer running in).
                h0 = h if bias_func is None else \
                    xp.broadcast_to(h_eff, (n_shots, n_pbits))[0]
                E[run] = self.i0 * (m[0] @ h0 + 0.5 * ((m[0] @ J) @ m[0]))

            if return_final:
                return backend.to_numpy(m if n_shots > 1 else m[0])

            all_I = backend.to_numpy(all_I)
            all_m = backend.to_numpy(all_m)
            all_mfilt = backend.to_numpy(all_mfilt)
            E = backend.to_numpy(E)

        if n_shots == 1:
            if return_filtered:
                return all_I, all_m[:, 0, :], E, all_mfilt[:, 0, :]
            return all_I, all_m[:, 0, :], E
        if return_filtered:
            return all_m, all_mfilt
        return all_m

    def copy(self):
        return CaSuDaSolver(
            self.Nt, self.dt, self.i0, self.expected_mean, self.seed,
            self.backend, self.tau, self.cache_J,
            self.use_sparse, self.use_numba, self.reuse_buffers,
            self.cache_static,
        )
