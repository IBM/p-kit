import numpy as np

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


def _validate_initial_state(initial_state, n_pbits):
    state = np.asarray(initial_state)

    if state.size != n_pbits:
        raise ValueError(f"initial_state must contain {n_pbits} values")

    state = state.reshape(n_pbits)

    if not np.all((state == -1) | (state == 1)):
        raise ValueError("initial_state must contain only -1 or +1")

    return state


def _csd_step(xp, m, J, h_eff, I_scale, threshold, dt, rnd):
    """One CSD update: I(m) -> stochastic flip probability -> new m.

    A free function (not a bound method) so a backend can JIT-compile it in
    isolation - PyTorch's compile() traces/caches a function object, and
    every CaSuDaSolver/backend combination should share the same compiled
    graph for this step rather than recompiling per solver instance.
    """
    field = m @ J
    I = I_scale * (field + h_eff)
    s = xp.exp(-dt * xp.exp(-m * (I + threshold)))
    m = m * xp.sign(s - rnd)
    return m, I


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, 'p-bits for probabilistic spin logic', Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, backend=None,
                 tau=0.1, cache_J=False):
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
        self._step = self.backend.compile(_csd_step)

    def clear_cache(self):
        self._J_cache = self._J_cache_key = None

    def _get_J(self, c):
        if not self.cache_J:
            return self.backend.asarray(c.J)

        key = (id(c), id(c.J), c.J.shape, getattr(c, '_j_version', 0))

        if self._J_cache is None or key != self._J_cache_key:
            self._J_cache = self.backend.asarray(c.J)
            self._J_cache_key = key

        return self._J_cache

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1,
              bias_func=None, return_filtered=False, initial_state=None):
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
            default return contract is unchanged.
        initial_state : array-like, optional
            Initial +/-1 state for ``m``, tiled across shots. When ``None``
            (default) ``m`` is initialized randomly, as before.

        Returns
        -------
        Default (``return_filtered=False``):
            ``n_shots == 1`` -> ``(all_I, all_m, E)``
            ``n_shots > 1``  -> ``all_m``
        With ``return_filtered=True``, ``all_mfilt`` is appended:
            ``n_shots == 1`` -> ``(all_I, all_m, E, all_mfilt)``
            ``n_shots > 1``  -> ``(all_m, all_mfilt)``
        """

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

                m, I = self._step(xp, m, J, h_eff, I_scale, threshold, self.dt, rnd)

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
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean,
                            self.seed, self.backend, self.tau, self.cache_J)
