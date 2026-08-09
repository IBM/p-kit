from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver
import numpy as np


class CaSuDaSolver(Solver):
    # K. Y. Camsari, B. M. Sutton, and S. Datta, 'p-bits for probabilistic spin logic', Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1,
              bias_func=None, return_filtered=False):
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
        xp = self.xp
        n_pbits = c.n_pbits

        J = xp.asarray(c.J)
        h = xp.asarray(c.h).flatten()  # Ensure h is 1D for proper broadcasting
        threshold = float(np.arctanh(self.expected_mean))
        tau = self.tau

        # m is (n_shots, n_pbits) — works for n_shots=1 too
        all_m = xp.zeros((self.Nt, n_shots, n_pbits))
        all_I = xp.zeros((self.Nt, n_pbits))
        all_mfilt = xp.zeros((self.Nt, n_shots, n_pbits))
        E = xp.zeros(self.Nt)
        m = xp.sign(0.5 - self.random((n_shots, n_pbits)))
        # Filtered companion state (leaky integrator). Holds step (run-1) when
        # bias_func is evaluated, giving the hook temporal memory of the past.
        m_filt = xp.zeros((n_shots, n_pbits))

        for run in range(self.Nt):
            h_eff = h if bias_func is None else xp.asarray(bias_func(run, m, m_filt))
            I = annealing_func(self, run) * (m @ J + h_eff)
            s = xp.exp(-self.dt * xp.exp(-m * (I + threshold)))
            m = m * xp.sign(s - self.random((n_shots, n_pbits)))
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
            E[run] = self.i0 * (xp.dot(m[0], h0) + 0.5 * xp.dot(xp.dot(m[0], J), m[0]))

        if self.device == 'cuda':
            all_I, all_m, E = all_I.get(), all_m.get(), E.get()
            all_mfilt = all_mfilt.get()

        if n_shots == 1:
            if return_filtered:
                return all_I, all_m[:, 0, :], E, all_mfilt[:, 0, :]
            return all_I, all_m[:, 0, :], E
        if return_filtered:
            return all_m, all_mfilt
        return all_m

    def copy(self):
        return CaSuDaSolver(self.Nt, self.dt, self.i0, self.expected_mean,
                            self.seed, self.device, self.tau)
