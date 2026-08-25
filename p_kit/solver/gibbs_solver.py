from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


class GibbsSolver(Solver):
    """Asynchronous single-p-bit-per-step Gibbs sampler for Ising-type models.

    S. Geman and D. Geman, 'Stochastic relaxation, Gibbs distributions, and
    the Bayesian restoration of images', IEEE Transactions on Pattern
    Analysis and Machine Intelligence, vol. PAMI-6, no. 6, pp. 721-741, 1984,
    doi: 10.1109/TPAMI.1984.4767596.

    Unlike CaSuDaSolver's synchronous, vectorized update of all p-bits at
    once, this solver resamples exactly one p-bit per timestep conditioned
    on the current state of all the others — the textbook asynchronous
    Gibbs/Metropolis-style update for PSL/Ising models. It is slower but
    provides a reference implementation to validate CaSuDa's synchronous
    approximation against.
    """

    # `dt` and `tau` are accepted (via the inherited Solver.__init__) only
    # for interface parity with CaSuDaSolver, so this solver is a drop-in
    # swap. They are not used here: Gibbs sampling uses a direct sigmoid
    # acceptance probability, not CaSuDa's leaky flip-rate model.

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1):
        """Run the asynchronous Gibbs p-bit sampler.

        Returns
        -------
        ``n_shots == 1`` -> ``(all_I, all_m, E)``
        ``n_shots > 1``  -> ``all_m``
        """

        xp = self.xp
        n_pbits = c.n_pbits

        J = xp.asarray(c.J)
        h = xp.asarray(c.h).flatten()  # Ensure h is 1D for proper broadcasting

        # m is (n_shots, n_pbits) — works for n_shots=1 too
        m = xp.sign(0.5 - self.random((n_shots, n_pbits)))

        # Unscaled local field per shot, kept in sync with m via a cheap
        # incremental update on each single-p-bit flip (see below) rather
        # than recomputing the full m @ J product every step — the point of
        # an asynchronous single-p-bit sampler is that a step touches only
        # one p-bit's worth of coupling, not the whole circuit's.
        field = m @ J + h

        all_m = xp.zeros((self.Nt, n_shots, n_pbits))
        all_I = xp.zeros((self.Nt, n_pbits))
        E = xp.zeros(self.Nt)

        for run in range(self.Nt):
            idx = int(self._random_gen.integers(0, n_pbits))
            a = annealing_func(self, run)
            # Pre-update field: the drive that decides this step's flip,
            # recorded as all_I[run] to match CaSuDaSolver's convention that
            # all_I[run] is the current which produced all_m[run].
            I_full = a * field
            p = 1 / (1 + xp.exp(-2 * I_full[:, idx]))
            u = self.random(n_shots)
            new = xp.where(u < p, 1.0, -1.0)
            delta = new - m[:, idx]
            m[:, idx] = new
            # J is symmetric, so column idx == row idx; flipping p-bit idx
            # by `delta` shifts every other p-bit's field by delta * J[idx].
            field = field + delta[:, None] * J[idx, :]

            all_I[run] = I_full[0]
            all_m[run] = m
            E[run] = self.i0 * (xp.dot(m[0], h) +
                                0.5 * xp.dot(xp.dot(m[0], J), m[0]))

        if self.device == 'cuda':
            all_I, all_m, E = all_I.get(), all_m.get(), E.get()

        if n_shots == 1:
            return all_I, all_m[:, 0, :], E
        return all_m

    def copy(self):
        return GibbsSolver(self.Nt, self.dt, self.i0, self.expected_mean,
                           self.seed, self.device, self.tau)
