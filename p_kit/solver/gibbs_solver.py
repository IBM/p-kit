from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.annealing import constant
from .base_solver import Solver


class GibbsSolver(Solver):
    """Asynchronous single-p-bit-per-sweep Gibbs sampler for Ising-type models.

    S. Geman and D. Geman, 'Stochastic relaxation, Gibbs distributions, and
    the Bayesian restoration of images', IEEE Transactions on Pattern
    Analysis and Machine Intelligence, vol. PAMI-6, no. 6, pp. 721-741, 1984,
    doi: 10.1109/TPAMI.1984.4767596.

    Unlike CaSuDaSolver's synchronous, vectorized update of all p-bits at
    once, this solver resamples each p-bit exactly once per sweep, in a
    random order, conditioned on the current state of all the others — the
    textbook asynchronous Gibbs/heat-bath update for PSL/Ising models.

    ``Nt`` counts *sweeps*, not individual single-p-bit updates: each sweep
    performs ``n_pbits`` sequential updates internally, but only the
    post-sweep state is recorded. This mirrors CaSuDaSolver's convention
    that every ``Nt`` tick updates every p-bit once, so the same ``Nt``
    (and the same ``i0``) represents comparable computational work and
    comparable mixing between the two solvers. It is slower to mix than
    CaSuDa's fully parallel update (sequential vs. synchronous within a
    sweep), but provides a reference implementation to validate CaSuDa's
    synchronous approximation against.
    """

    # `dt` and `tau` are accepted (via the inherited Solver.__init__) only
    # for interface parity with CaSuDaSolver, so this solver is a drop-in
    # swap. They are not used here: Gibbs sampling uses a direct sigmoid
    # acceptance probability, not CaSuDa's leaky flip-rate model.

    def solve(self, c: PCircuit, annealing_func=constant, n_shots=1):
        """Run the asynchronous Gibbs p-bit sampler.

        Each of the ``Nt`` steps is a full sweep: every p-bit is resampled
        exactly once, in a random order, using a heat-bath/Gibbs
        probability conditioned on the others' current state.

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
        # than recomputing the full m @ J product on every one of the
        # n_pbits flips within a sweep. This also lets the energy below be
        # computed in O(n_pbits) instead of the O(n_pbits^2) of m @ J @ m.
        field = m @ J + h

        all_m = xp.zeros((self.Nt, n_shots, n_pbits))
        all_I = xp.zeros((self.Nt, n_pbits))
        E = xp.zeros(self.Nt)

        for run in range(self.Nt):
            a = annealing_func(self, run)
            # Pre-sweep field: the drive that seeds this step's transition,
            # recorded as all_I[run] to match CaSuDaSolver's convention
            # that all_I[run] is the current which produced all_m[run].
            all_I[run] = (a * field)[0]

            for idx in self._random_gen.permutation(n_pbits):
                idx = int(idx)
                p = 1 / (1 + xp.exp(-2 * a * field[:, idx]))
                u = self.random(n_shots)
                new = xp.where(u < p, 1.0, -1.0)
                delta = new - m[:, idx]
                m[:, idx] = new
                # J is symmetric, so column idx == row idx; flipping p-bit
                # idx by `delta` shifts every p-bit's field by delta * J[idx].
                field = field + delta[:, None] * J[idx, :]

            all_m[run] = m
            # E = i0*(m@h + 0.5*m@J@m); since field = m@J+h, m@J@m reduces
            # to (field-h)@m, so E = 0.5*i0*m@(h+field) — O(n_pbits) using
            # the field already maintained above, no O(n_pbits^2) matmul.
            E[run] = 0.5 * self.i0 * xp.dot(m[0], h + field[0])

        if self.device == 'cuda':
            all_I, all_m, E = all_I.get(), all_m.get(), E.get()

        if n_shots == 1:
            return all_I, all_m[:, 0, :], E
        return all_m

    def copy(self):
        return GibbsSolver(self.Nt, self.dt, self.i0, self.expected_mean,
                           self.seed, self.device, self.tau)
