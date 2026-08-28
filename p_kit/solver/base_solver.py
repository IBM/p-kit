from p_kit.backends import NumpyBackend


class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0, seed=None, backend=None,
                 tau=0.1) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean
        self.seed = seed
        self.backend = backend if backend is not None else NumpyBackend()
        # tau: leaky-integrator constant for the filtered companion state m_filt
        # (0 < tau <= 1). m_filt tracks a running, graded [-1, 1] estimate of each
        # p-bit's activity — the "extend +/-1 with filters" primitive. It never
        # affects the hard +/-1 sampler unless a bias_func consumes it.
        self.tau = tau
        self._generator = self.backend.make_generator(self.seed)

    def random(self, shape):
        return self.backend.random(self._generator, shape)

    def solve(self, c, annealing_func=None, n_shots=1):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()
