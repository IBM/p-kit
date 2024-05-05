class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.expected_mean = expected_mean

    def solve(self):
        raise NotImplementedError()
