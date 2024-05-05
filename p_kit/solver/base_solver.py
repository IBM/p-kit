class Solver:
    def __init__(self, Nt, dt, i0, threshold=0) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0
        self.threshold = threshold

    def solve(self):
        raise NotImplementedError()
