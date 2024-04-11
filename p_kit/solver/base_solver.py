from p_kit.core.p_circuit import PCircuit


class Solver:
    def __init__(self, Nt, dt, i0) -> None:
        self.Nt = Nt
        self.dt = dt
        self.i0 = i0

    def solve(self, _c: PCircuit):
        raise NotImplementedError()
