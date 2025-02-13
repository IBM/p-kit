
def constant(solver, _run):
    return solver.i0

def linear(solver, run):
    return (0 - solver.i0) / solver.Nt * (run - 1) + solver.i0
