
import numpy as np

def constant(solver, _run):
    return solver.i0

def linear(solver, run):
    return (0 - solver.i0) / solver.Nt * (run - 1) + solver.i0

def execute(solver, circuit, annealing, n_shots):
    samples = []
    for _ in range(n_shots):
        _, time_points, _ = solver.solve(circuit, annealing_func=annealing)
        samples.append(time_points[-1, :])
    return np.array(samples)