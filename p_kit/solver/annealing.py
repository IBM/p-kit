import numpy as np
from joblib import Parallel, delayed

def constant(solver, _run):
    return solver.i0

def linear(solver, run):
    return (0 - solver.i0) / solver.Nt * (run - 1) + solver.i0

def execute(solver, circuit, annealing, n_shots, n_last_samples=50, n_jobs=-1):
    def job():
        _, time_points, _ = solver.copy().solve(circuit.copy(), annealing_func=annealing)
        return time_points[-n_last_samples-1:-1, :]

    samples = Parallel(n_jobs=n_jobs)(delayed(job)() for _ in range(n_shots))

    return np.array(samples).reshape((n_shots * n_last_samples, circuit.n_pbits))
