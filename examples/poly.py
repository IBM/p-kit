import numpy as np
from p_kit.library.poly import PolyOptimizer
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.solver.annealing import constant, execute
from p_kit.visualization.histplot import histplot

# Minimize 3*x^2 + 2*x + 5  over integers [0, 15] (n_bits=4)
# Global minimum on non-negative integers: x=0  (f=5)
coeffs = {
    ('x', 'x'): 3,
    ('x',): 2,
    (): 5,
}
circuit = PolyOptimizer(coeffs, variables=['x'], n_bits=4, minimize=True)

solver = CaSuDaSolver(Nt=int(1e4), dt=0.1, i0=1.0, expected_mean=0, seed=42)
samples = execute(solver, circuit, constant, n_shots=100, n_last_samples=10, n_jobs=-1)

def f(x):
    return 3 * x ** 2 + 2 * x + 5

results = [circuit.decode(row) for row in samples]
best = min(results, key=lambda r: f(r['x']))
print(f"Best x = {best['x']},  f(x) = {f(best['x'])}")

histplot(samples, decode=lambda m: circuit.decode(m)['x'])
