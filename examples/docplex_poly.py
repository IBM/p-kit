try:
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
except ImportError:
    raise SystemExit(0)  # skip gracefully if docplex is not installed
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.solver.annealing import constant, execute
from p_kit.visualization.histplot import histplot

# Build the docplex model: minimize 3*x^2 + 2*x + 5
mdl = Model(name="poly")
x = mdl.integer_var(lb=0, ub=15, name="x")
mdl.minimize(3 * x * x + 2 * x + 5)

# Convert to p-bit circuit (n_bits auto-inferred from ub=15 → 4 bits)
circuit = DocplexOptimizer(mdl)

solver = CaSuDaSolver(Nt=int(1e4), dt=0.1, i0=1.0, expected_mean=0, seed=42)
samples = execute(solver, circuit, constant, n_shots=100, n_last_samples=10, n_jobs=-1)

histplot(samples, decode=lambda m: circuit.decode(m)['x'])
