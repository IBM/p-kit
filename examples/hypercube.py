"""Visualize a p-circuit's {-1,+1}^n state space as a hypercube.

See examples/llm/llm_demo_005.py for a third case: a 128-p-bit reservoir,
whose state space is far too large to enumerate, so hypercube_plot instead
PCA-projects the sampled trajectory. That demo needs the optional
torch/transformers/numba dependencies (pip install p-kit[llm]), so it's
kept out of this dependency-free example.
"""

from p_kit.psl.gates import ANDGate, FullAdder
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import hypercube_plot

# n_pbits=3 -> a literal cube. The p-bit dynamics should spend most of
# their time on the 4 vertices where output == input1 AND input2.
and_gate = ANDGate()
solver = CaSuDaSolver(Nt=20000, dt=0.1667, i0=0.8, seed=0)
_, all_m, _ = solver.solve(and_gate)
hypercube_plot(all_m)

# n_pbits=5 -> too large to embed in 3D exactly, so hypercube_plot falls
# back to a nested/recursive-cube projection (edges still connect states
# exactly one flip apart).
full_adder = FullAdder()
solver = CaSuDaSolver(Nt=40000, dt=0.1667, i0=0.8, seed=0)
_, all_m, _ = solver.solve(full_adder)
hypercube_plot(all_m)
