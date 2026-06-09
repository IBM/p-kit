"""Half Adder example."""

from p_kit.psl.gates import HalfAdder
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np

adder = HalfAdder()
# Clamp inputs to 1+1 to demonstrate carry output
adder.h[0] = 10  # input1 = 1
adder.h[1] = 10  # input2 = 1

solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.9)
_, output, _ = solver.solve(adder)

print(f"Sum output (bit 2): {np.mean(output[:, 2]):.2f}")
print(f"Carry output (bit 3): {np.mean(output[:, 3]):.2f}")

histplot(output)
