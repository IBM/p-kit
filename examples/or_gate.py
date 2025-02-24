"""Module for pipelines."""

from p_kit.psl import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np


c = PCircuit(3)
# c.set_weight(0, 1, -2)
# c.set_weight(0, 2, -2)
# c.set_weight(1, 2, 1)
c.J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])

# Here, you can change biases.
# A high bias clamp a p-bit toward 1/0.
# (depending on the sign of the bias)
c.h = np.array([-1, -1, 2])

solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)

input, output, _ = solver.solve(c)

histplot(output)

# function characteristic of p_bit 2
vin_vout(input, output, p_bit=2)
