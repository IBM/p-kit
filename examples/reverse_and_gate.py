"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.reverse_solver.simple_solver import SimpleReverseSolver
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np


# c = PCircuit(3)
# # c.set_weight(0, 1, -2)
# # c.set_weight(0, 2, -2)
# # c.set_weight(1, 2, 1)
# c.J = np.array([[0,-1, 2],[-1, 0, 2],[2, 2, 0]])

# # Here, you can change biases.
# # A high bias clamp a p-bit toward 1/0.
# # (depending on the sign of the bias)
# c.h = np.array([1,1,-2])

solver = SimpleReverseSolver(truth_table=["000", "100", "010", "111"])

solver.solve()
