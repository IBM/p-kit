"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np


c = PCircuit(3)
# c.set_weight(0, 1, -2)
# c.set_weight(0, 2, -2)
# c.set_weight(1, 2, 1)
c.J = np.array([[0,-2, -2],[-2, 0, 1],[-2, 1, 0]])
c.h = np.array([2,-1,-1])

solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)

output = solver.solve(c)

histplot(output)
