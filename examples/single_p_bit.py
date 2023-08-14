"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np


c = PCircuit(1)
c.J = np.array([[0]])
c.h = np.array([0])

solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.8)

output = solver.solve(c)

histplot(output)
