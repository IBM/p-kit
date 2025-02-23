from p_kit.psl import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np
import matplotlib.pyplot as plt

c = PCircuit(3)

c.J = np.array([[0, 1, 2], [1, 0, -2], [2, -2, 0]])
c.h = np.array([1, -1, -2])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output, _ = solver.solve(c)

histplot(output)
