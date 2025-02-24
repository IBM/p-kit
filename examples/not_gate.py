from p_kit.psl import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np

c = PCircuit(2)

c.J = np.array([[0, -1], [-1, 0]])
c.h = np.array([0, 0])

solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output, _ = solver.solve(c)

histplot(output)

vin_vout(input, output, p_bit=1)
