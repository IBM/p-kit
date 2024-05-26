from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, plot3d, vin_vout
import numpy as np

c = PCircuit(6)

c.J = np.array([
    [0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0]
    ])

# A high bias clamp a p-bit toward 1/0.
c.h = np.array(
    [-1,-1,-1,-1, 2, 2]
    )

solver = CaSuDaSolver(Nt=100000, dt=0.1667, i0=0.8)

input, output = solver.solve(c)

histplot(output)

# 3d histogram, displaying the number of count of p0p1 as a function of p2p3
plot3d(output, A=[0, 1], B=[2, 3])
