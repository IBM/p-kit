"""Module for pipelines."""
from p_kit import PCircuit
import numpy as np

c = PCircuit(3)
c.J = np.array([[0,-2, -2],[-2, 0, 1],[-2, 1, 0]])
c.i0 = 0.3

c.solve(0)

print("I", c.I)
print("M", c.M)