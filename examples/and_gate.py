"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.visualization import histplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


c = PCircuit(3, i0=0.8)
# c.set_weight(0, 1, -2)
# c.set_weight(0, 2, -2)
# c.set_weight(1, 2, 1)
c.J = np.array([[0,-2, -2],[-2, 0, 1],[-2, 1, 0]])
c.h = np.array([2,-1,-1])

output = c.solve(10000, 0.1667)

histplot(output)
