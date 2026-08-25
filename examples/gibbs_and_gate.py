"""Module for pipelines."""

from p_kit.psl import PCircuit
from p_kit.solver.gibbs_solver import GibbsSolver
from p_kit.visualization import histplot, energyplot, vin_vout
import numpy as np


c = PCircuit(3)
# c.set_weight(0, 1, -2)
# c.set_weight(0, 2, -2)
# c.set_weight(1, 2, 1)
c.J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])

# Here, you can change biases.
# A high bias clamp a p-bit toward 1/0.
# (depending on the sign of the bias)
c.h = np.array([1, 1, -2])

# Asynchronous single-p-bit-per-step Gibbs sampling converges slower than
# CaSuDa's synchronous update, so use a larger Nt to well-sample the AND
# gate's stationary distribution. dt is kept for interface parity even
# though GibbsSolver does not use it.
solver = GibbsSolver(Nt=30000, dt=0.1667, i0=0.8)

input, output, energy = solver.solve(c)

histplot(output)

energyplot(output, energy)

# function characteristic of p_bit 2
vin_vout(input, output, p_bit=2)
