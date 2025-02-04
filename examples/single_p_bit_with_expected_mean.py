"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver

import numpy as np


c = PCircuit(1)
c.J = np.array([[0]])
c.h = np.array([0])

N = 100
means = []
expected_means = []
for expected_mean in range(-N, N, N // 10):
    expected_mean = expected_mean / N

    # When i0=0, the average of outputs should lean towards the expected_mean (default is zero)
    solver = CaSuDaSolver(Nt=100000, dt=0.1667, i0=0, expected_mean=expected_mean)

    _, output, _ = solver.solve(c)

    means.append(np.mean(output[:, 0]))
    expected_means.append(expected_mean)

print("Expected means:\n", expected_means)
print("Experimental means:\n", means)
