"""XOR gate example."""

from p_kit.psl.gates import XORGate
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot

gate = XORGate()

solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)
input_data, output, energy = solver.solve(gate)

histplot(output)
