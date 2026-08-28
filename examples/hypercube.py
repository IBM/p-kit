"""Visualize a p-circuit's {-1,+1}^n state space as a hypercube."""

import os
import sys

import numpy as np

from p_kit.psl.gates import ANDGate, FullAdder
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import hypercube_plot

# n_pbits=3 -> a literal cube. The p-bit dynamics should spend most of
# their time on the 4 vertices where output == input1 AND input2.
and_gate = ANDGate()
solver = CaSuDaSolver(Nt=20000, dt=0.1667, i0=0.8, seed=0)
_, all_m, _ = solver.solve(and_gate)
hypercube_plot(all_m)

# n_pbits=5 -> too large to embed in 3D exactly, so hypercube_plot falls
# back to a nested/recursive-cube projection (edges still connect states
# exactly one flip apart).
full_adder = FullAdder()
solver = CaSuDaSolver(Nt=40000, dt=0.1667, i0=0.8, seed=0)
_, all_m, _ = solver.solve(full_adder)
hypercube_plot(all_m)

# n_pbits=128 (SparsePBitLM's reservoir) -> {-1,+1}^128 has ~3.4e38
# vertices, far too many to enumerate. hypercube_plot instead PCA-projects
# the actual sequence of reservoir states visited while reading some text.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm"))
from llm_model import SparsePBitLM  # noqa: E402

CORPUS = "the cat sat on the mat. the dog sat by the door. " * 5

lm = SparsePBitLM(n_pbits=128, degree=6, seed=7)
lm.fit(CORPUS, sweeps=1, washout=0, ridge=0.05)
lm.reset()
history = np.array([
    lm.step(lm.char_to_id[ch]) for ch in CORPUS if ch in lm.char_to_id
])
hypercube_plot(history)
