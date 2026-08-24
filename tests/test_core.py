import numpy as np
import pytest

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver

def test_casuda_initial_state():
    c = PCircuit(3)
    solver = CaSuDaSolver(Nt=1, dt=0.0, i0=0.8, seed=1)
    state = np.array([1, -1, 1])

    _, output, _ = solver.solve(c, initial_state=state)

    assert np.array_equal(output[0], state)

def test_casuda_initial_state_shape():
    c = PCircuit(3)
    solver = CaSuDaSolver(Nt=1, dt=0.0, i0=0.8)

    with pytest.raises(ValueError):
        solver.solve(c, initial_state=np.array([1, -1]))

def test_casuda_initial_state_values():
    c = PCircuit(3)
    solver = CaSuDaSolver(Nt=1, dt=0.0, i0=0.8)

    with pytest.raises(ValueError):
        solver.solve(c, initial_state=np.array([1, 0, -1]))