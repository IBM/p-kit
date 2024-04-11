from p_kit.core.p_circuit import PCircuit
from p_kit.visualization.utils import dist_truth_table, truth_table
from .base_solver import ReverseSolver
from ..solver import CaSuDaSolver
from random import random
import numpy as np
from docplex.mp.model import Model
from docplex.cp.model import CpoModel


class SimpleReverseSolver(ReverseSolver):
    def solve(self):
        prob = CpoModel()
        J = self._get_J(prob)
        h = self._get_h(prob)
        circuit = PCircuit(n_pbits=self.n_pbits)
        circuit.J = J
        circuit.h = h
        solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)
        _, outputs = solver.solve(circuit, prob)
        table = truth_table(outputs)
        prob.minimize(dist_truth_table(table, self.truth_table, prob))
        ret = prob.solve(
               execfile='C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2211\cpoptimizer\\bin\\x64_win64\cpoptimizer.exe')
        print(ret)


