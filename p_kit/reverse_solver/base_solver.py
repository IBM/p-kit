from docplex.mp.vartype import ContinuousVarType, IntegerVarType, BinaryVarType
from docplex.mp.model import Model
from docplex.cp.model import CpoModel
import numpy as np

class ReverseSolver:
    def __init__(self, truth_table: list) -> None:
        self.truth_table= [int(s, 2) for s in truth_table]
        self.lb = -2
        self.up = 2
        self.n_pbits = len(truth_table[0])

    def _get_J(self, prob):
        keys = range(self.n_pbits)
        IntegerVarType.one_letter_symbol = lambda _: "I"
        # J = prob.integer_var_matrix(
        #     keys1=keys, keys2=keys, name="J", lb=self.lb, ub=self.up
        # )
        ret = []
        for l in keys:
            line = []
            for c in keys:
                var = prob.integer_var(self.lb, self.up, name=f'J_{l}{c}')
                line.append(var)
            ret.append(line)
        ret = np.array(ret)
        # print(ret)
        return ret
    
    def _get_h(self, prob):
        keys = range(self.n_pbits)
        # h = prob.integer_var_matrix(
        #     keys1=[1], keys2=keys, name="h", lb=self.lb, ub=self.up
        # )
        # h = np.array([h[key] for key in h])
        h = np.array([prob.integer_var(self.lb, self.up, name=f'h_{key}') for key in keys])
        return h

    def solve(self):
        raise NotImplementedError()
