"""Module for pipelines."""
from p_kit import PCircuit
import numpy as np

def tostr(m):
    return "1" if m == 1 else "0"

def tostr2(M):
    return tostr(M[0]) + tostr(M[1]) + tostr(M[2])

c = PCircuit(3, i0=0.8)
c.J = np.array([[0,-2, -2],[-2, 0, 1],[-2, 1, 0]])
c.h = np.array([2,-1,-1])


ret = {
    '000': 0,
    '001': 0,
    '010': 0,
    '011': 0,
    '100': 0,
    '101': 0,
    '110': 0,
    '111': 0
}


output = c.solve(10000, 0.1667)
print(output)
for m in output:
    s = tostr2(m)
    ret[s] = ret[s] + 1

print(ret)