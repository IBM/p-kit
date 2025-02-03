from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt

c = PCircuit(2)

c.J = np.array([[0,-1],[-1,0]])
c.h = np.array([0,0])

solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input,output = solver.solve(c)

histplot(output)

#3d Histogram plot for the p-bit
#plot3d(output, A=[0], B=[1])



vin_vout(input, output, p_bit=1)