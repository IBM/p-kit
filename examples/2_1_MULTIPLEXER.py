"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os


c = PCircuit(6)


#2:1 MUX using Hamiltonian
#c.J = np.array([[0,0,-1,-1,0,2,0],[0,0,-1,0,2,0,0],[-1,-1,0,0,2,0,0],[-1,0,0,0,0,2,0],[0,2,2,0,0,-1,2],[2,0,0,2,-1,0,2],[0,0,0,0,2,2,0]])
#c.h = np.array([1,1,1,1,-3,-3,2])


# Reduced number of p-bits using Hamiltonian
c.J = np.array([[0,1,0,0,2,0],[1,0,-1,2,-2,0],[0,-1,0,2,0,0],[0,2,2,0,-1,2],[2,-2,0,-1,0,2],[0,0,0,2,2,0]])
c.h = np.array([1,0,1,-3,-3,2])




solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output = solver.solve(c)

histplot(output)


#For printing the the values to the external files
#
# # Get the current working directory
# current_dir = os.getcwd()
# print("Current Directory:", current_dir)
#
# # Output array to a file in the current directory
# file_path = os.path.join(current_dir, 'output_MUX_NEW.txt')
#
# with open(file_path, 'w') as f:
#     for element in output:
#         f.write(str(element) + '\n')
#
# print(f"Array data saved to {file_path}")

