"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout, plot3d
import numpy as np
import matplotlib.pyplot as plt
import os

c = PCircuit(10)


#5bit relu new using AND gate and NOT gate
#p-bits, 1 to 6 are input bits except 2 (intermediate)
#p-bits, 7 to 11 are output bits
#7th bit is the sign-bit of output
"""
c.J = np.array([[0,-1,0,0,0,0,0,0,0,0,0],
                [-1,0,-1,-1,-1,-1,0,2,2,2,2],
                [0,-1,0,0,0,0,0,2,0,0,0],
                [0,-1,0,0,0,0,0,0,2,0,0],
                [0,-1,0,0,0,0,0,0,0,2,0],
                [0,-1,0,0,0,0,0,0,0,0,2],
                [0,0,0,0,0,0,0,0,0,0,0],
                [0,2,2,0,0,0,0,0,0,0,0],
                [0,2,0,2,0,0,0,0,0,0,0],
                [0,2,0,0,2,0,0,0,0,0,0],
                [0,2,0,0,0,2,0,0,0,0,0]])

#c.h = np.array([0,4,1,1,1,1,-2,-2,-2,-2,-2])
"""



#5bit relu new using NOT IMPLY LOGIC
#p-bits, 1 to 6 are input bits except 2
#p-bits, 7 to 11 are output bits
#10th bit is the output sign-bit

c.J = np.array([[0,1,2,0,0,0,0,0,0,0],
                [1,0,-2,1,-2,1,-2,1,-2,0],
                [2,-2,0,0,0,0,0,0,0,0],
                [0,1,0,0,2,0,0,0,0,0],
                [0,-2,0,2,0,0,0,0,0,0],
                [0,1,0,0,0,0,2,0,0,0],
                [0,-2,0,0,0,2,0,0,0,0],
                [0,1,0,0,0,0,0,0,2,0],
                [0,-2,0,0,0,0,0,2,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

c.h = np.array([1,-4,-2,1,-2,1,-2,1,-2,-2])

solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.9)
input, output = solver.solve(c)


'''
#histplot(output)
#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1,2,3,4,5], B=[6,7,8,9,10,11])
#OUTPUT ARRAY IN FILE

current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_RELU_5BIT_NEW_INV.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")

# Read data from the file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'output_RELU_10BIT_NEW_INV.txt')

with open(file_path, 'r') as f:
    lines = f.readlines()

# Combine every set of 5 lines
combined_lines = []
for i in range(0, len(lines), 2):
    combined_line = ' '.join(line.strip() for line in lines[i:i+2])
    combined_lines.append(combined_line)

# Write combined lines to a new file
combined_file_path = os.path.join(current_dir, 'Combined_output_RELU_10BIT_NEW_INV.txt')
with open(combined_file_path, 'w') as f:
    for line in combined_lines:
        f.write(line + '\n')

print(f"Combined data saved to {combined_file_path}")


#np.set_printoptions(threshold=np.inf)
#print(output)

#vin_vout(input, output, p_bit=3)
'''