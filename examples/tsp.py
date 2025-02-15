# Adapted from: 
# https://github.com/anirudhgha/p-bit/tree/master/Research/Travelling%20Salesman%20Problem

import matplotlib.pyplot as plt
import numpy as np

from p_kit.library.tsp import TSP
from p_kit.solver.annealing import constant, execute, linear
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization.tsp_graph import visualize_tsp_route
from p_kit.visualization.utils import tsp_hist

# Define city graph (distance matrix)
# city_graph = np.array([[0, 510, 480],
#                       [510, 0, 240],
#                       [480, 240, 0]])

city_graph = np.array([[0, 510, 480, 490],
              [510, 0, 240, 370],
              [480, 240, 0, 220],
              [490, 370, 220, 0]])
# city_graph = np.array([[  0.,1.,1.,1.,-10.,-10.,1.,-15.,-15.],
#  [  1.,0.,1.,-10.,1.,-10.,-15.,1.,-15.],
#  [  1.,1.,0.,-10.,-10.,1.,-15.,-15.,1.],
#  [  1.,-10.,-10.,0.,1.,1.,-20.,-20.,-20.],
#  [-10.,1.,-10.,1.,0.,1.,-20.,-20.,-20.],
#  [-10.,-10.,1.,1.,1.,0.,-20.,-20.,-20.],
#  [  1.,-15.,-15.,-20.,-20.,-20.,0.,1.,1.],
#  [-15.,1.,-15.,-20.,-20.,-20.,1.,0.,1.],
#  [-15.,-15.,1.,-20.,-20.,-20.,1.,1.,0.]])

# Convert to negative weights for optimization
city_graph *= -1

# Initialize p-circuit
circuit = TSP(city_graph=city_graph, tsp_modifier=0.33)

# Simulated Annealing Schedule
solver = CaSuDaSolver(Nt=int(1e6), dt=1 / (2 * len(city_graph)), i0=0.005, expected_mean=0, seed=None)

# Increase n_shots to have more reliable results
n_shots = 100
samples = execute(solver, circuit, constant, n_shots, n_last_samples=50, n_jobs=-1)

# Visualize Best path
hist = tsp_hist(samples, city_graph)
plt.bar(hist.keys(), hist.values())
plt.show()

best_path = max(hist, key=hist.get)
    
visualize_tsp_route(best_path, city_graph)
