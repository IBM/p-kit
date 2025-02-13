from p_kit.core import convert_city_graph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, plot3d

np.set_printoptions(linewidth=300)

# Define city graph (distance matrix)
city_graph = np.array([[0, 510, 480],
                      [510, 0, 240],
                      [480, 240, 0]])

# city_graph = np.array([[0, 510, 480, 490],
#               [510, 0, 240, 370],
#               [480, 240, 0, 220],
#               [490, 370, 220, 0]])

# Convert to negative weights for optimization
city_graph *= -1

# Initialize p-circuit
myp = convert_city_graph(city_graph=city_graph, tsp_modifier=0.33)

# Get interaction weights and biases
J, h = myp.J, myp.h
print(J.shape, h.shape)
print("Interaction Matrix (J):\n", J)

Nt = int(1e5)
def annealing(i0, run):
    v = (0 - i0) / Nt * (run - 1) + i0
    # v = i0 * (1.0001)**(run - 1)
    # print(v)
    return v

# Simulated Annealing Schedule
end_samples = []
solver = CaSuDaSolver(Nt=Nt, dt=1 / (2 * len(city_graph)), i0=0.005)

for i in range(100):
    _, samples, _ = solver.solve(myp, annealing_func=annealing)
    end_samples.append(samples[-1, :])

samples = np.array(end_samples)
# plot3d(samples, A=[0, 1, 2], B=[3, 4, 5])

# 🏆 Extract Best Path from Binary Matrix
def extract_tsp_path(sample_matrix):
    """ Extracts the TSP path from a binary matrix. """
    path = []
    for order in range(sample_matrix.shape[1]):  # Iterate through tour steps
        #print(sample_matrix[:, order])
        step = (sample_matrix[:, order])
        # city = np.argmax(step)  # Find city with value '1'

        maxes = np.argwhere(step == np.max(step)).flatten()
        city = np.random.choice(maxes)
        path.append(city)
    return path

print(samples)
hist = {}
for i in range(len(samples)):
    s = samples[i, :].reshape((len(city_graph), len(city_graph[0])))
    path = extract_tsp_path(s)
    #print(path)
    key = "".join([str(p) for p in path])
    if key in hist:
        hist[key] = hist[key] + 1
    else:
        hist[key] = 1

print(hist)
plt.bar(hist.keys(), hist.values())
# plt.show()
print(extract_tsp_path(np.array([s[-1, :]])))

# 📌 Graph-Based TSP Path Visualization
def visualize_tsp_route(path, city_graph):
    """ Visualizes the best TSP path using NetworkX. """
    Nm = len(city_graph)
    G = nx.DiGraph()

    # Define city positions in a circular layout
    pos = {i: (np.cos(2 * np.pi * i / Nm), np.sin(2 * np.pi * i / Nm)) for i in range(Nm)}

    # Add nodes (cities)
    for i in range(Nm):
        G.add_node(i, pos=pos[i])

    # Add edges based on best path
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i + 1], weight=-city_graph[path[i]][path[i + 1]])

    # Add edge to complete the tour (last city → first city)
    G.add_edge(path[-1], path[0], weight=-city_graph[path[-1]][path[0]], alpha=0.1)

    # Draw the graph
    
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='r', node_size=1000, font_size=15, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

   
    

# 🔹 Visualize Best Path
# plt.figure(figsize=(6, 6))
# for path, count in hist.items():
#     visualize_tsp_route([int(path[0]), int(path[1]), int(path[2])], city_graph)

# plt.title("TSP Solution Path")
plt.show()