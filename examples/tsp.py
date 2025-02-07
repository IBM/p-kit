from p_kit.core import convert_city_graph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from p_kit.solver.csd_solver import CaSuDaSolver

np.set_printoptions(linewidth=300)

# Define city graph (distance matrix)
city_graph = np.array([[0, 510, 480],
                      [510, 0, 240],
                      [480, 240, 0]])

# Convert to negative weights for optimization
city_graph *= -1

# Initialize p-circuit
myp = convert_city_graph(city_graph=city_graph, tsp_modifier=0.33)

# Get interaction weights and biases
J, h = myp.J, myp.h
print(J.shape, h.shape)
print("Interaction Matrix (J):\n", J)

# Simulated Annealing Schedule
solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.9, expected_mean=0, seed=930)
_, samples, _ = solver.solve(myp)

# 🏆 Extract Best Path from Binary Matrix
def extract_tsp_path(sample_matrix):
    """ Extracts the TSP path from a binary matrix. """
    path = []
    for order in range(sample_matrix.shape[1]):  # Iterate through tour steps
        city = np.argmax(sample_matrix[:, order])  # Find city with value '1'
        path.append(city)
    return path

print(samples)
final_sample = samples[-1, :].reshape((len(city_graph), len(city_graph[0])))
best_path = extract_tsp_path(final_sample)
print("Best TSP Path Found:", best_path)


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
    G.add_edge(path[-1], path[0], weight=-city_graph[path[-1]][path[0]])

    # Draw the graph
    plt.figure(figsize=(6, 6))
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='r', node_size=1000, font_size=15, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("TSP Solution Path")
    plt.show()

# 🔹 Visualize Best Path
visualize_tsp_route(best_path, city_graph)