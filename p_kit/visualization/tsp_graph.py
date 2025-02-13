"""Plot TSP graph"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Graph-Based TSP Path Visualization
def visualize_tsp_route(path, city_graph):
    """ Visualizes the best TSP path using NetworkX. """

    if(type(path).__name__ == "str"):
        path = [int(v) for v in path]

    plt.figure(figsize=(6, 6))
    n_cities = len(city_graph)
    G = nx.DiGraph()

    # Define city positions in a circular layout
    pos = {i: (np.cos(2 * np.pi * i / n_cities), np.sin(2 * np.pi * i / n_cities)) for i in range(n_cities)}

    # Add nodes (cities)
    for i in range(n_cities):
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
    plt.title("TSP Solution Path")
    plt.show()