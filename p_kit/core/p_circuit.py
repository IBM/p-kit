import numpy as np


class PCircuit():

    """Create and holds J and h parameters.

    Parameters
    ----------
    n_pbits: string
        Identifier of the pipeline (for log purposes).

    Attributes
    ----------
    h : np.array((n_pbits, 1))
        biases
    J : np.array((n_pbits, n_pbits))
        weights

    Notes
    -----
    .. versionadded:: 0.0.1

    """

    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits,))
        self.J = np.zeros((n_pbits, n_pbits))
    
    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        self.J[from_pbit, to_pbit] = weight
        if(sym):
            self.J[to_pbit, from_pbit] = weight


def convert_city_graph(city_graph=None, tsp_modifier=None):
            """
            build a j matrix for some travelling salesman problem graph. 

            designing a travelling salesman J:
            Rule 1: 1 between pbits of same city
            Rule 2: 1 between pbits of same order
            Rule 3: negative distances as weights between rows (ex. all p-bits in city-1-row to all cities in city-3-row)
            Rule 4: 0 connections from city_n-order_n to itself
            
            See excel sheet (building_J_for_tsp in shark tank 2020 purdue onedrive)
            """

            city_graph = np.asarray(city_graph)

            city_graph = np.divide(city_graph, np.amax(np.abs(city_graph)))

            if tsp_modifier is None:
                tsp_modifier = 1

            Nm_cities = len(city_graph[0])

            circuit = PCircuit(Nm_cities ** 2)

            # Rule 3: negative distances from one city to another
            for i in range(Nm_cities):
                for j in range(Nm_cities):
                    #if order(i) is one away from order(j)
                    if i == j:
                        continue
                    off_diag = np.ones(Nm_cities-1) * city_graph[j, i]
                    #set both off diagonals (one to the left and right of main diagonal) of weights to off_diag
                    weights_i_j = np.diag(off_diag, 1) + np.diag(off_diag, -1)

                    circuit.J[j * Nm_cities: j * Nm_cities + Nm_cities, i * Nm_cities: i * Nm_cities + Nm_cities] = weights_i_j[:,:]

            # Rule 1 - 1 between pbits of same city
            for i in range(Nm_cities):
                circuit.J[i * Nm_cities:i * Nm_cities + Nm_cities, i * Nm_cities:i * Nm_cities + Nm_cities] = tsp_modifier # dif

            # Rule 2 - 1 between pbits of same order
            for i in range(Nm_cities ** 2):
                for j in range(Nm_cities ** 2):
                    if i == j % Nm_cities or j == i % Nm_cities:
                        circuit.J[i, j] = tsp_modifier

            # Rule 4: 0s on the diagonal
            np.fill_diagonal(circuit.J, 0)

            return circuit

