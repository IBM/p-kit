import numpy as np
from p_kit.core import PCircuit


class TSP(PCircuit):

    """ Build a J matrix for some travelling salesman problem graph. 
    Designing a travelling salesman J:
    Rule 1: 1 between pbits of same city
    Rule 2: 1 between pbits of same order
    Rule 3: negative distances as weights between rows (ex. all p-bits in city-1-row to all cities in city-3-row)
    Rule 4: 0 connections from city_n-order_n to itself

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

    def __init__(self, city_graph=None, tsp_modifier=None):
        PCircuit.__init__(self, len(city_graph[0]) ** 2)
        self.tsp_modifier = 1 if tsp_modifier is None else tsp_modifier
        self._init_city_graph(city_graph)
    

    def _init_city_graph(self, city_graph=None):

            city_graph = np.asarray(city_graph)
            city_graph = np.divide(city_graph, np.amax(np.abs(city_graph)))

            n_cities = len(city_graph[0])

            # Rule 3: negative distances from one city to another
            for i in range(n_cities):
                for j in range(n_cities):
                    #if order(i) is one away from order(j)
                    if i == j:
                        continue
                    off_diag = np.ones(n_cities-1) * city_graph[j, i]
                    #set both off diagonals (one to the left and right of main diagonal) of weights to off_diag
                    weights_i_j = np.diag(off_diag, 1) + np.diag(off_diag, -1)

                    self.J[j * n_cities: j * n_cities + n_cities, i * n_cities: i * n_cities + n_cities] = weights_i_j[:,:]

            # Rule 1 - 1 between pbits of same city
            for i in range(n_cities):
                self.J[i * n_cities:i * n_cities + n_cities, i * n_cities:i * n_cities + n_cities] = self.tsp_modifier # dif

            # Rule 2 - 1 between pbits of same order
            for i in range(n_cities ** 2):
                for j in range(n_cities ** 2):
                    if i == j % n_cities or j == i % n_cities:
                        self.J[i, j] = self.tsp_modifier

            # Rule 4: 0s on the diagonal
            np.fill_diagonal(self.J, 0)


