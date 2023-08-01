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
        self.h = np.zeros((n_pbits, 1))
        self.J = np.zeros((n_pbits, n_pbits))
    
    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        self.J[from_pbit, to_pbit] = weight
        if(sym):
            self.J[to_pbit, from_pbit] = weight
    
                    