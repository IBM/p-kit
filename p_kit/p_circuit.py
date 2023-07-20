"""Module for pipelines."""
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
    .. versionadded:: 0.1.0

    """

    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits,1))
        self.J = np.zeros((n_pbits, n_pbits))
