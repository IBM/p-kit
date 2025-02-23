import numpy as np
from p_kit.utils.deprecation import deprecated

PCIRCUIT_EXAMPLE = """
    @psl.pcircuit(n_pbits=3)
    class MyCircuit:
        input1 = Port('input1')
        input2 = Port('input2')
        output = Port('output')
"""

@deprecated("@psl.pcircuit decorator", example=PCIRCUIT_EXAMPLE)
class PCircuit():
    """Create and holds J and h parameters.

    .. warning::
        **DEPRECATED**. This class is deprecated since version 0.1.0 and will be 
        removed in version 1.0.0. Use the @psl.pcircuit decorator instead.

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
    .. deprecated:: 1.0.0
        Use @psl.pcircuit decorator instead.
    """
    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits, 1))
        self.J = np.zeros((n_pbits, n_pbits))
    
    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        self.J[from_pbit, to_pbit] = weight
        if(sym):
            self.J[to_pbit, from_pbit] = weight