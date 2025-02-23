from .context import *
from .p_circuit import *

from typing import Type, TypeVar, Dict, Union, Tuple
import numpy as np

T = TypeVar("T")


def module(cls: Type[T]) -> Type[T]:
    """
    Decorator that transforms a class into a probabilistic circuit module.

    This decorator provides functionality for managing multiple circuit instances
    and synthesizing their combined matrices. The synthesize method supports both
    sparse and dense matrix formats.

    Args:
        cls (Type[T]): Class to be decorated

    Returns:
        Type[T]: Decorated class with module functionality

    Example:
        >>> @module
        >>> class MyCircuit:
        >>>     def __init__(self):
        >>>         self.gate1 = ANDGate()
        >>>         self.gate2 = ANDGate()
        >>>
        >>> # Create instance and synthesize matrices
        >>> circuit = MyCircuit()
        >>>
        >>> # Get sparse representation (default)
        >>> J_sparse, h_sparse = circuit.synthesize()
        >>>
        >>> # Get dense matrix representation
        >>> J_dense, h_dense = circuit.synthesize(format='dense')
    """
    context = ModuleContext()

    original_new = cls.__new__
    original_init = cls.__init__

    def __new__(cls, *args, **kwargs):
        instance = original_new(cls)
        instance._context = context
        return instance

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for attr_name, attr_value in vars(self).items():
            if hasattr(attr_value, "n_pbits"):
                self._context.register_instance(attr_value)

    def synthesize(self, format: str = "sparse") -> Union[
        Tuple[Dict[int, Dict[int, float]], Dict[int, float]],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Synthesizes the combined circuit matrices in either sparse or dense format.

        Args:
            format (str, optional): Output format - 'sparse' or 'dense'. Defaults to 'sparse'.
                - 'sparse': Returns adjacency list representation as nested dictionaries
                - 'dense': Returns numpy arrays

        Returns:
            If format=='sparse':
                Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
                    - J: Adjacency list where J[i][j] is the weight of edge i->j
                    - h: Dictionary where h[i] is the bias of node i
            If format=='dense':
                Tuple[np.ndarray, np.ndarray]:
                    - J: Dense coupling matrix
                    - h: Dense bias vector

        Raises:
            ValueError: If format is not 'sparse' or 'dense'
        """
        if format not in ["sparse", "dense"]:
            raise ValueError("Invalid format. Must be either 'sparse' or 'dense'")
        return self._context.synthesize(format=format)

    cls.__new__ = __new__
    cls.__init__ = __init__
    cls.synthesize = synthesize

    return cls


def pcircuit(n_pbits: int) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator that transforms a class into a probabilistic circuit `PCircuit`.

    This decorator adds circuit parameters (h and J matrices) and port management
    to the decorated class.

    Args:
        n_pbits (int): Number of ports/qubits in the circuit

    Returns:
        Callable[[Type[T]], Type[T]]: Decorator function

    Example:
        >>> @pcircuit(n_pbits=3)
        >>> class ANDGate:
        >>>     input1 = Port("input1")
        >>>     input2 = Port("input2")
        >>>     output = Port("output")
    """

    def decorator(cls: Type[T]) -> Type[T]:
        # Get port attributes from the class
        port_attrs = {
            name: attr for name, attr in cls.__dict__.items() if isinstance(attr, Port)
        }

        class WrappedCircuit(PCircuit, cls):
            def __init__(self, *args, **kwargs):
                # Initialize PCircuit first
                PCircuit.__init__(self, n_pbits, port_attrs)

                # Initialize the decorated class
                if hasattr(cls, "__init__"):
                    cls.__init__(self, *args, **kwargs)

                # Handle any custom h/J matrices defined in the decorated class
                if hasattr(cls, "h"):
                    if cls.h.shape != (n_pbits, 1):
                        raise ValueError(
                            f"h matrix must have shape ({n_pbits}, 1), got {cls.h.shape}"
                        )
                    self.h = cls.h

                if hasattr(cls, "J"):
                    if cls.J.shape != (n_pbits, n_pbits):
                        raise ValueError(
                            f"J matrix must have shape ({n_pbits}, {n_pbits}), got {cls.J.shape}"
                        )
                    self.J = cls.J

        return WrappedCircuit

    return decorator
