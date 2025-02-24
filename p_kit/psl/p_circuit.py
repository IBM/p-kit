from .port import *

import numpy as np
from typing import Dict, Any


class PCircuit:
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
    ports: dict[str, object]
        Circuit ports

    """

    def __init__(self, n_pbits: int, ports: Dict[str, Any] = None):
        self.n_pbits = n_pbits
        self.ports = ports #Kept for copy behavior

        self.h = np.zeros((n_pbits, 1))
        self.J = np.zeros((n_pbits, n_pbits))
        self._connections = {}

        if ports:
            self._initialize_ports(ports)

    def _initialize_ports(self, port_attrs: Dict[str, Any]) -> None:
        """Initialize ports from attributes dictionary."""
        # Create port index mapping
        port_indices = {name: idx for idx, name in enumerate(port_attrs.keys())}

        # Set up each port
        for name, port in port_attrs.items():
            new_port = Port(name=port.name)
            new_port.circuit = self
            new_port.index = port_indices[name]
            # Check ports name doesn't conflict with reserved attributes.
            assert name != "ports"
            assert name != "h"
            assert name != "J"
            setattr(self, name, new_port)

    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        self.J[from_pbit, to_pbit] = weight
        if sym:
            self.J[to_pbit, from_pbit] = weight

    def copy(self):
        new_circuit = PCircuit(self.n_pbits, self.ports)
        new_circuit.J = self.J
        new_circuit.h = self.h
        return new_circuit
