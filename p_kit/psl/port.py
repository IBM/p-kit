import numpy as np
from typing import Type, Union, TypeVar, Callable, Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod


class ConnectionStrategy(ABC):
    """Base strategy for port connections in quantum circuits."""

    @abstractmethod
    def assign_global_index(
        self,
        source_port: "Port",
        target_port: "Port",
        port_count: int,
        port_to_global: Dict["Port", int],
    ) -> int:
        # Each port gets unique index
        source_port.global_index = port_count
        port_to_global[source_port] = port_count
        return port_count + 1

    @abstractmethod
    def synthesize_connection_sparse(
        self, source_idx: int, target_idx: int, J_global: Dict[int, Dict[int, float]]
    ) -> None:
        """Add connection weights to global J matrix during synthesis (sparse format)."""
        pass

    @abstractmethod
    def synthesize_connection_dense(
        self, source_idx: int, target_idx: int, J_global: np.ndarray
    ) -> None:
        """Add connection weights to global J matrix during synthesis (dense format)."""
        pass

    def synthesize_connection(
        self,
        source_idx: int,
        target_idx: int,
        J_global: Union[Dict[int, Dict[int, float]], np.ndarray],
        format: str = "sparse",
    ) -> None:
        """Dispatch to appropriate synthesis method based on format."""
        if format == "sparse":
            self.synthesize_connection_sparse(source_idx, target_idx, J_global)
        else:
            self.synthesize_connection_dense(source_idx, target_idx, J_global)


class NoCopyConnection(ConnectionStrategy):
    """Direct connection between ports with shared global index."""

    def assign_global_index(self, source_port, target_port, port_count, port_to_global):
        source_port.global_index = port_count
        target_port.global_index = port_count
        port_to_global[source_port] = port_count
        port_to_global[target_port] = port_count
        return port_count + 1

    def synthesize_connection_sparse(
        self, source_idx: int, target_idx: int, J_global: Dict[int, Dict[int, float]]
    ) -> None:
        # No weights added for NO_COPY - they share a global index
        pass

    def synthesize_connection_dense(
        self, source_idx: int, target_idx: int, J_global: np.ndarray
    ) -> None:
        # No weights added for NO_COPY - they share a global index
        pass

    def synthesize_connection(self, source_idx, target_idx, J_global, format="sparse"):
        return super().synthesize_connection(source_idx, target_idx, J_global, format)


class VanillaCopyConnection(ConnectionStrategy):
    """Standard copy connection with weight 1.0."""

    def assign_global_index(self, source_port, target_port, port_count, port_to_global):
        return super().assign_global_index(
            source_port, target_port, port_count, port_to_global
        )

    def synthesize_connection_sparse(
        self, source_idx: int, target_idx: int, J_global: Dict[int, Dict[int, float]]
    ) -> None:
        J_global[source_idx][target_idx] = 1.0
        J_global[target_idx][source_idx] = 1.0

    def synthesize_connection_dense(
        self, source_idx: int, target_idx: int, J_global: np.ndarray
    ) -> None:
        J_global[source_idx, target_idx] = 1.0
        J_global[target_idx, source_idx] = 1.0

    def synthesize_connection(self, source_idx, target_idx, J_global, format="sparse"):
        return super().synthesize_connection(source_idx, target_idx, J_global, format)


class WeightedCopyConnection(ConnectionStrategy):
    """Copy connection with customizable weight."""

    def __init__(self, weight: float):
        self.weight = weight

    def assign_global_index(self, source_port, target_port, port_count, port_to_global):
        return super().assign_global_index(
            source_port, target_port, port_count, port_to_global
        )

    def synthesize_connection_sparse(
        self, source_idx: int, target_idx: int, J_global: Dict[int, Dict[int, float]]
    ) -> None:
        J_global[source_idx][target_idx] = self.weight
        J_global[target_idx][source_idx] = self.weight

    def synthesize_connection_dense(
        self, source_idx: int, target_idx: int, J_global: np.ndarray
    ) -> None:
        J_global[source_idx, target_idx] = self.weight
        J_global[target_idx, source_idx] = self.weight

    def synthesize_connection(self, source_idx, target_idx, J_global, format="sparse"):
        return super().synthesize_connection(source_idx, target_idx, J_global, format)


@dataclass
class Port:
    """
    Represents a port in a quantum circuit that can be connected to other ports.

    The Port class handles connections between different components of quantum circuits,
    managing both local and global indices for matrix operations.

    Attributes:
        name (str): Name identifier for the port
        circuit (Any): Reference to the parent circuit
        index (int): Local index within the parent circuit
        global_index (int): Global index in the synthesized system
        _connections (Dict): Connection information to other ports

    Example:
        >>> port1 = Port("input1")
        >>> port2 = Port("output1")
        >>> port1.connect(port2, ConnectionType.NO_COPY)
    """

    name: str
    circuit: Any = None
    index: int = None
    global_index: int = None
    _connections: Dict = field(default_factory=dict)

    def __hash__(self):
        """
        Generates a hash based on the combination of circuit reference and port name.

        Returns:
            int: Hash value for the port
        """
        return hash((id(self.circuit), self.name))

    def __eq__(self, other):
        """
        Compares two ports for equality based on circuit reference and name.

        Args:
            other (Port): Another port to compare with

        Returns:
            bool: True if ports are equal, False otherwise
        """
        if not isinstance(other, Port):
            return False
        return id(self.circuit) == id(other.circuit) and self.name == other.name

    def connect(self, other_port: "Port", strategy: ConnectionStrategy):
        """
        Establishes a connection between this port and another port.

        Args:
            other_port (Port): Target port to connect to
            strategy: Connection strategy defining the connection behavior
        """
        if self.circuit is None or other_port.circuit is None:
            raise ValueError("Both ports must be bound to circuits")
        if self.index is None or other_port.index is None:
            raise ValueError("Both ports must have assigned indices")

        if isinstance(strategy, type):
            self._connection_strategy = strategy()
        else:
            self._connection_strategy = strategy

        self._connected_port = other_port
