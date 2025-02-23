import numpy as np
from typing import Type, TypeVar, Callable, Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

T = TypeVar('T')

class ConnectionType(Enum):
    """
    Enumeration defining different types of connections between ports in quantum circuits.
    
    Types:
    - NO_COPY: Direct connection between ports with shared global index
    - VANILLA_COPY: Standard copy connection with weight 1.0
    - WEIGHTED_COPY: Copy connection with customizable weight
    """
    NO_COPY = "no_copy"
    VANILLA_COPY = "vanilla"
    WEIGHTED_COPY = "weighted"

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
    
    def connect(self, other_port: 'Port', connection_type: ConnectionType = ConnectionType.NO_COPY, 
                weight: float = 1.0):
        """
        Establishes a connection between this port and another port.
        
        Args:
            other_port (Port): Target port to connect to
            connection_type (ConnectionType): Type of connection to establish
            weight (float): Weight of the connection (used for WEIGHTED_COPY)
            
        Raises:
            ValueError: If either port is not bound to a circuit or lacks an index
        """
        if self.circuit is None or other_port.circuit is None:
            raise ValueError("Both ports must be bound to circuits")
        if self.index is None or other_port.index is None:
            raise ValueError("Both ports must have assigned indices")
        
        self._connections = {
            'type': connection_type,
            'target_circuit': other_port.circuit,
            'target_port': other_port.name,
            'weight': weight
        }