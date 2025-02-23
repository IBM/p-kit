from .port import *
from typing import Dict, List, Tuple, Union
import numpy as np

class ModuleContext:
    def __init__(self):
        self.instances: List[Any] = []
        self.port_to_global: Dict[Port, int] = {}
        
    def register_instance(self, instance: Any):
        self.instances.append(instance)
        
    def reset_indices(self):
        """Reset all global indices before synthesis."""
        for instance in self.instances:
            for port in self.get_circuit_ports(instance):
                port.global_index = None
        self.port_to_global.clear()
        
    def assign_global_indices(self) -> int:
        """Assigns global indices to all ports in registered instances."""
        port_count = 0
        for instance in self.instances:
            for port in self.get_circuit_ports(instance):
                if port._connections.get('type') == ConnectionType.NO_COPY:
                    if port.global_index is None:
                        port.global_index = port_count
                        target = getattr(port._connections['target_circuit'], 
                                      port._connections['target_port'])
                        target.global_index = port_count
                        self.port_to_global[port] = port_count
                        self.port_to_global[target] = port_count
                        port_count += 1
                else:
                    if port.global_index is None:
                        port.global_index = port_count
                        self.port_to_global[port] = port_count
                        port_count += 1
        return port_count

    def get_circuit_ports(self, instance) -> List[Port]:
        return [port for name, port in vars(instance).items() 
                if isinstance(port, Port)]

    def synthesize(self, format: str = 'sparse') -> Union[
            Tuple[Dict[int, Dict[int, float]], Dict[int, float]],
            Tuple[np.ndarray, np.ndarray]]:
        """
        Synthesizes global J and h matrices from all registered instances.
        
        Args:
            format (str): Output format - 'sparse' for adjacency list or 'dense' for numpy matrix
            
        Returns:
            If format=='sparse':
                Tuple[Dict[int, Dict[int, float]], Dict[int, float]]: 
                    - Adjacency list representation of J where J[i][j] is the weight of edge i->j
                    - Dictionary of h biases where h[i] is the bias of node i
            If format=='dense':
                Tuple[np.ndarray, np.ndarray]: Global J coupling matrix and h bias vector
        """
        self.reset_indices()
        
        total_ports = self.assign_global_indices()
        
        # Initialize data structures
        if format == 'sparse':
            J_global: Dict[int, Dict[int, float]] = {}
            h_global: Dict[int, float] = {}
        else:
            J_global = np.zeros((total_ports, total_ports))
            h_global = np.zeros((total_ports, 1))
        
        # Process instance couplings and biases
        for instance in self.instances:
            circuit_ports = self.get_circuit_ports(instance)
            for port1 in circuit_ports:
                gi = port1.global_index
                if gi is not None:
                    # Add bias
                    if format == 'sparse':
                        if instance.h is not None and port1.index < instance.h.shape[0]:
                            h_global[gi] = float(instance.h[port1.index])
                    else:
                        if instance.h is not None and port1.index < instance.h.shape[0]:
                            h_global[gi, 0] = instance.h[port1.index]
                    
                    # Add couplings
                    for port2 in circuit_ports:
                        gj = port2.global_index
                        if (gj is not None and gi != gj and 
                            instance.J is not None and 
                            port1.index < instance.J.shape[0] and 
                            port2.index < instance.J.shape[1]):
                            
                            weight = instance.J[port1.index, port2.index]
                            if weight != 0:  # Only store non-zero weights
                                if format == 'sparse':
                                    if gi not in J_global:
                                        J_global[gi] = {}
                                    J_global[gi][gj] = float(weight)
                                else:
                                    J_global[gi, gj] = weight
        
        # Process connections between instances
        for instance in self.instances:
            for port in self.get_circuit_ports(instance):
                if port._connections and port.global_index is not None:
                    conn = port._connections
                    other_port = getattr(conn['target_circuit'], conn['target_port'])
                    if other_port.global_index is not None:
                        gi = port.global_index
                        gj = other_port.global_index
                        
                        weight = None
                        if conn['type'] == ConnectionType.VANILLA_COPY:
                            weight = 1.0
                        elif conn['type'] == ConnectionType.WEIGHTED_COPY:
                            weight = float(conn['weight'])
                            
                        if weight is not None:
                            if format == 'sparse':
                                # Add both directions for undirected edges
                                if gi not in J_global:
                                    J_global[gi] = {}
                                if gj not in J_global:
                                    J_global[gj] = {}
                                J_global[gi][gj] = weight
                                J_global[gj][gi] = weight
                            else:
                                J_global[gi, gj] = weight
                                J_global[gj, gi] = weight
        
        return J_global, h_global