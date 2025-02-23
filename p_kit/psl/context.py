from .port import *
from typing import Dict, List, Tuple, Union, Any
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
                if (
                    hasattr(port, "_connection_strategy")
                    and getattr(port, "_connection_strategy")
                    and port.global_index is None
                ):
                    # Let the strategy handle index assignment
                    port_count = port._connection_strategy.assign_global_index(
                        port, port._connected_port, port_count, self.port_to_global
                    )
                elif port.global_index is None:
                    # Unconnected ports get their own index
                    port.global_index = port_count
                    self.port_to_global[port] = port_count
                    port_count += 1
        return port_count

    def get_circuit_ports(self, instance) -> List[Port]:
        return [port for name, port in vars(instance).items() if isinstance(port, Port)]

    def synthesize(self, format: str = "sparse") -> Union[
        Tuple[Dict[int, Dict[int, float]], Dict[int, float]],
        Tuple[np.ndarray, np.ndarray],
    ]:
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
        J_global = {} if format == "sparse" else np.zeros((total_ports, total_ports))
        h_global = {} if format == "sparse" else np.zeros((total_ports, 1))

        self._process_instance_matrices(J_global, h_global, format)
        self._process_connections(J_global, format)

        return J_global, h_global

    def _process_instance_matrices(self, J_global, h_global, format: str):
        """Process internal J and h matrices of instances."""
        for instance in self.instances:
            circuit_ports = self.get_circuit_ports(instance)
            for port1 in circuit_ports:
                gi = port1.global_index
                if gi is not None:
                    # Add bias
                    if instance.h is not None and port1.index < instance.h.shape[0]:
                        if format == "sparse":
                            h_global[gi] += float(instance.h[port1.index])
                        else:
                            h_global[gi, 0] += instance.h[port1.index]

                    # Add couplings
                    self._add_instance_couplings(
                        instance, port1, gi, circuit_ports, J_global, format
                    )

    def _add_instance_couplings(
        self, instance, port1, gi, circuit_ports, J_global, format
    ):
        """Add coupling terms from instance J matrix."""
        for port2 in circuit_ports:
            gj = port2.global_index
            if (
                gj is not None
                and gi != gj
                and instance.J is not None
                and port1.index < instance.J.shape[0]
                and port2.index < instance.J.shape[1]
            ):

                weight = instance.J[port1.index, port2.index]
                if weight != 0:  # Only store non-zero weights
                    if format == "sparse":
                        if gi not in J_global:
                            J_global[gi] = {}
                        J_global[gi][gj] = float(weight)
                    else:
                        J_global[gi, gj] = weight

    def _process_connections(self, J_global, format: str):
        """Process connections between instances."""
        for instance in self.instances:
            for port in self.get_circuit_ports(instance):
                if (
                    hasattr(port, "_connection_strategy")
                    and getattr(port, "_connection_strategy")
                    and port.global_index is not None
                ):
                    other_port = port._connected_port
                    if other_port.global_index is not None:
                        # Let the strategy handle matrix updates
                        port._connection_strategy.synthesize_connection(
                            port.global_index, other_port.global_index, J_global
                        )
