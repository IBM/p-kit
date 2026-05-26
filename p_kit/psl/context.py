from .port import *
from typing import Dict, List, Tuple, Union, Any
import numpy as np


class ModuleContext:
    def __init__(self):
        self.instances: List[Any] = []
        self.port_to_global: Dict[Port, int] = {}

    def register_instance(self, instance: Any):
        self.instances.append(instance)

    def register_submodule(self, submodule: Any):
        """Flatten a sub-module's instances into this context."""
        for instance in submodule._context.instances:
            if instance not in self.instances:
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
                strategy = getattr(port, "_connection_strategy", None)
                if strategy:
                    if port.global_index is None:
                        port_count = strategy.assign_global_index(
                            port, port._connected_port, port_count, self.port_to_global
                        )
                    elif (
                        isinstance(strategy, NoCopyConnection)
                        and port._connected_port.global_index is None
                    ):
                        # Port already indexed by a prior NoCopy; propagate index down the chain
                        target = port._connected_port
                        target.global_index = port.global_index
                        self.port_to_global[target] = port.global_index
                elif port.global_index is None:
                    port.global_index = port_count
                    self.port_to_global[port] = port_count
                    port_count += port.width
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

        J_global = {} if format == "sparse" else np.zeros((total_ports, total_ports))
        h_global = {} if format == "sparse" else np.zeros((total_ports, 1))

        self._process_instance_matrices(J_global, h_global, format)
        self._process_connections(J_global, format)

        return J_global, h_global

    def _process_instance_matrices(self, J_global, h_global, format: str):
        """Process internal J and h matrices of instances."""
        for instance in self.instances:
            # Skip module instances that have no J/h (they are port containers only)
            if not hasattr(instance, "J") or not hasattr(instance, "h"):
                continue
            circuit_ports = self.get_circuit_ports(instance)
            for port1 in circuit_ports:
                if port1.global_index is None:
                    continue
                for bit in range(port1.width):
                    gi = port1.global_index + bit
                    local_idx = port1.index + bit
                    if instance.h is not None and local_idx < instance.h.shape[0]:
                        val = float(instance.h.flat[local_idx])
                        if format == "sparse":
                            h_global[gi] = h_global.get(gi, 0.0) + val
                        else:
                            h_global[gi, 0] += val
                    self._add_instance_couplings(
                        instance, local_idx, gi, circuit_ports, J_global, format
                    )

    def _add_instance_couplings(
        self, instance, local_idx, gi, circuit_ports, J_global, format
    ):
        """Add coupling terms from instance J matrix for one source bit."""
        for port2 in circuit_ports:
            if port2.global_index is None:
                continue
            for bit2 in range(port2.width):
                gj = port2.global_index + bit2
                local_idx2 = port2.index + bit2
                if gi == gj:
                    continue
                if (
                    instance.J is None
                    or local_idx >= instance.J.shape[0]
                    or local_idx2 >= instance.J.shape[1]
                ):
                    continue
                weight = instance.J[local_idx, local_idx2]
                if weight != 0:
                    if format == "sparse":
                        J_global.setdefault(gi, {})[gj] = float(weight)
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
                        for bit in range(port.width):
                            port._connection_strategy.synthesize_connection(
                                port.global_index + bit,
                                other_port.global_index + bit,
                                J_global,
                                format,
                            )
