from .port import *

class ModuleContext:
    """
    Manages the context for a probabilistic circuit module, handling instance registration
    and matrix synthesis.
    
    The ModuleContext class coordinates the global indices of ports across multiple
    circuit instances and synthesizes the final system matrices.
    
    Attributes:
        instances (List[Any]): List of circuit instances in this context
        port_to_global (Dict[Port, int]): Mapping of ports to their global indices
    
    Example:
        >>> context = ModuleContext()
        >>> context.register_instance(my_circuit)
        >>> J, h = context.synthesize()
    """
    def __init__(self):
        self.instances: List[Any] = []
        self.port_to_global: Dict[Port, int] = {}
        
    def register_instance(self, instance: Any):
        """
        Registers a circuit instance in this context.
        
        Args:
            instance (Any): Circuit instance to register
        """
        self.instances.append(instance)
        
    def assign_global_indices(self) -> int:
        """
        Assigns global indices to all ports in registered instances.
        
        Returns:
            int: Total number of unique ports in the system
        """
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
        """
        Retrieves all ports from a circuit instance.
        
        Args:
            instance (Any): Circuit instance to get ports from
            
        Returns:
            List[Port]: List of ports in the circuit
        """
        return [port for name, port in vars(instance).items() 
                if isinstance(port, Port)]

    def synthesize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes global J and h matrices from all registered instances.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Global J coupling matrix and h bias vector
        """
        total_ports = self.assign_global_indices()
        
        J_global = np.zeros((total_ports, total_ports))
        h_global = np.zeros((total_ports, 1))
        
        for instance in self.instances:
            circuit_ports = self.get_circuit_ports(instance)
            for port1 in circuit_ports:
                gi = port1.global_index
                if gi is not None:
                    h_global[gi] = instance.h[port1.index]
                    
                    for port2 in circuit_ports:
                        gj = port2.global_index
                        if gj is not None and gi != gj:
                            J_global[gi, gj] = instance.J[port1.index, port2.index]
        
        for instance in self.instances:
            for port in self.get_circuit_ports(instance):
                if port._connections and port.global_index is not None:
                    conn = port._connections
                    other_port = getattr(conn['target_circuit'], conn['target_port'])
                    if other_port.global_index is not None:
                        gi = port.global_index
                        gj = other_port.global_index
                        
                        if conn['type'] == ConnectionType.VANILLA_COPY:
                            J_global[gi, gj] = 1
                            J_global[gj, gi] = 1
                        elif conn['type'] == ConnectionType.WEIGHTED_COPY:
                            J_global[gi, gj] = conn['weight']
                            J_global[gj, gi] = conn['weight']
        
        return J_global, h_global