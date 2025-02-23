from .context import *

def module(cls: Type[T]) -> Type[T]:
    """
    Decorator that transforms a class into a quantum circuit module.
    
    This decorator provides functionality for managing multiple circuit instances
    and synthesizing their combined matrices.
    
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
            if hasattr(attr_value, 'n_pbits'):
                self._context.register_instance(attr_value)
    
    def synthesize(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._context.synthesize()
    
    cls.__new__ = __new__
    cls.__init__ = __init__
    cls.synthesize = synthesize
    
    return cls

def pcircuit(n_pbits: int) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator that transforms a class into a parametric quantum circuit.
    
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
        port_attrs = {name: attr for name, attr in cls.__dict__.items() 
                     if isinstance(attr, Port)}
        
        if len(port_attrs) != n_pbits:
            raise ValueError(
                f"Circuit {cls.__name__} must define exactly {n_pbits} ports "
                f"(one for each index in h/J matrices). Found {len(port_attrs)} ports: "
                f"{list(port_attrs.keys())}"
            )
            
        port_indices = {name: idx for idx, name in enumerate(port_attrs.keys())}
        
        original_new = cls.__new__
        
        def __new__(cls, *args, **kwargs):
            instance = original_new(cls)
            
            instance.n_pbits = n_pbits
            instance.h = getattr(cls, 'h', np.zeros((n_pbits, 1)))
            instance.J = getattr(cls, 'J', np.zeros((n_pbits, n_pbits)))
            instance._connections = {}
            
            if hasattr(cls, 'h'):
                if cls.h.shape != (n_pbits, 1):
                    raise ValueError(
                        f"h matrix must have shape ({n_pbits}, 1), "
                        f"got {cls.h.shape}"
                    )
            
            if hasattr(cls, 'J'):
                if cls.J.shape != (n_pbits, n_pbits):
                    raise ValueError(
                        f"J matrix must have shape ({n_pbits}, {n_pbits}), "
                        f"got {cls.J.shape}"
                    )
            
            for name, port in port_attrs.items():
                new_port = Port(name=port.name)
                new_port.circuit = instance
                new_port.index = port_indices[name]
                setattr(instance, name, new_port)
            
            return instance
        
        cls.__new__ = __new__
        return cls
    return decorator