import pytest
import numpy as np
from p_kit import psl
from p_kit.psl.gates import ANDGate

@pytest.fixture
def three_cases():
    """Fixture providing an instance of ThreeCases."""
    @psl.module
    class ThreeCases:
        def __init__(self):
            # Case 1: No Copy Gate
            self.gate1 = ANDGate()
            self.gate2 = ANDGate()
            self.gate1.output.connect(self.gate2.input1, psl.ConnectionType.NO_COPY)
            
            # Case 2: Vanilla Copy Gate
            self.gate3 = ANDGate()
            self.gate4 = ANDGate()
            self.gate3.output.connect(self.gate4.input1, psl.ConnectionType.VANILLA_COPY)
            
            # Case 3: Weighted Copy Gate
            self.gate5 = ANDGate()
            self.gate6 = ANDGate()
            self.gate5.output.connect(self.gate6.input1, psl.ConnectionType.WEIGHTED_COPY, weight=0.5)
    
    return ThreeCases()

def test_no_copy_connection(three_cases):
    """Test NO_COPY connection between gate1 and gate2."""
    # Verify that the ports share the same global index
    assert three_cases.gate1.output.global_index == three_cases.gate2.input1.global_index
    
    # Verify connection type
    assert three_cases.gate1.output._connections['type'] == psl.ConnectionType.NO_COPY

def test_vanilla_copy_connection(three_cases):
    """Test VANILLA_COPY connection between gate3 and gate4."""
    
    # Verify connection type
    assert three_cases.gate3.output._connections['type'] == psl.ConnectionType.VANILLA_COPY
    
    # Get synthesized matrices
    J, _ = three_cases.synthesize()
    
    # Check coupling strength is 1.0 for vanilla copy
    i = three_cases.gate3.output.global_index
    j = three_cases.gate4.input1.global_index
    assert J[i, j] == 1.0
    assert J[j, i] == 1.0  # Should be symmetric

def test_weighted_copy_connection(three_cases):
    """Test WEIGHTED_COPY connection between gate5 and gate6."""
    
    # Verify connection type and weight
    assert three_cases.gate5.output._connections['type'] == psl.ConnectionType.WEIGHTED_COPY
    assert three_cases.gate5.output._connections['weight'] == 0.5
    
    # Get synthesized matrices
    J, _ = three_cases.synthesize()
    
    # Check coupling strength matches weight
    i = three_cases.gate5.output.global_index
    j = three_cases.gate6.input1.global_index
    assert J[i, j] == 0.5
    assert J[j, i] == 0.5  # Should be symmetric

def test_synthesis_matrix_shapes(three_cases):
    """Test the shapes of synthesized matrices."""
    J, h = three_cases.synthesize()
    
    # Count total unique ports (considering NO_COPY shares indices)
    total_ports = len({port.global_index for gate in 
                      [three_cases.gate1, three_cases.gate2, 
                       three_cases.gate3, three_cases.gate4,
                       three_cases.gate5, three_cases.gate6]
                      for port in [gate.input1, gate.input2, gate.output]
                      if port.global_index is not None})
    
    # Check matrix shapes
    assert J.shape == (total_ports, total_ports)
    assert h.shape == (total_ports, 1)

def test_synthesis_matrix_properties(three_cases):
    """Test properties of synthesized matrices."""
    J, h = three_cases.synthesize()
    
    # J matrix should be symmetric
    assert np.allclose(J, J.T)
    
    # Diagonal of J should be zero
    assert np.allclose(np.diag(J), 0)

# Optional: Test error cases
def test_invalid_connection_raises_error():
    """Test that invalid connections raise appropriate errors."""
    gate1 = ANDGate()
    gate2 = ANDGate()
    
    # Try to connect without setting circuit reference (should raise ValueError)
    with pytest.raises(ValueError, match="Both ports must be bound to circuits"):
        port1 = psl.Port("test1")
        port2 = psl.Port("test2")
        port1.connect(port2, psl.ConnectionType.NO_COPY)