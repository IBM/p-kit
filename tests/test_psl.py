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
            self.gate1.output.connect(self.gate2.input1, psl.NoCopyConnection)

            # Case 2: Vanilla Copy Gate
            self.gate3 = ANDGate()
            self.gate4 = ANDGate()
            self.gate3.output.connect(self.gate4.input1, psl.VanillaCopyConnection)

            # Case 3: Weighted Copy Gate
            self.gate5 = ANDGate()
            self.gate6 = ANDGate()
            self.gate5.output.connect(
                self.gate6.input1, psl.WeightedCopyConnection(weight=0.5)
            )

    return ThreeCases()


def test_no_copy_connection(three_cases):
    """Test NO_COPY connection between gate1 and gate2."""
    # Verify that the ports share the same global index
    assert (
        three_cases.gate1.output.global_index == three_cases.gate2.input1.global_index
    )

    # Verify connection type
    assert type(three_cases.gate1.output._connection_strategy) == psl.NoCopyConnection


def test_vanilla_copy_connection_dense(three_cases):
    """Test VANILLA_COPY connection between gate3 and gate4 in dense format."""

    # Verify connection type
    assert (
        type(three_cases.gate3.output._connection_strategy) == psl.VanillaCopyConnection
    )

    # Get synthesized matrices in dense format
    J, _ = three_cases.synthesize(format="dense")

    # Check coupling strength is 1.0 for vanilla copy
    i = three_cases.gate3.output.global_index
    j = three_cases.gate4.input1.global_index
    assert J[i, j] == 1.0
    assert J[j, i] == 1.0  # Should be symmetric


def test_vanilla_copy_connection_sparse(three_cases):
    """Test VANILLA_COPY connection between gate3 and gate4 in sparse format."""

    # Get synthesized matrices in sparse format
    J, _ = three_cases.synthesize(format="sparse")

    # Check coupling strength is 1.0 for vanilla copy
    i = three_cases.gate3.output.global_index
    j = three_cases.gate4.input1.global_index
    assert J[i][j] == 1.0
    assert J[j][i] == 1.0  # Should be symmetric

    # Verify that only non-zero connections are stored
    assert len(J[i]) > 0
    assert all(weight != 0 for weight in J[i].values())


def test_weighted_copy_connection_dense(three_cases):
    """Test WEIGHTED_COPY connection between gate5 and gate6 in dense format."""

    # Verify connection type and weight
    assert (
        type(three_cases.gate5.output._connection_strategy)
        == psl.WeightedCopyConnection
    )
    assert three_cases.gate5.output._connection_strategy.weight == 0.5

    # Get synthesized matrices
    J, _ = three_cases.synthesize(format="dense")

    # Check coupling strength matches weight
    i = three_cases.gate5.output.global_index
    j = three_cases.gate6.input1.global_index
    assert J[i, j] == 0.5
    assert J[j, i] == 0.5  # Should be symmetric


def test_weighted_copy_connection_sparse(three_cases):
    """Test WEIGHTED_COPY connection between gate5 and gate6 in sparse format."""

    # Get synthesized matrices in sparse format
    J, _ = three_cases.synthesize(format="sparse")

    # Check coupling strength matches weight
    i = three_cases.gate5.output.global_index
    j = three_cases.gate6.input1.global_index
    assert J[i][j] == 0.5
    assert J[j][i] == 0.5  # Should be symmetric


def test_synthesis_matrix_shapes_dense(three_cases):
    """Test the shapes of synthesized dense matrices."""
    J, h = three_cases.synthesize(format="dense")

    # Count total unique ports (considering NO_COPY shares indices)
    total_ports = len(
        {
            port.global_index
            for gate in [
                three_cases.gate1,
                three_cases.gate2,
                three_cases.gate3,
                three_cases.gate4,
                three_cases.gate5,
                three_cases.gate6,
            ]
            for port in [gate.input1, gate.input2, gate.output]
            if port.global_index is not None
        }
    )

    # Check matrix shapes
    assert J.shape == (total_ports, total_ports)
    assert h.shape == (total_ports, 1)


def test_synthesis_sparse_structure(three_cases):
    """Test the structure of synthesized sparse matrices."""
    J, h = three_cases.synthesize(format="sparse")

    # Verify J is a dictionary of dictionaries
    assert isinstance(J, dict)
    assert all(isinstance(adj, dict) for adj in J.values())

    # Verify h is a dictionary
    assert isinstance(h, dict)

    # Verify all weights are floats
    assert all(
        isinstance(weight, float) for adj in J.values() for weight in adj.values()
    )
    assert all(isinstance(bias, float) for bias in h.values())


def test_synthesis_matrix_properties_dense(three_cases):
    """Test properties of synthesized dense matrices."""
    J, h = three_cases.synthesize(format="dense")

    # J matrix should be symmetric
    assert np.allclose(J, J.T)

    # Diagonal of J should be zero
    assert np.allclose(np.diag(J), 0)


def test_synthesis_matrix_properties_sparse(three_cases):
    """Test properties of synthesized sparse matrices."""
    J, h = three_cases.synthesize(format="sparse")

    # Test symmetry
    for i in J:
        for j, weight in J[i].items():
            assert abs(J[j][i] - weight) < 1e-10

    # Test no self-loops (diagonal should be absent)
    for i in J:
        assert i not in J[i]


def test_format_validation(three_cases):
    """Test that invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid format"):
        three_cases.synthesize(format="invalid")


def test_sparse_dense_equivalence(three_cases):
    """Test that sparse and dense formats produce equivalent results."""
    J_sparse, h_sparse = three_cases.synthesize(format="sparse")
    J_dense, h_dense = three_cases.synthesize(format="dense")

    # Convert sparse to dense for comparison
    n = J_dense.shape[0]
    J_from_sparse = np.zeros((n, n))
    for i in J_sparse:
        for j, weight in J_sparse[i].items():
            J_from_sparse[i, j] = weight

    h_from_sparse = np.zeros((n, 1))
    for i, bias in h_sparse.items():
        h_from_sparse[i] = bias

    # Compare
    assert np.allclose(J_dense, J_from_sparse)
    assert np.allclose(h_dense, h_from_sparse)


# Optional: Test error cases
def test_invalid_connection_raises_error():
    """Test that invalid connections raise appropriate errors."""
    gate1 = ANDGate()
    gate2 = ANDGate()

    # Try to connect without setting circuit reference (should raise ValueError)
    with pytest.raises(ValueError, match="Both ports must be bound to circuits"):
        port1 = psl.Port("test1")
        port2 = psl.Port("test2")
        port1.connect(port2, psl.NoCopyConnection)
