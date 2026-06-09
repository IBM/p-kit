import pytest
import numpy as np
from p_kit import psl
from p_kit.psl.gates import ANDGate, ORGate


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


# ── Feature: Ports with width ─────────────────────────────────────────────────

@psl.pcircuit(n_pbits=5)
class WideBusGate:
    """A gate with a 4-bit input bus and a 1-bit output."""
    data_in = psl.Port("data_in", width=4)
    flag = psl.Port("flag", width=1)

    J = np.zeros((5, 5))
    h = np.zeros((5, 1))


def test_port_width_field():
    gate = WideBusGate()
    assert gate.data_in.width == 4
    assert gate.flag.width == 1


def test_port_global_indices_after_synthesis():
    @psl.module
    class WideCircuit:
        def __init__(self):
            self.gate = WideBusGate()

    wc = WideCircuit()
    wc.synthesize(format="dense")
    gate = wc.gate
    assert len(gate.data_in.global_indices) == 4
    gi = gate.data_in.global_indices
    assert gi == list(range(gi[0], gi[0] + 4))


def test_wide_port_connection_width_mismatch_raises():
    gate1 = ANDGate()
    gate2 = ANDGate()
    # output (width=1) vs input1 (width=1) — OK, but let's try width mismatch via Port directly
    with pytest.raises(ValueError, match="different widths"):
        @psl.module
        class Mismatch:
            def __init__(self):
                self.g1 = WideBusGate()
                self.g2 = ANDGate()
                # data_in (width=4) vs input1 (width=1)
                self.g1.data_in.connect(self.g2.input1, psl.NoCopyConnection)
        Mismatch()


def test_wide_port_index_offset():
    """data_in starts at local index 0, flag at local index 4."""
    gate = WideBusGate()
    assert gate.data_in.index == 0
    assert gate.flag.index == 4


# ── Feature: Modules with Ports ───────────────────────────────────────────────

@psl.module
class AndWrapper:
    """Module that exposes the AND gate's output as a named interface port."""
    result = psl.Port("result")

    def __init__(self):
        self.gate = ANDGate()
        self.result.connect(self.gate.output, psl.NoCopyConnection)


def test_module_port_shares_global_index():
    aw = AndWrapper()
    aw.synthesize(format="dense")
    assert aw.result.global_index == aw.gate.output.global_index


def test_module_port_in_synthesis_shape():
    aw = AndWrapper()
    J, h = aw.synthesize(format="dense")
    # AND gate has 3 p-bits; result shares output's index → still 3 unique indices
    assert J.shape == (3, 3)
    assert h.shape == (3, 1)


def test_module_with_port_connected_to_another_gate():
    # Connect via internal gate port; the NoCopy propagation carries the shared
    # index through the chain: result(0) == gate.output(0) == gate2.input1(0)
    @psl.module
    class Chain:
        def __init__(self):
            self.wrapper = AndWrapper()
            self.gate2 = ANDGate()
            self.wrapper.gate.output.connect(self.gate2.input1, psl.NoCopyConnection)

    chain = Chain()
    J, h = chain.synthesize(format="dense")
    # 3 (AND in wrapper) + 3 (gate2) - 1 (shared output/input1) = 5
    assert J.shape == (5, 5)
    # All three ports collapse to the same global index
    assert chain.wrapper.result.global_index == chain.wrapper.gate.output.global_index
    assert chain.wrapper.gate.output.global_index == chain.gate2.input1.global_index


# ── Feature: Recursive Synthesis ──────────────────────────────────────────────

@psl.module
class TwoAnds:
    """Two AND gates chained; output of first feeds input of second."""
    def __init__(self):
        self.and1 = ANDGate()
        self.and2 = ANDGate()
        self.and1.output.connect(self.and2.input1, psl.NoCopyConnection)


def test_recursive_synthesis_flat():
    """Sub-module instances are flattened into the parent context."""
    @psl.module
    class FourAnds:
        def __init__(self):
            self.pair1 = TwoAnds()
            self.pair2 = TwoAnds()

    fa = FourAnds()
    J, h = fa.synthesize(format="dense")
    # pair1: 3+3-1 = 5 unique pbits; pair2: 5 unique pbits; no cross-connection → 10 total
    assert J.shape == (10, 10)
    assert h.shape == (10, 1)


def test_recursive_synthesis_preserves_couplings():
    """Internal couplings of sub-modules are present in the global J matrix."""
    @psl.module
    class Nested:
        def __init__(self):
            self.sub = TwoAnds()

    n = Nested()
    J_nested, _ = n.synthesize(format="dense")

    standalone = TwoAnds()
    J_standalone, _ = standalone.synthesize(format="dense")

    assert J_nested.shape == J_standalone.shape
    assert np.allclose(J_nested, J_standalone)


def test_recursive_synthesis_sparse_dense_equivalence():
    @psl.module
    class Nested:
        def __init__(self):
            self.sub = TwoAnds()

    n = Nested()
    J_sparse, h_sparse = n.synthesize(format="sparse")
    J_dense, h_dense = n.synthesize(format="dense")

    size = J_dense.shape[0]
    J_from_sparse = np.zeros((size, size))
    for i, row in J_sparse.items():
        for j, w in row.items():
            J_from_sparse[i, j] = w

    assert np.allclose(J_dense, J_from_sparse)



# ── XOR Gate Tests ────────────────────────────────────────────────────────────

def test_xor_gate_structure():
    """Test XOR gate has correct structure."""
    from p_kit.psl.gates import XORGate
    
    gate = XORGate()
    assert gate.input1.width == 1
    assert gate.input2.width == 1
    assert gate.output.width == 1
    assert gate.aux.width == 1
    assert gate.J.shape == (4, 4)
    assert gate.h.shape == (4, 1)


def test_xor_gate_truth_table():
    """Test XOR gate produces correct truth table with high i0."""
    from p_kit.psl.gates import XORGate
    from p_kit.solver.csd_solver import CaSuDaSolver
    
    gate = XORGate()
    solver = CaSuDaSolver(Nt=5000, dt=0.1667, i0=0.95, seed=42)
    
    test_cases = [
        ([-1, -1], -1),  # 0 XOR 0 = 0
        ([-1, 1], 1),    # 0 XOR 1 = 1
        ([1, -1], 1),    # 1 XOR 0 = 1
        ([1, 1], -1),    # 1 XOR 1 = 0
    ]
    
    for inputs, expected_output in test_cases:
        gate.h[0] = inputs[0] * 10
        gate.h[1] = inputs[1] * 10
        _, output, _ = solver.solve(gate)
        
        # Output is at index 2 (order: input1, input2, output, aux)
        output_states = output[:, 2]
        most_common = 1 if np.mean(output_states) > 0 else -1
        assert most_common == expected_output, \
            f"XOR({inputs[0]}, {inputs[1]}) expected {expected_output}, got {most_common}"


# ── XNOR Gate Tests ───────────────────────────────────────────────────────────

def test_xnor_gate_structure():
    """Test XNOR gate has correct structure."""
    from p_kit.psl.gates import XNORGate
    
    gate = XNORGate()
    assert gate.input1.width == 1
    assert gate.input2.width == 1
    assert gate.output.width == 1
    assert gate.aux.width == 1
    assert gate.J.shape == (4, 4)
    assert gate.h.shape == (4, 1)


def test_xnor_gate_truth_table():
    """Test XNOR gate produces correct truth table with high i0."""
    from p_kit.psl.gates import XNORGate
    from p_kit.solver.csd_solver import CaSuDaSolver
    
    gate = XNORGate()
    solver = CaSuDaSolver(Nt=5000, dt=0.1667, i0=0.95, seed=42)
    
    test_cases = [
        ([-1, -1], 1),   # 0 XNOR 0 = 1
        ([-1, 1], -1),   # 0 XNOR 1 = 0
        ([1, -1], -1),   # 1 XNOR 0 = 0
        ([1, 1], 1),     # 1 XNOR 1 = 1
    ]
    
    for inputs, expected_output in test_cases:
        gate.h[0] = inputs[0] * 10
        gate.h[1] = inputs[1] * 10
        _, output, _ = solver.solve(gate)
        
        # Output is at index 2 (order: input1, input2, output, aux)
        output_states = output[:, 2]
        most_common = 1 if np.mean(output_states) > 0 else -1
        assert most_common == expected_output, \
            f"XNOR({inputs[0]}, {inputs[1]}) expected {expected_output}, got {most_common}"


# ── Half Adder Tests ──────────────────────────────────────────────────────────

def test_half_adder_structure():
    """Test Half Adder has correct structure."""
    from p_kit.psl.gates import HalfAdder
    
    gate = HalfAdder()
    assert gate.input1.width == 1
    assert gate.input2.width == 1
    assert gate.sumout.width == 1
    assert gate.carryout.width == 1
    assert gate.J.shape == (4, 4)
    assert gate.h.shape == (4, 1)


def test_half_adder_truth_table():
    """Test Half Adder produces correct sum and carry outputs."""
    from p_kit.psl.gates import HalfAdder
    from p_kit.solver.csd_solver import CaSuDaSolver
    
    gate = HalfAdder()
    solver = CaSuDaSolver(Nt=5000, dt=0.1667, i0=0.95, seed=42)
    
    test_cases = [
        ([-1, -1], -1, -1),
        ([-1, 1], 1, -1),
        ([1, -1], 1, -1),
        ([1, 1], -1, 1),
    ]
    
    for inputs, expected_sum, expected_carry in test_cases:
        gate.h[0] = inputs[0] * 10
        gate.h[1] = inputs[1] * 10
        _, output, _ = solver.solve(gate)
        
        sum_states = output[:, 2]
        carry_states = output[:, 3]
        
        sum_result = 1 if np.mean(sum_states) > 0 else -1
        carry_result = 1 if np.mean(carry_states) > 0 else -1
        
        assert sum_result == expected_sum
        assert carry_result == expected_carry


