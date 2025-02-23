from p_kit.psl import *
from p_kit.psl.gates import *
from p_kit.visualization import heatmap


@module
class Multiplier:
    def __init__(self, size: int):
        # First create all the AND gates and Full Adders
        # We need size*size AND gates for all possible bit combinations
        for i in range(size):
            for j in range(size):
                # Create and store AND gate instance
                setattr(self, f"AND_{i}_{j}", ANDGate())

                # Create and store Full Adder instance
                # We need fewer FAs than ANDs
                if i > 0:  # Skip first row
                    setattr(self, f"FA_{i}_{j}", FullAdder())

        # Now connect everything up...
        for i in range(size):
            for j in range(size):
                current_and = getattr(self, f"AND_{i}_{j}")

                # Relate all input spins with COPY gates
                if i < size - 1:
                    next_i_and = getattr(self, f"AND_{i+1}_{j}")
                    current_and.input1.connect(next_i_and.input1, VanillaCopyConnection)
                if j < size - 1:
                    next_j_and = getattr(self, f"AND_{i}_{j+1}")
                    current_and.input2.connect(next_j_and.input2, VanillaCopyConnection)

                # Connect Full Adders
                if i > 0:  # Skip first row as it's just AND gates
                    current_fa = getattr(self, f"FA_{i}_{j}")

                    # Connect AND output to FA
                    current_and.output.connect(current_fa.input1, VanillaCopyConnection)

                    # Connect previous carry if not first column
                    if j > 0:
                        prev_fa = getattr(self, f"FA_{i}_{j-1}")
                        prev_fa.carryout.connect(
                            current_fa.carryin, VanillaCopyConnection
                        )


circuit = Multiplier(8)

J, _ = circuit.synthesize(format="dense")

heatmap(J)
