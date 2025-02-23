from p_kit import psl, visualization
from p_kit.psl.gates import *


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
            self.gate6.input1, psl.WeightedCopyConnection(weight=-0.5)
        )


circuit = ThreeCases()

J, h = circuit.synthesize(format="dense")

visualization.heatmap(J)
