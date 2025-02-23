from p_kit import psl, visualization
from p_kit.psl.gates import *


@psl.module
class CoupledANDs:
    def __init__(self):
        # Two ANDs with all terminals coupled
        self.gate1 = ANDGate()
        self.gate2 = ANDGate()
        self.gate1.input1.connect(self.gate2.input1, psl.VanillaCopyConnection)
        self.gate1.input2.connect(self.gate2.input2, psl.VanillaCopyConnection)
        self.gate1.output.connect(self.gate2.output, psl.VanillaCopyConnection)


circuit = CoupledANDs()

J, _ = circuit.synthesize(format="dense")

visualization.heatmap(J)
