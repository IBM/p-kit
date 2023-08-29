# p-kit
Python library for the simulation of probabilistic circuits.

A probablilstic computers can be either an software emulator or implemented in hardware. Several hardware implementations are proposed: optical, Arduino, etc. 

There are similarities between the p-bits and the q-bits. We would like to explore the possibility of using p-bits in a smilar way to q-bits. This will allow Quantum circuits to be executed on a probabilistic computer instead of a quantum computer.

Please check the [wiki](https://github.com/IBM/p-kit/wiki) for more information on probabilistic computers.

# Install

There is no official release at the moment. Only dev version is available.
Folow these steps:

1. Clone the repo locally
2. Set up a Python environment (we recommend Anaconda for example). We support python 3.9 and 3.10.
3. Run `pip install .` and `pip install .[tests]` (for visualization and testing).

# Getting starting

Create a circuit

```
from p_kit.core import PCircuit
c = PCircuit(3)
```

Set J and h. P-bits are interconnected inside a graph.
J is the weight of the connection between the p-bits.
h contains the biases for each p-bits:
- a high bias forces the p-bit value towards 1
- a low bias forces the p-bit value towards -1
- a bias equals to zero, results in a equal probability for the p-bit of being 1 or -1.

```
import numpy as np
c.J = np.array([[0,-2, -2],[-2, 0, 1],[-2, 1, 0]])
c.h = np.array([2,-1,-1])
```

Create a solver, and solves the circuit
Nt stands for the number of runs.
Dt for the time interval for which the inputs are held constant.
i0 is the strength of the correlation. The closer to 1, the closer p-bit behaves to boolean logic. 

```
from p_kit.solver.csd_solver import CaSuDaSolver
solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)
output = solver.solve(c)
```

The size of the output is 10000 (Nt).
It contains the states taken by the p-bits during the 10000 runs.
Let's plot a histogram of it

```
from p_kit.visualization import histplot
histplot(output)
```

![image](https://github.com/IBM/p-kit/assets/6229031/7988ba75-3f37-4f14-ab31-05c7659619a2)

You can see here that the states are not randomly distributed.

