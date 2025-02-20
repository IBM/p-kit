# p-kit
Python library for the simulation of probabilistic circuits.

Probabilistic computers operate on probabilistic or p-bits that fluctuate between `-1` and `+1` randomly with probability given its neighbors:

$$\mathbb{P}(s_i=1|s_j)=\tanh\left(\sum_{j} \mathbf{W}_{ij} s_j + b_i\right)$$

By connection to the Boltzmann distribution, the binary vector $\vec{s}$ found with the highest probability minimizes the related Hamiltonian:

$$H(\vec{s})=\mathbf{-s^\intercal W s- h^\intercal s}$$

Probabilistic computers have been known to computer science literature as Boltzmann machines for a number of years but have recently re-emerged as an accelerator target for hardware engineers with the advent of [Coherent Ising Machines](https://www.nature.com/articles/s41534-017-0048-9), [Digital Annealers](https://www.fujitsu.com/global/services/business-services/digital-annealer/), [Oscillator Based Ising Machines](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_19), [Dynamical Ising Machines](https://arxiv.org/abs/2305.06414) and [special purpose accelerators for Simulated Bifurcation](https://ieeexplore.ieee.org/abstract/document/8892209).

### What are Probabilistic Circuits?

[TODO]

Please check the [wiki](https://github.com/IBM/p-kit/wiki) for more information on probabilistic computers.

## Install

There is no official release at the moment. Only dev version is available.
Folow these steps:

1. Clone the repo locally
2. Set up a Python environment (we recommend Anaconda for example). We support python 3.9 and 3.10.
3. Run `pip install .` and `pip install .[tests]` (for visualization and testing).

## Getting starting

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
_, output = solver.solve(c)
```

The size of the output is 10000 (Nt).
It contains the states taken by the p-bits during the 10000 runs.
Let's plot a histogram of it

```
from p_kit.visualization import histplot
histplot(output)
```

![image](https://github.com/IBM/p-kit/assets/6229031/43a6223c-9634-48ca-9eae-c4f7584aa9f8)

You can see here that the states are not randomly distributed.
