# p-kit
Python library for the simulation of probabilistic circuits.

Probabilistic computers operate on probabilistic or p-bits that fluctuate between `-1` and `+1` randomly with probability given its neighbors:

$$\mathbb{P}(s_i=1|s_j)=\tanh\left(\sum_{j} \mathbf{W}_{ij} s_j + b_i\right)$$

By connection to the Boltzmann distribution, the binary vector $\vec{s}$ found with the highest probability minimizes the related Hamiltonian:

$$H(\vec{s})=\mathbf{-s^\intercal W s- h^\intercal s}$$

Probabilistic computers have been known to computer science literature as Boltzmann machines for a number of years but have recently re-emerged as an accelerator target for hardware engineers with the advent of [Coherent Ising Machines](https://www.nature.com/articles/s41534-017-0048-9), [Digital Annealers](https://www.fujitsu.com/global/services/business-services/digital-annealer/), [Oscillator Based Ising Machines](https://link.springer.com/chapter/10.1007/978-3-030-19311-9_19), [Dynamical Ising Machines](https://arxiv.org/abs/2305.06414) and [special purpose accelerators for Simulated Bifurcation](https://ieeexplore.ieee.org/abstract/document/8892209).

### What are Probabilistic Circuits?

Probabilistic Circuits or p-circuits are any realization of ordinary sequential computation (both classical and quantum!) using p-bits. To date, there are two design frameworks used to implement probabilistic spin logic:

#### Probabilistic Spin Logic

This framework realises classical computation in an "invertible" format, that is, unlike ordinary classical logic - a digital circuit can be run both forwards and backwards by using Gibbs sampling. To instantiate a PSL circuit, we can use the one-to-one relationship between digital logic gates and probabilistic spin logic gates.

#### Probabilistic Approximate Optimization Algorithms

Drawing inspiration from Quantum Approximate Optimization algorithms, PAOA represents a subset of QAOA that can be simulated probabilistically - this class of algorithms includes Clifford group quantum emulation, smaller implementations of Shor's algorithm and unusual applications typically reserved for Quantum Computers.

Please check the [wiki](https://github.com/IBM/p-kit/wiki) for more information on probabilistic computers.

## Install

There is no official release at the moment. Only dev version is available.
Follow these steps:

1. Clone the repo locally
2. Set up a Python environment (we recommend Anaconda for example). We support python 3.9 and 3.10.
3. Run `pip install .` and `pip install .[tests]` (for visualization and testing).

## Getting Started

Create a circuit using the new decorator-based API:

```python
from p_kit import psl
from p_kit.psl import Port

@psl.pcircuit(n_pbits=3)
class MyCircuit:
    # Define ports
    p1 = Port("p1")
    p2 = Port("p2")
    p3 = Port("p3")
    
    # Define matrices
    J = np.array([[0, -2, -2],
                  [-2, 0, 1],
                  [-2, 1, 0]])
    
    h = np.array([[2], [-1], [-1]])
```

Create a module to combine circuits:

```python
@psl.module
class MyModule:
    def __init__(self):
        self.circuit = MyCircuit()
```

Create a solver and solve the circuit:
- Nt: number of runs
- Dt: time interval for constant inputs
- i0: correlation strength (closer to 1 = closer to boolean logic)

```python
from p_kit.solver.csd_solver import CaSuDaSolver

solver = CaSuDaSolver(Nt=10000, dt=0.1667, i0=0.8)
_, output = solver.solve(c)
```

Visualize the results:

```python
from p_kit.visualization import histplot
histplot(output)
```

![image](https://github.com/IBM/p-kit/assets/6229031/43a6223c-9634-48ca-9eae-c4f7584aa9f8)

The histogram shows the non-random distribution of states, indicating successful probabilistic computation.

## Advanced Features

### Connection Types
The library supports multiple types of connections between ports:

```python
from p_kit.psl import ConnectionType

# No copy connection (shared global index)
circuit1.p3.connect(circuit2.p1, ConnectionType.NO_COPY)

# Vanilla copy with weight 1.0
circuit1.p3.connect(circuit2.p1, ConnectionType.VANILLA_COPY)

# Weighted copy with custom weight
circuit1.p3.connect(circuit2.p1, ConnectionType.WEIGHTED_COPY, weight=0.5)
```

### Standard Gates
The library provides common gates:

```python
from p_kit.psl.gates import ANDGate, ORGate

@psl.module
class LogicCircuit:
    def __init__(self):
        self.and_gate = ANDGate()
        self.or_gate = ORGate()
        
        # Connect gates
        self.and_gate.output.connect(self.or_gate.input1)
```

For more information, visit our [documentation](https://github.com/IBM/p-kit/wiki).
