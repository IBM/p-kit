# Objective

Study if a LM (and later LLM) can be implemented over p-kit. 

Compared with a traditional Transformer LLM, a p-kit-based language model could have these potential advantages:
•	Sparse computation: fewer active connections than dense attention/MLP layers. 
•	Binary/stochastic state: potentially simpler and lower-precision hardware. 
•	Natural probabilistic sampling: generation and uncertainty are built into the dynamics. 
•	Hardware flexibility: same J,h model could target CPU, GPU, FPGA, MCU, or physical p-bit hardware. 
•	Energy efficiency potential: especially on dedicated stochastic hardware. 
•	No dependence on PyTorch/TensorFlow: simpler experimental stack and easier custom hardware mapping. 
•	Recurrent state by design: potentially efficient for sequential processing without full attention over all previous tokens. 

# Current implemenation

The current implementation is a proof-of-concept sparse p-bit reservoir language model.

The recurrent state is formed by stochastic p-bits with sparse pairwise couplings:

$$
P(s_i = +1) = \frac{1 + \tanh(\beta I_i)}{2}
$$

$$
I_i = \sum_j J_{ij}s_j + h_i(\text{token}) + \text{memory\_scale}\,s_i(\text{previous})
$$

Workflow
character
   -> sparse stochastic p-bit reservoir
   -> p-kit ReLU circuit
   -> reservoir + ReLU features
   -> ridge-regression readout
   -> next-character probabilities

Both the recurrent reservoir and ReLU block are represented as p-kit circuits and executed with the p-kit stochastic solver. Only the final linear readout is trained using NumPy ridge regression.

The architecture is similar to an Echo State Network / reservoir computer:

Reservoir (J) couplings are randomly generated and remain fixed.
The p-bit ReLU block is fixed.
Only the final readout weights are trained.
No backpropagation through the recurrent network is required.

The sparse p-bit architecture is intended to be hardware-friendly: the same model can currently be simulated in software and could later target dedicated p-bit hardware.

# Next steps