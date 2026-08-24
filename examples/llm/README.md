# Objective

Study if a language model (and later LLM) can be implemented over p-kit. 

Compared with a traditional Transformer LLM, a p-kit-based language model could have these potential advantages:
* Sparse computation: fewer active connections than dense attention/MLP layers. 
* Binary/stochastic state: potentially simpler and lower-precision hardware. 
* Natural probabilistic sampling: generation and uncertainty are built into the dynamics. 
* Hardware flexibility: same J,h model could target CPU, GPU, FPGA, MCU, or physical p-bit hardware. 
* Energy efficiency potential: especially on dedicated stochastic hardware. 
* No dependence on PyTorch/TensorFlow: simpler experimental stack and easier custom hardware mapping. 
* Recurrent state by design: potentially efficient for sequential processing without full attention over all previous tokens. 

# Current implementation

The current implementation is a proof-of-concept sparse p-bit [Echo State Network](https://en.wikipedia.org/wiki/Echo_state_network) / reservoir language model. The ESN/reservoir is a bit limited, but fits nicely with p-kit. It maps well to p-kit because the reservoir can be a fixed sparse stochastic p-bit network, while only a small readout is trained.

The recurrent state is formed by stochastic p-bits with sparse pairwise couplings:

$$
P(s_i = +1) = \frac{1 + \tanh(\beta I_i)}{2}
$$

$$
I_i = \sum_j J_{ij}s_j + h_i(\mathrm{token}) + \alpha s_i(\mathrm{previous})
$$

where $\alpha$ is the memory scale.

## Workflow

`character -> sparse stochastic p-bit reservoir -> p-kit ReLU circuit -> reservoir + ReLU features -> ridge-regression readout -> next-character probabilities`

Both the recurrent reservoir and ReLU block are represented as p-kit circuits and executed with the p-kit stochastic solver. Only the final linear readout is trained using NumPy ridge regression.

The architecture is similar to an Echo State Network / reservoir computer:

* Reservoir (J) couplings are randomly generated and remain fixed.
* The p-bit ReLU block is fixed.
* Only the final readout weights are trained.
* No backpropagation through the recurrent network is required.

The sparse p-bit architecture is intended to be hardware-friendly: the same model can currently be simulated in software and could later target dedicated p-bit hardware.

# Files

* `llm_model.py` — Sparse p-bit reservoir language model proof of concept. Uses a p-kit recurrent reservoir, p-kit ReLU block, and NumPy ridge-regression readout.
* `llm_demo_001.py` — Basic p-kit language-model demo showing training, next-character generation, sparsity, and p-kit execution.
* `llm_demo_002.py` — Comparison between `SparsePBitLM` and a tiny Hugging Face GPT trained from scratch on the same corpus and vocabulary. Compares accuracy, perplexity, training time, trainable parameters, and generated text.

# Future research

* Scale the p-bit reservoir and study quality vs number of p-bits, sparse connections, and trainable parameters.
* Add persistent solver state once supported, instead of feeding the previous state back through h.
* Test a more p-kit-native hybrid architecture:

`token => p-bit reservoir => p-kit ReLU / nonlinear p-bit blocks => probabilistic p-bit output layer => posterior next-token sampling`

