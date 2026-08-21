"""
Sparse p-bit reservoir language model - a proof of concept.

Reservoir computing: a recurrent network with mostly fixed internal connections
that converts the input sequence into a rich dynamic state.

This is a small demo model. The recurrent state is made of stochastic p-bits
with sparse pairwise couplings:

    P(s_i = +1) = (1 + tanh(beta * I_i)) / 2
    I_i = sum_j J_ij s_j + h_i(token) + memory_scale * s_i(previous)

The p-bit reservoir is sparse and thus hardware-friendly. A conventional linear
readout is trained with ridge regression to predict the next character.

Workflow:

    character
       -> sparse stochastic p-bit reservoir
       -> p-kit ReLU circuit
       -> reservoir + ReLU features
       -> ridge-regression readout
       -> next-character probabilities

Both the sparse recurrent reservoir and the ReLU block are represented as
p-kit circuits and executed with the p-kit stochastic solver. The final
ridge-regression readout uses NumPy.

This separation allows the same p-bit model to be currently simulated in 
software and in the future executed on dedicated p-bit hardware without 
redesigning the model.

Currently a workaround is used for the recurrent memory: feed the previous 
reservoir state back through h because CaSuDaSolver cannot yet start from a 
user-provided initial state. Once https://github.com/IBM/p-kit/issues/117 is
implemented, this can be replaced by passing self.state directly to the solver.
"""

from dataclasses import dataclass
import numpy as np

from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver


@dataclass
class TrainStats:
    n_samples: int
    vocabulary_size: int
    accuracy: float


class SparsePBitLM:

    def __init__(
        self,
        n_pbits=128,
        degree=6,
        input_fanout=12,
        recurrent_scale=0.40,
        input_scale=1.4,
        memory_scale=0.35,
        reservoir_steps=10,
        relu_steps=10,
        seed=1,
    ):
        self.n_pbits = int(n_pbits)
        self.degree = int(degree)
        self.input_fanout = int(input_fanout)
        self.input_scale = float(input_scale)
        self.memory_scale = float(memory_scale)

        self.rng = np.random.default_rng(seed)

        if degree >= n_pbits:
            raise ValueError("degree must be smaller than n_pbits")

        # ==========================================================
        # p-kit recurrent reservoir
        # ==========================================================

        self.circuit = PCircuit(n_pbits)

        # Sparse symmetric p-bit connectivity.
        target_edges = (n_pbits * degree) // 2
        edges = set()

        while len(edges) < target_edges:

            i = int(self.rng.integers(n_pbits))
            j = int(self.rng.integers(n_pbits))

            if i == j:
                continue

            edges.add(tuple(sorted((i, j))))

        for i, j in edges:

            weight = self.rng.normal(
                0.0,
                recurrent_scale / np.sqrt(degree),
            )

            self.circuit.set_weight(
                i,
                j,
                weight,
                sym=True,
            )

        self.reservoir_solver = CaSuDaSolver(
            Nt=reservoir_steps,
            dt=0.1667,
            i0=0.8,
            seed=seed,
        )

        # ==========================================================
        # p-kit ReLU circuit
        # ==========================================================

        self.relu = PCircuit(10)

        # ReLU-style circuit based on the p-kit ReLU example.
        self.relu.J = np.array([
            [0,  1,  2,  0,  0,  0,  0,  0,  0,  0],
            [1,  0, -2,  1, -2,  1, -2,  1, -2,  0],
            [2, -2,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  2,  0,  0,  0,  0,  0],
            [0, -2,  0,  2,  0,  0,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  0,  2,  0,  0,  0],
            [0, -2,  0,  0,  0,  2,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  0,  0,  0,  2,  0],
            [0, -2,  0,  0,  0,  0,  0,  2,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ], dtype=float)

        self.relu_base_h = np.array(
            [1, -4, -2, 1, -2, 1, -2, 1, -2, -2],
            dtype=float,
        )

        self.relu_solver = CaSuDaSolver(
            Nt=relu_steps,
            dt=0.1667,
            i0=0.8,
            seed=seed + 1,
        )

        # Five reservoir p-bits feed the ReLU circuit.
        self.relu_source_idx = self.rng.choice(
            n_pbits,
            size=5,
            replace=False,
        )

        self.state = self.rng.choice(
            (-1.0, 1.0),
            size=n_pbits,
        )

        self.vocab = None
        self.char_to_id = None
        self.token_h = None
        self.readout = None

    def _build_vocab(self, text):

        self.vocab = sorted(set(text))

        self.char_to_id = {
            c: i
            for i, c in enumerate(self.vocab)
        }

        fanout = min(
            self.input_fanout,
            self.n_pbits,
        )

        # Sparse character-dependent p-kit biases.
        self.token_h = np.zeros(
            (len(self.vocab), self.n_pbits),
            dtype=float,
        )

        for token in range(len(self.vocab)):

            idx = self.rng.choice(
                self.n_pbits,
                size=fanout,
                replace=False,
            )

            self.token_h[token, idx] = (
                self.rng.choice(
                    (-1.0, 1.0),
                    size=fanout,
                )
                * self.input_scale
            )

    def reset(self):

        self.state = self.rng.choice(
            (-1.0, 1.0),
            size=self.n_pbits,
        )

    def step(
        self,
        token_id,
        sweeps=1,
    ):
        """
        Advance the recurrent reservoir using the p-kit solver.
        """

        for _ in range(sweeps):

            # Token input + recurrent state feedback.
            #
            # The feedback through h is temporary until p-kit can
            # initialize CaSuDaSolver directly from self.state.
            self.circuit.h = (
                self.token_h[token_id]
                + self.memory_scale * self.state
            )

            _, trajectory, _ = (
                self.reservoir_solver.solve(
                    self.circuit
                )
            )

            self.state = (
                trajectory[-1].astype(float)
            )

        return self.state

    def relu_features(self):
        """
        Run the p-kit ReLU circuit on part of the reservoir state.
        """

        h = self.relu_base_h.copy()

        h[:5] += self.state[
            self.relu_source_idx
        ]

        self.relu.h = h

        _, trajectory, _ = (
            self.relu_solver.solve(
                self.relu
            )
        )

        return trajectory[-1].astype(float)

    def _features(
        self,
        token_id,
        sweeps=1,
    ):

        reservoir = self.step(
            token_id,
            sweeps=sweeps,
        )

        relu = self.relu_features()

        # Direct character feature helps the small linear readout.
        token_feature = np.zeros(
            len(self.vocab),
            dtype=float,
        )

        token_feature[token_id] = 1.0

        return np.concatenate([
            reservoir,
            relu,
            token_feature,
            [1.0],
        ])

    def fit(
        self,
        text,
        sweeps=1,
        washout=30,
        ridge=0.05,
    ):

        self._build_vocab(text)

        ids = np.asarray([
            self.char_to_id[c]
            for c in text
        ])

        self.reset()

        X = []
        y = []

        for t in range(len(ids) - 1):

            feature = self._features(
                ids[t],
                sweeps=sweeps,
            )

            if t >= washout:

                X.append(feature)
                y.append(ids[t + 1])

        X = np.asarray(X)
        y = np.asarray(y)

        Y = np.zeros(
            (len(y), len(self.vocab))
        )

        Y[
            np.arange(len(y)),
            y,
        ] = 1.0

        # Only this readout-training stage uses conventional NumPy.
        reg = ridge * np.eye(
            X.shape[1]
        )

        reg[-1, -1] = 0.0

        self.readout = np.linalg.solve(
            X.T @ X + reg,
            X.T @ Y,
        )

        prediction = np.argmax(
            X @ self.readout,
            axis=1,
        )

        return TrainStats(
            n_samples=len(y),
            vocabulary_size=len(self.vocab),
            accuracy=float(
                np.mean(prediction == y)
            ),
        )

    def next_char_probabilities(
        self,
        current_char,
        sweeps=1,
        temperature=1.0,
    ):

        token_id = self.char_to_id[
            current_char
        ]

        feature = self._features(
            token_id,
            sweeps=sweeps,
        )

        logits = (
            feature @ self.readout
        )

        logits /= max(
            float(temperature),
            1e-6,
        )

        logits -= np.max(logits)

        p = np.exp(logits)

        return p / np.sum(p)

    def generate(
        self,
        seed_text,
        n_chars=120,
        sweeps=1,
        temperature=0.15,
    ):

        self.reset()

        # Prime recurrent reservoir with prompt.
        for c in seed_text[:-1]:

            if c in self.char_to_id:

                self.step(
                    self.char_to_id[c],
                    sweeps=sweeps,
                )

        output = list(seed_text)

        current = seed_text[-1]

        for _ in range(n_chars):

            p = self.next_char_probabilities(
                current,
                sweeps=sweeps,
                temperature=temperature,
            )

            token_id = int(
                self.rng.choice(
                    len(self.vocab),
                    p=p,
                )
            )

            current = self.vocab[
                token_id
            ]

            output.append(current)

        return "".join(output)

    @property
    def n_connections(self):

        return int(
            np.count_nonzero(
                self.circuit.J
            )
        )

    @property
    def sparsity(self):

        return 1.0 - (
            self.n_connections
            / self.circuit.J.size
        )