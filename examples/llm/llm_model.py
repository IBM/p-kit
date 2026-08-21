"""
Sparse p-bit reservoir language model - a proof of concept.

Reservoir computing: a recurrent network with mostly fixed internal connections
that converts the input sequence into a rich dynamic state.

This is a small demo model. The recurrent state is made of stochastic p-bits
with sparse pairwise couplings:

    P(s_i = +1) = (1 + tanh(beta * I_i)) / 2
    I_i = sum_j J_ij s_j + h_i(token)

The p-bit reservoir is sparse and thus hardware-friendly. A conventional linear
readout is trained with ridge regression to predict the next character.

Workflow:

    input character
        -> token-dependent h
        -> sparse p-bit recurrent circuit
        -> stochastic p-kit solver
        -> reservoir state
        -> linear readout
        -> next-character probabilities

The J/h representation is compatible with the basic p-kit circuit model.
p-kit provides the p-bit circuit representation and stochastic solver, while
the readout training uses NumPy ridge-regression.

This separation allows the same p-bit model to be simulated in software today
and later executed on dedicated p-bit hardware without redesigning the model.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainStats:
    n_samples: int
    vocabulary_size: int
    accuracy: float


class SparsePBitLM:
    def __init__(
        self,
        n_pbits: int = 256,
        degree: int = 8,
        input_fanout: int = 16,
        beta: float = 1.4,
        recurrent_scale: float = 0.35,
        input_scale: float = 1.25,
        seed: int = 1,
    ):
        if degree >= n_pbits:
            raise ValueError("degree must be smaller than n_pbits")

        self.n_pbits = int(n_pbits)
        self.degree = int(degree)
        self.input_fanout = int(input_fanout)
        self.beta = float(beta)
        self.recurrent_scale = float(recurrent_scale)
        self.input_scale = float(input_scale)
        self.rng = np.random.default_rng(seed)

        self.src, self.dst, self.edge_w = self._make_sparse_recurrent_graph()
        self.state = self.rng.choice((-1.0, 1.0), size=self.n_pbits)

        self.vocab = None
        self.char_to_id = None
        self.input_idx = None
        self.input_w = None
        self.readout = None

    def _make_sparse_recurrent_graph(self):
        """Creates a directed sparse random p-bit coupling graph."""
        src = []
        dst = []
        weights = []

        for i in range(self.n_pbits):
            candidates = np.delete(np.arange(self.n_pbits), i)
            neighbours = self.rng.choice(
                candidates, size=self.degree, replace=False
            )
            src.extend(neighbours.tolist())
            dst.extend([i] * self.degree)
            weights.extend(
                self.rng.normal(
                    0.0,
                    self.recurrent_scale / np.sqrt(self.degree),
                    size=self.degree,
                ).tolist()
            )

        return (
            np.asarray(src, dtype=np.int32),
            np.asarray(dst, dtype=np.int32),
            np.asarray(weights, dtype=np.float32),
        )

    def _build_vocab(self, text: str):
        self.vocab = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}

        fanout = min(self.input_fanout, self.n_pbits)
        self.input_idx = []
        self.input_w = []

        for _ in self.vocab:
            idx = self.rng.choice(self.n_pbits, size=fanout, replace=False)
            w = self.rng.choice((-1.0, 1.0), size=fanout) * self.input_scale
            self.input_idx.append(idx.astype(np.int32))
            self.input_w.append(w.astype(np.float32))

    def reset(self):
        self.state = self.rng.choice((-1.0, 1.0), size=self.n_pbits)

    def _field(self, token_id: int):
        field = np.zeros(self.n_pbits, dtype=np.float32)

        # Sparse recurrent contribution.
        np.add.at(field, self.dst, self.edge_w * self.state[self.src])

        # Sparse token-dependent external field.
        idx = self.input_idx[token_id]
        field[idx] += self.input_w[token_id]

        return field

    def step(self, token_id: int, sweeps: int = 1):
        """
        Advances the stochastic p-bit reservoir.

        Each sweep updates every p-bit once, in random asynchronous order.
        """
        for _ in range(sweeps):
            order = self.rng.permutation(self.n_pbits)

            # Recompute local field during asynchronous updates.
            # This is deliberately simple for the reference implementation.
            for i in order:
                mask = self.dst == i
                recurrent = np.dot(
                    self.edge_w[mask],
                    self.state[self.src[mask]],
                )

                token_mask = self.input_idx[token_id]
                pos = np.flatnonzero(token_mask == i)
                external = (
                    float(self.input_w[token_id][pos[0]])
                    if len(pos)
                    else 0.0
                )

                I = recurrent + external
                p_plus = 0.5 * (1.0 + np.tanh(self.beta * I))
                self.state[i] = 1.0 if self.rng.random() < p_plus else -1.0

        return self.state

    @staticmethod
    def _softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def fit(
        self,
        text: str,
        sweeps: int = 1,
        washout: int = 20,
        ridge: float = 1e-2,
    ) -> TrainStats:
        if len(text) < 3:
            raise ValueError("training text is too short")

        self._build_vocab(text)
        ids = np.asarray([self.char_to_id[c] for c in text], dtype=np.int32)
        self.reset()

        features = []
        targets = []

        for t in range(len(ids) - 1):
            self.step(ids[t], sweeps=sweeps)
            if t >= washout:
                # Reservoir state + current-token feature + constant bias.
                token_feature = np.zeros(len(self.vocab), dtype=np.float64)
                token_feature[ids[t]] = 1.0
                features.append(
                    np.concatenate([self.state, token_feature, [1.0]])
                )
                targets.append(ids[t + 1])

        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int32)

        Y = np.zeros((len(y), len(self.vocab)), dtype=np.float64)
        Y[np.arange(len(y)), y] = 1.0

        # Ridge regression: W = (X'X + lambda I)^-1 X'Y
        reg = ridge * np.eye(X.shape[1])
        reg[-1, -1] = 0.0  # do not regularize bias
        self.readout = np.linalg.solve(X.T @ X + reg, X.T @ Y)

        pred = np.argmax(X @ self.readout, axis=1)
        accuracy = float(np.mean(pred == y))

        return TrainStats(
            n_samples=len(y),
            vocabulary_size=len(self.vocab),
            accuracy=accuracy,
        )

    def next_char_probabilities(
        self,
        current_char: str,
        sweeps: int = 1,
        temperature: float = 1.0,
    ):
        if self.readout is None:
            raise RuntimeError("fit() must be called first")
        if current_char not in self.char_to_id:
            raise ValueError(f"character {current_char!r} is not in vocabulary")

        self.step(self.char_to_id[current_char], sweeps=sweeps)
        token_feature = np.zeros(len(self.vocab), dtype=np.float64)
        token_feature[self.char_to_id[current_char]] = 1.0
        feature = np.concatenate([self.state, token_feature, [1.0]])
        logits = feature @ self.readout
        probs = self._softmax(logits / max(float(temperature), 1e-6))
        return probs

    def generate(
        self,
        seed_text: str,
        n_chars: int = 200,
        sweeps: int = 1,
        temperature: float = 0.8,
    ) -> str:
        if not seed_text:
            raise ValueError("seed_text must not be empty")

        self.reset()

        # Prime reservoir with seed.
        for c in seed_text[:-1]:
            if c in self.char_to_id:
                self.step(self.char_to_id[c], sweeps=sweeps)

        out = list(seed_text)
        current = seed_text[-1]

        for _ in range(n_chars):
            if current not in self.char_to_id:
                current = self.vocab[0]

            probs = self.next_char_probabilities(
                current,
                sweeps=sweeps,
                temperature=temperature,
            )
            token_id = int(self.rng.choice(len(self.vocab), p=probs))
            current = self.vocab[token_id]
            out.append(current)

        return "".join(out)

    def dense_J_h(self, current_char: str):
        """
        Builds the p-kit-compatible dense Ising representation of the reservoir.
    
        J contains the sparse recurrent p-bit couplings converted to a dense matrix.
        h contains the token-dependent input bias for current_char.
    
        This representation can be used to create a p-kit PCircuit. A future
        hardware backend should preferably use the sparse edge representation
        directly to avoid storing the full N x N J matrix.
        """
        if current_char not in self.char_to_id:
            raise ValueError(f"character {current_char!r} is not in vocabulary")
    
        J = np.zeros((self.n_pbits, self.n_pbits), dtype=np.float64)
    
        # Convert the sparse recurrent edge list into p-kit's dense J matrix.
        J[self.dst, self.src] = self.edge_w
    
        # Encode the current character as external p-bit biases h.
        h = np.zeros(self.n_pbits, dtype=np.float64)
        token_id = self.char_to_id[current_char]
        h[self.input_idx[token_id]] = self.input_w[token_id]
    
        return J, h
    
    
    def to_pkit_circuit(self, current_char: str):
        """
        Creates a p-kit PCircuit representing the reservoir for current_char.
    
        The recurrent couplings become circuit.J and the character-dependent
        input biases become circuit.h.
        """
        from p_kit.psl.p_circuit import PCircuit
    
        J, h = self.dense_J_h(current_char)
    
        circuit = PCircuit(self.n_pbits)
        circuit.J = J
        circuit.h = h
    
        return circuit
