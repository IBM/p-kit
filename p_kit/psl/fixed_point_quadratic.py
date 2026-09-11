import numpy as np

from p_kit.psl.p_circuit import PCircuit


class FixedPointQuadratic:
    """Compile E(x)=0.5*(x-mean)^T precision*(x-mean) to p-bits.

    Each real variable is encoded as
        x_i = sum_k beta[k] * s[i,k],  s[i,k] in {-1,+1}
    with beta[k] = clip*2**k/(2**bit_width-1).

    The compiled PCircuit samples exp(-E) up to an irrelevant constant.
    """

    def __init__(self, precision, mean=None, bit_width=6, clip=4.0):
        K = np.asarray(precision, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("precision must be a square matrix")
        if not np.allclose(K, K.T, atol=1e-10):
            raise ValueError("precision must be symmetric")
        if bit_width < 1:
            raise ValueError("bit_width must be >= 1")
        if clip <= 0:
            raise ValueError("clip must be > 0")

        self.precision = K
        self.mean = np.zeros(K.shape[0]) if mean is None else np.asarray(mean, dtype=float)
        if self.mean.shape != (K.shape[0],):
            raise ValueError("mean has incompatible shape")

        self.bit_width = int(bit_width)
        self.clip = float(clip)
        self.n_variables = K.shape[0]
        self.n_pbits = self.n_variables * self.bit_width
        self.beta = (
            self.clip * (1 << np.arange(self.bit_width))
            / ((1 << self.bit_width) - 1)
        ).astype(float)

    def groups(self, offset=0):
        offset = int(offset)
        return [
            np.arange(offset + i*self.bit_width,
                      offset + (i+1)*self.bit_width, dtype=int)
            for i in range(self.n_variables)
        ]

    def apply(self, circuit: PCircuit, offset=0, annotate=True):
        """Add the quadratic energy to an existing PCircuit."""
        offset = int(offset)
        if offset < 0 or offset + self.n_pbits > circuit.n_pbits:
            raise ValueError("quadratic primitive does not fit in circuit")

        K, beta = self.precision, self.beta
        groups = self.groups(offset)
        circuit.h[offset:offset+self.n_pbits] += (
            (K @ self.mean)[:, None] * beta[None, :]
        ).reshape(-1)

        B = np.outer(beta, beta)
        for i in range(self.n_variables):
            gi = groups[i]
            block = -K[i, i] * B
            block = block.copy()
            np.fill_diagonal(block, 0.0)
            circuit.J[np.ix_(gi, gi)] += block

        ii, jj = np.triu_indices(self.n_variables, 1)
        for i, j in zip(ii, jj):
            if K[i, j] == 0:
                continue
            gi, gj = groups[i], groups[j]
            block = -K[i, j] * B
            circuit.J[np.ix_(gi, gj)] += block
            circuit.J[np.ix_(gj, gi)] += block.T

        if annotate:
            current = list(getattr(circuit, "_pkit_block_groups", []))
            current.extend([g.copy() for g in groups])
            circuit._pkit_block_groups = current
        return circuit

    def to_pcircuit(self):
        c = PCircuit(self.n_pbits)
        return self.apply(c)

    def encode(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.n_variables:
            raise ValueError("last dimension must equal n_variables")
        M = (1 << self.bit_width) - 1
        q = np.rint((np.clip(x, -self.clip, self.clip)/self.clip + 1)*M/2).astype(int)
        bits = (q[..., None] >> np.arange(self.bit_width)) & 1
        return (2*bits - 1).reshape(x.shape[:-1] + (self.n_pbits,))

    def decode(self, spins):
        spins = np.asarray(spins)
        if spins.shape[-1] != self.n_pbits:
            raise ValueError("last dimension must equal n_pbits")
        s = spins.reshape(spins.shape[:-1] + (self.n_variables, self.bit_width))
        return np.tensordot(s, self.beta, axes=([-1], [0]))
