"""Exact block-Gibbs sampling for grouped p-bits.

Updates each p-bit block jointly by enumerating all possible block states
and sampling from the exact conditional distribution.
"""

import numpy as np
from scipy.sparse import csr_matrix
from p_kit.backends import NumpyBackend
from p_kit.psl.p_circuit import PCircuit
from p_kit.solver.base_solver import Solver

class BlockGibbsSolver(Solver):
    def __init__(self, Nt=4, dt=1., i0=1., expected_mean=0, seed=None,
                 backend=None, tau=1., shuffle_blocks=True):
        if backend is None:
            backend = NumpyBackend(dtype=np.float32)
        super().__init__(Nt=Nt, dt=dt, i0=i0, expected_mean=expected_mean,
                         seed=seed, backend=backend, tau=tau)
        self.shuffle_blocks = shuffle_blocks
        self._state_cache = {}

    def _states(self, width):
        if width < 1 or width >= 63:
            raise ValueError("Invalid block width")
        if width not in self._state_cache:
            q = np.arange(1 << width, dtype=np.uint64)[:, None]
            b = (q >> np.arange(width, dtype=np.uint64)) & 1
            self._state_cache[width] = 2.*b.astype(float)-1.
        return self._state_cache[width]

    @staticmethod
    def _groups(groups, n):
        out, used = [], set()
        for g in groups:
            g = np.asarray(g, dtype=int).ravel()
            if not len(g):
                continue
            if np.any(g < 0) or np.any(g >= n):
                raise ValueError("Block index outside circuit")
            if len(np.unique(g)) != len(g):
                raise ValueError("Duplicate index in block")
            if any(int(i) in used for i in g):
                raise ValueError("Blocks overlap")
            used.update(map(int, g))
            out.append(g)
        return out, used

    def solve(self, c: PCircuit, annealing_func=None, n_shots=1, groups=None,
              initial_state=None, clamped=None, return_final=True):
        J = np.asarray(c.J, float)
        h = np.asarray(c.h, float).reshape(-1)
        n = c.n_pbits
        if J.shape != (n, n) or h.shape != (n,):
            raise ValueError("Invalid PCircuit shape")
        if not np.allclose(J, J.T, atol=1e-10):
            raise ValueError("J must be symmetric")

        if groups is None:
            groups = getattr(c, "_pkit_block_groups", None)
        if groups is None:
            raise ValueError("No block groups supplied")

        blocks, used = self._groups(groups, n)
        clamped = {} if clamped is None else {int(i): float(v) for i, v in clamped.items()}
        for i, v in clamped.items():
            if i < 0 or i >= n or v not in (-1., 1.):
                raise ValueError("Invalid clamp")
        for g in blocks:
            if any(int(i) in clamped for i in g):
                raise ValueError("Clamped p-bit inside sampled block")
        blocks += [np.array([i]) for i in range(n) if i not in used and i not in clamped]

        if initial_state is None:
            M = np.where(self._generator.random((n_shots, n)) < .5, -1., 1.)
        else:
            M = np.asarray(initial_state, float)
            if M.ndim == 1:
                M = np.broadcast_to(M, (n_shots, n)).copy()
            elif M.shape == (1, n):
                M = np.broadcast_to(M, (n_shots, n)).copy()
            elif M.shape != (n_shots, n):
                raise ValueError("Invalid initial_state shape")
            else:
                M = M.copy()
            M = np.where(M >= 0, 1., -1.)

        for i, v in clamped.items():
            M[:, i] = v

        field = M @ J + h
        Js = csr_matrix(J)
        prepared = []

        for g in blocks:
            states = self._states(len(g))
            Jgg = J[np.ix_(g, g)]
            internal = .5*np.einsum("si,ij,sj->s", states, Jgg, states)
            cols = np.unique(Js[g, :].tocoo().col)
            W = J[np.ix_(g, cols)] if len(cols) else np.empty((len(g), 0))
            prepared.append((g, Jgg, states, internal, cols, W))

        trajectory = None
        if not return_final:
            trajectory = np.empty((self.Nt+1, n_shots, n))
            trajectory[0] = M

        for sweep in range(self.Nt):
            scale = self.i0 if annealing_func is None else float(annealing_func(self, sweep))
            order = self._generator.permutation(len(prepared)) if self.shuffle_blocks else range(len(prepared))
            for bi in order:
                g, Jgg, states, internal, cols, W = prepared[int(bi)]
                old = M[:, g].copy()
                outside = field[:, g]-old@Jgg
                logits = scale*(outside@states.T+internal)
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                u = self._generator.random(n_shots)
                choice = (np.cumsum(p, axis=1) < u[:, None]).sum(axis=1)
                new = states[np.minimum(choice, len(states)-1)]
                delta = new-old
                M[:, g] = new
                if len(cols):
                    field[:, cols] += delta@W
            for i, v in clamped.items():
                M[:, i] = v
            if trajectory is not None:
                trajectory[sweep+1] = M

        return M if return_final else trajectory

    def copy(self):
        return BlockGibbsSolver(
            Nt=self.Nt, dt=self.dt, i0=self.i0, expected_mean=self.expected_mean,
            seed=self.seed, backend=self.backend, tau=self.tau,
            shuffle_blocks=self.shuffle_blocks
        )