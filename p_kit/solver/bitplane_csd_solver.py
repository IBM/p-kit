"""Bit-plane CaSuDa sampling for fixed-point p-bit variables.

Updates one bit plane at a time across grouped fixed-point variables,
avoiding simultaneous updates of strongly coupled bits within each variable.
"""

import numpy as np

from p_kit.solver.annealing import constant
from p_kit.solver.csd_solver import CaSuDaSolver, _validate_initial_state

try:
    import numba
except ImportError:
    numba = None


if numba is not None:

    @numba.njit(cache=True)
    def _final_dense_bitplane_numba(
        m,
        J,
        h,
        anneal,
        rnd,
        plane_orders,
        bit_width,
        dt,
        threshold,
        tmp,
    ):
        ns, n = m.shape
        class_bit = n - 1
        n_coeff = class_bit // bit_width

        for run in range(len(anneal)):

            for op in range(bit_width):
                plane = plane_orders[run, op]

                # Compute one bit from every coefficient.
                # These p-bits are updated in parallel.
                for s in range(ns):
                    for coeff in range(n_coeff):
                        i = coeff * bit_width + plane

                        field = h[i]

                        for j in range(n):
                            field += m[s, j] * J[j, i]

                        I = anneal[run] * field

                        p = np.exp(
                            -dt
                            * np.exp(
                                -m[s, i]
                                * (I + threshold)
                            )
                        )

                        tmp[s, i] = (
                            m[s, i]
                            * (
                                1.0
                                if p - rnd[run, s, i] >= 0
                                else -1.0
                            )
                        )

                # Commit this plane before computing the next.
                for s in range(ns):
                    for coeff in range(n_coeff):
                        i = coeff * bit_width + plane
                        m[s, i] = tmp[s, i]

            # Update class p-bit last.
            for s in range(ns):
                i = class_bit
                field = h[i]

                for j in range(n):
                    field += m[s, j] * J[j, i]

                I = anneal[run] * field

                p = np.exp(
                    -dt
                    * np.exp(
                        -m[s, i]
                        * (I + threshold)
                    )
                )

                m[s, i] = (
                    m[s, i]
                    * (
                        1.0
                        if p - rnd[run, s, i] >= 0
                        else -1.0
                    )
                )

        return m


    @numba.njit(cache=True)
    def _final_sparse_bitplane_numba(
        m,
        indptr,
        indices,
        data,
        h,
        anneal,
        rnd,
        plane_orders,
        bit_width,
        dt,
        threshold,
        tmp,
    ):
        ns, n = m.shape
        class_bit = n - 1
        n_coeff = class_bit // bit_width

        for run in range(len(anneal)):

            for op in range(bit_width):
                plane = plane_orders[run, op]

                for s in range(ns):
                    for coeff in range(n_coeff):
                        i = coeff * bit_width + plane

                        field = h[i]

                        for k in range(
                            indptr[i],
                            indptr[i + 1],
                        ):
                            field += (
                                data[k]
                                * m[s, indices[k]]
                            )

                        I = anneal[run] * field

                        p = np.exp(
                            -dt
                            * np.exp(
                                -m[s, i]
                                * (I + threshold)
                            )
                        )

                        tmp[s, i] = (
                            m[s, i]
                            * (
                                1.0
                                if p - rnd[run, s, i] >= 0
                                else -1.0
                            )
                        )

                # Commit this entire bit plane.
                for s in range(ns):
                    for coeff in range(n_coeff):
                        i = coeff * bit_width + plane
                        m[s, i] = tmp[s, i]

            # Class p-bit.
            for s in range(ns):
                i = class_bit
                field = h[i]

                for k in range(
                    indptr[i],
                    indptr[i + 1],
                ):
                    field += (
                        data[k]
                        * m[s, indices[k]]
                    )

                I = anneal[run] * field

                p = np.exp(
                    -dt
                    * np.exp(
                        -m[s, i]
                        * (I + threshold)
                    )
                )

                m[s, i] = (
                    m[s, i]
                    * (
                        1.0
                        if p - rnd[run, s, i] >= 0
                        else -1.0
                    )
                )

        return m


class BitPlaneCaSuDaSolver(CaSuDaSolver):
    """
    Bit-plane asynchronous CaSuDa solver.

    Intended for circuits whose state layout is:

        coefficient 0: bit0 bit1 ... bit(B-1)
        coefficient 1: bit0 bit1 ... bit(B-1)
        ...
        coefficient N: bit0 bit1 ... bit(B-1)
        class p-bit

    For bit_width=6 the update order is conceptually:

        bit0 of ALL coefficients  -> commit
        bit1 of ALL coefficients  -> commit
        ...
        bit5 of ALL coefficients  -> commit
        class bit                 -> commit

    The order of the bit planes can optionally be shuffled at every
    CaSuDa timestep.

    P-bits belonging to different coefficients in the same bit plane
    are updated in parallel. The mutually coupled bits representing
    the same coefficient are therefore never updated simultaneously.

    This implementation currently supports return_final=True only.
    """

    def __init__(
        self,
        Nt,
        dt,
        i0,
        bit_width,
        expected_mean=0,
        seed=None,
        backend=None,
        tau=0.1,
        cache_J=False,
        use_sparse=False,
        reuse_buffers=False,
        cache_static=False,
        shuffle_planes=True,
    ):
        super().__init__(
            Nt=Nt,
            dt=dt,
            i0=i0,
            expected_mean=expected_mean,
            seed=seed,
            backend=backend,
            tau=tau,
            cache_J=cache_J,
            use_sparse=use_sparse,
            reuse_buffers=reuse_buffers,
            cache_static=cache_static,
        )

        self.bit_width = int(bit_width)
        self.shuffle_planes = bool(shuffle_planes)

        if self.bit_width < 1:
            raise ValueError(
                "bit_width must be >= 1"
            )

    def _plane_orders(self):
        if not self.shuffle_planes:
            return np.tile(
                np.arange(
                    self.bit_width,
                    dtype=np.int32,
                ),
                (self.Nt, 1),
            )

        keys = np.asarray(
            self.random(
                (
                    self.Nt,
                    self.bit_width,
                )
            )
        )

        return np.argsort(
            keys,
            axis=1,
        ).astype(np.int32)

    def _validate_layout(self, c):
        n = c.n_pbits

        if n < 2:
            raise ValueError(
                "BitPlaneCaSuDaSolver requires "
                "coefficient p-bits plus a final class p-bit."
            )

        coefficient_bits = n - 1

        if coefficient_bits % self.bit_width != 0:
            raise ValueError(
                f"(n_pbits - 1) must be divisible by bit_width. "
                f"Got n_pbits={n}, bit_width={self.bit_width}."
            )

    def _solve_final_bitplane(
        self,
        c,
        annealing_func,
        n_shots,
        initial_state,
    ):
        self._validate_layout(c)

        n = c.n_pbits

        h = np.asarray(
            c.h
        ).reshape(-1)

        J = self._get_static_J(c)

        anneal = self._get_static_anneal(
            annealing_func
        )

        threshold = float(
            np.arctanh(
                self.expected_mean
            )
        )

        if initial_state is None:
            m = np.sign(
                0.5
                - self.random(
                    (
                        n_shots,
                        n,
                    )
                )
            )

        else:
            state = _validate_initial_state(
                initial_state,
                n,
            )

            m = np.tile(
                state,
                (
                    n_shots,
                    1,
                )
            )

        m = np.asarray(
            m,
            dtype=h.dtype,
        )

        plane_orders = self._plane_orders()

        rnd = np.asarray(
            self.random(
                (
                    self.Nt,
                    n_shots,
                    n,
                )
            ),
            dtype=h.dtype,
        )

        if self.reuse_buffers:
            tmp = self.backend.buffer(
                "bitplane_tmp",
                (
                    n_shots,
                    n,
                ),
            )
        else:
            tmp = np.empty(
                (
                    n_shots,
                    n,
                ),
                dtype=h.dtype,
            )

        if self.use_numba:
            if numba is None:
                raise RuntimeError(
                    "Numba is required for the compiled "
                    "BitPlaneCaSuDaSolver path."
                )

            if self.use_sparse:
                m = _final_sparse_bitplane_numba(
                    m,
                    J.indptr,
                    J.indices,
                    J.data,
                    h,
                    anneal,
                    rnd,
                    plane_orders,
                    self.bit_width,
                    self.dt,
                    threshold,
                    tmp,
                )

            else:
                m = _final_dense_bitplane_numba(
                    m,
                    J,
                    h,
                    anneal,
                    rnd,
                    plane_orders,
                    self.bit_width,
                    self.dt,
                    threshold,
                    tmp,
                )

            return (
                m[0]
                if n_shots == 1
                else m
            )

        # ----------------------------------------------------------
        # NumPy fallback
        # ----------------------------------------------------------

        class_bit = n - 1
        n_coeff = (
            class_bit
            // self.bit_width
        )

        for run in range(self.Nt):

            scale = anneal[run]

            for plane in plane_orders[run]:

                indices = (
                    np.arange(n_coeff)
                    * self.bit_width
                    + plane
                )

                if self.use_sparse:
                    field = (
                        J[indices]
                        .dot(m.T)
                        .T
                    )
                else:
                    field = (
                        m
                        @ J[:, indices]
                    )

                I = scale * (
                    field
                    + h[indices]
                )

                old = m[:, indices]

                p = np.exp(
                    -self.dt
                    * np.exp(
                        -old
                        * (
                            I
                            + threshold
                        )
                    )
                )

                rr = self.random(
                    old.shape
                )

                m[:, indices] = (
                    old
                    * np.where(
                        p - rr >= 0,
                        1.0,
                        -1.0,
                    )
                )

            # ----------------------------------------------
            # Update class bit after all coefficient planes.
            # ----------------------------------------------

            if self.use_sparse:
                field = np.asarray(
                    J[class_bit]
                    .dot(m.T)
                ).reshape(-1)
            else:
                field = (
                    m
                    @ J[:, class_bit]
                )

            I = scale * (
                field
                + h[class_bit]
            )

            old = m[:, class_bit]

            p = np.exp(
                -self.dt
                * np.exp(
                    -old
                    * (
                        I
                        + threshold
                    )
                )
            )

            rr = self.random(
                old.shape
            )

            m[:, class_bit] = (
                old
                * np.where(
                    p - rr >= 0,
                    1.0,
                    -1.0,
                )
            )

        return (
            m[0]
            if n_shots == 1
            else m
        )

    def solve(
        self,
        c,
        annealing_func=constant,
        n_shots=1,
        bias_func=None,
        return_filtered=False,
        initial_state=None,
        return_final=False,
    ):
        if not return_final:
            raise ValueError(
                "BitPlaneCaSuDaSolver currently requires "
                "return_final=True."
            )

        if return_filtered:
            raise ValueError(
                "return_filtered cannot be used with "
                "return_final=True."
            )

        if bias_func is not None:
            raise NotImplementedError(
                "bias_func is not currently supported by "
                "BitPlaneCaSuDaSolver."
            )

        return self._solve_final_bitplane(
            c,
            annealing_func,
            n_shots,
            initial_state,
        )

    def copy(self):
        return BitPlaneCaSuDaSolver(
            Nt=self.Nt,
            dt=self.dt,
            i0=self.i0,
            bit_width=self.bit_width,
            expected_mean=self.expected_mean,
            seed=self.seed,
            backend=self.backend,
            tau=self.tau,
            cache_J=self.cache_J,
            use_sparse=self.use_sparse,
            reuse_buffers=self.reuse_buffers,
            cache_static=self.cache_static,
            shuffle_planes=self.shuffle_planes,
        )
