import numpy as np
from p_kit.psl.p_circuit import PCircuit


class PolyOptimizer(PCircuit):
    """Encode a multivariate polynomial (linear + quadratic terms only) into J/h.

    Each integer variable is binary-expanded: x = sum_k(2^k * b_k), b_k in {0,1},
    then mapped to p-bits via b_k = (s_k + 1) / 2, s_k in {-1, +1}.

    Parameters
    ----------
    coeffs : dict
        Maps monomial tuples of variable names to float coefficients.
        Order within a tuple does not matter; {('x','y'): c} == {('y','x'): c}.
        - Constant:         {(): 5}           (ignored)
        - Linear:           {('x',): 2}
        - Quadratic (x^2):  {('x', 'x'): 3}
        - Cross-term (x*y): {('x', 'y'): 1}
        Degree > 2 raises ValueError.
    variables : list of str
        Variable names. Defines ordering and total p-bit count.
    n_bits : int, default 4
        Bit-width per variable. Each variable ranges over [0, 2^n_bits - 1].
    minimize : bool, default True
        If True, negate J and h so the solver minimizes the polynomial.
        Set to False to maximize instead.

    Attributes
    ----------
    h : np.array((n_pbits,))
        Bias vector.
    J : np.array((n_pbits, n_pbits))
        Coupling matrix.

    Notes
    -----
    .. versionadded:: 0.0.1
    """

    def __init__(self, coeffs, variables, n_bits=4, minimize=True):
        self.variables = list(variables)
        self.n_bits = n_bits
        self.minimize = minimize
        self.coeffs = self._normalize(coeffs)
        PCircuit.__init__(self, len(variables) * n_bits)
        self._encode()

    @staticmethod
    def _normalize(coeffs):
        result = {}
        for mono, val in coeffs.items():
            if len(mono) > 2:
                raise ValueError(
                    f"Monomial {mono} has degree {len(mono)}; only linear and quadratic terms are supported."
                )
            key = tuple(sorted(mono))
            result[key] = result.get(key, 0) + val
        return result

    def _idx(self, var, bit):
        return self.variables.index(var) * self.n_bits + bit

    def _encode(self):
        n = self.n_bits
        # Offset introduced by the (s+1)/2 binary expansion: x = (1/2)*sum_k 2^k*s_k + C
        C = (2 ** n - 1) / 2

        for mono, coeff in self.coeffs.items():
            if len(mono) == 0:
                continue

            elif len(mono) == 1:
                # Linear: coeff * x_v = (coeff/2) * sum_k 2^k * s_{v,k} + const
                v = mono[0]
                for k in range(n):
                    self.h[self._idx(v, k)] += coeff / 2 * (2 ** k)

            elif mono[0] == mono[1]:
                # Quadratic x_v^2:
                #   h contribution:  coeff * C * 2^k  per bit k
                #   J contribution (j<k):  coeff/2 * 2^j * 2^k
                # The (j,k) and (k,j) terms of the double sum both survive, so
                # the s_j*s_k coefficient is 2 * coeff/4 * 2^j * 2^k. Solvers
                # read J through the 1/2 * s.J.s convention (see CaSuDaSolver),
                # under which a symmetric pair contributes exactly J[j,k] --
                # not 2*J[j,k] -- so w is that coefficient, undivided.
                v = mono[0]
                for k in range(n):
                    self.h[self._idx(v, k)] += coeff * C * (2 ** k)
                for j in range(n):
                    for k in range(j + 1, n):
                        w = coeff / 2 * (2 ** j) * (2 ** k)
                        ij, ik = self._idx(v, j), self._idx(v, k)
                        self.J[ij, ik] += w
                        self.J[ik, ij] += w

            else:
                # Cross-term x_u * x_v (u != v):
                #   h contribution:  coeff * C/2 * 2^k  per bit k, for both variables
                #   J contribution:  coeff/4 * 2^j * 2^k
                # b_j*b_k expands to (1 + s_j + s_k + s_j*s_k)/4, so the pair
                # coefficient is coeff/4 * 2^j * 2^k. See the x_v^2 branch for
                # why that goes into J undivided.
                u, v = mono
                for k in range(n):
                    wk = 2 ** k
                    self.h[self._idx(u, k)] += coeff * C / 2 * wk
                    self.h[self._idx(v, k)] += coeff * C / 2 * wk
                for j in range(n):
                    for k in range(n):
                        w = coeff / 4 * (2 ** j) * (2 ** k)
                        ij, ik = self._idx(u, j), self._idx(v, k)
                        self.J[ij, ik] += w
                        self.J[ik, ij] += w

        if self.minimize:
            self.J = -self.J
            self.h = -self.h

        np.fill_diagonal(self.J, 0)

    def decode(self, samples):
        """Convert p-bit samples {-1, +1} to integer variable values.

        Parameters
        ----------
        samples : array-like of shape (n_pbits,)

        Returns
        -------
        dict mapping variable name to integer value in [0, 2^n_bits - 1]
        """
        bits = ((np.asarray(samples) + 1) / 2).astype(int)
        return {
            var: int(sum(int(bits[self._idx(var, k)]) * (2 ** k) for k in range(self.n_bits)))
            for var in self.variables
        }
