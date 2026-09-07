import numpy as np
import pytest
from functools import partial

from conftest import requires_module
from p_kit.library.poly import PolyOptimizer

requires_docplex = partial(requires_module, name="docplex")


# --- PolyOptimizer ---

def test_n_pbits():
    circuit = PolyOptimizer({('x',): 1}, ['x'], n_bits=4)
    assert circuit.n_pbits == 4


def test_n_pbits_multivar():
    circuit = PolyOptimizer({('x', 'y'): 1}, ['x', 'y'], n_bits=3)
    assert circuit.n_pbits == 6


def test_j_symmetric():
    circuit = PolyOptimizer({('x', 'x'): 3}, ['x'], n_bits=4, minimize=False)
    assert np.allclose(circuit.J, circuit.J.T)


def test_j_diagonal_zero():
    circuit = PolyOptimizer({('x', 'x'): 3}, ['x'], n_bits=4, minimize=False)
    assert np.allclose(np.diag(circuit.J), 0)


def test_linear_h():
    # 2*x with n_bits=1: h[0] = 2/2 * 2^0 = 1
    circuit = PolyOptimizer({('x',): 2}, ['x'], n_bits=1, minimize=False)
    assert np.isclose(circuit.h[0], 1.0)


def test_quadratic_h():
    # 3*x^2 with n_bits=2, C=1.5: h[k] = 3 * 1.5 * 2^k
    circuit = PolyOptimizer({('x', 'x'): 3}, ['x'], n_bits=2, minimize=False)
    assert np.isclose(circuit.h[0], 4.5)
    assert np.isclose(circuit.h[1], 9.0)


def test_quadratic_J():
    # 3*x^2 with n_bits=2: J[0,1] = 3/2 * 2^0 * 2^1 = 3.0
    circuit = PolyOptimizer({('x', 'x'): 3}, ['x'], n_bits=2, minimize=False)
    assert np.isclose(circuit.J[0, 1], 3.0)


def _landscape(circuit, coeffs, variables, n_bits):
    """Yield (f(x), F(s)) over every p-bit state.

    F is the quantity the solvers actually see: h.s + 1/2 s.J.s, matching
    CaSuDaSolver's energy readout and the gradient its update rule ascends.
    """
    import itertools
    h = np.asarray(circuit.h).reshape(-1)
    for bits in itertools.product([-1, 1], repeat=len(variables) * n_bits):
        s = np.array(bits, dtype=float)
        F = s @ h + 0.5 * (s @ circuit.J @ s)
        vals = circuit.decode(s)
        f = 0.0
        for mono, c in coeffs.items():
            if len(mono) == 1:
                f += c * vals[mono[0]]
            elif len(mono) == 2:
                f += c * vals[mono[0]] * vals[mono[1]]
        yield f, F


@pytest.mark.parametrize("coeffs,variables,n_bits", [
    ({('x',): -0.75, ('y',): -0.75, ('x', 'y'): 1}, ['x', 'y'], 1),
    ({('x', 'x'): 1, ('x',): -3}, ['x'], 2),
    ({('x', 'x'): 2, ('y', 'y'): -1, ('x', 'y'): 3, ('x',): 1}, ['x', 'y'], 2),
])
def test_encoding_is_faithful(coeffs, variables, n_bits):
    """J/h must reproduce the polynomial up to an additive constant.

    Comparing only argmin is not enough: a mis-scaled coupling flattens the
    landscape into ties that an argmin check passes by arbitrary tie-break.
    """
    circuit = PolyOptimizer(coeffs, variables, n_bits=n_bits, minimize=False)
    resid = np.array([F - f for f, F in _landscape(circuit, coeffs, variables, n_bits)])
    assert np.allclose(resid, resid[0])


def test_cross_term_ranks_optimum_strictly():
    # min of -0.75x - 0.75y + xy over {0,1}^2 is -0.75 at (1,0) and (0,1);
    # (1,1) scores -0.5 and must stay strictly worse. A halved xy coupling
    # ties all three.
    coeffs = {('x',): -0.75, ('y',): -0.75, ('x', 'y'): 1}
    circuit = PolyOptimizer(coeffs, ['x', 'y'], n_bits=1, minimize=True)
    # minimize=True negates, and the solvers ascend F, so the optimum is argmax F
    best = max(F for _, F in _landscape(circuit, coeffs, ['x', 'y'], 1))
    ranked = sorted(
        (F, f) for f, F in _landscape(circuit, coeffs, ['x', 'y'], 1)
    )
    assert np.isclose(ranked[-1][1], -0.75)
    assert not np.isclose(ranked[-1][0], ranked[-3][0])
    assert np.isclose(best, ranked[-1][0])


def test_minimize_negates():
    c_min = PolyOptimizer({('x',): 2, ('x', 'x'): 3}, ['x'], n_bits=2, minimize=True)
    c_max = PolyOptimizer({('x',): 2, ('x', 'x'): 3}, ['x'], n_bits=2, minimize=False)
    assert np.allclose(c_min.h, -c_max.h)
    assert np.allclose(c_min.J, -c_max.J)


def test_monomial_order_invariant():
    c1 = PolyOptimizer({('x', 'y'): 1}, ['x', 'y'], n_bits=2, minimize=False)
    c2 = PolyOptimizer({('y', 'x'): 1}, ['x', 'y'], n_bits=2, minimize=False)
    assert np.allclose(c1.J, c2.J)
    assert np.allclose(c1.h, c2.h)


def test_degree_3_raises():
    with pytest.raises(ValueError):
        PolyOptimizer({('x', 'x', 'x'): 1}, ['x'], n_bits=4)


def test_decode_zero():
    circuit = PolyOptimizer({('x',): 1}, ['x'], n_bits=4)
    assert circuit.decode(np.array([-1, -1, -1, -1]))['x'] == 0


def test_decode_one():
    circuit = PolyOptimizer({('x',): 1}, ['x'], n_bits=4)
    assert circuit.decode(np.array([1, -1, -1, -1]))['x'] == 1


def test_decode_eight():
    circuit = PolyOptimizer({('x',): 1}, ['x'], n_bits=4)
    assert circuit.decode(np.array([-1, -1, -1, 1]))['x'] == 8


def test_decode_multivar():
    circuit = PolyOptimizer({('x', 'y'): 1}, ['x', 'y'], n_bits=2)
    # x bits: [1,-1] → x=1; y bits: [-1,1] → y=2
    result = circuit.decode(np.array([1, -1, -1, 1]))
    assert result['x'] == 1
    assert result['y'] == 2


# --- DocplexOptimizer ---

@requires_docplex
def test_docplex_linear():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.integer_var(lb=0, ub=3, name='x')
    mdl.minimize(2 * x)
    circuit = DocplexOptimizer(mdl)
    expected = PolyOptimizer({('x',): 2}, ['x'], n_bits=2, minimize=True)
    assert np.allclose(circuit.h, expected.h)
    assert np.allclose(circuit.J, expected.J)


@requires_docplex
def test_docplex_quadratic():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.integer_var(lb=0, ub=3, name='x')
    mdl.minimize(3 * x * x + 2 * x)
    circuit = DocplexOptimizer(mdl)
    expected = PolyOptimizer({('x', 'x'): 3, ('x',): 2}, ['x'], n_bits=2, minimize=True)
    assert np.allclose(circuit.h, expected.h)
    assert np.allclose(circuit.J, expected.J)


@requires_docplex
def test_docplex_infer_n_bits_integer():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.integer_var(lb=0, ub=15, name='x')
    mdl.minimize(x)
    circuit = DocplexOptimizer(mdl)
    assert circuit.n_bits == 4


@requires_docplex
def test_docplex_infer_n_bits_binary():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.binary_var(name='x')
    mdl.minimize(x)
    circuit = DocplexOptimizer(mdl)
    assert circuit.n_bits == 1
    assert circuit.n_pbits == 1


@requires_docplex
def test_docplex_maximize():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.integer_var(lb=0, ub=3, name='x')
    mdl.maximize(2 * x)
    circuit = DocplexOptimizer(mdl)
    expected = PolyOptimizer({('x',): 2}, ['x'], n_bits=2, minimize=False)
    assert np.allclose(circuit.h, expected.h)


@requires_docplex
def test_docplex_continuous_raises():
    from docplex.mp.model import Model
    from p_kit.library.docplex_optimizer import DocplexOptimizer
    mdl = Model()
    x = mdl.continuous_var(lb=0, ub=1, name='x')
    mdl.minimize(x)
    with pytest.raises(ValueError, match="continuous"):
        DocplexOptimizer(mdl)
