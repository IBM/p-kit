import math
from p_kit.library.poly import PolyOptimizer


class DocplexOptimizer(PolyOptimizer):
    """Build a PolyOptimizer from a DOcplex linear/quadratic model.

    Only the objective function is encoded into J/h — constraints are ignored.
    Objective must be linear or quadratic (degree ≤ 2).
    The optimization sense (minimize/maximize) is read from the model.

    Parameters
    ----------
    model : docplex.mp.model.Model
    n_bits : int, optional
        Bit-width per variable. If None, auto-inferred from variable bounds:
        binary variables → 1, integer [0, ub] → ceil(log2(ub + 1)).
        All variables share the same n_bits; for mixed models, pass an explicit value.

    Notes
    -----
    .. versionadded:: 0.0.1
    """

    def __init__(self, model, n_bits=None):
        coeffs, variables = DocplexOptimizer._parse(model)
        minimize = model.is_minimized()
        if n_bits is None:
            n_bits = DocplexOptimizer._infer_n_bits(model, variables)
        super().__init__(coeffs, variables, n_bits=n_bits, minimize=minimize)

    @staticmethod
    def _parse(model):
        obj = model.objective_expr
        coeffs = {}
        variables = [v.name for v in model.iter_variables()]

        # linear_part exists on QuadExpr; fall back to obj itself for LinearExpr
        linear_part = getattr(obj, 'linear_part', obj)
        for var, coeff in linear_part.iter_terms():
            coeffs[(var.name,)] = coeffs.get((var.name,), 0) + coeff
        if linear_part.constant:
            coeffs[()] = linear_part.constant

        if hasattr(obj, 'iter_quad_triplets'):
            for v1, v2, coeff in obj.iter_quad_triplets():
                key = tuple(sorted([v1.name, v2.name]))
                coeffs[key] = coeffs.get(key, 0) + coeff

        return coeffs, variables

    @staticmethod
    def _infer_n_bits(model, variables):
        var_map = {v.name: v for v in model.iter_variables()}
        max_bits = 1
        for name in variables:
            v = var_map[name]
            if v.is_binary():
                bits = 1
            elif v.is_integer():
                ub = v.ub
                if math.isinf(ub):
                    raise ValueError(
                        f"Variable '{name}' has no finite upper bound; specify n_bits explicitly."
                    )
                bits = max(1, math.ceil(math.log2(ub + 1)))
            else:
                raise ValueError(
                    f"Variable '{name}' is continuous; only binary and integer variables are supported."
                )
            max_bits = max(max_bits, bits)
        return max_bits
