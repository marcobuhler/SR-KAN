import jax.numpy as jnp, jax.random as jr
from typing import Callable, List, Tuple, Dict
import chex, optimistix, sympy
from .mv_standard import *
from .mv_polynomial import *
from .mv_trig import *
from srkan.function_libraries.function_utils import optimize_expr_constants, SympyEvaluator


def fit_bf(expr: Callable, x: chex.Array, y: chex.Array, key: chex.PRNGKey):
    w0 = jr.truncated_normal(key, 1e-3, 1e-2, shape=(expr[0]))
    w0 = w0.at[0].set(1.0)
    expr = expr[1]
    solver = optimistix.BFGS(rtol=1e-6, atol=1e-8, norm=optimistix.two_norm)  #
    res = optimistix.least_squares(expr, solver, w0, args=(x, y), throw=False, max_steps=120000)
    return [res.value, jnp.mean(expr(res.value, (x, y)) ** 2)]


def brute_force(x: chex.Array, y: chex.Array, key: chex.PRNGKey) -> Tuple[List, Dict]:
    assert x.shape[-1] == 2 or x.shape[-1] == 3 or x.shape[-1] == 4, "Brute-force only works for 2 or 3 or 4 variables"
    res = []
    y = y.ravel()
    if x.shape[-1] == 2:
        functions = bf_standard_params_2 | bf_poly_params_2
        for expr in functions.values():
            res.append(fit_bf(expr, x, y, key))
        return res, functions
    if x.shape[-1] == 3:
        functions = bf_standard_params_3 | bf_poly_params_3
        for expr in functions.values():
            res.append(fit_bf(expr, x, y, key))
        return res, functions
    if x.shape[-1] == 4:
        functions = bf_standard_params_4
        for expr in functions.values():
            res.append(fit_bf(expr, x, y, key))
        return res, functions


def run_brute_force(x, y, key):
    res, func = brute_force(x, y, key)
    mse = jnp.array([r[1] for r in res])
    func = list(func.values())
    ind = jnp.nanargmin(mse)
    expr = func[ind][1](res[ind][0], args=None, str_expr=True)
    expr = SympyEvaluator(sympy.sympify(expr))
    expr = optimize_expr_constants(expr, x, y, use_ls=False)
    return expr
