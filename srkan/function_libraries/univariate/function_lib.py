import jax.numpy as jnp, jax.random as jr
from typing import Callable, List, Tuple, Dict
import chex
import optimistix, difflib
from scipy.signal import find_peaks
from .trigonometric import *
from .polynomial import *
from .logarithms import *
from .standard_functions import *
from .special_functions import *
from .hyperbolic import *


def get_intial_weights(expr: Callable, x: chex.Array, y: chex.Array, key: chex.PRNGKey):
    assert x.shape == y.shape
    num_weights = expr[0][0]
    expr = expr[1]

    # Initial guesses for the least squares
    rand_weight = jr.truncated_normal(key, 0, 1)
    if expr.__name__ == "linear":
        x_mean, y_mean = jnp.mean(x), jnp.mean(y)
        slope = jnp.sum((x - x_mean) * (y - y_mean)) / jnp.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        return jnp.array([slope, intercept])
    if expr.__name__ in ["sin", "cos"]:
        peaks = find_peaks(y)[0]
        period = 2 * jnp.pi * len(peaks) / (jax.lax.abs(x.min()) + jax.lax.abs(x.max()))
        return jnp.array([y.max() - y.mean(), period, rand_weight, y.mean()])
    if expr.__name__ == "sin2":
        peaks = find_peaks(y)[0]
        period = 2 * jnp.pi * len(peaks) // 2 / (jax.lax.abs(x.min()) + jax.lax.abs(x.max()))
        return jnp.array([y.max() - y.mean(), period, rand_weight, y.mean()])
    if expr.__name__ in ["dsin", "dcos"]:
        y_temp = 1 / y
        y_temp_mean = jnp.mean(y_temp)
        y_temp_max = jnp.max(y_temp)
        w0_guess = y_temp_max - y_temp_mean
        peaks = find_peaks(y_temp)[0]
        num_peaks = len(peaks)
        if num_peaks > 0:
            period = 2 * jnp.pi * num_peaks / (jax.lax.abs(x.min()) + jax.lax.abs(x.max()))
        else:
            period = 1.0
        w1_guess = period
        w2_guess = rand_weight
        w3_guess = y_temp_mean
        return jnp.array([w0_guess, w1_guess, w2_guess, w3_guess])
    elif expr.__name__ == "tan":
        return jnp.array([1, 1, 0.5 * (x.max() + x.min()), 0.5 * (y.max() + y.min())])
    elif expr.__name__ in ["x_sin", "x_cos"]:
        peaks = find_peaks(y / x)[0]
        period = jnp.floor(2 * jnp.pi * len(peaks) / (jax.lax.abs(x.min()) + jax.lax.abs(x.max())))
        return jnp.array([jnp.max(y / x) - jnp.mean(y / x), period, rand_weight, jnp.mean(y / x)])
    elif expr.__name__ == "x_tan":
        return jnp.array([1, 1, 0.5 * (jnp.max(x / 2) + jnp.min(x / 2)), 0.5 * (jnp.max(y / x) + jnp.min(y / x))])
    elif expr.__name__ in ["sin_x2", "cos_x2"]:
        peaks = find_peaks(y)[0]
        period = jnp.floor(2 * jnp.pi * len(peaks) / (2 * jax.lax.square(x).min() + 2 * jax.lax.square(x).max()))
        return jnp.array([jnp.max(y) - jnp.mean(y), period, rand_weight, jnp.mean(y)])
    elif expr.__name__ in ["x2_sin", "x2_cos"]:
        peaks = find_peaks(y / jax.lax.square(x))[0]
        period = jnp.floor(2 * jnp.pi * len(peaks) / (jax.lax.abs(x.min()) + jax.lax.abs(x.max())))
        return jnp.array(
            [
                (y / jax.lax.square(x)).max() - (y / jax.lax.square(x)).mean(),
                period,
                rand_weight,
                (y / jax.lax.square(x)).mean(),
            ]
        )
    elif expr.__name__ in ["exp_sin"]:
        y_ = jax.lax.log(y) if jnp.isnan(jax.lax.log(y)).sum() == 0 else y
        peaks = find_peaks(y_)[0]
        period = jnp.floor(2 * jnp.pi * len(peaks) / (jax.lax.abs(x.min()) + jax.lax.abs(x.max())))
        return jnp.array([1.0, y_.max() - y_.mean(), period, rand_weight, y_.mean()])
    elif expr.__name__ in ["x_exp_sin"]:
        y_ = jnp.log(y - x) if jnp.isnan(jax.lax.log(y - x)).sum() == 0 else y
        peaks = find_peaks(y_)[0]
        period = jnp.floor(2 * jnp.pi * len(peaks) / (jax.lax.abs(x.min()) + jax.lax.abs(x.max())))
        return jnp.array([1.0, 1.0, y_.max() - y_.mean(), period, rand_weight, y_.mean(), 1.0])
    elif expr.__name__ == "exp":
        a = -1.0 if y[0] > y[-1] else 1.0
        return jnp.array([a, 1.0, y.min()])
    elif expr.__name__ in ["log", "log10"]:
        a = -1.0 if y[0] > y[-1] else 1.0
        return jnp.array([a, 1.0, 1])
    elif expr.__name__ in ["sqrt", "inv_sqrt"]:
        rand_weight = jr.truncated_normal(key, 0.1, 1, 2)
        b = 1.0 if jnp.all(x > 0) else jnp.abs(jnp.min(x)) + 0.01
        return jnp.array([rand_weight[0], b, rand_weight[1]])
    elif expr.__name__ in ["sqrt2", "inv_sqrt2"]:
        rand_weight = jr.truncated_normal(key, 0.1, 1, 2)
        b = 1.0
        return jnp.array([rand_weight[0], b, rand_weight[1]])
    elif expr.__name__ in ["inv_x", "n_inv_x"]:
        b = 1.0 if jnp.all(x > 0) else jnp.abs(jnp.min(x)) + 0.01
        return jnp.array([1.0, b])
    elif expr.__name__ in ["inv_x2", "n_inv_x2"]:
        b = 1.0
        return jnp.array([1.0, b])
    elif expr.__name__ == "michaelis_menten":
        ymax = y.max()
        try:
            km = x[y > (ymax / 2)][jnp.argmin(y[y > (ymax / 2)])]
        except:
            km = 1
        return jnp.array([ymax, km])
    else:
        return jr.truncated_normal(key, -1, 1, shape=num_weights)


def fit_single_expression(expr: Callable, x: chex.Array, y: chex.Array, key: chex.PRNGKey):
    """
    Fits a single symbolic expression to input-output data using a least-squares optimizer.

    Parameters
    ----------
    expr : Callable
        A tuple or callable where `expr[1]` is the function to be optimized (e.g., a symbolic model).
    x : chex.Array
        Input data array (independent variable).
    y : chex.Array
        Target data array (dependent variable).
    key : chex.PRNGKey
        A JAX random key used to initialize weights.

    Returns
    -------
    list
        A list containing:
        - The optimized parameter values (array).
        - The mean squared error (float) between the predicted and true `y` values after fitting.

    Notes
    -----
    - Uses `get_intial_weights` to generate initial parameters based on the input expression and data.
    - Optimization is performed using the `optimistix.BFGS` solver via least-squares minimization.
    - Returns both the final parameter values and the mean squared error as a measure of fit quality.
    """
    w0 = get_intial_weights(expr, x, y, key)
    expr = expr[1]
    solver = optimistix.BFGS(rtol=1e-4, atol=1e-6, norm=optimistix.two_norm)
    if expr.__name__ in ["sin", "cos"]:
        res1 = optimistix.least_squares(expr, solver, w0, args=(x, y), throw=False, max_steps=2048)
        value1 = res1.value
        del res1

        w0 = w0.at[0].set(w0[0] * (-1))
        res2 = optimistix.least_squares(expr, solver, w0, args=(x, y), throw=False, max_steps=2048)
        value2 = res2.value
        del res2
        if jnp.mean(jax.lax.square(expr(value1, (x, y)))) < jnp.mean(jax.lax.square(expr(value2, (x, y)))):
            return [value1, jnp.mean(jax.lax.square(expr(value1, (x, y))))]
        else:
            return [value2, jnp.mean(jax.lax.square(expr(value2, (x, y))))]
    else:
        res = optimistix.least_squares(expr, solver, w0, args=(x, y), throw=False, max_steps=2048)
        value = res.value
        del res
    return [value, jnp.mean(jax.lax.square(expr(value, (x, y))))]


all_expr = standard_fct_params | polynomial_params | trig_params | special_params | hyperbolic_params | log_params


def fit_function_library(x: chex.Array, y: chex.Array, key: chex.PRNGKey, functions: list[str]) -> Tuple[List, Dict]:
    """
    Fits a set of symbolic functions to input data

    Parameters
    ----------
    x : chex.Array
        Input data array (typically features or independent variable).
    y : chex.Array
        Target data array (dependent variable).
    key : chex.PRNGKey
        A JAX PRNG key for any random operations during fitting.
    functions : list of str
        A list of function names to fit, each of which must exist in the `all_expr` dictionary.

    Returns
    -------
    Tuple[List, Dict]
        - A list of fit results, one for each function (as returned by `fit_single_expression`).
        - A dictionary mapping each function name to its corresponding symbolic expression.

    Notes
    -----
    - If all elements of `x` are positive, all functions (including logarithmic ones) are used.
    - If any values in `x` are non-positive, logarithmic functions are removed to avoid domain errors.
    - Logarithmic functions are filtered using the `log_params` keys.

    Raises
    ------
    KeyError
        If a function in `functions` does not exist in the `all_expr` dictionary.

    """
    res = []
    expr_func = {}
    for func in functions:
        expr_func[func] = all_expr[func]

    if jnp.all(x > 0):  # No problem with log
        for expr in expr_func.values():
            res.append(fit_single_expression(expr, x, y, key))
        return res, expr_func
    else:  # Remove log and sqrt functions, because they fail
        for param in log_params.keys():
            if param in expr_func:
                del expr_func[param]
        for expr in expr_func.values():
            res.append(fit_single_expression(expr, x, y, key))
        return res, expr_func


def create_library(functions: list[str], exclude_functions: list[str]):
    """
    Expands a list of symbolic function categories or names into a complete list of individual functions


    Parameters
    ----------
    functions : list of str
        A list of function names or special keywords. The special keywords include:
            - "all": includes all available functions in the library.
            - "standard": expands to all standard functions.
            - "polynomial": expands to polynomial-related functions.
            - "hyperbolic": expands to hyperbolic functions.
            - "special": expands to special functions (e.g., error function, gamma, etc.).
            - "logarithm": expands to logarithmic functions.
            - "trigonometric": expands to trigonometric functions.

    Returns
    -------
    list of str
        A list of functions included in the requested library.

    Raises
    ------
    AssertionError
        If any of the requested or expanded function names are not found in the `all_expr` dictionary.

    Notes
    -----
    - Uses `difflib.get_close_matches` to suggest a possible match if an unknown function name is encountered.
    - The output list contains no duplicates.
    """

    if "all" in functions:
        functions = list(all_expr.keys())
    if "standard" in functions:
        functions.remove("standard")
        functions.extend(list(standard_fct_params.keys()))
    if "polynomial" in functions:
        functions.remove("polynomial")
        functions.extend(list(polynomial_params.keys()))
    if "hyperbolic" in functions:
        functions.remove("hyperbolic")
        functions.extend(list(hyperbolic_params.keys()))
    if "special" in functions:
        functions.remove("special")
        functions.extend(list(special_params.keys()))
    if "logarithm" in functions:
        functions.remove("logarithm")
        functions.extend(list(log_params.keys()))
    if "trigonometric" in functions:
        functions.remove("trigonometric")
        functions.extend(list(trig_params.keys()))

    functions = list(set(functions))
    for func in functions:
        assert (
            func in all_expr.keys()
        ), f"Function {func} is not in the library. Closest match: {difflib.get_close_matches(func, all_expr.keys())}"

    if exclude_functions is not None:
        for func in exclude_functions:
            assert (
                func in all_expr.keys()
            ), f"Function to exclude '{func}' is not in the library. Closest match: {difflib.get_close_matches(func, all_expr.keys())}"
            functions.remove(func)
    return functions
