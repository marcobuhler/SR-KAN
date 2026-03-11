import chex
import jax.numpy as jnp
import jax


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def square(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} *x_**2+{w[1]}"
    else:
        x, y = args
        assert len(w) == 2, "A square function has 2 parameters"
        y_pred = w[0] * jnp.square(x) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def cube(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} *x_**3+{w[1]}"
    else:
        x, y = args
        assert len(w) == 2, "A cube function has 2 parameters"
        y_pred = w[0] * jax.lax.integer_pow(x, 3) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def pow4(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} *x_**4+{w[1]}"
    else:
        x, y = args
        assert len(w) == 2, "A pow4 function has 2 parameters"
        y_pred = w[0] * jax.lax.integer_pow(x, 4) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1 / ({w[0]} *x_**2 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = 1 / (w[0] * jax.lax.square(x) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def n_inv_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"-1 / ({w[0]} *x_**2 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = -1 / (w[0] * jax.lax.square(x) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_x3(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1/({w[0]} * x_**3 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = 1 / (w[0] * jax.lax.integer_pow(x, 3) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def n_inv_x3(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"-1/({w[0]} * x_**3 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = -1 / (w[0] * jax.lax.integer_pow(x, 3) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_x4(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1 / ({w[0]}*x_**4 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = 1 / (w[0] * jax.lax.integer_pow(x, 4) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def n_inv_x4(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"-1 / ({w[0]}*x_**4 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv function has 2 parameters"
        y_pred = -1 / (w[0] * jax.lax.integer_pow(x, 4) + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def poly3(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_**3 + {w[1]}*x_**2 + {w[2]}*x_ + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A poly3 function has 4 parameters"
        y_pred = w[0] * jnp.power(x, 3) + w[1] * jnp.square(x) + w[2] * x + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


polynomial_params = {
    "square": [[2, 1.5], square],
    "poly3": [[4, 3], poly3],
    "cube": [[2, 2], cube],
    "pow4": [[2, 2.5], pow4],
    "inv_x2": [[2, 1.5], inv_x2],
    "n_inv_x2": [[2, 1.5], n_inv_x2],
    "inv_x3": [[2, 2], inv_x3],
    "n_inv_x3": [[2, 2], n_inv_x3],
    "inv_x4": [[2, 2.5], inv_x4],
    "n_inv_x4": [[2, 2.5], n_inv_x4],
}
