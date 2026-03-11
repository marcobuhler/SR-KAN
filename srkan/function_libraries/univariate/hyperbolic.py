import jax.numpy as jnp
import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def tanh(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} + {w[1]}*tanh({w[2]}*x_)"
    else:
        x, y = args
        assert len(w) == 3, "A tanh function has 4 parameters"
        y_pred = w[0] + w[1] * jnp.tanh(w[2] * x)
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def cosh(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} + {w[1]}*cosh({w[2]}*x_)"
    else:
        x, y = args
        assert len(w) == 3, "A cosh function has 4 parameters"
        y_pred = w[0] + w[1] * jnp.cosh(w[2] * x)
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sinh(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} + {w[1]}*sinh({w[2]}*x_)"
    else:
        x, y = args
        assert len(w) == 3, "A sinh function has 4 parameters"
        y_pred = w[0] + w[1] * jnp.sinh(w[2] * x)
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


hyperbolic_params = {"tanh": [[3, 4], tanh], "cosh": [[3, 4], cosh], "sinh": [[3, 4], sinh]}
