import jax.numpy as jnp
import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def log(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * log({w[1]} * x_) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A log function has 3 parameters"
        y_pred = w[0] * jnp.log(w[1] * x) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def log_squared(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * log({w[1]} * x_**2) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A log_squared function has 3 parameters"
        y_pred = w[0] * jnp.log(w[1] * x**2) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def log10(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * log10({w[1]} * x_) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A log10 function has 3 parameters"
        y_pred = w[0] * jnp.log10(w[1] * x) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


log_params = {
    "log": [[3, 2], log],
    "log10": [[3, 3], log10],
    "log_squared": [[3, 3], log_squared],
}
