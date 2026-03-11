import chex
import jax.numpy as jnp


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def sinx1x2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*sin({w[1]}*x_0*x_1+{w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4
        y_pred = w[0] * jnp.sin(w[1] * x[:, 0] * x[:, 1] + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def cosx1x2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*cos({w[1]}*x_0*x_1+{w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4
        y_pred = w[0] * jnp.cos(w[1] * x[:, 0] * x[:, 1] + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


bf_trig_params_2 = {
    "sinx1x2": [4, sinx1x2],
    "cosx1x2": [4, cosx1x2],
}
