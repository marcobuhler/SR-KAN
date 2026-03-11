import jax.numpy as jnp
import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def sx1sx2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*(x_0*x_1)**2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * jnp.square(x[:, 0] * x[:, 1]) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sx1x2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_0**2*x_1 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * jnp.square(x[:, 0]) * x[:, 1] + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x1sx2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_0*x_1**2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 0] * jnp.square(x[:, 1]) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sx1_sx2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*(x_0/x_1)**2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * jnp.square(x[:, 0] / x[:, 1]) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sx2_sx1(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*(x_1/x_0)**2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * jnp.square(x[:, 1] / x[:, 0]) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x1_sx2(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_0/(x_1**2) + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 0] / (x[:, 1] ** 2) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x2_sx1(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_1/(x_0**2) + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 1] / (x[:, 0] ** 2) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sx1sx2sx3(w, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*(x_0*x_1*x_2)**2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * jnp.square(x[:, 0] * x[:, 1] * x[:, 2]) + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


bf_poly_params_2 = {
    "square_x1x2": [2, sx1sx2],
    "sx1x2": [2, sx1x2],
    "x1sx2": [2, x1sx2],
    "sx1_sx2": [2, sx1_sx2],
    "sx2_sx1": [2, sx2_sx1],
    "x1_sx2": [2, x1_sx2],
    "x2_sx1": [2, x2_sx1],
}
bf_poly_params_3 = {
    "sx1sx2sx3": [2, sx1sx2sx3],
}
