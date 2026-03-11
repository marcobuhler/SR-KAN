import jax.numpy as jnp
import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def constant(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}"
    else:
        x, y = args
        assert len(w) == 1, "A constant function has 1 parameter"
        y_pred = w[0] * jnp.ones(x.shape)
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def linear(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_ + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2, "A square function has 2 parameters"
        y_pred = w[0] * x + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x2_m_x(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}* x_**2*(x_+{w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A x2_x function has 2 parameters"
        y_pred = w[0] * jnp.square(x) * (x + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x2_p_x(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}* x_**2+{w[1]}*x_+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A x2_p_x function has 2 parameters"
        y_pred = w[0] * jnp.square(x) + w[1] * x + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def exp(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * exp({w[1]} * x_) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A exp function has 3 parameters"
        y_pred = w[0] * jnp.exp(w[1] * x) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def exp2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * exp({w[1]} * x_**2) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A exp function has 3 parameters"
        y_pred = w[0] * jnp.exp(w[1] * x**2) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def double_exp(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * exp({w[1]} * x_) + {w[2]} * exp({w[3]} * x_) + {w[4]}"
    else:
        x, y = args
        assert len(w) == 5, "A double_exp function has 5 parameters"
        y_pred = w[0] * jnp.exp(w[1] * x) + w[2] * jnp.exp(w[3] * x) + w[4]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def damped_sine(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * exp(-{w[1]} * x_) * sin({w[2]} * x_ + {w[3]}) + {w[4]}"
    else:
        x, y = args
        assert len(w) == 5, "A damped_sine function has 5 parameters"
        y_pred = w[0] * jnp.exp(-w[1] * x) * jnp.sin(w[2] * x + w[3]) + w[4]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def gaussian(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * exp(-{w[1]} * (x_ - {w[2]})**2) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A gaussian function has 4 parameters"
        # w[1] (width) should be positive, but optimizers can handle this.
        y_pred = w[0] * jnp.exp(-w[1] * (x - w[2]) ** 2) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sigmoid(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        # Note: f-string requires {{}} to escape braces
        return f"{w[0]} / (1 + exp(-{w[1]} * (x_ - {w[2]}))) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A sigmoid function has 4 parameters"
        y_pred = w[0] / (1.0 + jnp.exp(-w[1] * (x - w[2]))) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def power_law(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_**{w[1]} + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A power law function has 3 parameters"
        # Note: This can produce NaNs if x is negative and w[1] is not an integer.
        # It's safer if you know x is positive.
        y_pred = w[0] * jnp.power(x, w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sqrt(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*sqrt(x_+{w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A sqrt function has 3 parameters"
        y_pred = w[0] * jnp.sqrt(x + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_x(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1/({w[0]}*x_+{w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv_x function has 2 parameters"
        y_pred = 1 / (w[0] * x + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}/(x_**2+{w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv_x function has 2 parameters"
        y_pred = w[0] / (x**2 + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def n_inv_x(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"-1/({w[0]}*x_+{w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv_x function has 2 parameters"
        y_pred = -1 / (w[0] * x + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def n_inv_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"-1/({w[0]}*x_**2+{w[1]})"
    else:
        x, y = args
        assert len(w) == 2, "A inv_x function has 2 parameters"
        y_pred = -1 / (w[0] * x**2 + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_sqrt(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1/({w[0]}*sqrt(x_+{w[1]})+{w[2]})"
    else:
        x, y = args
        assert len(w) == 3, "A inv_sqrt function has 2 parameters"
        y_pred = 1 / (w[0] * jnp.sqrt(x + w[1]) + w[2])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sqrt2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"sqrt({w[0]}*x_**2+{w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3, "A sqrt function has 3 parameters"
        y_pred = jnp.sqrt(w[0] * x**2 + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def inv_sqrt2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"1/(sqrt({w[0]}*x_**2+{w[1]})+{w[2]})"
    else:
        x, y = args
        assert len(w) == 3, "A inv_sqrt function has 2 parameters"
        y_pred = 1 / (jnp.sqrt(w[0] * x**2 + w[1]) + w[2])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def rational_poly(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        # Parentheses are critical for the string expression
        return f"({w[0]} * x_ + {w[1]}) / (x_**2 + {w[2]} * x_ + {w[3]})"
    else:
        x, y = args
        assert len(w) == 4, "A rational_poly function has 4 parameters"

        numerator = w[0] * x + w[1]
        denominator = x**2 + w[2] * x + w[3]
        y_pred = numerator / (denominator + 1e-8)

        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


standard_fct_params = {
    "constant": [[1, 0], constant],
    "linear": [[2, 1], linear],
    "x2_m_x": [[3, 2], x2_m_x],
    "inv_x": [[2, 1], inv_x],
    "n_inv_x": [[2, 1], n_inv_x],
    "inv_x2": [[2, 1.5], inv_x2],
    "n_inv_x2": [[2, 1.5], n_inv_x2],
    "x2_p_x": [[3, 2], x2_p_x],
    "sqrt": [[3, 1], sqrt],
    "sqrt2": [[3, 1.5], sqrt2],
    "inv_sqrt": [[3, 1], inv_sqrt],
    "inv_sqrt2": [[3, 1.5], inv_sqrt2],
    "exp": [[3, 2.5], exp],
    "exp2": [[3, 3], exp2],
    "gaussian": [[4, 3], gaussian],
    "sigmoid": [[4, 3], sigmoid],
    "power_law": [[3, 3], power_law],
    "double_exp": [[5, 4], double_exp],
    "damped_sine": [[5, 4], damped_sine],
    "rational_poly": [[4, 2.5], rational_poly],
}
