import jax.numpy as jnp
import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def sin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * sin({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A sin function has 4 parameters"
        y_pred = w[0] * jnp.sin(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sin2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * sin({w[1]} * x_ + {w[2]})**2 + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A sin function has 4 parameters"
        y_pred = w[0] * jnp.square(jnp.sin(w[1] * x + w[2])) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def dsin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f" 1/({w[0]} *sin({w[1]} * x_ + {w[2]}) + {w[3]})"
    else:
        x, y = args
        assert len(w) == 4, "A dsin function has 4 parameters"
        y_pred = 1 / (w[0] * jnp.sin(w[1] * x + w[2]) + w[3])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def tan(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * tan({w[1]} * (x_ - {w[2]})) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A tan function has 4 parameters"
        y_pred = w[0] * jnp.tan(w[1] * (x - w[2])) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def cos(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * cos({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A cos function has 4 parameters"
        y_pred = w[0] * jnp.cos(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def dcos(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f" 1/({w[0]} *cos({w[1]} * x_ + {w[2]}) + {w[3]})"
    else:
        x, y = args
        assert len(w) == 4, "A dcos function has 4 parameters"
        y_pred = 1 / (w[0] * jnp.cos(w[1] * x + w[2]) + w[3])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x_sin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_*sin({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A x_sin function has 4 parameters"
        y_pred = w[0] * x * jnp.sin(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x2_sin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_**2*sin({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A x2_sin function has 4 parameters"
        y_pred = w[0] * jnp.square(x) * jnp.sin(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x_cos(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_*cos({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A x_cos function has 4 parameters"
        y_pred = w[0] * x * jnp.cos(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x_tan(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_*tan({w[1]} * (x_ - {w[2]})) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A x_tan function has 4 parameters"
        y_pred = w[0] * x * jnp.tan(w[1] * (x - w[2])) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def sin_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} *sin({w[1]} * x_**2 + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A sin_x2 function has 4 parameters"
        y_pred = w[0] * jnp.sin(w[1] * x**2 + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def exp_sin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*exp({w[1]}*sin({w[2]} * x_ + {w[3]})+ {w[4]}) "
    else:
        x, y = args
        assert len(w) == 5, "A exp_sin function has 5 parameters"
        y_pred = w[0] * jnp.exp(w[1] * jnp.sin(w[2] * x + w[3]) + w[4])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x_exp_sin(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*x_ + {w[1]}*exp({w[2]}*sin({w[3]} * x_)+ {w[4]} + {w[5]})+ {w[6]}"
    else:
        x, y = args
        assert len(w) == 7, "A x_exp_sin function has 7 parameters"
        y_pred = w[0] * x + w[1] * jnp.exp(w[2] * jnp.sin(w[3] * x + w[4]) + w[5]) + w[6]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def x2_cos(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_**2*cos({w[1]} * x_ + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A x2_cos function has 4 parameters"
        y_pred = w[0] * x**2 * jnp.cos(w[1] * x + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def cos_x2(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} *cos({w[1]} * x_**2 + {w[2]}) + {w[3]}"
    else:
        x, y = args
        assert len(w) == 4, "A cos_x2 function has 4 parameters"
        y_pred = w[0] * jnp.cos(w[1] * x**2 + w[2]) + w[3]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def exp_cos(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]}*exp({w[1]}*cos({w[2]} * x_ + {w[3]})+ {w[4]}) "
    else:
        x, y = args
        assert len(w) == 5, "A exp_cos function has 5 parameters"
        y_pred = w[0] * jnp.exp(w[1] * jnp.cos(w[2] * x + w[3]) + w[4])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


def fourier_sum(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * sin({w[1]} * x_ + {w[2]}) + {w[3]} * sin({w[4]} * x_ + {w[5]}) + {w[6]}"
    else:
        x, y = args
        assert len(w) == 7, "A fourier_sum function has 7 parameters"

        term1 = w[0] * jnp.sin(w[1] * x + w[2])
        term2 = w[3] * jnp.sin(w[4] * x + w[5])

        y_pred = term1 + term2 + w[6]

        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


trig_params = {
    "sin": [[4, 3], sin],
    "tan": [[4, 3], tan],
    "dsin": [[4, 3], dsin],
    "sin2": [[4, 3], sin2],
    "cos": [[4, 3], cos],
    "dcos": [[4, 3], dcos],
    "x_sin": [[4, 4], x_sin],
    "x_cos": [[4, 4], x_cos],
    "x_tan": [[4, 4], x_tan],
    "sin_x2": [[4, 4], sin_x2],
    "x2_sin": [[4, 4], x2_sin],
    "exp_sin": [[5, 5], exp_sin],
    "x_exp_sin": [[7, 6], x_exp_sin],
    "cos_x2": [[4, 4], cos_x2],
    "x2_cos": [[4, 4], x2_cos],
    "exp_cos": [[5, 5], exp_cos],
    "fourier_sum": [[7, 6], fourier_sum],
}
