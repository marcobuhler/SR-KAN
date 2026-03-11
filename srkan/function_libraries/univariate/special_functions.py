import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def michaelis_menten(w: chex.Array, args, str_expr: bool = False):
    if str_expr:
        return f"{w[0]} * x_ / ({w[1]}+x_)"
    else:
        x, y = args
        assert len(w) == 2, "A michaelis menten function has 3 parameters"
        y_pred = w[0] * x / (w[1] + x)
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        else:
            return y_pred


special_params = {"michaelis_menten": [[2, 2], michaelis_menten]}
