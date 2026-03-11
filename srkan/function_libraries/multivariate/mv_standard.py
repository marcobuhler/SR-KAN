import chex


def check_input(x: chex.Array, y: chex.Array):
    assert y.shape == x.shape, f"The shape of the prediction {x.shape} does not match the input shape {y.shape}"


def x1x2(w, args, str_expr: bool = False):
    """w[0]*x0 * x1 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 0] * x[:, 1] + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1d2(w, args, str_expr: bool = False):
    """w[0]*x0 / x1 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0/({w[2]}*x_1 + {w[1]})"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (w[2] * x[:, 1] + w[1])
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1x2(w, args, str_expr: bool = False):
    """w[0]/x0 * x1 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0+ {w[1]})*x_1+{w[2]} "
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] + w[1]) * x[:, 1] + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1d2(w, args, str_expr: bool = False):
    """w[0]/x0 / x1 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1 + {w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1x2dx1px2(w, args, str_expr: bool = False):
    """w[0]*x0 * x1/(w[1]*x0+x1+w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1/(x_0+x_1+{w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] * x[:, 1] / (x[:, 0] + x[:, 1] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1px2dx1x2(w, args, str_expr: bool = False):
    """w[0]*x0 + x1/(w[1]*x0*x1+w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}*(x_0+x_1)/(x_0*x_1+{w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * (x[:, 0] + x[:, 1]) / (x[:, 0] * x[:, 1] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1x2x3(w, args, str_expr: bool = False):
    """w[0]*x0 * x1 * x2 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1*x_2 + {w[1]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 0] * x[:, 1] * x[:, 2] + w[1]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1x2d3(w, args, str_expr: bool = False):
    """w[0]*x0 * x1 / x2 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1/(x_2 + {w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] * x[:, 1] / (x[:, 2] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1d2x3(w, args, str_expr: bool = False):
    """w[0]*x0 / x1 * x2 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1+{w[1]})*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] + w[1]) * x[:, 2] + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1d2d3(w, args, str_expr: bool = False):
    """w[0]*x0 / x1 / x2 + w[1]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1*x_2 + {w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] * x[:, 2] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1x2x3(w, args, str_expr: bool = False):
    """w[0]/x0 * x1 * x2 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0+{w[1]})*x_1*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] + w[1]) * x[:, 1] * x[:, 2] + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1x2d3(w, args, str_expr: bool = False):
    """w[0]/x0 * x1 / x2 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_2+{w[1]})*x_1+ {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 2] + w[1]) * x[:, 1] + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1d2x3(w, args, str_expr: bool = False):
    """w[0]/x0 / x1 * x2 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1+{w[1]})*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] + w[1]) * x[:, 2] + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def d1d2d3(w, args, str_expr: bool = False):
    """w[0]/x0 / x1 / x2 + w[1]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1*x_2 + {w[1]})+{w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] * x[:, 2] + w[1]) + w[2]
        if y is not None:
            check_input(y, y_pred)
            return y_pred - y
        return y_pred


def x1x2x3x4(w, args, str_expr: bool = False):
    """w[0]*x0*x1*x2*x3 + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1*x_2*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 2
        y_pred = w[0] * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def x1x2x3d4(w, args, str_expr: bool = False):
    """w[0]*x0*x1*x2 / (x3 + w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1*x_2/(x_3 + {w[1]}) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] * x[:, 1] * x[:, 2] / (x[:, 3] + w[1]) + w[2]
        return y_pred - y if y is not None else y_pred


def x1x2d3x4(w, args, str_expr: bool = False):
    """w[0]*x0*x1 / (x2 + w[1]) * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1/(x_2 + {w[1]})*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] * x[:, 1] / (x[:, 2] + w[1]) * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def x1x2d3d4(w, args, str_expr: bool = False):
    """w[0]*x0*x1 / (x2*x3 + w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0*x_1/(x_2*x_3 + {w[1]}) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] * x[:, 1] / (x[:, 2] * x[:, 3] + w[1]) + w[2]
        return y_pred - y if y is not None else y_pred


def x1d2x3x4(w, args, str_expr: bool = False):
    """w[0]*x0 / (x1 + w[1]) * x2 * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1 + {w[1]})*x_2*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] + w[1]) * x[:, 2] * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def x1d2x3d4(w, args, str_expr: bool = False):
    """w[0]*x0 / (x1*x3 + w[1]) * x2 + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1*x_3 + {w[1]})*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] * x[:, 3] + w[1]) * x[:, 2] + w[2]
        return y_pred - y if y is not None else y_pred


def x1d2d3x4(w, args, str_expr: bool = False):
    """w[0]*x0 / (x1*x2 + w[1]) * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1*x_2 + {w[1]})*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] * x[:, 2] + w[1]) * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def x1d2d3d4(w, args, str_expr: bool = False):
    """w[0]*x0 / (x1*x2*x3 + w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}*x_0/(x_1*x_2*x_3 + {w[1]}) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] * x[:, 0] / (x[:, 1] * x[:, 2] * x[:, 3] + w[1]) + w[2]
        return y_pred - y if y is not None else y_pred


def d1x2x3x4(w, args, str_expr: bool = False):
    """w[0]/(x0 + w[1]) * x1 * x2 * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0 + {w[1]})*x_1*x_2*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] + w[1]) * x[:, 1] * x[:, 2] * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def d1x2x3d4(w, args, str_expr: bool = False):
    """w[0]/(x0*x3 + w[1]) * x1 * x2 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_3 + {w[1]})*x_1*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 3] + w[1]) * x[:, 1] * x[:, 2] + w[2]
        return y_pred - y if y is not None else y_pred


def d1x2d3x4(w, args, str_expr: bool = False):
    """w[0]/(x0*x2 + w[1]) * x1 * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_2 + {w[1]})*x_1*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 2] + w[1]) * x[:, 1] * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def d1x2d3d4(w, args, str_expr: bool = False):
    """w[0]/(x0*x2*x3 + w[1]) * x1 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_2*x_3 + {w[1]})*x_1 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 2] * x[:, 3] + w[1]) * x[:, 1] + w[2]
        return y_pred - y if y is not None else y_pred


def d1d2x3x4(w, args, str_expr: bool = False):
    """w[0]/(x0*x1 + w[1]) * x2 * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1 + {w[1]})*x_2*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] + w[1]) * x[:, 2] * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def d1d2x3d4(w, args, str_expr: bool = False):
    """w[0]/(x0*x1*x3 + w[1]) * x2 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1*x_3 + {w[1]})*x_2 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] * x[:, 3] + w[1]) * x[:, 2] + w[2]
        return y_pred - y if y is not None else y_pred


def d1d2d3x4(w, args, str_expr: bool = False):
    """w[0]/(x0*x1*x2 + w[1]) * x3 + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1*x_2 + {w[1]})*x_3 + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] * x[:, 2] + w[1]) * x[:, 3] + w[2]
        return y_pred - y if y is not None else y_pred


def d1d2d3d4(w, args, str_expr: bool = False):
    """w[0]/(x0*x1*x2*x3 + w[1]) + w[2]"""
    if str_expr:
        return f"{w[0]}/(x_0*x_1*x_2*x_3 + {w[1]}) + {w[2]}"
    else:
        x, y = args
        assert len(w) == 3
        y_pred = w[0] / (x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] + w[1]) + w[2]
        return y_pred - y if y is not None else y_pred


bf_standard_params_2 = {
    "x1x2": [2, x1x2],
    "x1d2": [3, x1d2],
    "d1x2": [3, d1x2],
    "d1d2": [3, d1d2],
    "x1x2dx1px2": [3, x1x2dx1px2],
    "x1px2dx1x2": [3, x1px2dx1x2],
}

bf_standard_params_3 = {
    "x1x2x3": [2, x1x2x3],
    "x1x2d3": [3, x1x2d3],
    "x1d2x3": [3, x1d2x3],
    "x1d2d3": [3, x1d2d3],
    "d1x2x3": [3, d1x2x3],
    "d1x2d3": [3, d1x2d3],
    "d1d2x3": [3, d1d2x3],
    "d1d2d3": [3, d1d2d3],
}
bf_standard_params_4 = {
    "x1x2x3x4": [2, x1x2x3x4],
    "x1x2x3d4": [3, x1x2x3d4],
    "x1x2d3x4": [3, x1x2d3x4],
    "x1x2d3d4": [3, x1x2d3d4],
    "x1d2x3x4": [3, x1d2x3x4],
    "x1d2x3d4": [3, x1d2x3d4],
    "x1d2d3x4": [3, x1d2d3x4],
    "x1d2d3d4": [3, x1d2d3d4],
    "d1x2x3x4": [3, d1x2x3x4],
    "d1x2x3d4": [3, d1x2x3d4],
    "d1x2d3x4": [3, d1x2d3x4],
    "d1x2d3d4": [3, d1x2d3d4],
    "d1d2x3x4": [3, d1d2x3x4],
    "d1d2x3d4": [3, d1d2x3d4],
    "d1d2d3x4": [3, d1d2d3x4],
    "d1d2d3d4": [3, d1d2d3d4],
}
