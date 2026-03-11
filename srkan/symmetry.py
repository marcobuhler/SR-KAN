import jax.numpy as jnp
from .simplification_nn import Simplification_Net
import chex
import jax

"""Adapted from https://github.com/SJ001/AI-Feynman"""


def get_error(x, y):
    assert x.shape == y.shape
    return jnp.mean(jnp.abs(x - y))


def evaluate_net(net, data_pairs):
    return jax.vmap(jax.vmap(net), in_axes=(0))(data_pairs)


# checks if f(x,y)=f(x-y)
def neg_translational_symmetry(net: Simplification_Net, x: chex.Array, y: chex.Array):
    data_pairs = []
    n_vars = x.shape[-1]
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pass
            else:
                a = 0.5 * min(jnp.std(x[:, i]), jnp.std(x[:, j]))
                data_pairs.append(x.at[:, [i, j]].set(x[:, [i, j]] + a))

    data_pairs = jnp.array(data_pairs)
    errors = jax.vmap(get_error, in_axes=(0, None))(evaluate_net(net, data_pairs), y)
    return errors


# checks if f(x,y)=f(x-y)
def pos_translational_symmetry(net: Simplification_Net, x: chex.Array, y: chex.Array):
    data_pairs = []
    n_vars = x.shape[-1]
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pass
            else:
                a = 0.5 * min(jnp.std(x[:, i]), jnp.std(x[:, j]))
                data_pairs.append(x.at[:, [i, j]].set(x[:, [i, j]] + jnp.array([a, -a])))

    data_pairs = jnp.array(data_pairs)
    errors = jax.vmap(get_error, in_axes=(0, None))(evaluate_net(net, data_pairs), y)
    return errors


# checks if f(x,y)=f(x*y)
def mult_translational_symmetry(net: Simplification_Net, x: chex.Array, y: chex.Array):
    data_pairs = []
    n_vars = x.shape[-1]
    a = 1.2
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pass
            else:
                data_pairs.append(x.at[:, [i, j]].set(x[:, [i, j]] * jnp.array([a, 1 / a])))

    data_pairs = jnp.array(data_pairs)
    errors = jax.vmap(get_error, in_axes=(0, None))(evaluate_net(net, data_pairs), y)
    return errors


# checks if f(x,y)=f(x/y)
def div_translational_symmetry(net: Simplification_Net, x: chex.Array, y: chex.Array):
    data_pairs = []
    n_vars = x.shape[-1]
    a = 1.2
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pass
            else:
                data_pairs.append(x.at[:, [i, j]].set(x[:, [i, j]] * a))

    data_pairs = jnp.array(data_pairs)
    errors = jax.vmap(get_error, in_axes=(0, None))(evaluate_net(net, data_pairs), y)
    return errors


def evaluate_symmetries(net, x: chex.Array, y: chex.Array, val_error: float, verbose: bool = False):
    ind = []
    nvars = x.shape[-1]
    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                pass
            else:
                ind.append([i, j])

    best_error = 7
    symmetry_type = 0
    best_ind = None
    types = ["Negative symmetry", "Positive symmetry", "Multiplicative symmetry", "Divisive symmetry"]

    y = y / y.std()
    neg_errors = neg_translational_symmetry(net, x, y) / val_error
    if neg_errors.min() < best_error:
        symmetry_type = 1
        best_error = neg_errors.min()
        best_ind = ind[jnp.argmin(neg_errors)]

    pos_errors = pos_translational_symmetry(net, x, y) / val_error
    if pos_errors.min() < best_error:
        symmetry_type = 2
        best_error = pos_errors.min()
        best_ind = ind[jnp.argmin(pos_errors)]

    mult_errors = mult_translational_symmetry(net, x, y) / val_error
    if mult_errors.min() < best_error:
        symmetry_type = 3
        best_error = mult_errors.min()
        best_ind = ind[jnp.argmin(mult_errors)]

    div_errors = div_translational_symmetry(net, x, y) / val_error
    if div_errors.min() < best_error:
        symmetry_type = 4
        best_error = div_errors.min()
        best_ind = ind[jnp.argmin(div_errors)]

    if symmetry_type == 0 and verbose:
        print("No symmetry found")
    elif verbose:
        print(f"{types[symmetry_type-1]} found between variables {best_ind}")
    return best_ind, symmetry_type
