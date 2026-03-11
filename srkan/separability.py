import jax.numpy as jnp
import jax, chex
import equinox as eqx
from itertools import combinations


def check_separability(net: eqx.Module, x: chex.Array, y: chex.Array, val_error: float, verbose: bool = False) -> tuple:
    """
    Checks for additive and multiplicative separability

    Adapted from https://github.com/SJ001/AI-Feynman

    Parameters
    ----------
    net : eqx.Module
        Trained simplification network used as a interpolator of the data

    xtrain : Array
        Training input data

    xtest : Array
        Test input data

    ytest : Array
        Test target data

    Returns
    -------
    additive_indeces, additive_error, multiplicative_indeces, multiplicative_error
        The indeces and associated error which are first additively separable and then multiplicatively separable.
        Returns None if no separability is found, meaning the error is larger then the threshold.

    """
    y = y / y.std()
    n_vars = x.shape[-1]
    errors_add = []
    errors_mult = []
    indeces = []
    fact_vary_all = x.at[:, :].set(x.mean(0))
    t1 = jax.vmap(net)(fact_vary_all)

    for i in range(1, n_vars):  # Loop over the variables
        c = combinations(range(n_vars), i)
        for j in c:  # Loop over all variable combinations
            rest_indx = list(filter(lambda x: x not in j, range(n_vars)))
            fact_vary_one = x.at[:, rest_indx].set(x[:, rest_indx].mean(0))
            fact_vary_rest = x.at[:, j].set(x[:, j].mean(0))
            t2 = jax.vmap(net)(fact_vary_one)
            t3 = jax.vmap(net)(fact_vary_rest)
            indeces.append([list(j), rest_indx])
            errors_add.append((jnp.abs(y - (t2 + t3 - t1)).mean()) / val_error)
            errors_mult.append((jnp.abs(y - (t2 * t3 / t1)).mean()) / val_error)

    errors_add = jnp.array(errors_add)
    errors_mult = jnp.array(errors_mult)
    ind_add = indeces[errors_add.argmin()] if errors_add.min() <= 10 else None
    ind_mult = indeces[errors_mult.argmin()] if errors_mult.min() <= 10 else None
    if ind_add is not None or ind_mult is not None:
        if float(errors_add.min()) < float(errors_mult.min()):
            if verbose:
                print(f"Additive separability found between variables: {ind_add[0]} and {ind_add[1]}")
            return ind_add, float(errors_add.min()), 0
        else:
            if verbose:
                print(f"Multiplicative separability found between variables: {ind_mult[0]} and {ind_mult[1]}")
            return ind_mult, float(errors_mult.min()), 1
    else:
        if verbose:
            print("No separability found")
        return None, float(errors_mult.min()), 1


def do_separability(net, x: chex.Array, y: chex.Array, ind: list, categ: int):
    ymean = jax.vmap(net)(x.at[:, :].set(x.mean(0)))
    x_prime = x[:, ind[0]]
    x_prime2 = x[:, ind[1]]
    y_prime2 = jax.vmap(net)(x.at[:, ind[0]].set(x[:, ind[0]].mean(0)))
    if categ == 0:
        y_prime = jax.vmap(net)(x.at[:, ind[1]].set(x[:, ind[1]].mean(0))) - ymean
    if categ == 1:
        y_prime = jax.vmap(net)(x.at[:, ind[1]].set(x[:, ind[1]].mean(0))) / ymean
    return x_prime, y_prime * y.std(), x_prime2, y_prime2 * y.std()
