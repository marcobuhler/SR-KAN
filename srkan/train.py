import equinox as eqx
import jax, optax, chex, optimistix
import jax.numpy as jnp, jax.random as jr
from typing import Union

from .model.kan import KAN
from .utils import dataloader


def fit_adam(
    model: KAN,
    x: chex.Array,
    y: chex.Array,
    steps: int = 5000,
    lr: Union[list, float] = [1e-2, 1e-6],
    batch_size: int = 32,
    lamb: float = 1e-3,
    l1: float = 1.0,
    entropy: float = 2.0,
    base_reg: float = 1.0,
    key: chex.PRNGKey = jr.key(2),
    verbose: bool = False,
) -> KAN:
    "Trains a KAN model"

    @eqx.filter_value_and_grad
    def grad_loss(model: KAN, x: chex.Array, y: chex.Array):
        y_pred = jax.vmap(model)(x)
        reg_loss = model.reg_loss(x, l1, entropy, base_reg)
        return jnp.mean(jax.lax.abs(y - y_pred)) + lamb * reg_loss

    @eqx.filter_jit
    def make_step(xi: chex.Array, yi: chex.Array, model: KAN, opt_state):
        loss, grads = grad_loss(model, xi, yi)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    if type(lr) == list:
        assert len(lr) == 2, "Provide two lr for linear schedule"
        schedule = optax.linear_schedule(lr[0], lr[1], steps)
        optim = optax.adam(schedule)
    else:
        assert type(lr) == float, "Provide a float for the learning rate"
        optim = optax.adam(lr)

    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for step, (xi, yi) in zip(range(steps), dataloader((x, y), batch_size, key=key)):
        loss, model, opt_state = make_step(xi, yi, model, opt_state)
        if (step % 1000) == 0 or step == steps - 1:
            if verbose:
                print(f"Step: {step}, Loss: {loss}")

    return model


def fit_bfgs(
    model: KAN,
    x: chex.Array,
    y: chex.Array,
    lamb: float = 1e-3,
    l1: float = 1.0,
    entropy: float = 2.0,
    base_reg: float = 1.0,
    max_steps: int = 8192,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    verbose: bool = False,
) -> KAN:
    "Trains a KAN model"

    @eqx.filter_jit
    def grad_loss(params, args):
        static, x, y = args
        model = eqx.combine(params, static)
        y_pred = jax.vmap(model)(x)
        reg_loss = model.reg_loss(x, l1, entropy, base_reg)
        return jnp.mean(jax.lax.abs(y - y_pred)) + lamb * reg_loss

    params, static = eqx.partition(model, eqx.is_inexact_array)

    solver = optimistix.BFGS(rtol, atol, norm=optimistix.two_norm)
    sol = optimistix.minimise(grad_loss, solver, params, args=(static, x, y), throw=False, max_steps=max_steps)
    model = sol.value
    if verbose:
        print(f"Final Loss BFGS: {grad_loss(model,args=(static, x, y) )}")
        print(f"Steps taken: {sol.stats["num_steps"]}")
    model = eqx.combine(model, static)
    return model


def fit_kan(
    model: KAN,
    x: chex.Array,
    y: chex.Array,
    verbose: bool = False,
    key: chex.PRNGKey = jr.key(3),
    adam: bool = True,
    bfgs: bool = True,
    regularization_params: list[float] = [1e-4, 1, 2, 1],
):
    lamb, l1, entropy, base_reg = regularization_params
    if adam:
        model = fit_adam(model, x, y, verbose=verbose, key=key, lamb=lamb, l1=l1, entropy=entropy, base_reg=base_reg)
    if bfgs:
        model = fit_bfgs(model, x, y, verbose=verbose, lamb=lamb, l1=l1, entropy=entropy, base_reg=base_reg)
    return model
