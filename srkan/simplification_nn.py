import equinox as eqx
import jax, chex, optax
import jax.numpy as jnp, jax.random as jr

from .utils import dataloader


class Simplification_Net(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_size: int, out_size: int, width_size: int = None, depth: int = 4, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16 * in_size if width_size is None else width_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, y):
        return self.mlp(y)


def fit_helper(
    model: Simplification_Net,
    x: chex.Array,
    xval: chex.Array,
    y: chex.Array,
    yval: chex.Array,
    steps: int = 2000,
    lr: list[float] = [1e-2, 1e-6],
    batch_size: int = 32,
    seed: int = 123,
    verbose: bool = False,
) -> Simplification_Net:

    @eqx.filter_value_and_grad
    def grad_loss(model: Simplification_Net, x: chex.Array, y: chex.Array):
        y_pred = jax.vmap(model)(x)
        reg = 0
        for l in model.mlp.layers:
            reg += jnp.abs(l.weight).mean()
            reg += jnp.abs(l.bias).mean()
        return jnp.mean(jnp.abs(y - y_pred)) + 1e-3 * reg

    @eqx.filter_jit
    def make_step(xi: chex.Array, yi: chex.Array, model: Simplification_Net, opt_state):
        loss, grads = grad_loss(model, xi, yi)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    schedule = optax.linear_schedule(lr[0], lr[1], steps)
    optim = optax.adabelief(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for step, (xi, yi) in zip(range(steps), dataloader((x, y), batch_size, key=jr.key(seed))):
        loss, model, opt_state = make_step(xi, yi, model, opt_state)
        if (step % 1000) == 0 or step == steps - 1:
            if verbose:
                print(f"Step: {step}, Loss: {loss}")
                print(f"Val loss: {jnp.abs(jax.vmap(model)(xval)-yval).mean()}")
    if verbose:
        print(f"Final loss {loss}")
    return model, loss
