import chex, jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class RBF(eqx.Module):
    grid: chex.Array
    inv_denominator: chex.Array
    w: chex.Array
    start_val: chex.Array

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 5,
        inp_dim: int = 1,
        out_dim: int = 1,
        scale: int = 0.1,
        key: chex.PRNGKey = jr.key(14),
    ):
        super().__init__()
        iKey, sKey, wKey = jr.split(key, 3)
        self.grid = jnp.linspace(grid_min, grid_max, num_grids)
        self.inv_denominator = jnp.ones((num_grids, out_dim, inp_dim))
        self.start_val = jnp.ones((num_grids, out_dim, inp_dim))
        self.w = scale * jr.truncated_normal(wKey, shape=(num_grids, out_dim, inp_dim), lower=-2, upper=2)

    def __call__(self, x):
        return self.w * (
            self.start_val - jax.lax.square(jax.lax.tanh((x - self.grid) * self.inv_denominator))
        )  # RSWAF function
