from .rbf import RBF
from typing import Union, Callable
import equinox as eqx
import jax.nn as jnn, jax.random as jr
import jax.numpy as jnp
import jax, chex


class KANLayer(eqx.Module):
    """
    A Kolmogorov Arnold Network (KAN) layer combining Radial Basis Functions (RBFs) with a base neural network layer.

    This layer applies a learnable RBF transformation to the input, optionally combines it with a base neural
    network output, and returns the result.

    Attributes:
        rbf (RBF): The radial basis function transformation component.
        base_activation (Callable): The activation function used in the base network path.
        base_linear (eqx.nn.Linear): Linear layer applied to the activated inputs in the base path.
        use_base_update (bool): Whether to include the base path output in the final result.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
    """

    rbf: RBF
    base_activation: Callable
    base_linear: eqx.nn.Linear
    use_base_update: bool
    input_dim: int
    output_dim: int

    def __init__(
        self,
        input_dim: Union[int, list],
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int = 5,
        base_activation=jnn.silu,
        use_base_update: bool = True,
        scale: float = 0.1,
        *,
        key,
    ) -> None:
        """
        Initializes a KANLayer with given dimensions and RBF configuration.

        Args:
            input_dim (Union[int, list]): Number of input features.
            output_dim (int): Number of output features.
            grid_min (float): Minimum value for RBF grid initialization.
            grid_max (float): Maximum value for RBF grid initialization.
            num_grids (int): Number of grid points per input dimension. Default is 5.
            base_activation (Callable): Activation function for base network. Default is `jax.nn.silu`.
            use_base_update (bool): Whether to include the base linear output. Default is True.
            scale (float): Scaling factor for RBF initialization. Default is 0.1.
            key: JAX PRNG key for random initialization.
        """
        super().__init__()

        sum_key, base_key = jr.split(key, 2)

        if grid_max == None and grid_min == None:
            grid_min = -input_dim
            grid_max = input_dim
        grid_max = grid_max * jnp.ones((output_dim, input_dim))
        grid_min = grid_min * jnp.ones((output_dim, input_dim))

        self.rbf = RBF(grid_min, grid_max, num_grids, input_dim, output_dim, scale, sum_key)
        self.base_activation = base_activation
        self.use_base_update = use_base_update
        self.base_linear = eqx.nn.Linear(int(input_dim), int(output_dim), use_bias=False, key=base_key)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x: chex.Array, mult: bool = False):
        """
        Applies the KAN layer to an input tensor `x`.

        Args:
            x (chex.Array): Input array of shape (batch_size, input_dim).
            mult (bool): If True, compute product over RBF outputs; otherwise sum. Default is False.

        Returns:
            chex.Array: Transformed output of shape (batch_size, output_dim).
        """
        if mult:
            ret = self.rbf(x).sum(0)
            ret = jnp.prod(ret, -1)
        else:
            ret = self.rbf(x).sum((0, -1))
        if self.use_base_update:
            ret = ret + self.base_linear(self.base_activation(x))
        return ret

    def get_response(self, x_range: list, input_index: int, output_index: int, num_pts: int = 100):
        """
        RBF response for a selected input and output index over a range of values.

        Args:
            x_range (list): Range `[min, max]` for the selected input feature.
            input_index (int): Index of the input dimension to plot.
            output_index (int): Index of the output dimension to observe.
            num_pts (int): Number of points to evaluate in the range. Default is 100.

        Returns:
            Tuple[chex.Array, chex.Array]: The x-values from the specified input index and
            corresponding transformed y-values from the specified output index.
        """
        assert input_index < self.input_dim, "Input index out of range"
        assert output_index < self.output_dim, "Output index out of range"
        x = jnp.linspace(*x_range, num_pts).reshape(num_pts, self.input_dim)
        y = jax.vmap(self.rbf)(x).sum(1)[:, output_index]
        if self.use_base_update:
            y += self.base_linear.weight[output_index] * (self.base_activation(x))
        return x[:, input_index], y[:, input_index]
