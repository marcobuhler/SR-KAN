import equinox as eqx
import math, sympy, jax, chex
import jax.random as jr, jax.numpy as jnp, jax.nn as jnn
import matplotlib.pyplot as plt
from typing import Callable, List, Union

from ..function_libraries.univariate.function_lib import fit_function_library
from .kan_layer import KANLayer
from ..function_libraries.function_utils import SympyEvaluator, optimize_expr_constants


class KAN(eqx.Module):
    """
    Main KAN class

    Parameters
    ----------
    layers_hidden : List[int]
        A list of integers specifying the number of neurons per layer, including input and output dimensions.
    grid_min : Union[float, list, None], optional
        Minimum values for each layer's grid, used in defining the RBF kernels. Can be a float or list per layer.
    grid_max : Union[float, list, None], optional
        Maximum values for each layer's grid. Should match `grid_min` in shape.
    num_grids : int, default=5
        Number of grid points used per dimension in the RBF basis functions.
    use_base_update : bool, default=True
        Whether to use the base linear transformation along with the RBF representation in each layer.
    base_activation : Callable, default=jax.nn.leaky_relu
        Activation function applied before the base linear transformation.
    scale : float, default=0.01
        Scaling factor for initialization of parameters.
    variable_names : list[str], optional
        List of names for input variables, used for symbolic expression extraction and visualization.
    key : chex.PRNGKey
        JAX random key used to initialize the layers and weights.

    Attributes
    ----------
    layers : eqx.nn.Sequential
        A sequence of `KANLayer`s that define the model.
    grid_min, grid_max : float or list
        Grid range for each layer.
    layer_size : list
        Architecture specification of the model (same as `layers_hidden`).
    num_grids : int
        Number of grid points for each kernel.
    use_base_update : bool
        Indicates if base linear update is used alongside RBF.
    base_activation : Callable
        Activation function used before base linear transformation.
    scale : float
        Parameter scaling factor.
    key : chex.PRNGKey
        Random key used for reproducibility.
    variable_names : list[str]
        Variable names used in symbolic expression extraction.
    """

    layers: eqx.nn.Sequential
    grid_min: float
    grid_max: float
    layer_size: list
    num_grids: float
    use_base_update: bool
    base_activation: Callable
    scale: float
    key: chex.PRNGKey
    variable_names: list[str]
    mult: bool

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: Union[float, list, None] = None,
        grid_max: Union[float, list, None] = None,
        num_grids: int = 5,
        use_base_update: bool = True,
        base_activation=jnn.leaky_relu,
        scale: float = 0.01,
        variable_names: list[str] = None,
        mult: bool = False,
        *,
        key,
    ) -> None:
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.layer_size = layers_hidden
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.scale = scale
        self.key = key
        self.variable_names = variable_names
        keys = jr.split(key, len(layers_hidden))
        self.mult = mult

        assert len(grid_max) == len(grid_min), "Provide the same number of grid_min as grid_max"
        assert (
            type(layers_hidden[0]) == int and type(layers_hidden[-1]) == int
        ), "Input layer and output layer can only be integer valued"

        if not len(layers_hidden[:-1]) == len(grid_min) == len(grid_max):
            grid_max = [*grid_max, *[None for _ in range(len(layers_hidden[1:-1]))]]
            grid_min = [*grid_min, *[None for _ in range(len(layers_hidden[1:-1]))]]

        self.layers = eqx.nn.Sequential(
            [
                KANLayer(
                    jnp.array(in_dim),
                    jnp.array(out_dim),
                    grid_min=gmin,
                    grid_max=gmax,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    scale=scale,
                    key=layer_key,
                )
                for in_dim, out_dim, gmin, gmax, layer_key in zip(
                    layers_hidden[:-1], layers_hidden[1:], grid_min, grid_max, keys
                )
            ]
        )

    def __call__(self, y: chex.Array, args=None):
        for layer in self.layers:
            y = layer(y, self.mult)
        return y

    def reg_loss(self, x, l1: int = 1.0, entropy: int = 1.0, base_reg: int = 1.0):
        """
        Computes a regularization loss for the KAN model to encourage sparsity, compactness, and interpretability.

        Parameters
        ----------
        x : chex.Array
            Input data array used to compute the activations across all layers.
        l1 : float, default=1.0
            Weight for the L1 regularization term, applied to both the RBF activations and kernel weights.
        entropy : float, default=1.0
            Weight for the entropy-based regularization term, which promotes sparse and diverse feature usage.
        base_reg : float, default=1.0
            Additional L1 penalty applied to the base linear transformation weights (if used).

        Returns
        -------
        float
            The total regularization loss, combining RBF activation sparsity, entropy penalties, and optional base weight penalties.

        """

        _, xout = jax.vmap(self.get_x_range)(x)
        reg = 0
        for i, layer in enumerate(self.layers):
            vec = jnp.abs(xout[i]).mean(0)
            rbf = vec.sum()  # + jnp.abs(layer.rbf.w).sum()

            p_row = vec / (jnp.sum(vec, 1, keepdims=True) + 1e-4)
            p_col = vec / (jnp.sum(vec, 0, keepdims=True) + 1e-4)
            entropy_row = -jnp.mean(jnp.sum(p_row * jnp.log(p_row + 1e-4), axis=1))
            entropy_col = -jnp.mean(jnp.sum(p_col * jnp.log(p_col + 1e-4), axis=0))
            reg = reg + l1 * rbf + entropy * (entropy_row + entropy_col)
            if layer.use_base_update:
                reg = reg + base_reg * jnp.abs(layer.base_linear.weight).sum()

        return reg

    def get_x_range(self, y: chex.Array) -> tuple[list]:
        """
        Propagates input `y` through the network while capturing intermediate representations from each layer,
        specifically the RBF activations before and after base linear transformations.

        Parameters
        ----------
        y : chex.Array
            The input array to the model (e.g., a batch of samples). Shape should match the model's input layer.

        Returns
        -------
        tuple of lists
            - `xin`: A list of inputs to each layer (before transformation).
            - `xout_spline`: A list of intermediate RBF outputs, summed over the batch dimension,
            capturing the contribution of each kernel before any base linear transformation.

        Notes
        -----
        - For each layer, RBF activations are computed, and the sum is stored.
        - If `use_base_update` is enabled, the base linear transformation is applied after the RBF computation.
        - This method is primarily used for symbolic analysis and regularization, where understanding the
        internal activations is important.
        """

        xin, xout_rbf = [], []
        for layer in self.layers:
            xin.append(y)
            rbf = layer.rbf(y)
            xout_rbf.append(rbf.sum(axis=0))
            if self.mult:
                ret = rbf.sum(axis=0)
                rbf = jnp.prod(ret, axis=-1)
            else:
                rbf = rbf.sum(axis=(0, -1))
            if layer.use_base_update:
                y = rbf + layer.base_linear(layer.base_activation(y))
            else:
                y = rbf

        return xin, xout_rbf

    def extract_symbolic_expression(
        self,
        x: chex.Array,
        y: chex.Array,
        functions: list[str],
        criteria: str = "score",
        key: chex.PRNGKey = jr.key(123),
        sharey: bool = True,
        plot: bool = False,
        save: bool = False,
        name: str = "Fit",
    ):
        """
        Extracts symbolic expressions that approximate each layer of the trained KAN model
        by fitting interpretable functions to intermediate representations using least squares fit.

        Parameters
        ----------
        x : chex.Array
            Input data array (e.g., features) used to evaluate and extract symbolic layer approximations.
        y : chex.Array
            Target data array used for symbolic expression fitting and final evaluation.
        functions : list[str]
            List of symbolic function names or (e.g., "square", "inv_x") to use during fitting.
        criteria : str, default="score"
            Selection criterion for choosing the best symbolic expression per node:
                - "best": selects the expression with the lowest mean squared error (MSE).
                - "score": selects based on MSE scaled by expression complexity.
        key : chex.PRNGKey, default=jr.key(123)
            JAX random key used during fitting for reproducibility.
        sharey : bool, default=True
            If plotting is enabled, determines whether subplots share the same y-axis scale.
        plot : bool, default=False
            Whether to generate plots comparing actual node outputs and symbolic fits.
        save : bool, default=False
            Whether to save the generated plots as `.svg` files.
        name : str, default="Fit"
            Prefix for saving plot filenames when `save=True`.

        Returns
        -------
        final_layer : list of sympy.Expr
            A list of symbolic expressions representing each output node of the network's final layer.
        expr_mse : list of float
            Mean squared errors corresponding to each symbolic expression in the final layer.

        Notes
        -----
        - The function fits candidate symbolic expressions to each internal RBF node output.
        - Symbolic expressions are propagated layer by layer to approximate the entire model.
        - Final expressions are cleaned, simplified, and variable names are substituted.
        - Expression constants are further optimized using `optimize_expr_constants`.
        - Optionally generates and saves plots comparing fitted symbolic expressions to actual node outputs.
        """
        if criteria not in ["best", "score"]:
            raise ValueError(f"Criteria '{criteria}' must be 'best' or 'score'.")

        variable_names = self.variable_names or list(range(x.shape[-1]))

        xrange, _ = jax.vmap(self.get_x_range)(x)
        curve_fits = []

        for layer_ind, layer in enumerate(self.layers):
            _xmin, _xmax = jnp.min(jnp.array(xrange[layer_ind]), axis=0), jnp.max(jnp.array(xrange[layer_ind]), axis=0)
            layer_fits = []

            for j in range(layer.output_dim.sum()):
                output_fits = []

                if plot:
                    fig, ax = plt.subplots(1, int(layer.input_dim), sharey=sharey, figsize=(layer.input_dim * 5, 5))

                for i in range(layer.input_dim):
                    x_, y_ = layer.get_response([_xmin, _xmax], i, j)
                    res, expr_func = fit_function_library(x_, y_, key, functions)

                    mse = jnp.array([r[1] for r in res])
                    expr_funcs = list(expr_func.values())
                    names = list(expr_func.keys())

                    if criteria == "best":
                        ind = jnp.nanargmin(mse)
                    else:
                        scores = jnp.array([mse[k] * 5 ** expr_funcs[k][0][1] for k in range(len(mse))])
                        ind = jnp.nanargmin(scores)

                    expr = expr_funcs[ind][1](res[ind][0], args=None, str_expr=True).replace("x_", f"x_{layer_ind}_{i}")
                    output_fits.append(sympy.simplify(expr))

                    if plot:
                        plot_ax = ax if layer.input_dim == 1 else ax[i]
                        plot_ax.plot(x_, expr_funcs[ind][1](res[ind][0], (x_, None)), label="Fitted")
                        plot_ax.plot(x_, y_, label="Actual")
                        plot_ax.set_title(f"Node {i} {j}: {names[ind]}")
                        plot_ax.legend()
                        plt.tight_layout()

                if save:
                    plt.savefig(f"{name}_layer_{layer_ind}_{j}.svg", dpi=200, transparent=True)
                if plot:
                    plt.show()

                layer_fits.append(output_fits)
            curve_fits.append(layer_fits)

        # Construct symbolic expressions for the model
        if self.mult:
            symbolic_layers = [
                [math.prod(curve_fits[ind][j]) for j in range(layer.output_dim.sum())]
                for ind, layer in enumerate(self.layers)
            ]
        else:
            symbolic_layers = [
                [sum(curve_fits[ind][j]) for j in range(layer.output_dim.sum())]
                for ind, layer in enumerate(self.layers)
            ]

        # Substitute expressions for each layer
        for ind in range(1, len(self.layers)):
            for j in range(self.layers[ind].output_dim.sum()):
                for i in range(self.layers[ind].input_dim):
                    symbolic_layers[ind][j] = symbolic_layers[ind][j].replace(
                        sympy.symbols(f"x_{ind}_{i}"), sympy.sympify(f"({symbolic_layers[ind - 1][i]})")
                    )

        # Replace variable names in the final layer
        final_layer = symbolic_layers[-1]
        for i in range(x.shape[-1]):
            for j in range(self.layers[-1].output_dim.sum()):
                final_layer[j] = final_layer[j].replace(sympy.symbols(f"x_{0}_{i}"), sympy.symbols(f"x_{i}"))

        # Optimize and compute MSE for final expressions
        jax_expressions = [optimize_expr_constants(SympyEvaluator(func), x, y) for func in final_layer]
        expr_mse = [expr(x, y, mse=True) for expr in jax_expressions]

        for i in range(x.shape[-1]):
            for j in range(self.layers[-1].output_dim.sum()):
                final_layer[j] = final_layer[j].replace(
                    sympy.symbols(f"x_{i}"), sympy.symbols(f"x_{variable_names[i]}")
                )
        return final_layer, expr_mse


def standard_sum_kan(x: chex.Array, key: chex.PRNGKey, variable_names, activation_function, num_grids) -> KAN:
    return KAN(
        [x.shape[-1], 1],
        key=key,
        num_grids=num_grids,
        base_activation=activation_function,
        scale=0.01,
        grid_max=[jnp.ceil(x).max(0)],
        grid_min=[jnp.floor(x).min(0)],
        use_base_update=True,
        variable_names=variable_names,
        mult=False,
    )


def standard_mult_kan(x: chex.Array, key: chex.PRNGKey, variable_names, activation_function, num_grids) -> KAN:
    return KAN(
        [x.shape[-1], 1],
        key=key,
        num_grids=num_grids,
        base_activation=activation_function,
        scale=1,
        grid_max=[jnp.ceil(x).max(0)],
        grid_min=[jnp.floor(x).min(0)],
        use_base_update=False,
        variable_names=variable_names,
        mult=True,
    )
