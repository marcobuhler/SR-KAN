from .kan import KAN
import equinox as eqx
import chex
import jax.random as jr


class CompositeKan(eqx.Module):
    """
    A composite model that combines multiple Kolmogorov–Arnold Networks (KANs) for function approximation.

    This class allows for the parallel use of multiple KAN models (list of KANs) and aggregates
    their outputs for enhanced representation capacity. It also supports joint regularization
    loss computation and symbolic expression extraction.

    Methods:
        __call__(y):
            Applies both KANs to input `y` and returns the sum of their outputs.

        reg_loss(x, l1=1.0, entropy=1.0, base_reg=1.0):
            Computes the sum of regularization losses from both KANs.

            Args:
                x (array-like): Input data for regularization computation.
                l1 (float): L1 regularization weight.
                entropy (float): Entropy regularization weight.
                base_reg (float): Base regularization weight.

            Returns:
                float: Combined regularization loss from both KANs.

        extract_symbolic_expression(x, y, functions, criteria="score", key=jr.key(123),
                                    sharey=True, plot=False, save=False, name="Fit"):
            Extracts and combines symbolic expressions from both KANs to represent the learned function.

            Args:
                x (chex.Array): Input features.
                y (chex.Array): Target output values.
                functions (list[str]): List of symbolic functions to use in the expression.
                criteria (str): Selection criteria for best expression ("score", etc.).
                key (chex.PRNGKey): PRNG key for randomness control.
                sharey (bool): Whether to share the target variable across terms.
                plot (bool): Whether to plot the symbolic fit.
                save (bool): Whether to save the symbolic representation.
                name (str): Name for the symbolic model (used in saving and plots).

            Returns:
                tuple[list[str], float]: A list containing the combined symbolic expression
                                         and the total MSE from both KANs.
    """

    kan: list[KAN]
    mult: bool

    def __init__(self, kan, mult):
        self.kan = kan
        self.mult = mult

    def __call__(self, y):
        if self.mult:
            predictions = 1
        else:
            predictions = 0
        for kan in self.kan:
            if self.mult:
                predictions *= kan(y)
            else:
                predictions += kan(y)
        return predictions

    def reg_loss(self, x, l1: int = 1.0, entropy: int = 1.0, base_reg: int = 1.0):
        reg = 0
        for kan in self.kan:
            reg += kan.reg_loss(x, l1, entropy, base_reg)
        return reg

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
        if self.mult:
            expr = 1
        else:
            expr = 0

        mse = 0

        for kan in self.kan:
            temp_expr, temp_mse = kan.extract_symbolic_expression(
                x, y, functions, criteria, key, sharey, plot, save, name
            )
            if self.mult:
                expr *= temp_expr[0]
            else:
                expr += temp_expr[0]
            mse += temp_mse[0]
        return [expr], mse
