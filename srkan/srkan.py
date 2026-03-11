import jax.numpy as jnp, jax.random as jr, jax.nn as jnn
import chex, sympy, warnings, jax, copy, gc
from typing import Callable

from .function_libraries import (
    SympyEvaluator,
    optimize_expr_constants,
    create_library,
    run_brute_force,
    all_expr,
    rescale_expression_to_physical,
    integerize_exponents,
    perturb_and_optimize,
    backward_elimination,
)
from .utils import train_test_split
from .simplification_nn import Simplification_Net, fit_helper
from .separability import do_separability, check_separability
from .model import standard_mult_kan, standard_sum_kan
from .model import CompositeKan
from .symmetry import evaluate_symmetries
from .train import fit_kan


class regressor:
    """
    Symbolic regression framework leveraging Kolmogorov-Arnold Networks (KANs) and brute-force search
    to discover mathematical expressions from data.

    Attributes:
        key (chex.PRNGKey): JAX random key for stochastic operations.
        result_threshold (float): Mean Squared Error (MSE) target; fitting stops if reached.
        simpl_threshold (float): Validation error threshold for the simplification neural network.
        verbosity (int): Logging level (0: silent, 1: basic, 2: verbose).
        do_rounding (bool): If True, rounds numerical constants in the final expression.
        scale_x (bool): If True, standardizes input features before fitting.
        scale_y (bool): If True, standardizes target values before fitting.
        unscale (bool): If True, rescales the discovered expression back to physical units.
        functions (list[str]): Library of base functions allowed in symbolic expressions (e.g., "sin", "exp").
        exclude_functions (list[str]): List of functions to strictly exclude from the library.
        brute_force (bool): If True, performs an initial brute-force search for small input dimensions.
        simplifications (bool): If True, trains a helper network to detect symmetries and separability.
        manipulate_output (list[str]): List of output transformations to try (e.g., ["inv", "log"]).
        combination_kan_types (list[list[str]]): Configurations for mixed KANs (e.g., [["sum", "mult"]]).
        use_adam (bool): If True, uses the Adam optimizer for KAN training.
        use_bfgs (bool): If True, uses the BFGS optimizer for KAN fine-tuning.
        activation_function (Callable): Activation function used within KAN nodes.
        regularization_params (list[float]): Regularization coefficients for KAN training [L1, entropy, ...].
        n_grids (list[int]): Grid sizes for KAN spline interpolation [sum_kan_grid, mult_kan_grid].
        plot (bool): If True, generates plots during the fitting process.
        rand_constants (bool): If True, perturbs constants to escape local minima during refinement.
        random_iterations (int): Number of random restart iterations for constant optimization.
        backward_elim (bool): If True, iteratively prunes terms from the final expression to improve sparsity.
    """

    def __init__(
        self,
        key: chex.PRNGKey = jr.key(123),
        result_threshold: float = 1e-3,
        simpl_threshold: float = 1e-2,
        verbosity: int = 1,
        do_rounding: bool = True,
        scale_x: bool = True,
        scale_y: bool = True,
        unscale: bool = True,
        functions: list[str] = ["linear", "constant", "square", "inv_x", "michaelis_menten", "sqrt"],
        exclude_functions: list[str] = None,
        brute_force: bool = True,
        simplifications: bool = True,
        manipulate_output: list[str] = ["inv", "square", "sqrt", "log"],
        combination_kan_types: list[list[str]] = [["sum", "sum"], ["sum", "mult"], ["mult", "mult"]],
        use_adam: bool = True,
        use_bfgs: bool = True,
        activation_function: Callable = jnn.gelu,
        regularization_params: list[float] = [1e-4, 1, 2, 1],
        n_grids: list[int] = [5, 10],
        plot: bool = False,
        rand_constants: bool = True,
        random_iterations: int = 10,
        backward_elim: bool = False,
    ):
        self.key = key
        self.result_threshold = result_threshold
        self.simpl_threshold = simpl_threshold
        self.verbosity = verbosity
        self.plot = plot
        self.do_rounding = do_rounding
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.unscale = unscale
        self.all_functions = list(all_expr.keys())
        self.functions = create_library(functions, exclude_functions)
        self.best_mse = jnp.inf
        self.best_eq = None
        self.brute_force = brute_force
        self.simplifications = simplifications
        self.manipulate_output = manipulate_output
        self.activation_function = activation_function
        self.combination_kan_types = combination_kan_types
        self.adam = use_adam
        self.bfgs = use_bfgs
        self.regularization_params = regularization_params
        self.n_grids = n_grids
        self.backward_elim = backward_elim
        self.rand_constants = rand_constants
        self.random_iterations = random_iterations

        assert use_adam or use_bfgs, "Select one or both optimizers, adam and/or bfgs to train the KAN's"
        if manipulate_output is not False:
            for manipulation in manipulate_output:
                assert manipulation in [
                    "inv",
                    "square",
                    "sqrt",
                    "log",
                ], f'Manipulation has to be in ["inv", "square", "sqrt", "log"], provided was {manipulation}'
        for comb in combination_kan_types:  # Checking whether all combinations are valid
            assert all(type(t) == str for t in comb) == True, "Please provide a list[list[str]]"
            for kan_type in comb:
                assert kan_type in [
                    "sum",
                    "mult",
                ], f"Only 'sum' and 'mult' are currently valid options. Provided '{kan_type}'"

        if self.verbosity == 2:
            self.verbose = True
        else:
            self.verbose = False

    def fit(self, x: chex.Array, y: chex.Array):
        """Fit the symbolic regressor to the data and return the discovered expression.

        Args:
            x: Input features of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1) or (n_samples, n_outputs).

        Returns:
            A sympy expression representing the discovered equation, or None if no
            expression meeting the result_threshold was found.
        """
        assert y.shape[0] == x.shape[0], "Inputs do not match axis 0"
        if y.shape[-1] > 1:
            expr = [self._fit(x, y[:, [i]]) for i in range(y.shape[-1])]
            return expr
        else:
            if self.scale_x:
                xstd = x.std(0)
                variables_list = [sympy.Symbol(f"x_{i}") for i in range(len(xstd))]
                std_dev_map = dict(zip(variables_list, xstd))
            if self.scale_y:
                ystd = y.std()
            expr = self._fit(x / xstd if self.scale_x else x, y / ystd if self.scale_y else y)

            # Fail Fallback
            if expr is None:
                return None

            # Unscale data
            if self.unscale:
                if self.scale_x:
                    expr = rescale_expression_to_physical(expr, std_dev_map)
                if self.scale_y:
                    expr = (expr * ystd).doit()
                print(f"New Expression: {expr.doit()}")
            expr = integerize_exponents(expr)
            expr = SympyEvaluator(expr)

            expr = optimize_expr_constants(expr, x, y, verbose=self.verbose, tol=[1e-10, 1e-12], use_ls=True)

            # Retry with different random constants
            if expr(x, y, mse=True) > self.result_threshold and self.rand_constants:
                print("Perturbing")
                expr = perturb_and_optimize(
                    expr, self.key, x, y, self.result_threshold, self.random_iterations, self.verbose
                )
            elif jnp.isnan(expr(x, y, mse=True)) and self.rand_constants:
                print("Perturbing")
                expr = perturb_and_optimize(
                    expr, self.key, x, y, self.result_threshold, self.random_iterations, self.verbose
                )
            expr = SympyEvaluator(sympy.simplify(expr.sympy()))
            if self.backward_elim:
                expr = backward_elimination(expr, x, y, verbose=True)
            print("Final mse: ", expr(x, y, mse=True))
            expr = sympy.simplify(expr.sympy())
            print("Final equation: ", expr)
            gc.collect()
            jax.clear_caches()
            return expr

    def _fit(self, x: chex.Array, y: chex.Array):
        assert y.shape[0] == x.shape[0], "Inputs do not match axis 0"
        if y.shape[0] < 50:
            warnings.warn(
                "Input data contains less than 50 samples! The algorithm may fail depending on the problem complexity."
            )
        keys = jr.split(self.key, 3)

        # Brute force
        simplification_key, symmetry_key, separability_key, bf_key = jr.split(keys[0], 4)
        if self.brute_force and x.shape[-1] in [2, 3, 4]:
            if self.verbosity > 0:
                print("Start brute force search")
            expr = run_brute_force(x, y, bf_key)
            if self.check_success(expr, x, y):
                return expr.round() if self.do_rounding else expr.sympy()

        # Mult KAN
        if self.verbosity > 0:
            print("")
            print("Fitting multiplication KAN to input data")
        fit, _ = self.fit_and_extract_act(x, y, range(x.shape[-1]), jr.split(self.key, 1)[0], mult=True)
        expr = sympy.sympify(f"{fit[0]}")
        expr = optimize_expr_constants(SympyEvaluator(expr), x, y, use_ls=False)
        if self.check_success(expr, x, y):
            return expr.round() if self.do_rounding else expr.sympy()

        # Sum KAN
        if self.verbosity > 0:
            print("")
            print("Fitting summation KAN to input data")
        fit, _ = self.fit_and_extract_act(x, y, range(x.shape[-1]), jr.split(self.key, 1)[0])
        expr = sympy.sympify(f"{fit[0]}")
        expr = optimize_expr_constants(SympyEvaluator(expr), x, y, use_ls=False)
        if self.check_success(expr, x, y):
            return expr.round() if self.do_rounding else expr.sympy()

        # Combination KANs
        if self.verbosity > 0:
            print("")
            print("Fitting combination KANs to input data")
        for types in self.combination_kan_types:
            for multiply_kans in [False, True]:
                keys = jr.split(self.key, len(self.combination_kan_types))
                fit, _ = self.fit_ck_and_extract_act(
                    x, y, range(x.shape[-1]), jr.split(self.key, 1)[0], types=types, mult_kans=multiply_kans
                )
                expr = sympy.sympify(f"{fit[0]}")
                expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
                if self.check_success(expr, x, y):
                    return expr.round() if self.do_rounding else expr.sympy()

        # Simplifications
        if x.shape[-1] > 1 and self.simplifications:
            if self.verbosity > 0:
                print("")
                print("No simple equation found, starting to search for simplifications")
            net, val_error = self.simpl_nn_train(x, y, simplification_key, verbose=self.verbose)
            self.val_error = val_error
            if val_error < self.simpl_threshold:
                ind, categ = evaluate_symmetries(net, x, y, val_error, verbose=self.verbosity)
                if ind is not None:  # Translational symmetry found
                    expr = self._symmetry(x, y, categ, ind, separability_key)
                    if expr is not None:  # Success
                        return expr
                ind, _, categ = check_separability(net, x, y, val_error, verbose=self.verbosity)
                if ind is not None:  # Separability found
                    expr = self._separability(net, x, y, ind, categ, symmetry_key)
                    if expr is not None:  # Success
                        return expr
            else:
                if self.verbosity > 0:
                    print(val_error)
                    print("Network training failed...")

        keys = jr.split(keys[0], 4)
        if self.manipulate_output is not False:
            for manipulation in self.manipulate_output:
                if manipulation == "square":
                    if self.verbosity > 0:
                        print("")
                        print("Fitting y^2")
                    expr = self._fit_new_regressor(x, jnp.square(y), keys[0], scale=True)
                    expr = sympy.sympify(f"sqrt({expr})")
                    expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
                    if self.check_success(expr, x, y):
                        return expr.round() if self.do_rounding else expr.sympy()

                if not jnp.any(y == 0):
                    if manipulation == "inv":
                        if self.verbosity > 0:
                            print("")
                            print("Fitting 1/y")
                        expr = self._fit_new_regressor(x, 1 / y, keys[2], scale=True)
                        expr = sympy.sympify(f"1/({expr})")
                        expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
                        if self.check_success(expr, x, y):
                            return expr.round() if self.do_rounding else expr.sympy()

                if jnp.all(y > 0):
                    if manipulation == "sqrt":
                        if self.verbosity > 0:
                            print("")
                            print("Fitting sqrt(y)")
                        expr = self._fit_new_regressor(x, jnp.sqrt(y), keys[1], scale=True)
                        expr = sympy.sympify(f"({expr})^2")
                        expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
                        if self.check_success(expr, x, y):
                            return expr.round() if self.do_rounding else expr.sympy()
                    if manipulation == "log":
                        if self.verbosity > 0:
                            print("")
                            print("Fitting log(y)")
                        expr = self._fit_new_regressor(x, jnp.log(y), keys[3], scale=True)
                        expr = sympy.sympify(f"exp({expr})")
                        expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
                        if self.check_success(expr, x, y):
                            return expr.round() if self.do_rounding else expr.sympy()

        # If the result threshold is not reached:
        if self.verbosity > 0:
            print("")
            print("Returning the best so far: ", self.best_eq)
        return self.best_eq

    def _symmetry(self, x: chex.Array, y: chex.Array, categ: int, ind, key):
        xp1 = jnp.delete(x, jnp.array(ind), -1)
        if categ == 1:  # neg symmetry
            xp2 = jax.lax.sub(x[:, ind[0]], x[:, ind[1]]).reshape(-1, 1)
        if categ == 2:  # pos symmetry
            xp2 = jax.lax.add(x[:, ind[0]], x[:, ind[1]]).reshape(-1, 1)
        if categ == 3:  # mult symmetry
            xp2 = jax.lax.mul(x[:, ind[0]], x[:, ind[1]]).reshape(-1, 1)
        if categ == 4:  # div symmetry
            xp2 = jax.lax.div(x[:, ind[0]], x[:, ind[1]]).reshape(-1, 1)

        xp = jnp.hstack([xp1, xp2])
        if self.verbosity > 0:
            print("")
            print(f"Fitting new regressor with {ind}")
        expr = self._fit_new_regressor(xp, y, key)
        if expr is not None:
            expr = self.change_translational_names(expr, xp.shape[-1], ind, categ)
            expr = sympy.simplify(expr)
            expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
            if self.check_success(expr, x, y):
                return expr.round() if self.do_rounding else expr.sympy()

    def _separability(
        self, net: Simplification_Net, x: chex.Array, y: chex.Array, ind: list, categ: int, key: chex.PRNGKey
    ):
        xp1, yp1, xp2, yp2 = do_separability(net, x, y, ind, categ)
        ind_0 = jnp.array(ind[0])
        ind_1 = jnp.array(ind[1])
        keys = jr.split(key, 2)
        if self.verbosity > 0:
            print("")
            print(f"Fit new regressor to x {ind_0}")
        expr1 = self._fit_new_regressor(xp1, yp1, keys[0])
        if expr1 is not None:
            expr1 = self.change_variable_names(expr1, ind_0)
        if self.verbosity > 0:
            print("")
            print(f"Fit new regressor to x {ind_1}")
        expr2 = self._fit_new_regressor(xp2, yp2, keys[1])
        if expr2 is not None:
            expr2 = self.change_variable_names(expr2, ind_1)

        if expr1 is not None and expr2 is not None:
            expr = f"{expr1+expr2}" if categ == 0 else f"{expr1*expr2}"
            expr = sympy.simplify(expr)
            expr = optimize_expr_constants(SympyEvaluator(expr), x, y)
            if self.check_success(expr, x, y):
                return expr.round() if self.do_rounding else expr.sympy()
        return None

    def check_success(self, expr: SympyEvaluator, x: chex.Array, y: chex.Array) -> bool:
        mse = expr(x, y, mse=True)
        if self.verbosity > 0:
            print("")
            print(f"Equation Found with mse: {mse}")
            print(expr.round() if self.do_rounding else expr.sympy())

        if mse < self.best_mse:
            self.best_mse = mse
            self.best_eq = expr.round() if self.do_rounding else expr.sympy()
        return mse <= self.result_threshold

    def change_translational_names(self, expr, shape, names, categ):
        if categ == 1:
            term = sympy.symbols(f"x_{names[0]}") - sympy.symbols(f"x_{names[1]}")
        elif categ == 2:
            term = sympy.symbols(f"x_{names[0]}") + sympy.symbols(f"x_{names[1]}")
        elif categ == 3:
            term = sympy.symbols(f"x_{names[0]}") * sympy.symbols(f"x_{names[1]}")
        elif categ == 4:
            term = sympy.symbols(f"x_{names[0]}") / sympy.symbols(f"x_{names[1]}")

        n = [i for i in jnp.arange(0, shape + 1) if i not in names]
        for i in range(len(n)):
            expr = expr.replace(sympy.symbols(f"x_{i}"), sympy.symbols(f"x_old_{n[i]}"))
        expr = expr.replace(sympy.symbols(f"x_{shape-1}"), term)
        for i in range(len(n)):
            expr = expr.replace(sympy.symbols(f"x_old_{n[i]}"), sympy.symbols(f"x_{n[i]}"))
        if self.verbosity > 0:
            print("")
            print(f"Replacing x_{shape-1} with {term}")
        return expr

    def _fit_new_regressor(self, x, y, key, scale=False):
        new_reg = copy.deepcopy(self)
        new_reg.key = key
        new_reg.best_mse = jnp.inf
        new_reg.best_eq = None
        new_reg.manipulate_output = False
        new_reg.scale_x = False
        new_reg.scale_y = scale
        expr = new_reg.fit(x, y)
        del new_reg
        return expr

    @staticmethod
    def change_variable_names(expr, names):
        for i in range(len(names)):
            expr = expr.replace(sympy.symbols(f"x_{i}"), sympy.symbols(f"x_old_{i}"))
        for i in range(len(names)):
            expr = expr.replace(sympy.symbols(f"x_old_{i}"), sympy.symbols(f"x_{names[i]}"))
        return expr

    def fit_ck_and_extract_act(self, x, y, var_names, key, types, mult_kans):
        keys = jr.split(key, 2 + len(types))
        kans = []
        for i, t in enumerate(types):
            kans.append(
                standard_sum_kan(
                    x,
                    keys[i],
                    variable_names=var_names,
                    activation_function=self.activation_function,
                    num_grids=self.n_grids[0],
                )
                if t == "sum"
                else standard_mult_kan(
                    x,
                    keys[i],
                    variable_names=var_names,
                    activation_function=self.activation_function,
                    num_grids=self.n_grids[1],
                )
            )
        model = CompositeKan(kans, mult=mult_kans)
        return self._fit_and_extract(model, x, y, keys[1])

    def fit_and_extract_act(self, x, y, var_names, key, mult=False):
        keys = jr.split(key, 3)
        if mult:
            model = standard_mult_kan(
                x,
                keys[0],
                variable_names=var_names,
                activation_function=self.activation_function,
                num_grids=self.n_grids[1],
            )
        else:
            model = standard_sum_kan(
                x,
                keys[0],
                variable_names=var_names,
                activation_function=self.activation_function,
                num_grids=self.n_grids[0],
            )
        return self._fit_and_extract(model, x, y, keys[1])

    def _fit_and_extract(self, model, x, y, key):
        keys = jr.split(key, 2)
        model = fit_kan(
            model,
            x,
            y,
            verbose=self.verbose,
            key=keys[0],
            adam=self.adam,
            bfgs=self.bfgs,
            regularization_params=self.regularization_params,
        )
        fit, mse = model.extract_symbolic_expression(x, y, self.functions, plot=self.plot, key=keys[2])
        del model
        return fit, mse

    def simpl_nn_train(self, x: chex.Array, y: chex.Array, key: chex.PRNGKey, verbose: bool = False):
        split_key, model_key = jr.split(key)
        y = y / y.std()
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, split_key)
        batchsize = 32 if len(x) < 500 else int(x.shape[0] / 20)
        if self.verbosity > 0:
            print("")
            print("Starting to train the simplification network")
        net = Simplification_Net(x.shape[-1], y.shape[-1], 64, 2, key=model_key)
        net, _ = fit_helper(
            net, xtrain, xtest, ytrain, ytest, 5000, batch_size=batchsize, lr=[1e-2, 1e-4], verbose=verbose
        )
        val_error = jnp.abs(ytest - jax.vmap(net)(xtest)).mean()
        if self.verbosity > 0:
            print("Val Error: ", val_error)
        if val_error > self.simpl_threshold:
            if self.verbosity > 0:
                print("Retraining the Network for better performance")
            net, _ = fit_helper(
                net, xtrain, xtest, ytrain, ytest, 5000, batch_size=batchsize, lr=[1e-4, 1e-10], verbose=verbose
            )
            val_error = jnp.abs(ytest - jax.vmap(net)(xtest)).mean()

        return net, val_error
