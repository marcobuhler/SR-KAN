# SR-KAN: Symbolic Regression via Kolmogorov-Arnold Networks

This repo is the code used for the publication KAN-SR: A Kolmogorov-Arnold Network Guided Symbolic Regression Framework (https://arxiv.org/abs/2509.10089). **SR-KAN** is a Symbolic Regression framework built on **JAX**. It leverages the interpretable structure of **Kolmogorov-Arnold Networks (KANs)** combined with physics-inspired simplification strategies (symmetry and separability detection) to recover closed-form mathematical expressions from data.

Unlike black-box models, SR-KAN is designed to discover the underlying governing equations of a system, prioritizing parsimony and physical interpretability.

The current implementation is limited to single-layer KANs, but will be expanded in the future to deeper structures.

## Features

* **KAN-Based Search:** Utilizes Summation, Multiplication, and composite KANs to approximate complex functions.
* **Physics-Inspired Simplification:** Automatically detects symmetries and separabilities to decompose high-dimensional problems into simpler sub-problems.
* **Robust Symbolic Extraction:** Converts continuous neural approximations into discrete symbolic expressions using a customizable library of functions.
* **End-to-End Optimization:** Includes brute-force search for small dimensions, constant perturbation (random restart), and backward elimination (sparsity pruning).
* **JAX Accelerated:** Fully differentiable and GPU/TPU compatible for fast training.

## Example
```python 
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sklearn.metrics import r2_score
from symkan import regressor, SympyEvaluator
```
# 1. Configuration: Enable 64-bit precision for scientific accuracy
```python 
jax.config.update("jax_enable_x64", True)
```
# 2. Prepare Data
# Example: y = x0 / (1 + x1^2)
```python 
key = jr.key(42)
x_np = np.random.uniform(0.1, 3.0, (1000, 2))
y_np = x_np[:, 0] / (1 + x_np[:, 1]**2) #+0.01*np.random.normal(size=(1000,)) # Add noise

x = jnp.array(x_np)
y = jnp.array(y_np).reshape(-1, 1)
```
# 3. Initialize and Fit the Regressor
# We use a strict threshold and enable backward elimination for a clean equation
```python 
model = regressor(
    key=jr.key(0),
    result_threshold=1e-10,       # Stop if MSE < 1e-10 (change in the case of added noise)
    functions=["all"],            # Use all available library functions
    do_rounding=True,             # Round constants (e.g., 0.999 -> 1.0)
    backward_elim=True,           # Prune unnecessary terms
    manipulate_output=["inv"],    # Try fitting 1/y to handle rational functions
    combination_kan_types=[["mult", "mult"]], # Search strategy
    verbosity=1
)

print("Starting Symbolic Regression...")
expression = model.fit(x, y)

# 4. Evaluate and Print Results
print(f"\n recovered Expression: {expression}")

# Evaluate the expression numerically on the data
evaluator = SympyEvaluator(expression)
y_pred = evaluator(x, None, mse=False)

r2 = r2_score(y_np, y_pred)
print(f"R² Score: {r2:.6f}")
```

## Hyperparameter Details

The `regressor` class offers fine-grained control over the symbolic search. The parameters are organized below by their specific role in the SR-KAN pipeline.

### Search Space & Architecture

* **`functions`** (`list[str]`)
    Explicitly defines the library of elementary operations (e.g., `['sin', 'exp', 'inv', 'linear']`) available for reconstructing the final expression. This directly dictates the expressivity and interpretability of the model. Use `['all']` to include the full default library.

* **`combination_kan_types`** (`list[list[str]]`)
    Controls the structural inductive bias of the intermediate neural approximation. It specifies the architecture of the KAN layers tested during the search (e.g., `[['sum', 'mult']]` tests a summation layer followed by a multiplication layer). Multiple architectures can be implemented, simply define the architecture by listing the type of layer inside a nested list: `[['sum','sum','sum'], ['mult','sum']]` first tests for three parallel summation layers and if the threshold is not reached it tests a two parallel layers with one multiplication and one summation layer. Deeper KANs are not yet implemented, as it was not found to be needed for the experiments investigated.

* **`manipulate_output`** (`list[str]`)
    A list of transformations (e.g., `['inv', 'sqrt', 'log']`) applied to the target variable $\mathbf{y}$ prior to fitting. This allows the model to recover relationships like $y = e^{f(x)}$ by fitting $\log(y) = f(x)$.

* **`brute_force`** (`bool`)
    For low-dimensional problems ($d \le 4$), this flag enables an initial exhaustive search of simple functional forms. This allows the system to quickly identify low-complexity targets without engaging the full neural training phase.

### Simplification & Decomposition

* **`simplifications`** (`bool`)
    Enables the structural simplification pipeline. When `True`, the system trains an auxiliary neural network to detect underlying symmetries (e.g., translational, multiplicative) and variable separability. This is critical for decomposing high-dimensional problems into simpler sub-problems.

* **`simpl_threshold`** (`float`)
    *(Default: `1e-2`)* Sets the rigor of the simplification detection. It represents the maximum acceptable validation error for the auxiliary network; if the network error exceeds this threshold, the simplification hypothesis is rejected to prevent creating invalid sub-problems.

### Optimization & Convergence

* **`result_threshold`** (`float`)
    *(Default: `1e-3`)* The primary termination criterion. The search concludes successfully if a recovered symbolic expression achieves a Mean Squared Error (MSE) below this value. For noise-free scientific data, lower this value (e.g., `1e-15`) to force higher precision.

* **`n_grids`** (`list[int]`)
    *(Default: `[5, 10]`)* Controls the fitting fidelity of the spline-based KANs by setting the number of interpolation grid points. Higher values allow for fitting more complex functions but increase the risk of overfitting.

* **`regularization_params`** (`list[float]`)
    Imposes $\ell_1$ and entropy penalties on the KAN activation functions. This encourages the network to learn sparse and smooth representations, which are easier to convert into symbolic equations.

### Refinement & Pruning

* **`rand_constants`** (`bool`)
    Enables a stochastic refinement stage. If an initial candidate expression fails to meet the `result_threshold`, its constants are iteratively perturbed and re-optimized using a local random restart strategy to escape local minima.

* **`backward_elim`** (`bool`)
    Activates the "Backward Elimination" phase. After a solution is found, this routine recursively prunes terms that do not significantly contribute to predictive accuracy, enforcing Occam's razor to maximize interpretability.


### Cite
If you utilized `SR-KAN` for your own academic work, please use the following citation:

```
@misc{buhler2025,
      title={KAN-SR: A Kolmogorov-Arnold Network Guided Symbolic Regression Framework}, 
      author={Marco Andrea Bühler and Gonzalo Guillén-Gosálbez},
      year={2025},
      eprint={2509.10089},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.10089}, 
}
```