# gaussx

> Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX.

Built on [lineax](https://github.com/patrick-kidger/lineax), [equinox](https://github.com/patrick-kidger/equinox), and [matfree](https://github.com/pnkraemer/matfree).

**New here?** Start with the [Vision](vision.md) to understand why gaussx exists, then read the [Architecture](architecture.md) to see how it's organized.

## Installation

```bash
pip install gaussx
```

Or with `uv`:

```bash
uv add gaussx
```

## Quickstart

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

# Structured operators
A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
B = lx.DiagonalLinearOperator(jnp.array([3.0, 4.0]))
K = gaussx.Kronecker(A, B)

# Primitives exploit structure automatically
v = jnp.ones(4)
x = gaussx.solve(K, v)       # per-factor solve
ld = gaussx.logdet(K)         # n2*logdet(A) + n1*logdet(B)
L = gaussx.cholesky(K)        # Kronecker(chol(A), chol(B))
t = gaussx.trace(K)           # trace(A) * trace(B)
```

## API Notes

Several of the newer public APIs have explicit requirements that are worth calling out up front:

- `gaussx.kronecker_posterior_predictive(...)` requires `K_test_diag_factors=` when you want predictive variances.
- `gaussx.ssm_to_naturals(...)` validates that `Q[0]` matches `P_0` so the joint prior is internally consistent.
- `gaussx.ImplicitKernelOperator(...)` only advertises symmetry and PSD to `lineax` when those tags are provided explicitly.

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

mean, var = gaussx.kronecker_posterior_predictive(
	[Kx, Ky],
	y,
	noise_var=1e-2,
	grid_shape=(nx, ny),
	K_cross_factors=[Kx_star, Ky_star],
	K_test_diag_factors=[jnp.ones(nx_star), jnp.ones(ny_star)],
)

theta_1, theta_2 = gaussx.ssm_to_naturals(A, Q, mu_0, P_0=Q[0])

kernel_op = gaussx.ImplicitKernelOperator(
	kernel_fn,
	X,
	tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
)
```

## Examples

- [Basics](notebooks/basics.ipynb) — operators, primitives, JAX transforms
- [Operator Zoo](notebooks/operator_zoo.ipynb) — every operator type with structure visualization
- [Woodbury Solve](notebooks/woodbury_solve.ipynb) — step-by-step Woodbury identity
- [Kronecker Eigendecomposition](notebooks/kronecker_eigen.ipynb) — per-factor eigen/cholesky/sqrt
- [Kernel Regression](notebooks/kernel_regression.ipynb) — GP regression with hyperparameter optimization
- [GP on a 2D Grid](notebooks/gp_2d_grid.ipynb) — Kronecker structure for spatial data
- [Sparse Variational GP](notebooks/sparse_variational_gp.ipynb) — inducing points with ELBO optimization
- [Structured GP](notebooks/structured_gp.ipynb) — Kronecker and low-rank comparison
- [Solver Comparison](notebooks/solver_comparison.ipynb) — DenseSolver vs CGSolver
- [Differentiating Through Solve](notebooks/differentiating_solve.ipynb) — jax.grad through gaussx primitives

## Links

- [API Reference](api/reference.md)
- [GitHub](https://github.com/jejjohnson/gaussx)
