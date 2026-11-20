# gaussx

% The shared sections below are included from README.md (gh-399); edit them there.

```{include} ../README.md
:start-after: <!-- --8<-- [start:intro] -->
:end-before: <!-- --8<-- [end:intro] -->
```

**New here?** Start with the [Vision](vision.md) to understand why gaussx exists, then read the [Architecture](architecture.md) to see how it's organized.

```{include} ../README.md
:start-after: <!-- --8<-- [start:install] -->
:end-before: <!-- --8<-- [end:install] -->
```

```{include} ../README.md
:start-after: <!-- --8<-- [start:quickstart] -->
:end-before: <!-- --8<-- [end:quickstart] -->
```

```{include} ../README.md
:start-after: <!-- --8<-- [start:gp-example] -->
:end-before: <!-- --8<-- [end:gp-example] -->
```

```{include} ../README.md
:start-after: <!-- --8<-- [start:inside] -->
:end-before: <!-- --8<-- [end:inside] -->
```

```{include} ../README.md
:start-after: <!-- --8<-- [start:api-notes] -->
:end-before: <!-- --8<-- [end:api-notes] -->
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

The full list is in the Examples section of the navigation.

## Links

- [API reference](https://jejjohnson.github.io/gaussx/reference/)
- [GitHub](https://github.com/jejjohnson/gaussx)
