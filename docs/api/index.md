# API Reference

gaussx is structured linear algebra, Gaussian distributions, and exponential-family
primitives for JAX, built on [lineax](https://github.com/patrick-kidger/lineax),
[Equinox](https://github.com/patrick-kidger/equinox), and
[matfree](https://github.com/pnkraemer/matfree). The reference is organised by the
package's layered architecture rather than dumped as one flat page:

| Section | Layer | What's inside |
|---------|-------|---------------|
| [Primitives](primitives.md) | 0 | Pure functions with structural dispatch — `solve`, `logdet`, `cholesky`, `trace`, `diag`, `sqrt`, `inv`, `eig`, `svd`, root decompositions |
| [Operators & Tags](operators.md) | 1 | `Kronecker`, `BlockDiag`, `LowRankUpdate`, `Toeplitz`, block-tridiagonal and kernel operators, plus the structural tags that drive dispatch |
| [Solvers & Preconditioners](solvers.md) | 1.5 | Solver strategy objects (`DenseSolver`, `CGSolver`, `BBMMSolver`, SLQ logdets), the `linear_solve` front door, and preconditioners |
| [Linear-Algebra Utilities](linalg.md) | — | `safe_cholesky`, `symmetrize`, Woodbury and Schur identities, matrix-RHS solves, tridiagonal solves |
| [Distributions & Exponential Family](distributions.md) | 2 | `MultivariateNormal` / `MultivariateNormalPrecision`, Gaussian sugar ops, KL divergences, natural-parameter conversions |
| [Gaussian Processes](gp.md) | 3 | Conditioning, whitening, prediction caches, Matheron updates, ELBOs, LOVE / LOO, OILMM projections |
| [Kernels & Approximations](kernels.md) | 3 | Nyström and RFF operators, EigenPro preconditioning, HSIC / MMD, grid + interpolation helpers |
| [Quadrature & Moment Matching](quadrature.md) | 3 | Integrators (Gauss-Hermite, unscented, Taylor, MC), likelihoods, kernel expectations, uncertain-input GP prediction |
| [State-Space Models & Kalman](ssm.md) | 3 | SDE kernels, Kalman filter / RTS smoother (sequential, parallel, infinite-horizon), SpInGP, CVI sites |
| [Bayesian Inference & Ensembles](inference.md) | 3 | Bayesian linear regression, Newton / natural-gradient updates, ensemble Kalman primitives (localization, inflation, ETKF) |

## Conventions

A few patterns hold across the whole package:

- **Operators are lineax operators.** Every structured matrix extends
  [`lineax.AbstractLinearOperator`](https://docs.kidger.site/lineax/api/operators/)
  and is an immutable `equinox.Module` pytree — safe under `jit` / `grad` / `vmap`.
  Dense matrices enter the system as `lx.MatrixLinearOperator(A, tags)`.

- **Tags drive dispatch.** Primitives inspect operator *structure* (Kronecker,
  block-diagonal, low-rank, …) and *properties*
  (`lineax.positive_semidefinite_tag`, symmetric, triangular) to pick the cheapest
  algorithm. Tag your operators — an untagged dense PSD matrix falls back to LU
  where a tagged one gets Cholesky.

- **`solver=None` means structural dispatch.** Functions that accept an optional
  `solver:`[`AbstractSolverStrategy`](solvers.md) use the structure-aware default
  when it is `None`; pass `CGSolver()`, `BBMMSolver()`, or a `ComposedSolver` to
  override the numerical path without touching the math.

- **Lazy over dense.** Primitives like `inv`, `sqrt`, and `cholesky` return
  *operators*, not arrays, wherever structure allows; nothing is materialized until
  `.as_matrix()` is called. When a structured path has to densify, a
  `DenseFallbackWarning` is emitted.

- **Pure functions.** Outside the operator classes everything is a pure function:
  arrays and operators in, arrays and operators out. PRNG keys are explicit
  arguments for every stochastic routine.

Every public class and function carries a Google-style docstring with shapes in
[jaxtyping](https://docs.kidger.site/jaxtyping/) notation; tensor contraction and
reshaping inside the package go through [einx](https://github.com/fferflo/einx).
