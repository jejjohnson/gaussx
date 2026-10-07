# API Reference

gaussx is structured linear algebra, Gaussian distributions, and exponential-family
primitives for JAX, built on [lineax](https://github.com/patrick-kidger/lineax),
[Equinox](https://github.com/patrick-kidger/equinox), and
[matfree](https://github.com/pnkraemer/matfree). The reference is organised by the
package's layered architecture rather than dumped as one flat page:

| Section | Layer | What's inside |
|---------|-------|---------------|
| [Primitives](primitives.md) | 0 | Pure functions with structural dispatch — `solve`, `logdet`, `cholesky`, `trace`, `diag`, `sqrt`, `inv`, `eig`, `svd`, root decompositions |
| [Operators & Tags](operators.md) | 1 | `Kronecker`, `BlockDiag`, `LowRankUpdate`, `Toeplitz`, block-tridiagonal, interpolated and masked operators, grid helpers, plus the structural tags that drive dispatch |
| [Sparse Operators](sparse.md) | 1 | `SparseOperator` on a static, hashable `SparsityPattern`: traced values, host-side symbolic pattern algebra (`union`, `congruence`); sparse Cholesky with a cached symbolic analysis, Takahashi selected inverse and exact gradients |
| [GMRF Precision Builders](gmrf.md) | 1 | Precisions for latent Gaussian models: iid, RW1 / RW2, AR(1), Besag / BYM2 with `generalized_variance_scale`, SPDE Matérn on meshes and grids, `fem_matrices`, `fem_projector` |
| [Solvers & Preconditioners](solvers.md) | 1.5 | Solver strategy objects (`DenseSolver`, `CGSolver`, `BBMMSolver`, SLQ logdets), the `linear_solve` front door, and preconditioners |
| [Linear-Algebra Utilities](linalg.md) | — | `safe_cholesky`, `symmetrize`, Woodbury and Schur identities, matrix-RHS solves, tridiagonal solves |
| [Sketching](sketching.md) | — | Random subspace embeddings: Gaussian, orthonormal, sparse-sign, SRHT and row-sampling sketches; `hadamard_transform` |
| [Randomized Linear Algebra](randomized.md) | — | Range finder, randomized SVD / eigh, randomized Nyström (`randomized_nystrom`), randomly pivoted Cholesky (`rp_cholesky`) with landmark pivots |
| [Distributions & Exponential Family](distributions.md) | 2 | `MultivariateNormal` / `MultivariateNormalPrecision`, Gaussian sugar ops, KL divergences, natural-parameter conversions |
| [Gaussian Processes](gp.md) | 3 | Conditioning, whitening, prediction caches, Matheron updates, ELBOs, LOVE / LOO, OILMM projections |
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
  `.as_matrix()` is called. Where a structured operator nevertheless has to be
  materialized — `cholesky` of a `SumOfKroneckers` — a `DenseFallbackWarning` points
  you at the matrix-free alternative.

- **Pure functions.** Outside the operator classes everything is a pure function:
  arrays and operators in, arrays and operators out. PRNG keys are explicit
  arguments for every stochastic routine.

### Naming

Each rule below is enforced for new names. Every rename it implied keeps the
old name as a deprecated alias: `gaussx.<old>` returns the same object as the
new name and emits a `DeprecationWarning` naming the replacement. The aliases
will be removed in gaussx 0.7.0.
`tests/test_naming.py` checks the mechanical parts (CamelCase and spelling).

| Family | Rule | Renamed (old → new) |
|---|---|---|
| Solves | `solve_<rhs-shape>` when variants differ only in the right-hand side's shape (`solve_columns`, `solve_rows`, `solve_matrix`); `<structure-or-algorithm>_solve` when the name says which structure or algorithm is used (`woodbury_solve`, `kronecker_sum_solve`, `tridiagonal_solve`, `discrete_lyapunov_solve`, `linear_solve`) | `solve_tridiagonal` → `tridiagonal_solve`, `solve_tridiagonal_batched` → `tridiagonal_solve_batched` |
| KL divergence | Every KL function computes **KL(first ‖ second)**, and its docstring's first line says so. Operator-level Gaussian helpers use the `gaussian_*` prefix (`gaussian_log_prob`, `gaussian_entropy`, `gaussian_kl`). `gauss_kl` keeps its GPflow name; `kl_divergence` is the exponential-family form; `AbstractMultivariateNormal.kl` is the distribution-level form | `dist_kl_divergence` → `gaussian_kl` |
| Parameter conversions | `mean_cov` = `(mean, covariance operator)`; `mean_chol` = `(mean, lower Cholesky factor array)` | `meanvar_to_natural` → `mean_chol_to_natural`, `natural_to_meanvar` → `natural_to_mean_chol`, `meanvar_to_expectation` → `mean_chol_to_expectation`, `expectation_to_meanvar` → `expectation_to_mean_chol` |
| Spelling | US English (`-ize`, `-ization`) in public identifiers, matching jax, numpy and scipy and the majority of existing names (`symmetrize`, `localization_matrix`, `randomized_svd`, …). Prose and private names may use either | `DiagonalisedOperator` → `DiagonalizedOperator`, `as_diagonalised` → `as_diagonalized`, `discretise_mfd` → `discretize_mfd`, `discretise_mfd_sequence` → `discretize_mfd_sequence` |
| Result containers | `*Result`: immutable output of a one-shot computation; `*State`: the carry of a sequential or recursive algorithm; `*Cache`: a precomputation reused across later calls; `*Decomposition`: a matrix factorization; `*Params`: model parameters; domain nouns (`GaussianSites`) are allowed | `EigenFactorization` → `EigenDecomposition` |
| Classes vs factories | CamelCase is reserved for classes, so `isinstance` works on every CamelCase name. A function that builds and returns some other type is snake_case | `Circulant` → `circulant`, `SumOperator` → `sum_operator`, `ScaledOperator` → `scaled_operator`, `ProductOperator` → `product_operator` |

### Deprecation policy

gaussx is pre-1.0, but pyrox and other downstream code pin against it, so
deprecated APIs are removed on a schedule rather than ad hoc:

1. **Every deprecation names its removal version** in the warning message and
   in the docstring, together with its replacement (for example, "SumKronecker
   is deprecated and will be removed in gaussx 0.7.0; use SumOfKroneckers").
   Warnings go through `gaussx._deprecation` (`warn_deprecated`,
   `renamed_kwargs`, or the `RENAMED` table for renamed public names), so
   they are a `GaussxDeprecationWarning`, a `DeprecationWarning` subclass,
   attributed to the caller's line.
2. **Window.** A deprecation is removed in the first minor release (`0.x.0`)
   after the release that first ships its warning. Before 1.0, release-please
   bumps only the patch version for `feat`/`fix`, so a minor release happens
   only with a breaking change. The PR that removes the deprecated code is
   that breaking change (`feat!:`/`refactor!:`). Every deprecation current
   today is due in **gaussx 0.7.0**.
3. **Guard tests.** `tests/test_deprecations.py` reads every
   `warn_deprecated` call in the package. It fails if a message names no
   removal version, and it fails once `gaussx.__version__` reaches a version
   that a message promises, so CI goes red on the release that is due to
   remove the code.
4. **Docs.** Deprecated names leave the main `members:` lists and move to a
   "Deprecated aliases" block at the end of their page.
5. **Changelog.** A PR that only deprecates uses the `deprecate:` commit
   type, which release-please lists under "Deprecations". A `feat`/`fix` PR
   that also deprecates something adds a `deprecate(...)` line to its squash
   commit through a `BEGIN_COMMIT_OVERRIDE` block in the PR body.

Every public class and function carries a Google-style docstring with shapes in
[jaxtyping](https://docs.kidger.site/jaxtyping/) notation; tensor contraction and
reshaping inside the package go through [einx](https://github.com/fferflo/einx).

## See also

- [Architecture](../architecture.md) — the layered stack, the dispatch flow chart,
  per-primitive fast-path coverage, and where gaussx sits relative to
  [pyrox-gp](https://github.com/jejjohnson/pyrox),
  [finitevolX](https://github.com/jejjohnson/finitevolX), and
  [spectraldiffx](https://github.com/jejjohnson/spectraldiffx).
- [Vision](../vision.md) — why the library exists and what it deliberately is not.
- [Unified Solvers](../design/unified-solvers.md) — how the solver substrate is
  shared with the PDE packages.
