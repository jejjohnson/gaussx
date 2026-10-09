---
name: structured-gaussian-linalg
description: Write covariance, precision and Gaussian code in JAX on gaussx instead of dense jnp.linalg — structured operators (Kronecker, Kronecker sums, block-diagonal, block-tridiagonal, low-rank + diagonal, Toeplitz, sparse), structure-aware solve / logdet / Cholesky / sampling, multivariate normals and KL, GP marginal likelihoods and predictions, Kalman filters and smoothers, quadrature, ensemble Kalman and natural-gradient updates. Use whenever a task solves a linear system, takes a log-determinant, factorises or samples from a covariance, or builds a GP / state-space / Bayesian model in JAX, in a project that uses (or could use) gaussx.
---

# Structured Gaussian linear algebra with gaussx

Your covariance matrix has structure. `jnp.linalg.solve`, `cho_solve` and
`slogdet` throw it away and pay O(N³); gaussx keeps it and pays for the
structure instead (a Kronecker product of two 100 × 100 factors: two small
solves, not one 10,000 × 10,000 one). Before writing a solve, a
factorisation, a Woodbury step, a Kalman loop or a Gaussian density, look it
up:

1. **The capability index** lists every public name with a one-line summary,
   grouped by layer, plus the lineax, matfree and optimistix APIs gaussx
   builds on: <https://jejjohnson.github.io/gaussx/capabilities/>. Or search
   the installed version:

   ```python
   import inspect

   import gaussx

   for name in gaussx.__all__:
       doc = (inspect.getdoc(getattr(gaussx, name)) or "").split("\n")[0]
       print(f"gaussx.{name}: {doc}")
   ```

2. Compose what exists. If something is *almost* there, pass an operator or
   a `solver=` before writing a replacement.

## Which layer to enter

| You have… | Enter at | Use |
|---|---|---|
| A matrix with structure and a right-hand side | Layer 0–1 | build the operator, then `gaussx.solve` / `logdet` / `cholesky` / `sqrt` / `inv` / `diag` / `trace` |
| A bare `v ↦ A v` function (a PDE operator, a Jacobian) | Layer 1.5 | `gaussx.linear_solve((matvec, shape), b, solver=..., preconditioner=...)` or `as_linear_operator` |
| A problem too big for exact factorisations | Layer 1.5 | `solver=gaussx.CGSolver()` / `BBMMSolver()` / `SLQLogdet()`, preconditioned by `NystromPreconditioner` / `PartialCholeskyPreconditioner` |
| A Gaussian to evaluate, condition, sample or compare | Layer 2 | `gaussian_log_prob`, `gaussian_entropy`, `gaussian_kl`, `conditional`, `sample_mvn`; as objects (NumPyro-compatible), `MultivariateNormal` / `MultivariateNormalPrecision` |
| A GP, an SSM, a quadrature or an ensemble / natural-gradient update | Layer 3 | `gaussx` GP (`predict_mean`, `collapsed_elbo`, `love_cache`, …), SSM (`kalman_filter`, `rts_smoother`, `MaternSDE`, …), quadrature (`GaussHermiteIntegrator`, …), inference (`enkf_analysis`, `blr_full_update`, …) |

## The rules your code must keep

- **Operators, not arrays.** Covariances and precisions are
  `lineax.AbstractLinearOperator`s. A dense matrix enters as
  `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`: the tag is what
  lets gaussx (and lineax) use Cholesky instead of LU and take the PSD paths.
- **Say what the structure is.** `gaussx.Kronecker(K1, K2)`, not
  `jnp.kron(K1, K2)`; `gaussx.low_rank_plus_diag(d, U, psd=True)`, not
  `jnp.diag(d) + U @ U.T`; `gaussx.BlockDiag(...)`, `gaussx.BlockTriDiag(...)`,
  `gaussx.Toeplitz(column)`, `gaussx.SparseOperator(values, pattern)`. Lineax's
  algebra (`A + B`, `2.0 * A`, `A @ B`) keeps operators lazy.
- **Never materialise to solve.** `x.as_matrix()` followed by a dense solve
  undoes all of the above. A `DenseFallbackWarning` from gaussx means a path
  had to densify, and names the matrix-free alternative.
- **Pick numerics with `solver=`, not by rewriting the math.** Functions that
  solve or take a logdet accept `solver=` (`None` = structural dispatch);
  stochastic ones (`BBMMSolver`, `SLQLogdet`) also take a `key=`.
- **JAX rules.** Everything is a pure function of operators and arrays, safe
  under `jit`, `grad` and `vmap`; PRNG keys are explicit arguments; float32
  inputs stay float32.

## Worked example

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag


def rbf(x, lengthscale):  # (n,) → (n, n)
    d = x[:, None] - x[None, :]
    return jnp.exp(-0.5 * (d / lengthscale) ** 2) + 1e-6 * jnp.eye(x.shape[0])


x1, x2 = jnp.linspace(0, 1, 30), jnp.linspace(0, 1, 40)  # a 30 × 40 grid
y = jr.normal(jr.key(0), (30 * 40,))  # (N,), N = 1200


def neg_mll(lengthscale, noise=0.1):
    # K_y = K₁ ⊗ K₂ + σ² I ⊗ I, never materialised: O(n₁³ + n₂³), not O(N³).
    K_y = gaussx.SumOfKroneckers(
        gaussx.Kronecker(
            lx.MatrixLinearOperator(rbf(x1, lengthscale), PSD),  # (30, 30)
            lx.MatrixLinearOperator(rbf(x2, lengthscale), PSD),  # (40, 40)
        ),
        gaussx.Kronecker(
            lx.MatrixLinearOperator(noise * jnp.eye(30), PSD),
            lx.MatrixLinearOperator(jnp.eye(40), PSD),
        ),
    )  # (N, N)
    return -gaussx.gaussian_log_prob(jnp.zeros_like(y), K_y, y)  # ()


value, grad = jax.jit(jax.value_and_grad(neg_mll))(0.3)  # matches the dense K_y

# Low rank + diagonal (D + U Uᵀ): Woodbury solve, determinant-lemma logdet.
U = jr.normal(jr.key(1), (1000, 5))  # (n, k)
A = gaussx.low_rank_plus_diag(jnp.full(1000, 0.5), U, psd=True)  # (n, n), O(n k²)
x = gaussx.solve(A, jnp.ones(1000))  # (n,)
ld = gaussx.logdet(A)  # ()

# A linear-Gaussian state-space model: filter, then smooth.
F = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # (N, N) transition
H = jnp.array([[1.0, 0.0]])  # (M, N) observation model
obs = jr.normal(jr.key(2), (50, 1))  # (T, M)
state = gaussx.kalman_filter(
    F, H, 0.01 * jnp.eye(2), 0.1 * jnp.eye(1), obs, jnp.zeros(2), jnp.eye(2)
)  # filtered / predicted means and covariances, log_likelihood
means, covs = gaussx.rts_smoother(state, F)  # (T, N), (T, N, N)
```

## Don't write it — use gaussx

| Don't write… | Use |
|---|---|
| `jnp.linalg.solve(K, b)`, `cho_solve(cho_factor(K), b)`, `solve_triangular` pairs | `gaussx.solve(lx.MatrixLinearOperator(K, PSD), b)`, or on the structured operator |
| `jnp.linalg.slogdet(K)[1]`, `2 * jnp.sum(jnp.log(jnp.diag(L)))` | `gaussx.logdet(op)`; from a factor you already have, `gaussx.cholesky_logdet(L)` |
| `jnp.linalg.inv(K) @ B` | `gaussx.solve_matrix(op, B)` (or `inv(op)`, which stays lazy) |
| A jitter-retry loop around `jnp.linalg.cholesky` | `gaussx.safe_cholesky(op)` / `gaussx.add_jitter(op)` |
| `jnp.kron(A, B)` then a solve / logdet / sample | `gaussx.Kronecker(A, B)`; `A ⊗ I + I ⊗ B` is `KroneckerSum`; `Σᵢ Aᵢ ⊗ Bᵢ` is `SumOfKroneckers` |
| The Woodbury identity or the matrix-determinant lemma by hand | `gaussx.low_rank_plus_diag` / `low_rank_plus_identity` / `LowRankUpdate`, `woodbury_solve` |
| `-0.5 * (r @ solve(K, r) + logdet + n log 2π)` | `gaussx.gaussian_log_prob(mean, op, y)` |
| A Gaussian KL, entropy or conditional by hand | `gaussx.gaussian_kl`, `gaussian_entropy`, `conditional` |
| `mean + L @ jr.normal(...)` sampling | `gaussx.sample_mvn(mean, op, key=key)` (structure-aware square roots) |
| A Kalman predict / update loop, an RTS smoother | `gaussx.kalman_filter`, `rts_smoother`, `parallel_kalman_filter`; SDE kernels (`MaternSDE`, …) and `discretize_mfd` for GP-as-SSM |
| A CG loop, a Lanczos loop, Hutchinson probes | `solver=gaussx.CGSolver()` / `BBMMSolver()`, `gaussx.SLQLogdet`, `gaussx.trace_and_diag`; `matfree` below them |
| Gauss–Hermite / unscented / cubature points and weights | `gaussx.GaussHermiteIntegrator`, `UnscentedIntegrator`, `CubatureIntegrator`, `expected_log_likelihood` |
| GP predictive mean / variance, a collapsed ELBO, whitening | `gaussx.build_prediction_cache` + `predict_mean` / `predict_variance`, `collapsed_elbo`, `whiten_covariance`, `love_cache` |
| An EnKF / ETKF analysis, localisation, inflation | `gaussx.enkf_analysis`, `etkf_transform`, `localization_matrix`, `inflate_*` |
| Natural ↔ mean parameter algebra | `gaussx.mean_cov_to_natural`, `natural_to_mean_cov`, `GaussianExpFam` |

A dense `jnp.linalg` call is fine on a small matrix with no structure to
exploit; the rule is not "never", it is "not on a matrix gaussx can do
better".

## Self-check before you finish

- Every covariance / precision is a lineax operator, PSD-tagged where it is
  PSD, and built with the structure it has.
- No `.as_matrix()` feeds a solve, logdet, Cholesky or sample; no
  `DenseFallbackWarning` is ignored.
- Results match a dense reference on a small case
  (`jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(op.as_matrix(), b))`),
  and the code runs under `jax.jit` and `jax.grad`.

If gaussx genuinely lacks what you need, keep your addition small and shaped
like gaussx (an operator in, an operator or array out, pure, `solver=` where
it solves) and consider proposing it upstream at
<https://github.com/jejjohnson/gaussx/issues>.
