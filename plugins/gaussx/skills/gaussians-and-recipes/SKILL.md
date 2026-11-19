---
name: gaussians-and-recipes
description: Write Gaussian models and algorithms in JAX on gaussx — Gaussian log-densities, KL, entropy, conditioning and sampling on structured covariances; Gaussian Markov random fields with structured precisions; natural and expectation parameters; likelihoods and Gauss–Hermite, unscented and cubature integrators; GP prediction, sparse and variational GP bounds; Kalman filters and RTS smoothers, SDE kernels, nonlinear filtering; expected log-likelihoods and EP moment matching; ensemble Kalman (EnKF, ETKF, EKI), localisation and inflation; natural-gradient, Newton and Laplace updates. Use when a task evaluates or samples a Gaussian, runs a Kalman filter or smoother, predicts with a GP, integrates a likelihood under a Gaussian, or updates an ensemble or a Gaussian approximation, in a project that uses (or could use) gaussx.
---

# Gaussians and recipes with gaussx

gaussx ships the Gaussian itself and the algorithms that are linear algebra on Gaussians.
Every one of them takes covariance operators and keeps their structure, and takes a `solver=` or an integrator where the numerics can change.

This skill covers rungs ③ models and ④ algorithms of gaussx, with the integrators from rung ②, in the vocabulary of the `role-ladder` skill.
The operators and the verbs are in `gaussx:structured-gaussian-linalg`, and the solver strategies in `gaussx:solvers-and-preconditioners`.

## Start from

- **A Gaussian** is a mean and a covariance operator: `gaussx.gaussian_log_prob(loc, cov, value)`, `gaussian_kl(p_loc, p_cov, q_loc, q_cov)`, `gaussian_entropy(cov)`, `conditional(loc, cov, obs_idx, obs_values)`, `sample_mvn(mean, cov, key=key)`.
- **A structured precision** is a field: `gaussx.GaussianMRF(loc, precision)`, an equinox module with `log_prob` and `sample`. The NumPyro distribution classes (`MultivariateNormal`, `LGSSM`, …) live in `pyrox_gauss`.
- **An expectation under a Gaussian** is `gaussx.expected_log_likelihood(likelihood, gaussx.GaussianState(mean, cov), integrator)`.
- **A state-space model** is filtered by `gaussx.kalman_filter(...)`, which returns a `FilterState`, and smoothed by `gaussx.rts_smoother(state, transition)`.
- **A GP** is predicted from `gaussx.build_prediction_cache(K_y, y)` with `predict_mean` and `predict_variance`. gaussx never evaluates a kernel; the caller supplies the matrices.
- **An ensemble** `(J, N)` is updated by `gaussx.enkf_analysis(particles, obs_particles, observation, obs_noise, key=key)`.

## Rules

1. The covariance stays an operator from end to end. A Kronecker covariance then gives a Kronecker-cost density.
2. Pass the numerics down: `solver=` where an algorithm solves, an integrator where it integrates. A function above gaussx takes them as arguments too.
3. Only `GaussianLikelihood` has a closed-form expectation. Every other likelihood needs an integrator.
4. PRNG keys are explicit arguments.
5. These are single steps and single passes. The loop over iterations or assimilation cycles belongs to the caller.

## Worked example

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
mean = jnp.zeros(3)
cov = lx.MatrixLinearOperator(
    jnp.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]), PSD
)
value = jnp.array([0.1, -0.2, 0.3])

log_prob = gaussx.gaussian_log_prob(mean, cov, value)  # ()
draws = gaussx.sample_mvn(mean, cov, key=jr.key(0), num_samples=4)  # (4, 3)
assert jnp.isfinite(log_prob) and draws.shape == (4, 3)

F = jnp.array([[1.0, 0.1], [0.0, 1.0]])  # (N, N) transition
H = jnp.array([[1.0, 0.0]])  # (M, N) observation model
obs = jr.normal(jr.key(1), (20, 1))  # (T, M)
state = gaussx.kalman_filter(
    F, H, 0.01 * jnp.eye(2), 0.1 * jnp.eye(1), obs, jnp.zeros(2), jnp.eye(2)
)
means, covs = gaussx.rts_smoother(state, F)  # (T, N), (T, N, N)
assert means.shape == (20, 2) and jnp.isfinite(state.log_likelihood)
```

## Don't write it

| Don't write | Use |
|---|---|
| `-0.5 * (r @ solve(K, r) + logdet + n log 2π)` | `gaussx.gaussian_log_prob(mean, op, y)` |
| A Gaussian KL, entropy or conditional by hand | `gaussx.gaussian_kl`, `gaussian_entropy`, `conditional` |
| `mean + L @ jr.normal(...)` sampling | `gaussx.sample_mvn(mean, op, key=key)` |
| A Kalman predict and update loop, an RTS smoother | `gaussx.kalman_filter`, `rts_smoother`, `parallel_kalman_filter`; `MaternSDE` and `discretize_mfd` for a GP as a state-space model |
| Gauss–Hermite, unscented or cubature points and weights | `gaussx.GaussHermiteIntegrator`, `UnscentedIntegrator`, `CubatureIntegrator`, `expected_log_likelihood` |
| A GP predictive mean or variance, a collapsed ELBO, whitening | `gaussx.build_prediction_cache` with `predict_mean` and `predict_variance`, `collapsed_elbo`, `unwhiten`, `love_cache` |
| An EnKF or ETKF analysis, localisation, inflation | `gaussx.enkf_analysis`, `etkf_transform`, `localization_matrix`, `inflate_rtps` |
| Natural and mean parameter algebra | `gaussx.mean_cov_to_natural`, `natural_to_mean_cov`, `GaussianExpFam` |

## The shelf

Each entry is one file with the names it covers, a worked example, the rules and what goes wrong.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ② strategy | [Integrators](shelf/integrators.md) | Gauss–Hermite, unscented, cubature, Taylor, Monte Carlo | 13 |
| ③ model | [Gaussians](shelf/gaussians.md) | densities, KL, entropy, conditioning, sampling | 15 |
| ③ model | [GMRF fields](shelf/distributions.md) | The GMRF fields `GaussianMRF`, `IntrinsicGMRF`, `ConstrainedGMRF` and `BYM2GMRF` | 4 |
| ③ model | [Exponential-family forms](shelf/expfam.md) | natural, expectation and mean-covariance conversions | 16 |
| ③ model | [Likelihoods](shelf/likelihoods.md) | Gaussian, Poisson, Bernoulli, Student-t and the rest | 10 |
| ③ model | [SDE kernels](shelf/sde-kernels.md) | Matérn, periodic, quasi-periodic, sums and products, and their discretisation | 17 |
| ④ algorithm | [GP conditioning and prediction](shelf/gp.md) | conditionals, prediction caches, LOVE, Kronecker GPs, leave-one-out | 20 |
| ④ algorithm | [Variational GPs](shelf/variational-gp.md) | ELBOs, the collapsed bound, whitening | 9 |
| ④ algorithm | [Kalman filtering and smoothing](shelf/kalman.md) | sequential, parallel, mean-field and infinite-horizon filters and smoothers | 23 |
| ④ algorithm | [Nonlinear filtering and sites](shelf/nonlinear-filtering.md) | nonlinear Kalman, CVI site updates, SpInGP | 14 |
| ④ algorithm | [Expectations and moment matching](shelf/expectations.md) | expected log-likelihood, EP moments, psi statistics, uncertain inputs | 22 |
| ④ algorithm | [Natural-gradient and Newton updates](shelf/natural-gradient.md) | BLR, damped natural updates, Gauss–Newton, Laplace | 19 |
| ④ algorithm | [Ensemble Kalman](shelf/ensemble-kalman.md) | analysis, ETKF, EKI, localisation, inflation | 21 |
<!-- shelf:end -->

## Before you finish

- Every covariance is an operator with its structure, and no density, KL or sample was written by hand.
- Every algorithm that can take a `solver=` or an integrator receives it from the caller's arguments.
- The result was checked on a small case against a closed form or a dense reference.

<!-- shelf:below:begin -->
## The units below this skill

A card is enough to call a unit.
Load its skill only when the card's last line applies.

### Structures and primitives · ⓪ ① · skill `gaussx:structured-gaussian-linalg`

- **Provides.** Covariance and precision operators that keep their structure, and the verbs that use it.
- **Nouns.** `gaussx.Kronecker`, `gaussx.SumOfKroneckers`, `gaussx.BlockDiag`, `gaussx.BlockTriDiag`, `gaussx.Toeplitz`, `gaussx.SparseOperator`, `gaussx.low_rank_plus_diag`; any `lineax.AbstractLinearOperator`, tagged.
- **Verbs.** `gaussx.solve`, `gaussx.logdet`, `gaussx.cholesky`, `gaussx.sqrt`, `gaussx.inv`, `gaussx.diag`, `gaussx.trace`; `gaussx.solve_matrix`, `gaussx.woodbury_solve`, `gaussx.safe_cholesky`.
- **Adverbs.** `solver=` on `gaussx.solve`; left out, the verb dispatches on the structure.
- **Keep.** A dense matrix enters as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`; factors are operators, not arrays.
- **Never.** Call `as_matrix()` and then a dense `jnp.linalg` solve, or build `jnp.kron(A, B)`.
- **One rung down.** Open `gaussx:structured-gaussian-linalg` to build a new structure or to see which verb has a fast path for which noun.

### Strategies · ②, setup · skill `gaussx:solvers-and-preconditioners`

- **Provides.** The ways a solve or a log-determinant is computed, as objects a caller passes in.
- **Nouns.** `gaussx.AutoSolver`, `gaussx.DenseSolver`, `gaussx.CGSolver`, `gaussx.PreconditionedCGSolver`, `gaussx.BBMMSolver`, `gaussx.MINRESSolver`, `gaussx.LSMRSolver`, `gaussx.SparseCholeskySolver`; `gaussx.SLQLogdet`, `gaussx.DenseLogdet`; `gaussx.NystromPreconditioner`, `gaussx.PartialCholeskyPreconditioner`.
- **Verbs.** `strategy.solve(operator, vector)` and `strategy.logdet(operator, key=key)`; `gaussx.linear_solve(operator, vector, solver=...)` for an operator or a `(matvec, shape)` pair.
- **Adverbs.** `preconditioner=` on the CG family; `key=` for a stochastic log-determinant, or `gaussx.KeyedSolver`; `gaussx.ComposedSolver` to take the solve from one strategy and the log-determinant from another.
- **Keep.** A function above gaussx takes `solver: gaussx.AbstractSolverStrategy | None = None` and hands it down unchanged; `None` means structural dispatch.
- **Never.** Write a CG, Lanczos or Hutchinson loop, or fix a solver inside a function that could take one.
- **One rung down.** Open `gaussx:solvers-and-preconditioners` to choose between strategies, tune a preconditioner or build a randomized factorisation.
<!-- shelf:below:end -->

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Nothing else in the stack: this is a base skill.

Used by (pointers, not skills to load): argminax, filterax, finitevolX, gauss_flows, integrix, kernellib, manipy, plumax, pyrox, radtrax, spectraldiffx, vardax.

<!-- skill-tree:end -->
