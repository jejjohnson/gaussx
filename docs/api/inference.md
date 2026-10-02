# Bayesian Inference & Ensembles

Layer 3 recipes for conjugate updates, second-order variational steps, and
ensemble data assimilation. All covariances are operators, so the updates
inherit structured solves; all stochastic routines take explicit PRNG keys.

## Bayesian linear regression

Closed-form Gaussian posterior updates — full covariance or diagonal-only —
plus the marginal likelihood and expected log-likelihood that score them.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [blr_full_update, blr_diag_update, log_marginal_likelihood, gaussian_expected_log_lik]

## Newton & natural-gradient updates

Second-order variational steps: Newton's method on the variational objective,
Gauss-Newton curvature (exact diagonal or Hutchinson-estimated), damped
natural-gradient steps, and the PSD projection that keeps Riemannian updates
on the manifold.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [newton_update, damped_natural_update, gauss_newton_precision, ggn_diagonal, hutchinson_hessian_diag, riemannian_psd_correction, cavity_distribution, trace_correction]

## Precision-form Laplace (INLA)

`laplace_mode` is the inner loop of INLA: the Gaussian approximation
$\mathcal N(\hat x, H^{-1})$ of a latent Gaussian model at fixed
hyperparameters $\theta$, computed entirely in precision form. With
$\eta = Ax + o$, $g = \partial_\eta\log p(y\mid\eta)$ and
$W = -\operatorname{diag}(\partial^2_\eta\log p)$, each Newton step solves

$$
(Q + A^\top WA)\,x_{t+1} = Q\mu + A^\top(g + WAx_t),
$$

with a matrix whose structure never changes: an identity or row-selection
projector keeps a `BlockTriDiag` prior banded, and anything else becomes a
`SparseOperator` on a pattern whose sparse Cholesky analysis is shared by every
step and every $\theta$. Hard sum-to-zero (null-space) constraints of an
`IntrinsicGMRF` are applied by kriging each solve. At the mode,

$$
\log\tilde\pi(y\mid\theta) = \log p(y\mid A\hat x + o)
  - \tfrac12(\hat x-\mu)^\top Q(\hat x-\mu) + \tfrac12\log|Q| - \tfrac12\log|H|,
$$

with the constraint corrections of Rue et al. (2009) for intrinsic priors. The
mode is differentiated implicitly ($\partial_\theta\hat x =
H^{-1}\partial_\theta\nabla\ell$, via `jax.lax.custom_root`) and the
log-determinants through their structured or sparse (Takahashi) VJPs, so
`jax.grad` and the reverse-over-reverse Hessian that `theta_design` uses are
exact.

```python
import jax
import jax.numpy as jnp
import numpy as np

import gaussx as gx

# Poisson counts with an RW2 seasonal effect; θ = log τ
n_days = 364  # even: rw2_structure(n) pads odd n with one node (see below)
t = jnp.arange(n_days, dtype=float)
counts = jnp.asarray(np.random.default_rng(0).poisson(np.exp(1 + np.sin(t / 58))))
rw2_null = jnp.column_stack([jnp.ones(n_days), t - t.mean()])


def log_marginal(log_tau):
    prior = gx.IntrinsicGMRF(
        jnp.zeros(n_days),
        jnp.exp(log_tau),
        gx.rw2_structure(n_days),
        null_space=rw2_null,
        constraint="hard",
    )
    result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts))
    return result.log_marginal  # H stays BlockTriDiag


value, grad = jax.value_and_grad(log_marginal)(0.0)  # exact: implicit diff + block Cholesky
```

The likelihood holds the observations, so `laplace_mode(prior, likelihood)`
takes no separate `y`. For an odd number of nodes `rw2_structure(n)` returns
`n + 1` rows (a decoupled padding node): build the field on `n + 1` nodes with
`null_space` zero on the padding row, observe the first `n` through a row
selection `SparseOperator.from_coo(np.arange(n), np.arange(n), jnp.ones(n), (n, n + 1))`
(which keeps $H$ banded), and drop `mode[n]`.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [laplace_mode, LaplaceResult]

## Ensemble covariances, gain & analysis

Bessel-corrected empirical (cross-)covariances from ensemble members, the
ensemble Kalman gain built from them, and the analysis step that applies it.

The gain functions are the *pieces*; `enkf_analysis` is the *step* -- the
stochastic (perturbed-observation) update that turns a prior ensemble and an
observation into a posterior ensemble. `etkf_transform` is its deterministic
square-root counterpart.

A caveat worth stating up front: the Gaussian assumption in an ensemble Kalman
filter is a property of the **coordinates**, not of the algorithm. Applied to a
non-Gaussian prior the update is biased, and the bias does not shrink with
ensemble size. Conjugating the update with a bijection that Gaussianises the
prior -- warp, analyse, warp back -- removes it.

That conjugated update is exact Bayes only under conditions worth stating
precisely, since they are easy to over-claim. It holds **in the population
limit** -- with a finite ensemble the gain is empirical and the perturbations
are Monte Carlo, so the result is an estimate regardless -- and only when the
observation model is **affine with additive Gaussian noise** in the same latent
coordinates that Gaussianise the prior. A Gaussian conditional likelihood is not
sufficient on its own: `y = z² + ε` has Gaussian noise and a non-Gaussian
posterior that no Kalman update reproduces. Outside those conditions
conjugation is an approximation with no guaranteed ordering against the
physical-space update -- usually much better, but a badly matched warp can make
the latent joint less Gaussian and do worse.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ensemble_covariance, ensemble_cross_covariance, ensemble_kalman_gain, enkf_analysis, etkf_transform]

## Ensemble Kalman inversion

`eki_step` is the same Kalman update as `enkf_analysis` with two knobs added,
and reduces to it exactly at `dt=1`. It is the *inverse problem* reading of the
ensemble filter: one fixed observation, no time axis, and a schedule of tempered
steps instead of a sequence of assimilation windows.

`dt` is the observation-side tempering step, replacing `R` by `R / dt`. Over a
schedule with `sum(dt) = 1` the composition is exactly one Bayesian update in
the linear-Gaussian population limit -- the precisions add -- so the sum
condition is what makes a schedule a tempering path rather than a heuristic.
`step` is the state-side operator in the gradient-flow view: it multiplies each
member's increment, so a `BlockDiag` of scaled identities gives a different rate
per state block. It changes the trajectory, not the fixed point.

The two helpers cover the standard variations. `tikhonov_augment` puts a prior
`N(m0, C0)` into the step by observation augmentation (TEKI) -- a helper rather
than a flag, so `C0` stays an operator and the step itself knows nothing about
priors. `discrepancy_step_size` is the tuning-parameter-free data misfit
controller of Iglesias & Yang (2021): a pure function of the ensemble misfits,
so it belongs here rather than in whatever drives the iteration.

The iteration loop, the stopping rule, and the forward model itself are all out
of scope: these are array-in / array-out steps.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [eki_step, tikhonov_augment, discrepancy_step_size]

## Localization & inflation

The standard fixes for small-ensemble rank deficiency: Schur-product
localization with a taper (Gaspari-Cohn by default) and multiplicative /
RTPP / RTPS inflation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [localization_matrix, localized_kalman_gain, gaspari_cohn, inflate_multiplicative, inflate_rtpp, inflate_rtps]

## Distances

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [euclidean_distance, haversine_distance]
