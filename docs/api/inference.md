# Bayesian Inference & Ensembles

Layer 3 recipes for conjugate updates, second-order variational steps, and
ensemble data assimilation. All covariances are operators, so the updates
inherit structured solves; all stochastic routines take explicit PRNG keys.

## Bayesian learning rule

Damped natural-parameter updates of the Bayesian learning rule (BLR) — full
covariance or diagonal-only — plus the marginal likelihood and expected
log-likelihood that score Gaussian posteriors.

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
      members: [newton_update, damped_natural_update, gauss_newton_operator, gauss_newton_precision, ggn_diagonal, hutchinson_hessian_diag, riemannian_psd_correction, cavity_distribution, trace_correction]

### Natural-parameter conventions

Two conventions for the second natural parameter coexist. Both share the first one, `nat1` $= \boldsymbol{\Lambda}_x \boldsymbol{\mu}_x$. Feeding the output of one family into the other silently flips the sign and halves the precision. Convert with `nat2` $\mapsto -2$ `nat2`, from $-\frac{1}{2} \boldsymbol{\Lambda}_x$ to $\boldsymbol{\Lambda}_x$.

| Function | Convention of `nat2` |
|---|---|
| `blr_diag_update`, `blr_full_update` | `nat2` $= -\frac{1}{2} \boldsymbol{\Lambda}_x$ by default; $+\boldsymbol{\Lambda}_x$ with `convention="precision"` |
| `mean_cov_to_natural`, `natural_to_mean_cov` | `nat2` $= -\frac{1}{2} \boldsymbol{\Lambda}_x$ |
| `newton_update`, `cavity_distribution` | $+\boldsymbol{\Lambda}_x$ |
| `damped_natural_update` | either (it is linear), as long as both arguments share it |

## Precision-form Laplace (INLA)

`laplace_mode` is deprecated and will be removed in gaussx 0.7.0: the loop now lives in `argminax.solve.laplace_mode`, which iterates gaussx's single step (below) and returns the same fields. The algebra on this page describes both, and the examples call argminax.

`laplace_mode` is the inner loop of INLA: the Gaussian approximation $\mathcal{N}(\hat{\mathbf{x}}, \boldsymbol{\Lambda}_{x \mid y}^{-1})$ of a latent Gaussian model at fixed hyperparameters $\boldsymbol{\theta}$, computed entirely in precision form. With the linear predictor $\boldsymbol{\eta} = \mathbf{A} \mathbf{x} + \mathbf{o}$, $\mathbf{g} = \nabla_{\boldsymbol{\eta}} \log p(\mathbf{y} \mid \boldsymbol{\eta})$ and the site precision $\boldsymbol{\Lambda}_\text{site} = -\mathrm{diag}(\nabla^2_{\boldsymbol{\eta}} \log p(\mathbf{y} \mid \boldsymbol{\eta}))$ (floored at $10^{-6}$), each Newton step solves

$$
\boldsymbol{\Lambda}_{x \mid y} \mathbf{x}^{(j+1)} = \boldsymbol{\Lambda}_x \boldsymbol{\mu}_x + \mathbf{A}^\top (\mathbf{g} + \boldsymbol{\Lambda}_\text{site} \mathbf{A} \mathbf{x}^{(j)})
$$

with $\boldsymbol{\Lambda}_{x \mid y} = \boldsymbol{\Lambda}_x + \mathbf{A}^\top \boldsymbol{\Lambda}_\text{site} \mathbf{A}$, a matrix whose structure never changes: an identity or row-selection
projector keeps a `BlockTriDiag` prior banded, and anything else becomes a
`SparseOperator` on a pattern whose sparse Cholesky analysis is shared by every
step and every $\boldsymbol{\theta}$. Hard sum-to-zero (null-space) constraints of an
`IntrinsicGMRF` are applied by kriging each solve. At the mode,

$$
\log \tilde{\pi}(\mathbf{y} \mid \boldsymbol{\theta}) = \log p(\mathbf{y} \mid \mathbf{A} \hat{\mathbf{x}} + \mathbf{o}) - \frac{1}{2} (\hat{\mathbf{x}} - \boldsymbol{\mu}_x)^\top \boldsymbol{\Lambda}_x (\hat{\mathbf{x}} - \boldsymbol{\mu}_x) + \frac{1}{2} \log \det \boldsymbol{\Lambda}_x - \frac{1}{2} \log \det \boldsymbol{\Lambda}_{x \mid y}
$$

with the constraint corrections of Rue et al. (2009) for intrinsic priors. The
mode is differentiated implicitly ($\partial_{\boldsymbol{\theta}} \hat{\mathbf{x}} = -\boldsymbol{\Lambda}_{x \mid y}^{-1} \partial_{\boldsymbol{\theta}} \nabla_{\mathbf{x}} \mathcal{J}$, via `jax.lax.custom_root`) and the
log-determinants through their structured or sparse (Takahashi) VJPs, so
`jax.grad` and the reverse-over-reverse Hessian that `theta_design` uses are
exact.

```python
import argminax as amx
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
    result = amx.solve.laplace_mode(prior, gx.PoissonLikelihood(counts))
    return result.log_marginal  # H stays BlockTriDiag


value, grad = jax.value_and_grad(log_marginal)(
    0.0
)  # exact: implicit diff + block Cholesky
```

The likelihood holds the observations, so `laplace_mode(prior, likelihood)`
takes no separate `y`. For an odd number of nodes `rw2_structure(n)` returns
`n + 1` rows (a decoupled padding node): build the field on `n + 1` nodes with
`null_space` zero on the padding row, observe the first `n` through a row
selection `SparseOperator.from_coo(np.arange(n), np.arange(n), jnp.ones(n), (n, n + 1))`
(which keeps $\boldsymbol{\Lambda}_{x \mid y}$ banded), and drop `mode[n]`.

One Newton step is public, so a caller can run its own loop around it (`argminax.solve.laplace_mode` does): `laplace_model` builds the `LaplaceModel`, whose `newton` is the Newton target and `hessian` the site information and $\boldsymbol{\Lambda}_{x \mid y}$, and `laplace_result` refactorises at the mode and assembles the log-marginal, so the result is the one `laplace_mode` returns.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LaplaceResult, laplace_model, LaplaceModel, laplace_result]

## Low-rank VB mean correction (INLA)

Under a skewed likelihood (Bernoulli, Poisson with small counts) the Laplace mode is a biased estimate of the posterior mean. `vb_mean_correction` (Van Niekerk & Rue, 2024; R-INLA's default `control.vb` correction) keeps the Laplace covariance $\boldsymbol{\Sigma} = \boldsymbol{\Lambda}_{x \mid y}^{-1}$ and shifts the mean within a $p$-dimensional subspace spanned by the columns of $\mathbf{S}$, $\bar{\mathbf{x}} = \hat{\mathbf{x}} + \boldsymbol{\Sigma} \mathbf{S} \boldsymbol{\delta}$, with $\boldsymbol{\delta} \in \mathbb{R}^p$ maximising

$$
\mathrm{ELBO}(\boldsymbol{\delta}) = \sum_m \mathbb{E}_{\eta_m \sim \mathcal{N}(\mu_{\eta,m}(\boldsymbol{\delta}), v_m)} [\log p(y_m \mid \eta_m)] - \frac{1}{2} (\bar{\mathbf{x}} - \boldsymbol{\mu}_x)^\top \boldsymbol{\Lambda}_x (\bar{\mathbf{x}} - \boldsymbol{\mu}_x)
$$

with $\boldsymbol{\mu}_\eta = \mathbf{A} \bar{\mathbf{x}} + \mathbf{o}$ and $v_m = [\mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^\top]_{mm}$. The expectations are 1-D Gauss-Hermite per site, the predictor variances come from the same Takahashi selected inverse as the marginal variances, and the Newton steps on $\boldsymbol{\delta}$ are $p$-dimensional, so it costs $p$ solves with the Laplace factor. For a Gaussian likelihood the correction is exactly zero. Typically $\mathbf{S}$ spans the fixed effects:

```python
import argminax as amx
import jax.numpy as jnp
import numpy as np

import gaussx as gx

# Rare detections along a transect: logit p_i = β₀ + u_i, u an RW2 under
# Σu = Σ(t − t̄)u = 0 and β₀ ~ N(0, 10³); latent x = (u, β₀)
n = 40
t = np.arange(n, dtype=float)
p_true = 1 / (1 + np.exp(2 - np.sin(t / 5)))
detected = jnp.asarray(np.random.default_rng(0).random(n) < p_true, float)
R = np.asarray(gx.rw2_structure(n).as_matrix())
rows, cols = np.nonzero(np.tril(R))
Q = gx.SparseOperator.from_coo(
    np.r_[rows, n],
    np.r_[cols, n],
    jnp.r_[2.0 * R[rows, cols], 1e-3],
    (n + 1, n + 1),
    symmetric=True,
)
V = jnp.zeros((n + 1, 2)).at[:n, 0].set(1.0).at[:n, 1].set(t - t.mean())
prior = gx.IntrinsicGMRF(jnp.zeros(n + 1), 1.0, Q, V)
A = gx.SparseOperator.from_coo(  # η = u + β₀
    np.r_[np.arange(n), np.arange(n)],
    np.r_[np.arange(n), np.full(n, n)],
    jnp.ones(2 * n),
    (n, n + 1),
)
detection = gx.BernoulliLikelihood(detected)

res = amx.solve.laplace_mode(prior, detection, projector=A)
fixed_effect_idx = jnp.array([n])
mean_vb = gx.vb_mean_correction(
    res, prior, detection, projector=A, subspace=fixed_effect_idx
)
```

As with `laplace_mode`, the likelihood holds the observations, so there is no
separate `y` argument; pass the same `projector` and `offset` as to
`laplace_mode`.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [vb_mean_correction]

## Ensemble covariances, gain & analysis

Empirical (cross-)covariances from ensemble members (maximum likelihood by default, `bessel=True` for unbiased), the
ensemble Kalman gain built from them, and the analysis step that applies it.

The gain functions are the *pieces*; `enkf_analysis` is the *step* -- the
stochastic (perturbed-observation) update that turns a prior ensemble and an
observation into a posterior ensemble. `etkf_analysis` is its deterministic
square-root counterpart, and `etkf_transform` returns that update's weights. With fewer observations than members, both take a rank-$D_y$ route that costs $O(|\mathcal{E}| D_y^2)$ rather than an $(|\mathcal{E}|, |\mathcal{E}|)$ eigendecomposition, and `etkf_analysis` also never forms the transform.

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

### Local ETKF

`obs_weights` replaces $\boldsymbol{\Sigma}_\varepsilon^{-1}$ by $\mathbf{D}_\rho^{1/2} \boldsymbol{\Sigma}_\varepsilon^{-1} \mathbf{D}_\rho^{1/2}$ with $\mathbf{D}_\rho = \mathrm{diag}(\boldsymbol{\rho})$, which for a diagonal $\boldsymbol{\Sigma}_\varepsilon$ is $\rho_k / [\boldsymbol{\Sigma}_\varepsilon]_{kk}$ (Hunt, Kostelich & Szunyogh 2007). A zero weight is the same as deleting that observation, so the local ETKF is a `vmap` of `etkf_transform` over grid points, with one row of tapered distances per point.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx

import gaussx

key = jax.random.key(0)
particles = jax.random.normal(key, (20, 30))  # (J, N): members by grid points
grid = jnp.linspace(0.0, 10.0, 30)
sites = jnp.array([1.0, 4.0, 7.0, 9.0])
y = jnp.array([0.3, -0.1, 0.4, 0.2])
obs_noise = lx.DiagonalLinearOperator(0.2 * jnp.ones(4))

obs_particles = particles[:, jnp.searchsorted(grid, sites)]  # (J, M)
distances = jnp.abs(grid[:, None] - sites[None, :])  # (N, M)
rho = gaussx.gaspari_cohn(distances, 3.0)  # zero beyond the 3.0 radius

w_mean, W = jax.vmap(
    lambda row: gaussx.etkf_transform(obs_particles, y, obs_noise, obs_weights=row)
)(rho)  # (N, J), (N, J, J)

mean = einx.mean("[member] state", particles)
anomalies = particles - mean
new_mean = mean + einx.dot("state member, member state -> state", w_mean, anomalies)
new_anomalies = einx.dot("state j k, k state -> j state", W, anomalies)
analysis = new_mean + new_anomalies  # (J, N)
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ensemble_covariance, ensemble_cross_covariance, ledoit_wolf_shrinkage, ensemble_kalman_gain, enkf_analysis, etkf_analysis, etkf_transform, ensemble_rts_step]

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
      members: [localization_matrix, localized_kalman_gain, gaspari_cohn, gaussian_taper, soar_taper, inflate_multiplicative, inflate_rtpp, inflate_rtps]

## Distances

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [euclidean_distance, haversine_distance]

## Deprecated aliases

These names still work but warn on use and will be removed in gaussx 0.7.0 (see the [deprecation policy](index.md#deprecation-policy)). Use `argminax.solve.laplace_mode` instead of `laplace_mode`.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [laplace_mode]
