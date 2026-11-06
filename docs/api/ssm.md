# State-Space Models & Kalman

Layer 3 recipes for linear-Gaussian state-space models. Stationary 1-D GP kernels with rational spectral densities admit exact SDE representations

$$
\mathrm{d}\mathbf{x}(t) = \mathbf{F}_\text{c} \mathbf{x}(t) \mathrm{d}t + \mathbf{L}_\text{c} \mathrm{d}\mathbf{b}(t)
$$

$$
f(t) = \mathbf{H} \mathbf{x}(t)
$$

turning $O(N^3)$ GP inference into $O(N D_x^3)$ Kalman filtering. This page covers the SDE kernel zoo, the filters and smoothers (sequential, parallel associative-scan, in covariance or square-root form, and steady-state), and the natural-parameter /
site machinery for non-conjugate likelihoods.

## SDE kernels

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SDEKernel, SDEParams, ConstantSDE, MaternSDE, PeriodicSDE, QuasiPeriodicSDE, CosineSDE, IntegratedWienerSDE, ProductSDE, SumSDE, sde_autocovariance]

### Stationary and non-stationary kernels

Most of the zoo is stationary: the process is assumed started in its
stationary distribution, so `SDEKernel.initial_covariance` returns
$\boldsymbol{\Sigma}_{x,\infty}$ and `sde_autocovariance` can report $k(\tau)$.

`IntegratedWienerSDE` is not. Its drift is nilpotent, the marginal
variance grows without bound, and the covariance depends on both times
rather than on their difference — so there is no $\boldsymbol{\Sigma}_{x,\infty}$ and no
$k(\tau)$. It reports `stationary = False` and `sde_params().P_inf =
None`, and the filter is started instead from an explicit
`initial_covariance` (diffuse by default). That is the local linear
trend prior at `order=1`: smooth, and linear unless the data push back,
with no lengthscale to choose.

Consumers should branch on `SDEKernel.stationary` rather than on whether
`P_inf` is `None` — the two are different questions, since a stationary
kernel may have no *closed form* for $\boldsymbol{\Sigma}_{x,\infty}$ (a learned drift, say).
A `SumSDE` may mix the two: it is stationary only if all its components
are, and stacks their initial covariances block-diagonally. A
`ProductSDE` needs both factors' $\boldsymbol{\Sigma}_{x,\infty}$ and rejects a
non-stationary one.

### Discretisation

Turning the continuous-time SDE into $\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \boldsymbol{\eta}_t$ takes two routes, and which one applies depends on whether a stationary covariance exists.

The default, `SDEKernel.discretise`, uses the *stationary* route $\boldsymbol{\Sigma}_{\eta,t} = \boldsymbol{\Sigma}_{x,\infty} - \mathbf{F}_t \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}_t^\top$ ([`process_noise_covariance`](#gaussx.process_noise_covariance), documented under *Process noise* below). It is exact and cheap, and every stationary kernel in the zoo above supplies the $\boldsymbol{\Sigma}_{x,\infty}$ it needs. `IntegratedWienerSDE` overrides `discretise` with an exact closed form instead — its nilpotent drift makes the exponential terminate — so it needs neither route.

`discretize_mfd` is the fallback for when that covariance is not available — most importantly when $\mathbf{F}_\text{c}$ is a **learned parameter** rather than derived from a kernel. Recovering $\boldsymbol{\Sigma}_{x,\infty}$ then means solving the Lyapunov equation $\mathbf{F}_\text{c} \boldsymbol{\Sigma}_{x,\infty} + \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}_\text{c}^\top + \mathbf{B}_\text{c} = \mathbf{0}$, with $\mathbf{B}_\text{c} = \mathbf{L}_\text{c} \mathbf{Q}_\text{c} \mathbf{L}_\text{c}^\top$, which has a unique solution **only if $\gamma_i(\mathbf{F}_\text{c}) + \gamma_j(\mathbf{F}_\text{c}) \neq 0$ for every pair of eigenvalues**. That condition fails for any undamped oscillatory mode, where $\gamma = \pm i\omega$ and so $\gamma + \bar\gamma = 0$ identically — `CosineSDE` has exactly that drift, and sidesteps it with a closed form that a learned $\mathbf{F}_\text{c}$ cannot use.

Matrix-fraction decomposition needs no $\boldsymbol{\Sigma}_{x,\infty}$ at all: it recovers both $\mathbf{F}_t$ and $\boldsymbol{\Sigma}_{\eta,t}$ from a single $2 D_x \times 2 D_x$ matrix exponential (Van Loan 1978) and is well defined for **every** $\mathbf{F}_\text{c}$. Reach for it when the drift is fitted; keep the stationary route otherwise. Note the obstruction is degeneracy of that Sylvester system, not instability — constraining $\mathbf{F}_\text{c}$ to be Hurwitz would not fix it, and would forbid the oscillatory modes MFD exists to support.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [discretize_mfd, discretize_mfd_sequence]

## Nonlinear filters

`nonlinear_kalman_filter` and `nonlinear_rts_smoother` take *callables*
rather than matrices, and moment-match them through an
[integrator](quadrature.md). The choice of integrator is the choice of
filter — one loop yields the whole textbook family:

| integrator | filter |
|---|---|
| `TaylorIntegrator` | extended Kalman filter (EKF) |
| `UnscentedIntegrator` | unscented Kalman filter (UKF) |
| `CubatureIntegrator` | cubature Kalman filter (CKF), degree 3 |
| `FifthOrderCubatureIntegrator` | degree-5 cubature filter |
| `GaussHermiteIntegrator` | Gauss-Hermite Kalman filter (GHKF) |
| `MonteCarloIntegrator` | Monte-Carlo Kalman filter |

The gain is $\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1}$, built from the integrator's *cross-covariance* $\boldsymbol{\Sigma}_{xy,t}$, so no Jacobian is formed anywhere — that is what makes the EKF and the UKF the same code. An integrator that does not supply a cross-covariance is rejected rather than silently producing a zero gain.

Two behaviours differ from `kalman_filter`, both deliberately:

- **`log_likelihood` is a moment-matched surrogate**, not the exact
  marginal likelihood, because $\boldsymbol{\Sigma}_{y,t}$ is the matched innovation covariance
  rather than the true one. It coincides with the exact value when the
  maps are affine. Maximising it to tune hyperparameters is standard
  practice for nonlinear Gaussian filters, but it is a surrogate.
- **The covariance update defaults to Joseph form** (`joseph=True`). $\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1}$ is only approximately the optimal gain, and $\boldsymbol{\Sigma}_{x,t \mid t-1} - \mathbf{G}_t \boldsymbol{\Sigma}_{y,t} \mathbf{G}_t^\top$ is guaranteed PSD only *for* the optimal gain, whereas Joseph form is a sum of two PSD terms for any $\mathbf{G}_t$. The effective observation matrix it needs is the statistical-linearisation gain $\mathbf{H}_t = \boldsymbol{\Sigma}_{xy,t}^\top \boldsymbol{\Sigma}_{x,t \mid t-1}^{-1}$ — what [`statistical_linear_regression`](quadrature.md) returns as `A` — and its noise is $\boldsymbol{\Sigma}_{\varepsilon,t} + \boldsymbol{\Sigma}_{\text{lin},t}$, not $\boldsymbol{\Sigma}_{\varepsilon,t}$, with $\boldsymbol{\Sigma}_{\text{lin},t}$ the linearisation residual.

  With that residual included the two forms are analytically identical for any consistent matched joint, not merely for affine maps. **`joseph` is therefore a numerical choice, not a modelling one**: it selects how the same covariance is computed, and should not change results beyond floating point. Dropping $\boldsymbol{\Sigma}_{\text{lin},t}$ — the naive reading of Joseph here — would instead understate the posterior by $\mathbf{G}_t \boldsymbol{\Sigma}_{\text{lin},t} \mathbf{G}_t^\top$.

With affine `dynamics` and `obs_fn` the filter reproduces `kalman_filter`
— means, covariances *and* log-likelihood — and the smoother reproduces
`rts_smoother`, for every *deterministic* rule, since each is exact for
affine maps. `MonteCarloIntegrator` is the exception: it propagates
finite-sample empirical moments, so it converges to the linear filter at
the usual $O(1/\sqrt{n})$ rate rather than matching it. A discrepancy
there is sampling error, not a bug.

### Driving your own loop

The wrappers above are a `jax.lax.scan` over three public per-step
functions, which are usable on their own when the loop is the part you
want to control — an irregular time grid, a custom gating rule, a filter
interleaved with something else, or a bank of filters:

| function | step |
|---|---|
| `nonlinear_kalman_predict` | $\boldsymbol{\mu}_{x,t \mid t-1} = \mathbb{E}[f(\mathbf{x}_{t-1})]$, $\boldsymbol{\Sigma}_{x,t \mid t-1} = \mathrm{Cov}[f(\mathbf{x}_{t-1})] + \boldsymbol{\Sigma}_{\eta,t}$ |
| `nonlinear_kalman_update` | match $h$, then $\mathbf{G}_t = \boldsymbol{\Sigma}_{xy,t} \boldsymbol{\Sigma}_{y,t}^{-1}$ and the Gaussian update |
| `sigma_point_kalman_update` | the same update from evaluations of $h$ the caller already has |
| `nonlinear_rts_step` | one RTS backward correction |

Each takes the same `integrator`, so the choice of filter carries through
unchanged. `nonlinear_kalman_update` returns its log-likelihood increment
rather than accumulating, leaving the accumulation to the caller.

`sigma_point_kalman_update` is for a forward model that runs outside the update, as in unscented Kalman inversion: it takes the points and their evaluations, with no `obs_fn` to call.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [nonlinear_kalman_filter, nonlinear_rts_smoother, nonlinear_kalman_predict, nonlinear_kalman_update, sigma_point_kalman_update, nonlinear_rts_step, masked_moment_inputs]

## Kalman filtering & smoothing

The forward filter and RTS smoother, their $O(\log N)$ parallel
(associative-scan) counterparts, and the steady-state (infinite-horizon)
variants built on the discrete algebraic Riccati equation.

`parallel_kalman_filter(square_root=True)` runs the square-root associative scan of Yaghoobi, Corenflos, Hassan & Särkkä (2022): elements carry factors $\mathbf{L}_\text{cond}$, $\mathbf{L}_\text{lik}$ of $\boldsymbol{\Sigma}_\text{cond} = \mathbf{L}_\text{cond} \mathbf{L}_\text{cond}^\top$ and $\boldsymbol{\Lambda}_\text{lik} = \mathbf{L}_\text{lik} \mathbf{L}_\text{lik}^\top$, every element and combination is a QR (`tria`) of stacked factors, and the returned covariances are Gram matrices, PSD by construction. It is the robust choice
for float32 chains with tiny observation noise or steps, where the
covariance form can go indefinite (gh-454); `psd_project=True` is the
cheaper eigenvalue-clipping safety net.

### Observation masks

`kalman_filter` and `parallel_kalman_filter` take an optional `mask`,
dispatched on its rank:

| shape | meaning |
|---|---|
| `(T,)` | **Per-step gate.** `False` runs the predict step only and contributes nothing to the log-likelihood — the usual way to predict on a merged train/test grid. |
| `(T, M)` | **Per-channel gate**, for partially observed multivariate series where different channels are measured at different times. |

The per-channel path marginalises unobserved channels *exactly*: row $i$ of $\mathbf{H}_t$ is zeroed, a unit block is substituted into $\boldsymbol{\Sigma}_{\varepsilon,t}$, and the residual entry is set to zero. The innovation covariance is then block-diagonal in the observed/masked split, so column $i$ of the gain vanishes and the masked channel cannot move the state — the posterior reproduces the row-deleted filter to machine precision, with no branching. A dummy block also contributes $-\frac{1}{2} \log 2\pi$ per masked channel to the full-vector density, which is stripped per step, so `log_likelihood` is the exact marginal $\log p(\mathbf{y}_\text{obs})$ and is invariant to both the mask pattern and the dummy variance.

An all-`False` row of a `(T, M)` mask is equivalent to a `False` entry in the
`(T,)` form, and `M == 1` is unambiguous either way. Masked entries of
`observations` are never read, so they may be `NaN`. Operator-typed
`obs_model` / `obs_noise` are materialised under a `(T, M)` mask, since
zeroing rows is inherently dense; `parallel_kalman_filter(psd_project=True)`
supports the `(T,)` mask only and raises `NotImplementedError` otherwise. `rts_smoother` needs no mask of
its own — it consumes filtered/predicted moments, which are already
mask-aware.

### Mean-field (block-diagonal) filtering

When the state decomposes into $L$ independent blocks of size $d$ (e.g. a multi-output temporal GP with one SDE per output), `meanfield_kalman_filter` and `meanfield_rts_smoother` run $L$ parallel $d$-state filters under `jax.vmap` — $O(T L d^3)$ total instead of the full filter's $O(T L^3 d^3)$. The trade-off is the mean-field approximation: inputs and posterior covariance are projected onto their diagonal blocks, so posterior *cross-block* covariance is dropped (the returned $(T, D_x, D_x)$ covariances are exactly zero off-block, and the log-likelihood is the sum of per-block log-likelihoods). The approximation is exact when the true cross-block dynamics are zero, which makes the decoupled case a useful consistency check against `kalman_filter`. A `BlockDiag` operator whose sub-operators match the blocking is split structurally, without materialising the full $(D_x, D_x)$ matrix; `parallel=True` routes each block through the associative-scan filter/smoother.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EmissionModel, FilterState, kalman_filter, rts_smoother, kalman_gain, parallel_kalman_filter, parallel_rts_smoother, meanfield_kalman_filter, meanfield_rts_smoother, infinite_horizon_filter, infinite_horizon_smoother, dare, DAREResult, pairwise_marginals]

## SpInGP

State-space (sparse-in-time) GP inference: marginal likelihood and posterior
through the SSM representation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [spingp_log_likelihood, spingp_posterior]

## Precision-form chains

The joint prior or posterior of a Gauss-Markov chain has a block-tridiagonal precision $\boldsymbol{\Lambda}_x$. The **UDL factorisation** $\boldsymbol{\Lambda}_x = \mathbf{U} \mathbf{D} \mathbf{U}^\top$ — unit upper block-bidiagonal $\mathbf{U}$, block-diagonal $\mathbf{D}$ — is computed last block first by Schur complementation, for blocks $t = 0, \ldots, T-1$ of size $D_x$,

$$
\mathbf{D}_{T-1} = [\boldsymbol{\Lambda}_x]_{T-1,T-1}
$$

$$
\mathbf{U}_t^\top = \mathbf{D}_{t+1}^{-1} [\boldsymbol{\Lambda}_x]_{t+1,t}
$$

$$
\mathbf{D}_t = [\boldsymbol{\Lambda}_x]_{t,t} - [\boldsymbol{\Lambda}_x]_{t+1,t}^\top \mathbf{D}_{t+1}^{-1} [\boldsymbol{\Lambda}_x]_{t+1,t}
$$

in $O(T D_x^3)$. Unlike the banded Cholesky factor behind [`cholesky`](primitives.md), its blocks have a physical reading: they *are* the state-space model with that precision, $\mathbf{F}_t = -\mathbf{U}_t^\top$ and $\boldsymbol{\Sigma}_{\eta,t}^{-1} = \mathbf{D}_t$ for $t \ge 1$, with $\boldsymbol{\Sigma}_{x,0}^{-1} = \mathbf{D}_0$. One pass therefore turns a precision-form posterior — the output of `spingp_posterior`, or a CVI / natural-gradient site update — into a chain that the Kalman machinery can sample, predict and interpolate from, without materialising the $(T D_x) \times (T D_x)$ covariance.

[`MarkovGaussian`](distributions.md#gaussx.MarkovGaussian) is the
distribution that carries both views: its canonical parameterisation is the
generative tuple $(\mathbf{F}_t, \mathbf{c}_t, \boldsymbol{\Sigma}_{\eta,t}, \boldsymbol{\mu}_{x,0}, \boldsymbol{\Sigma}_{x,0})$, and `to_precision_form` /
`from_precision_form` round-trip through the UDL factors exactly.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [UDLDecomposition, udl_decomposition, udl_to_ssm_params, udl_from_ssm_params]

## Sites & natural parameters

Conjugate-computation VI (CVI) site updates and the conversions between SSM
moment, expectation, and natural parameterizations used by non-conjugate
temporal inference.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianSites, cvi_update_sites, sites_to_precision, cavity_from_marginal, site_natural_from_tilted, site_mean_var_from_natural, expectations_to_ssm, naturals_to_ssm, ssm_to_expectations, ssm_to_naturals]

## Variational SDE linearisation

Variational treatments of a nonlinear SDE $\mathrm{d}\mathbf{x}(t) = f(\mathbf{x}(t)) \mathrm{d}t + \mathbf{L}_\text{c} \mathrm{d}\mathbf{b}(t)$ with diffusion covariance $\mathbf{B}_\text{c} = \mathbf{L}_\text{c} \mathbf{Q}_\text{c} \mathbf{L}_\text{c}^\top$ (Archambeau, Cornford, Opper & Shawe-Taylor, 2007) fit a Gaussian path $q(\mathbf{x}_t) = \mathcal{N}(\boldsymbol{\mu}_{x,t}, \boldsymbol{\Sigma}_{x,t})$ with a linear drift $\mathbf{F}_{\text{c},t} \mathbf{x} + \mathbf{c}_t$. `linearize_sde` returns the drift that is optimal along the path, $\mathbf{F}_{\text{c},t} = \mathbb{E}_q[\partial f / \partial \mathbf{x}] = \boldsymbol{\Sigma}_{xf,t}^\top \boldsymbol{\Sigma}_{x,t}^{-1}$ (Stein's lemma, so $f$ is never differentiated) and $\mathbf{c}_t = \boldsymbol{\mu}_{f,t} - \mathbf{F}_{\text{c},t} \boldsymbol{\mu}_{x,t}$; `sde_kl_divergence` returns the drift path-KL $\frac{1}{2} \sum_t \Delta t_t \mathbb{E}_q \Vert f(\mathbf{x}_t) - \mathbf{F}_{\text{c},t} \mathbf{x}_t - \mathbf{c}_t \Vert^2_{\mathbf{B}_{\text{c},t}^{-1}}$ between the linear and the nonlinear SDE. Both take any gaussx integrator for the expectations.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LinearizedSDE, linearize_sde, sde_kl_divergence]

## Process noise

The exact discretisation $\boldsymbol{\Sigma}_{\eta,t} = \boldsymbol{\Sigma}_{x,\infty} - \mathbf{F}_t \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}_t^\top$, which is the forward direction of the discrete Lyapunov equation that [`discrete_lyapunov_solve`](linalg.md#gaussx.discrete_lyapunov_solve) inverts. The congruence is delegated to [`cov_transform`](linalg.md#gaussx.cov_transform), so passing *operators* returns a lazy operator with structure intact — matched `Kronecker` factors stay factorised, a diagonal $\boldsymbol{\Sigma}_{x,\infty}$ skips its $(D_x, D_x)$ materialization. Passing arrays returns an array.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [process_noise_covariance]
