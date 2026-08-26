# State-Space Models & Kalman

Layer 3 recipes for linear-Gaussian state-space models. Stationary 1-D GP
kernels with rational spectral densities admit exact SDE representations

$$
\dot{x}(t) = F\,x(t) + L\,w(t), \qquad f(t) = H\,x(t),
$$

turning $O(N^3)$ GP inference into $O(N d^3)$ Kalman filtering. This page
covers the SDE kernel zoo, the filters and smoothers (sequential, parallel
associative-scan, square-root, and steady-state), and the natural-parameter /
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
$P_\infty$ and `sde_autocovariance` can report $K(\tau)$.

`IntegratedWienerSDE` is not. Its drift is nilpotent, the marginal
variance grows without bound, and the covariance depends on both times
rather than on their difference — so there is no $P_\infty$ and no
$K(\tau)$. It reports `stationary = False` and `sde_params().P_inf =
None`, and the filter is started instead from an explicit
`initial_covariance` (diffuse by default). That is the local linear
trend prior at `order=1`: smooth, and linear unless the data push back,
with no lengthscale to choose.

Consumers should branch on `SDEKernel.stationary` rather than on whether
`P_inf` is `None` — the two are different questions, since a stationary
kernel may have no *closed form* for $P_\infty$ (a learned drift, say).
A `SumSDE` may mix the two: it is stationary only if all its components
are, and stacks their initial covariances block-diagonally. A
`ProductSDE` needs both factors' $P_\infty$ and rejects a
non-stationary one.

### Discretisation

Turning the continuous-time SDE into $x_k = A x_{k-1} + q_k$ takes two
routes, and which one applies depends on whether a stationary covariance
exists.

The default, `SDEKernel.discretise`, uses the *stationary* route
$Q = P_\infty - A P_\infty A^\top$ ([`process_noise_covariance`](#gaussx.process_noise_covariance),
documented under *Process noise* below). It is
exact and cheap, and every stationary kernel in the zoo above supplies
the $P_\infty$ it needs. `IntegratedWienerSDE` overrides `discretise`
with an exact closed form instead — its nilpotent drift makes the
exponential terminate — so it needs neither route.

`discretise_mfd` is the fallback for when that covariance is not
available — most importantly when $F$ is a **learned parameter** rather
than derived from a kernel. Recovering $P_\infty$ then means solving the
Lyapunov equation $F P + P F^\top + Q_c = 0$, which has a unique solution
**only if $\lambda_i(F) + \lambda_j(F) \neq 0$ for every pair of
eigenvalues**. That condition fails for any undamped oscillatory mode,
where $\lambda = \pm i\omega$ and so $\lambda + \bar\lambda = 0$
identically — `CosineSDE` has exactly that drift, and sidesteps it with a
closed form that a learned $F$ cannot use.

Matrix-fraction decomposition needs no $P_\infty$ at all: it recovers
both $A$ and $Q$ from a single $2d \times 2d$ matrix exponential (Van
Loan 1978) and is well defined for **every** $F$. Reach for it when the
drift is fitted; keep the stationary route otherwise. Note the
obstruction is degeneracy of that Sylvester system, not instability —
constraining $F$ to be Hurwitz would not fix it, and would forbid the
oscillatory modes MFD exists to support.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [discretise_mfd, discretise_mfd_sequence]

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

The gain is $K = C S^{-1}$, built from the integrator's *cross-covariance*
$C$, so no Jacobian is formed anywhere — that is what makes the EKF and
the UKF the same code. An integrator that does not supply a
cross-covariance is rejected rather than silently producing a zero gain.

Two behaviours differ from `kalman_filter`, both deliberately:

- **`log_likelihood` is a moment-matched surrogate**, not the exact
  marginal likelihood, because $S$ is the matched innovation covariance
  rather than the true one. It coincides with the exact value when the
  maps are affine. Maximising it to tune hyperparameters is standard
  practice for nonlinear Gaussian filters, but it is a surrogate.
- **The covariance update defaults to Joseph form** (`joseph=True`).
  $K = C S^{-1}$ is only approximately the optimal gain, and
  $P^- - K S K^\top$ is guaranteed PSD only *for* the optimal gain,
  whereas Joseph form is a sum of two PSD terms for any $K$. The effective
  observation matrix it needs is the statistical-linearisation gain
  $H_{\text{eff}} = C^\top (P^-)^{-1}$ — what
  [`statistical_linear_regression`](quadrature.md) returns as `A` — and
  its noise is $R + \Omega$, not $R$, with $\Omega$ the linearisation
  residual.

  With that residual included the two forms are analytically identical for
  any consistent matched joint, not merely for affine maps. **`joseph` is
  therefore a numerical choice, not a modelling one**: it selects how the
  same covariance is computed, and should not change results beyond
  floating point. Dropping $\Omega$ — the naive reading of Joseph here —
  would instead understate the posterior by $K \Omega K^\top$.

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
| `nonlinear_kalman_predict` | $m^-, P^- = \mathcal{T}[f](m, P)$, $P^- \mathrel{+}= Q$ |
| `nonlinear_kalman_update` | match $h$, then $K = C S^{-1}$ and the Gaussian update |
| `nonlinear_rts_step` | one RTS backward correction |

Each takes the same `integrator`, so the choice of filter carries through
unchanged. `nonlinear_kalman_update` returns its log-likelihood increment
rather than accumulating, leaving the accumulation to the caller.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [nonlinear_kalman_filter, nonlinear_rts_smoother, nonlinear_kalman_predict, nonlinear_kalman_update, nonlinear_rts_step, masked_moment_inputs]

## Kalman filtering & smoothing

The forward filter and RTS smoother, their $O(\log N)$ parallel
(associative-scan) counterparts, and the steady-state (infinite-horizon)
variants built on the discrete algebraic Riccati equation.

### Observation masks

`kalman_filter` and `parallel_kalman_filter` take an optional `mask`,
dispatched on its rank:

| shape | meaning |
|---|---|
| `(T,)` | **Per-step gate.** `False` runs the predict step only and contributes nothing to the log-likelihood — the usual way to predict on a merged train/test grid. |
| `(T, M)` | **Per-channel gate**, for partially observed multivariate series where different channels are measured at different times. |

The per-channel path marginalises unobserved channels *exactly*: row $i$ of
$H_t$ is zeroed, a unit block is substituted into $R_t$, and the residual
entry is set to zero. The innovation covariance is then block-diagonal in the
observed/masked split, so column $i$ of the gain vanishes and the masked
channel cannot move the state — the posterior reproduces the row-deleted
filter to machine precision, with no branching. A dummy block also
contributes $-\tfrac12 \log 2\pi$ per masked channel to the full-vector
density, which is stripped per step, so `log_likelihood` is the exact marginal
$\log p(y_{\mathrm{obs}})$ and is invariant to both the mask pattern and the
dummy variance.

An all-`False` row of a `(T, M)` mask is equivalent to a `False` entry in the
`(T,)` form, and `M == 1` is unambiguous either way. Masked entries of
`observations` are never read, so they may be `NaN`. Operator-typed
`obs_model` / `obs_noise` are materialised under a `(T, M)` mask, since
zeroing rows is inherently dense; `form="sqrt"` supports the `(T,)` mask only
and raises `NotImplementedError` otherwise. `rts_smoother` needs no mask of
its own — it consumes filtered/predicted moments, which are already
mask-aware.

### Mean-field (block-diagonal) filtering

When the state decomposes into $L$ independent blocks of size $d$ (e.g. a
multi-output temporal GP with one SDE per output), `meanfield_kalman_filter`
and `meanfield_rts_smoother` run $L$ parallel $d$-state filters under
`jax.vmap` — $O(T\,L\,d^3)$ total instead of the full filter's
$O(T\,L^3 d^3)$. The trade-off is the mean-field approximation: inputs and
posterior covariance are projected onto their diagonal blocks, so posterior
*cross-block* covariance is dropped (the returned $(T, D, D)$ covariances are
exactly zero off-block, and the log-likelihood is the sum of per-block
log-likelihoods). The approximation is exact when the true cross-block
dynamics are zero, which makes the decoupled case a useful consistency check
against `kalman_filter`. A `BlockDiag` operator whose sub-operators match the
blocking is split structurally, without materialising the full $(D, D)$
matrix; `parallel=True` routes each block through the associative-scan
filter/smoother.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EmissionModel, FilterState, kalman_filter, rts_smoother, kalman_gain, parallel_kalman_filter, parallel_rts_smoother, meanfield_kalman_filter, meanfield_rts_smoother, infinite_horizon_filter, infinite_horizon_smoother, InfiniteHorizonState, dare, DAREResult, pairwise_marginals]

## SpInGP

State-space (sparse-in-time) GP inference: marginal likelihood and posterior
through the SSM representation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [spingp_log_likelihood, spingp_posterior]

## Sites & natural parameters

Conjugate-computation VI (CVI) site updates and the conversions between SSM
moment, expectation, and natural parameterizations used by non-conjugate
temporal inference.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianSites, cvi_update_sites, sites_to_precision, cavity_from_marginal, site_natural_from_tilted, site_mean_var_from_natural, expectations_to_ssm, naturals_to_ssm, ssm_to_expectations, ssm_to_naturals]

## Process noise

The exact discretisation $Q = P_\infty - A P_\infty A^\top$, which is the
forward direction of the discrete Lyapunov equation that
[`discrete_lyapunov_solve`](linalg.md#gaussx.discrete_lyapunov_solve) inverts.
The congruence is delegated to
[`cov_transform`](linalg.md#gaussx.cov_transform), so passing *operators*
returns a lazy operator with structure intact — matched `Kronecker` factors stay
factorised, a diagonal $P_\infty$ skips its $(N, N)$ materialization. Passing
arrays returns an array.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [process_noise_covariance]
