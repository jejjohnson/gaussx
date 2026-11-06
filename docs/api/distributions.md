# Distributions & Exponential Family

Layer 2: Gaussian distributions over structured covariance operators, the sugar
operations that probabilistic code actually calls, and the exponential-family
(natural-parameter) view used by variational and EP-style inference.

## Multivariate normal distributions

NumPyro-compatible distributions whose covariance (or precision) is a lineax
operator, so `sample` / `log_prob` inherit every structured fast path.
`MultivariateNormalPrecision` carries $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ directly — the natural home for natural-parameter guides, where materializing $\boldsymbol{\Sigma}$ would be wasted work. Both require `numpyro` to be installed.

Both subclass `AbstractMultivariateNormal`, so code written against the base
works for either parameterisation. It provides `covariance_operator` and
`precision_operator` (one native, the other a lazy `inv`), numpyro's dense
`covariance_matrix` / `precision_matrix` / `scale_tril`, and `p.kl(q)` for any
mix of the two classes:

$$
\mathrm{KL}[p \Vert q] = \frac{1}{2} (\mathrm{tr}(\boldsymbol{\Lambda}_q \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)^\top \boldsymbol{\Lambda}_q (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p) - D_x + \log \det \boldsymbol{\Sigma}_q - \log \det \boldsymbol{\Sigma}_p)
$$

with $\boldsymbol{\Lambda}_q = \boldsymbol{\Sigma}_q^{-1}$ and $D_x$ the event size.

When `q` is precision-parameterised, $\boldsymbol{\Lambda}_q$ is its native operator, so
the trace and quadratic terms need no solve. `MultivariateNormalPrecision.variance`
takes `diag_inv`'s exact structured paths (for example, the $O(N d^3)$ selected
inverse of a `BlockTriDiag` precision) and densifies only an unstructured
precision.

With a stochastic `solver` (e.g. `CGSolver()`), `log_prob` and `entropy`
estimate the log-determinant from the strategy's fixed seed, so every call
reuses the same probes. That is fine for SVI gradients, but averaging calls
reduces no variance and MCMC targets a fixed pseudo-likelihood. To vary the
probes, pass `solver=KeyedSolver(CGSolver(), key)` with a fresh key per step
(see [Solvers](solvers.md)). The functional `gaussian_log_prob`,
`gaussian_entropy` and `kl_standard_normal` take `key=` directly.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractMultivariateNormal, MultivariateNormal, MultivariateNormalPrecision]

## Gaussian Markov random fields

`GaussianMRF` is $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$ with a structured precision, and `IntrinsicGMRF` the improper $\mathcal{N}(\boldsymbol{\mu}, (\tau \mathbf{R})^+)$ of RW1 / RW2 / Besag (ICAR) priors, with a structure matrix $\mathbf{R}$ and its null space $\mathbf{V}$ of $D_c$ columns. Every operation dispatches on the precision's structure ([GMRF precision builders](gmrf.md) return the right one):

- **sample**, in order: a factorisable precision (dense, `BlockTriDiag`, or a `SparseOperator` through its sparse Cholesky factor) gives $\boldsymbol{\mu} + \mathbf{L}_{\boldsymbol{\Lambda}}^{-\top} \mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$; a Kronecker-structured one (`Kronecker`, `KroneckerSum`, `SpectralFunction`) the symmetric root in its factor eigenvectors; anything else with `precision_factors` $\boldsymbol{\Lambda} = \sum_k \mathbf{F}_k^\top \mathbf{F}_k$ is sampled by **perturbation-optimisation**: $\mathbf{x} = \boldsymbol{\Lambda}^{-1} \sum_k \mathbf{F}_k^\top \mathbf{z}_k$ has covariance $\boldsymbol{\Lambda}^{-1} \boldsymbol{\Lambda} \boldsymbol{\Lambda}^{-1} = \boldsymbol{\Lambda}^{-1}$, one CG solve per draw and no factorisation (graph precisions come factored, $\mathbf{B}^\top \mathrm{diag}(\mathbf{w}) \mathbf{B}$ from an incidence matrix).
- **log_prob**: $\frac{1}{2} \log \det \boldsymbol{\Lambda} - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) - \frac{D_x}{2} \log 2\pi$, with $\log \det \boldsymbol{\Lambda}$ from `log_det_precision` when known in closed form (no stochastic estimate inside an MCMC density), else `logdet_strategy`, `solver` or the structural `logdet`. The intrinsic density is $\frac{D_x - D_c}{2} \log \tau - \frac{\tau}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{R} (\mathbf{x} - \boldsymbol{\mu})$, plus the $\tau$-free $\frac{1}{2} \log \det_+ \mathbf{R} - \frac{D_x - D_c}{2} \log 2\pi$ ([`pseudo_logdet`](primitives.md)) with `include_normalizer`.
- **marginal variances**: `diag_inv` (Takahashi on a sparse or banded factor, factor eigenvectors on a grid).
- **observations** $\mathbf{y} = \mathbf{H} \mathbf{x} + \boldsymbol{\varepsilon}$: `condition_on_observations` returns the posterior GMRF, $\boldsymbol{\Lambda} + \mathbf{H}^\top \boldsymbol{\Lambda}_\varepsilon \mathbf{H}$ (sparse on the union pattern for a sparse prior and `A`, matrix-free for a grid prior) with its mean from one solve.
- **constraints** $\mathbf{A}_c \mathbf{x} = \mathbf{e}$: `condition_on_constraints` returns a `ConstrainedGMRF`, conditioning by kriging (Rue & Held, 2005, §2.3.3): $\mathbf{x} \mapsto \mathbf{x} - \boldsymbol{\Lambda}^{-1} \mathbf{A}_c^\top (\mathbf{A}_c \boldsymbol{\Lambda}^{-1} \mathbf{A}_c^\top)^{-1} (\mathbf{A}_c \mathbf{x} - \mathbf{e})$ maps a draw to an exact draw from $\mathbf{x} \mid \mathbf{A}_c \mathbf{x} = \mathbf{e}$, with corrected marginal variances and the constrained density. `IntrinsicGMRF` defaults to the hard constraint `null_spaceᵀx = 0`; `constraint="soft"` puts a tight Gaussian on `null_spaceᵀx` instead, for NUTS.

An odd-length `rw2_structure(n)` has one decoupled padding node: build the
`IntrinsicGMRF` on all `n + 1` nodes with `null_space` zero on it, and drop
it from draws. All four require `numpyro` to be installed.

**BYM2.** Use `BYM2GMRF(R_star, tau, phi, null_space)` for the BYM2 pair $(\mathbf{b}, \mathbf{u}^\star)$, not an `IntrinsicGMRF` on `bym2_precision`. The joint precision $\boldsymbol{\Lambda}(\tau, \phi)$ has the null vector $(\sqrt{\phi / \tau} \mathbf{1}, \mathbf{1})$, which depends on $\boldsymbol{\theta} = (\tau, \phi)$. So the bracket `IntrinsicGMRF` drops by default is not constant (it holds $\frac{n}{2} \log \frac{\tau}{1 - \phi}$), and projecting draws along `null_space` does not remove the singular direction. `BYM2GMRF` uses the exact density on $\{\mathbf{V}^\top \mathbf{u}^\star = \mathbf{0}\}$, $-\frac{1}{2} \mathbf{r}^\top \boldsymbol{\Lambda} \mathbf{r} + \frac{n}{2} \log \frac{\tau}{1 - \phi}$ with $\mathbf{r} = (\mathbf{b}, \mathbf{u}^\star) - \boldsymbol{\mu}$ (plus the constant $\frac{1}{2} \log \det_+ \mathbf{R}^\star - \frac{2n - D_c}{2} \log 2\pi$ with `include_normalizer`), and draws $\mathbf{u}^\star$ from the ICAR on $\mathbf{R}^\star$, then $\mathbf{b}$ from its definition. It is an `IntrinsicGMRF`, so `laplace_mode` takes it as a prior, and its log-marginal is then correct in $(\tau, \phi)$:

```python
scale = gx.generalized_variance_scale(R, V)  # R: the Besag structure, n areas
R_star = scale * R
# θ-free, so once: log det₊(s R) = log det₊ R + (n − D_c) log s, with D_c = 1 here
log_pdet = gx.pseudo_logdet(R, structure="laplacian") + (n - 1) * jnp.log(scale)


def log_marginal(tau, phi):
    prior = gx.BYM2GMRF(R_star, tau, phi, V, include_normalizer=True, log_pdet=log_pdet)
    return gx.laplace_mode(
        prior, gx.PoissonLikelihood(counts), projector=A
    ).log_marginal
```

```python
import einx
import jax
import jax.numpy as jnp
import numpy as np

import gaussx as gx

# Gap-fill a cloudy sea-surface-temperature snapshot on an H × W grid
H, W = 30, 40
kappa, tau, alpha = gx.matern_spde_params(range=8.0, sigma=1.0, nu=1.0, d=2)
prior = gx.GaussianMRF(
    jnp.zeros(H * W), gx.spde_precision_grid((H, W), kappa, tau, alpha=alpha)
)
clear_idx = np.flatnonzero(np.random.default_rng(0).random(H * W) > 0.6)
n_clear = clear_idx.size
sst = jnp.sin(jnp.arange(H * W) / 50.0)  # stand-in for the measured field
A = gx.SparseOperator.from_coo(
    np.arange(n_clear), clear_idx, jnp.ones(n_clear), (n_clear, H * W)
)
post = prior.condition_on_observations(A, noise_precision=1 / 0.2**2, y=sst[clear_idx])
filled = einx.id("(h w) -> h w", post.loc, h=H)
sd = einx.id("(h w) -> h w", jnp.sqrt(post.marginal_variances()), h=H)
draws = post.sample(jax.random.PRNGKey(0), (20,))  # 20 plausible gap-filled fields

# An ICAR field with a hard sum-to-zero constraint per island group: the
# mainland path 0 - 1 - 2 - 3 and the island pair 4 - 5
senders, receivers = np.array([1, 2, 3, 5]), np.array([0, 1, 2, 4])
degree = np.bincount(np.r_[senders, receivers], minlength=6).astype(float)
R = gx.SparseOperator.from_coo(
    np.r_[np.arange(6), senders],
    np.r_[np.arange(6), receivers],
    jnp.asarray(np.r_[degree, -np.ones(4)]),
    (6, 6),
    symmetric=True,
)
null_space = jnp.zeros((6, 2)).at[:4, 0].set(0.5).at[4:, 1].set(2**-0.5)
icar = gx.IntrinsicGMRF(jnp.zeros(6), 2.0, R, null_space)  # kernellib: graph_null_space
u = icar.sample(jax.random.PRNGKey(1))  # null_spaceᵀ u == 0 to machine precision
lp = icar.log_prob(u)  # ((D_x − D_c) / 2) log τ − (τ / 2) uᵀ R u, with D_x − D_c = 4
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianMRF, IntrinsicGMRF, ConstrainedGMRF, BYM2GMRF]

## Sequential distributions

The linear-Gaussian state-space model as a *density* rather than a set of
functions: `log_prob` is the Kalman marginal likelihood $\log p(\mathbf{y}_{1:T})$ (delegated to [`kalman_filter`](ssm.md)), `sample` is ancestral forward simulation, and `event_shape` is $(T, D_y)$ so `log_prob` returns a scalar with no `.to_event()` wrapping. That makes them usable directly as a NumPyro likelihood site.

`A`, `H`, `Q` and `R` (the transition $\mathbf{F}$, the observation operator $\mathbf{H}$ and the covariances $\boldsymbol{\Sigma}_\eta$ and $\boldsymbol{\Sigma}_\varepsilon$) each take a dense array *or* a lineax operator, exactly matching [`kalman_filter`](ssm.md)'s contract — a structured $\boldsymbol{\Sigma}_\eta$ / $\boldsymbol{\Sigma}_\varepsilon$ keeps its structure through the Cholesky in `sample`, and $\mathbf{H}$ through the sandwich in `variance`.

`MaskedLGSSM` carries a $(T, D_y)$ observation mask and returns the **exact** marginal $\log p(\mathbf{y}_\text{obs})$ — not a bound — because $p(\mathbf{y}_\text{miss} \mid \mathbf{y}_\text{obs})$ is closed-form Gaussian.
`LGSSMFactory` is the `mask -> MaskedLGSSM` callable for conditional use; it
is an `equinox.Module` rather than a closure so the state-space parameters
stay visible to `equinox.filter_grad`.

To use one as a normalizing-flow base, wrap it with `gauss_flows.NumpyroBase`,
which already adapts any numpyro distribution — no bespoke adapter class is
needed on either side. All three require `numpyro` to be installed.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LGSSM, MaskedLGSSM, LGSSMFactory]

`MarkovGaussian` is the density over the *states* of a Gauss-Markov chain rather than its observations: $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_{x,0}, \boldsymbol{\Sigma}_{x,0})$, $\mathbf{x}_{t+1} = \mathbf{F}_t \mathbf{x}_t + \mathbf{c}_t + \boldsymbol{\eta}_{t+1}$, with `event_shape` $(T, D_x)$. `sample` and `log_prob` use the chain factorisation at $O(T D_x^3)$ and never form the joint covariance, so it serves as a prior over a latent trajectory or as a structured variational guide. Its precision is block-tridiagonal; `to_precision_form` / `from_precision_form` convert to and from the $(\boldsymbol{\mu}_x, \boldsymbol{\Lambda}_x)$ layout that [`spingp_posterior`](ssm.md) returns, through the [UDL factorisation](ssm.md#precision-form-chains), so a precision-form
posterior becomes a sampleable chain in one $O(T D_x^3)$ pass. Requires
`numpyro`.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MarkovGaussian]

## Gaussian sugar ops

$$
\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2} \log \det \boldsymbol{\Sigma} - \frac{D_x}{2} \log 2\pi
$$

evaluated through structured `solve` + `logdet`, plus entropy, quadratic
forms, KL divergences, conditioning, and the numerically stable Joseph-form
covariance update.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [gaussian_log_prob, gaussian_entropy, quadratic_form, kl_standard_normal, gaussian_kl, conditional, linear_gaussian_update, LinearGaussianPosterior, information_diagnostics, InformationDiagnostics, joseph_update, add_jitter, project]

## Structured sampling

`sample_mvn` draws $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L} \mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\mathbf{L} \mathbf{L}^\top = \boldsymbol{\Sigma}$, picking the factor $\mathbf{L}$ from the structure of the covariance operator: element-wise for a diagonal, per factor for a Kronecker product, per block for a block-diagonal, FFT circulant embedding for a Toeplitz, and $\mathbf{L}_{\mathbf{A}_0} \mathbf{z}_1 + \mathbf{U} \mathbf{D}^{1/2} \mathbf{z}_2$ for a low-rank update $\mathbf{A}_0 + \mathbf{U} \mathbf{D} \mathbf{U}^\top$, so a $10^4 \times 10^4$ Kronecker covariance is never factorised densely. Every branch is exact. A `SumOfKroneckers` therefore takes the dense symmetric square root: its matrix-free sampler, `sum_of_kroneckers_sample`, is a truncated Lanczos approximation and stays an explicit opt-in. The one-off helpers `toeplitz_sample` and `kronecker_sum_sample` also remain available for callers that want them directly.

`sample_joint_conditional` draws a partitioned Gaussian $(\mathbf{x}_A, \mathbf{x}_B)$ jointly without forming its $(N + M)^2$ covariance: the observed block $B$ through `sample_mvn`, so it keeps its structure, and the other, $A$, through the gain $\boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1}$ plus a square root of the Schur complement. Given an observed value it also returns conditional draws that share the same noise, which makes them exactly the [`matheron_update`](gp.md) of the joint draws.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [sample_mvn, sample_joint_conditional]

## Exponential family

The Gaussian in natural form: the natural parameters $(\boldsymbol{\Lambda} \boldsymbol{\mu}, -\frac{1}{2} \boldsymbol{\Lambda})$, held in the fields `eta1` and `eta2`. Conversions between mean/covariance, natural, and expectation
parameterizations — multivariate (operator-aware) and univariate (per-site
diagonal) — plus the log-partition, Fisher information, and sufficient
statistics that natural-gradient and EP updates are built from.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GaussianExpFam
        - to_mean_cov
        - mean_cov_to_natural
        - natural_to_mean_cov
        - mean_chol_to_natural
        - natural_to_mean_chol
        - mean_chol_to_expectation
        - expectation_to_mean_chol
        - expectation_to_natural
        - natural_to_expectation
        - log_partition
        - fisher_info
        - sufficient_stats
        - kl_divergence

## Deprecated aliases

These names still work but warn on use and will be removed in gaussx 0.7.0
(see the [deprecation policy](index.md#deprecation-policy)).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [to_natural, to_expectation]
