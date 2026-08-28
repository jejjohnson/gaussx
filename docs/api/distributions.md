# Distributions & Exponential Family

Layer 2: Gaussian distributions over structured covariance operators, the sugar
operations that probabilistic code actually calls, and the exponential-family
(natural-parameter) view used by variational and EP-style inference.

## Multivariate normal distributions

NumPyro-compatible distributions whose covariance (or precision) is a lineax
operator, so `sample` / `log_prob` inherit every structured fast path.
`MultivariateNormalPrecision` carries $\Lambda = \Sigma^{-1}$ directly — the
natural home for natural-parameter guides, where materializing $\Sigma$ would
be wasted work. Both require `numpyro` to be installed.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MultivariateNormal, MultivariateNormalPrecision]

## Sequential distributions

The linear-Gaussian state-space model as a *density* rather than a set of
functions: `log_prob` is the Kalman marginal likelihood $\log p(y_{1:T})$
(delegated to [`kalman_filter`](ssm.md)), `sample` is ancestral forward
simulation, and `event_shape` is $(T, M)$ so `log_prob` returns a scalar with
no `.to_event()` wrapping. That makes them usable directly as a NumPyro
likelihood site.

`A`, `H`, `Q` and `R` each take a dense array *or* a lineax operator, exactly
matching [`kalman_filter`](ssm.md)'s contract — a structured $Q$ / $R$ keeps its
structure through the Cholesky in `sample`, and $H$ through the sandwich in
`variance`.

`MaskedLGSSM` carries a $(T, M)$ observation mask and returns the **exact**
marginal $\log p(y_{\mathrm{obs}})$ — not a bound — because
$p(y_{\mathrm{miss}} \mid y_{\mathrm{obs}})$ is closed-form Gaussian.
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

`MarkovGaussian` is the density over the *states* of a Gauss-Markov chain
rather than its observations: $x_0 \sim \mathcal{N}(\mu_0, P_0)$,
$x_{k+1} = A_k x_k + b_k + \varepsilon_k$, with `event_shape` $(T, d)$.
`sample` and `log_prob` use the chain factorisation at $O(T d^3)$ and never
form the joint covariance, so it serves as a prior over a latent trajectory
or as a structured variational guide. Its precision is block-tridiagonal;
`to_precision_form` / `from_precision_form` convert to and from the
$(\mu, \Lambda)$ layout that [`spingp_posterior`](ssm.md) returns, through
the [UDL factorisation](ssm.md#precision-form-chains), so a precision-form
posterior becomes a sampleable chain in one $O(T d^3)$ pass. Requires
`numpyro`.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MarkovGaussian]

## Gaussian sugar ops

$$
\log \mathcal{N}(x \mid \mu, \Sigma)
= -\tfrac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)
  - \tfrac12 \log|\Sigma| - \tfrac{N}{2}\log 2\pi
$$

evaluated through structured `solve` + `logdet`, plus entropy, quadratic
forms, KL divergences, conditioning, and the numerically stable Joseph-form
covariance update.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [gaussian_log_prob, gaussian_entropy, quadratic_form, kl_standard_normal, dist_kl_divergence, conditional, joseph_update, add_jitter, project]

## Exponential family

The Gaussian in natural form: $\eta_1 = \Lambda\mu$, $\eta_2 = -\tfrac12
\Lambda$. Conversions between mean/covariance, natural, and expectation
parameterizations — multivariate (operator-aware) and univariate (per-site
diagonal) — plus the log-partition, Fisher information, and sufficient
statistics that natural-gradient and EP updates are built from.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GaussianExpFam
        - to_natural
        - to_expectation
        - mean_cov_to_natural
        - natural_to_mean_cov
        - meanvar_to_natural
        - natural_to_meanvar
        - meanvar_to_expectation
        - expectation_to_meanvar
        - expectation_to_natural
        - natural_to_expectation
        - log_partition
        - fisher_info
        - sufficient_stats
        - kl_divergence
