# Gaussian Processes

Layer 3 recipes for GP inference: conditioning, whitening, prediction caches,
pathwise (Matheron) sampling, variational bounds, and cross-validation — all
expressed over covariance *operators* so structured kernels keep their fast
paths end to end. The modelling shell (kernels with hyperparameter priors,
NumPyro sites) lives downstream; gaussx owns the math.

## Conditioning & prediction

The standard posterior

$$
\mu_* = K_{*f}\,K^{-1}(y - \mu), \qquad
\Sigma_{**} = K_{**} - K_{*f}\,K^{-1}K_{f*}
$$

plus a precomputed-cache variant for repeated test-time queries and a
Kronecker-structured path for separable kernels on grids.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [base_conditional, build_prediction_cache, PredictionCache, predict_mean, predict_variance, conditional_interpolate, kronecker_posterior_predictive, kronecker_mll]

## Pathwise sampling

Matheron's rule turns joint prior draws $(a, b)$ into posterior draws:
$a + \mathrm{Cov}(a,b)\,\mathrm{Cov}(b,b)^{-1}(\beta - b)$.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [matheron_update]

## Whitening

The whitened parameterization $u = Lv$, $v \sim \mathcal{N}(0, I)$ that keeps
sparse-variational optimization well-conditioned, and the whitened SVGP
predictive that consumes it.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [unwhiten, unwhiten_covariance, whiten_covariance, whitened_svgp_predict, svgp_variance_adjustment]

## Variational bounds & KL

ELBOs for Gaussian and Monte-Carlo variational families, the collapsed
(Titsias) sparse bound, and the Gaussian-to-Gaussian KL term.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [variational_elbo_gaussian, variational_elbo_mc, collapsed_elbo, gauss_kl]

## Cross-validation & diagnostics

LOVE-style cached predictive variances and closed-form leave-one-out
cross-validation from a single factorization.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [love_cache, love_variance, LOVECache, leave_one_out_cv, LOOResult]

## Multi-output projections

The orthogonal instantaneous linear mixing model (OILMM): project multi-output
observations into independent latent processes and back.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [oilmm_project, oilmm_back_project]
