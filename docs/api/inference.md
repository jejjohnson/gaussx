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

## Ensemble covariances & Kalman gain

Bessel-corrected empirical (cross-)covariances from ensemble members and the
ensemble Kalman gain built from them.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [ensemble_covariance, ensemble_cross_covariance, ensemble_kalman_gain, etkf_transform]

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
