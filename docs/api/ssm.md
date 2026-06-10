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
      members: [SDEKernel, SDEParams, ConstantSDE, MaternSDE, PeriodicSDE, QuasiPeriodicSDE, CosineSDE, ProductSDE, SumSDE, sde_autocovariance]

## Kalman filtering & smoothing

The forward filter and RTS smoother, their $O(\log N)$ parallel
(associative-scan) counterparts, and the steady-state (infinite-horizon)
variants built on the discrete algebraic Riccati equation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [EmissionModel, FilterState, kalman_filter, rts_smoother, kalman_gain, parallel_kalman_filter, parallel_rts_smoother, infinite_horizon_filter, infinite_horizon_smoother, InfiniteHorizonState, dare, DAREResult, pairwise_marginals]

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

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [process_noise_covariance]
