# Quadrature & Moment Matching

Gaussian integration: $\mathbb{E}_{x \sim \mathcal{N}(\mu, \Sigma)}[f(x)]$ via
deterministic rules (Gauss-Hermite, unscented / cubature, Taylor) or Monte
Carlo, behind one `AbstractIntegrator` interface. Everything that needs an
expectation — expected log-likelihoods, EP tilted moments, uncertain-input GP
predictions — takes an integrator argument, so swapping the rule never touches
the model.

## State & integrators

`GaussianState` pairs a mean with a covariance *operator*; integrators
propagate functions of it.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianState, PropagationResult, AbstractIntegrator, GaussHermiteIntegrator, UnscentedIntegrator, TaylorIntegrator, MonteCarloIntegrator, AssumedDensityFilter]

## Quadrature rules

The raw point sets behind the integrators, for when a recipe needs direct
control.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [gauss_hermite_points, cubature_points, sigma_points]

## Likelihoods

Observation models with quadrature-friendly `log_prob` surfaces, shared by the
expectation helpers and the SSM / CVI recipes.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractLikelihood, GaussianLikelihood, HeteroscedasticGaussianLikelihood, BernoulliLikelihood, PoissonLikelihood, SoftmaxLikelihood, StudentTLikelihood]

## Expectations & EP moments

Expected log-likelihoods (the ELL term of every ELBO), generic mean / gradient /
cost expectations, and the tilted-moment matching at the heart of expectation
propagation.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [elbo, expected_log_likelihood, log_likelihood_expectation, mean_expectation, gradient_expectation, cost_expectation, ep_tilted_moments]

## Kernel expectations & uncertain-input GPs

The $\Psi$-statistics $\Psi_0 = \mathbb{E}[k(x,x)]$, $\Psi_1 = \mathbb{E}[k(x,
X)]$, $\Psi_2 = \mathbb{E}[k(x,\cdot)k(x,\cdot)^\top]$ and the GP / SVGP / VGP /
BGPLVM predictive equations for inputs that are themselves Gaussian.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [kernel_expectations, compute_psi_statistics, AnalyticalPsiStatistics, uncertain_gp_predict, uncertain_gp_predict_mc, uncertain_svgp_predict, uncertain_vgp_predict, uncertain_bgplvm_predict]
