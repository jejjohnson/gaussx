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
