"""Tests for `enkf_analysis`, the stochastic ensemble Kalman analysis step."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import (
    enkf_analysis,
    ensemble_kalman_gain,
    localization_matrix,
)
from gaussx._testing import random_pd_matrix


# ---------------------------------------------------------------------------
# Helpers: a small linear-Gaussian problem with an analytic posterior
# ---------------------------------------------------------------------------


def _linear_gaussian_problem(key, n_state=3, n_obs=2):
    """Prior N(mu, P), observation model (H, R), and the exact posterior."""
    k_mu, k_p, k_r, k_h, k_y = jr.split(key, 5)
    mu = jr.normal(k_mu, (n_state,))
    prior_cov = random_pd_matrix(k_p, n_state)
    obs_noise = random_pd_matrix(k_r, n_obs)
    obs_model = jr.normal(k_h, (n_obs, n_state))
    y = jr.normal(k_y, (n_obs,))

    innovation = obs_model @ prior_cov @ obs_model.T + obs_noise  # (M, M)
    gain = prior_cov @ obs_model.T @ jnp.linalg.inv(innovation)  # (N, M)
    post_mean = mu + gain @ (y - obs_model @ mu)
    post_cov = prior_cov - gain @ obs_model @ prior_cov
    return mu, prior_cov, obs_model, obs_noise, y, post_mean, post_cov, gain


def _draw_ensemble(key, mean, cov, n_ens):
    chol = jnp.linalg.cholesky(cov)
    return mean[None, :] + jr.normal(key, (n_ens, mean.shape[0])) @ chol.T


# ---------------------------------------------------------------------------
# Consistency: converges to the exact Kalman posterior as J -> infinity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_converges_to_exact_kalman_posterior(getkey):
    key = getkey()
    mu, prior_cov, obs_model, obs_noise, y, post_mean, post_cov, _ = (
        _linear_gaussian_problem(key)
    )
    k_ens, k_analysis = jr.split(getkey())
    n_ens = 200_000

    prior = _draw_ensemble(k_ens, mu, prior_cov, n_ens)  # (J, N)
    analysis = enkf_analysis(
        prior,
        prior @ obs_model.T,
        y,
        lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag),
        key=k_analysis,
    )

    mean_err = jnp.linalg.norm(jnp.mean(analysis, axis=0) - post_mean)
    mean_err /= jnp.linalg.norm(post_mean)
    cov_err = jnp.linalg.norm(jnp.cov(analysis.T, bias=False) - post_cov)
    cov_err /= jnp.linalg.norm(post_cov)

    assert mean_err < 0.02
    assert cov_err < 0.02


# ---------------------------------------------------------------------------
# Spread: the perturbation is what makes the analysis covariance (I-KH)P
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_analysis_spread_is_not_underdispersive(getkey):
    """The stochastic update gives (I-KH)P; the deterministic one is smaller.

    Dropping the observation perturbation would pass a mean-only test but
    shrink the analysis covariance to (I-KH)P(I-KH)^T -- short by K R K^T.
    This test is what catches that.
    """
    key = getkey()
    mu, prior_cov, obs_model, obs_noise, y, _, post_cov, gain = (
        _linear_gaussian_problem(key)
    )
    k_ens, k_analysis = jr.split(getkey())
    n_ens = 200_000
    noise_op = lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag)

    prior = _draw_ensemble(k_ens, mu, prior_cov, n_ens)  # (J, N)
    obs_prior = prior @ obs_model.T  # (J, M)
    analysis = enkf_analysis(prior, obs_prior, y, noise_op, key=k_analysis)

    # The deterministic (unperturbed) update, using the exact gain so the
    # contrast is analytic rather than a second Monte-Carlo estimate.
    deterministic = prior + (y[None, :] - obs_prior) @ gain.T  # (J, N)

    stochastic_cov = jnp.cov(analysis.T, bias=False)
    deterministic_cov = jnp.cov(deterministic.T, bias=False)

    # The stochastic analysis reproduces the true posterior covariance ...
    stochastic_err = jnp.linalg.norm(stochastic_cov - post_cov)
    stochastic_err /= jnp.linalg.norm(post_cov)
    assert stochastic_err < 0.02

    # ... while the deterministic one collapses to the Joseph-squared form.
    spread = jnp.eye(mu.shape[0]) - gain @ obs_model
    joseph = spread @ prior_cov @ spread.T  # (I - KH) P (I - KH)^T
    joseph_err = jnp.linalg.norm(deterministic_cov - joseph)
    joseph_err /= jnp.linalg.norm(joseph)
    assert joseph_err < 0.02

    # The shortfall is exactly K R K^T, so it is strictly under-dispersive.
    assert jnp.trace(deterministic_cov) < jnp.trace(stochastic_cov)
    assert jnp.allclose(post_cov - joseph, gain @ obs_noise @ gain.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Perturbation plumbing: determinism and the perturbed_obs / key equivalence
# ---------------------------------------------------------------------------


def _small_problem(getkey, n_ens=64, n_state=4, n_obs=3):
    k_prior, k_obs, k_y, k_r = jr.split(getkey(), 4)
    prior = jr.normal(k_prior, (n_ens, n_state))
    obs_prior = jr.normal(k_obs, (n_ens, n_obs))
    y = jr.normal(k_y, (n_obs,))
    noise = lx.MatrixLinearOperator(
        random_pd_matrix(k_r, n_obs), lx.positive_semidefinite_tag
    )
    return prior, obs_prior, y, noise


def test_same_key_is_deterministic(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    key = getkey()
    a = enkf_analysis(prior, obs_prior, y, noise, key=key)
    b = enkf_analysis(prior, obs_prior, y, noise, key=key)
    assert jnp.array_equal(a, b)


def test_different_keys_differ(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    a = enkf_analysis(prior, obs_prior, y, noise, key=getkey())
    b = enkf_analysis(prior, obs_prior, y, noise, key=getkey())
    assert not jnp.allclose(a, b)


def test_perturbed_obs_reproduces_the_key_path(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    key = getkey()
    from_key = enkf_analysis(prior, obs_prior, y, noise, key=key)

    # Rebuild the same realisation by hand and pass it in explicitly.
    n_ens, n_obs = obs_prior.shape
    chol = jnp.linalg.cholesky(noise.as_matrix())
    eps = jr.normal(key, (n_ens, n_obs), dtype=prior.dtype) @ chol.T
    from_explicit = enkf_analysis(
        prior, obs_prior, y, noise, perturbed_obs=y[None, :] + eps
    )
    assert jnp.allclose(from_key, from_explicit, atol=1e-12)


# ---------------------------------------------------------------------------
# Shapes, gain convention, localization
# ---------------------------------------------------------------------------


def test_shapes_with_distinct_state_and_obs_dims(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey, n_ens=32, n_state=7, n_obs=3)
    analysis = enkf_analysis(prior, obs_prior, y, noise, key=getkey())
    assert analysis.shape == (32, 7)


@pytest.mark.parametrize("bessel", [True, False])
def test_matches_gain_applied_by_hand(getkey, bessel):
    """Pins the update formula and the Bessel convention against the gain."""
    prior, obs_prior, y, noise = _small_problem(getkey)
    perturbed = y[None, :] + 0.1 * jr.normal(getkey(), obs_prior.shape)

    analysis = enkf_analysis(
        prior, obs_prior, y, noise, perturbed_obs=perturbed, bessel=bessel
    )
    gain = ensemble_kalman_gain(prior, obs_prior, noise, bessel=bessel)
    expected = prior + (perturbed - obs_prior) @ gain.T
    assert jnp.allclose(analysis, expected, atol=1e-10)


def test_bessel_flag_changes_the_result(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    perturbed = y[None, :] + 0.1 * jr.normal(getkey(), obs_prior.shape)
    kwargs = dict(perturbed_obs=perturbed)
    with_bessel = enkf_analysis(prior, obs_prior, y, noise, bessel=True, **kwargs)
    without = enkf_analysis(prior, obs_prior, y, noise, bessel=False, **kwargs)
    assert not jnp.allclose(with_bessel, without)


def test_all_ones_localization_matches_no_localization(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    perturbed = y[None, :] + 0.1 * jr.normal(getkey(), obs_prior.shape)
    n_state, n_obs = prior.shape[1], obs_prior.shape[1]

    plain = enkf_analysis(prior, obs_prior, y, noise, perturbed_obs=perturbed)
    localized = enkf_analysis(
        prior,
        obs_prior,
        y,
        noise,
        perturbed_obs=perturbed,
        localization=jnp.ones((n_state, n_obs)),
        obs_localization=jnp.ones((n_obs, n_obs)),
    )
    assert jnp.allclose(plain, localized, atol=1e-8)


def test_localization_suppresses_distant_updates(getkey):
    """A taper with a short radius must damp the far end of the state vector."""
    n_ens, n_state, n_obs = 128, 20, 1
    k_prior, k_y = jr.split(getkey(), 2)
    prior = jr.normal(k_prior, (n_ens, n_state))
    obs_prior = prior[:, :1]  # observe the first coordinate
    y = jr.normal(k_y, (n_obs,))
    noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(n_obs))
    perturbed = jnp.broadcast_to(y[None, :], (n_ens, n_obs))

    coords = jnp.arange(n_state, dtype=prior.dtype)[:, None]
    rho_xy = localization_matrix(coords, coords[:1], c=3.0)  # (N, M)

    plain = enkf_analysis(prior, obs_prior, y, noise, perturbed_obs=perturbed)
    localized = enkf_analysis(
        prior, obs_prior, y, noise, perturbed_obs=perturbed, localization=rho_xy
    )
    far_plain = jnp.abs(plain[:, -1] - prior[:, -1]).mean()
    far_local = jnp.abs(localized[:, -1] - prior[:, -1]).mean()
    assert far_local < 1e-12  # taper is exactly zero beyond the radius
    assert far_plain > far_local


# ---------------------------------------------------------------------------
# Errors and edge cases
# ---------------------------------------------------------------------------


def test_requires_exactly_one_perturbation_source(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    with pytest.raises(ValueError, match="exactly one of 'key'"):
        enkf_analysis(prior, obs_prior, y, noise)
    with pytest.raises(ValueError, match="exactly one of 'key'"):
        enkf_analysis(
            prior,
            obs_prior,
            y,
            noise,
            key=getkey(),
            perturbed_obs=jnp.zeros_like(obs_prior),
        )


def test_rejects_mismatched_ensemble_size(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    with pytest.raises(ValueError, match="same ensemble size"):
        enkf_analysis(prior, obs_prior[:-1], y, noise, key=getkey())


def test_rejects_mismatched_observation_shape(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    with pytest.raises(ValueError, match="observation must have shape"):
        enkf_analysis(prior, obs_prior, y[:-1], noise, key=getkey())


def test_rejects_mismatched_perturbed_obs_shape(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    with pytest.raises(ValueError, match="perturbed_obs must have shape"):
        enkf_analysis(
            prior, obs_prior, y, noise, perturbed_obs=jnp.zeros((obs_prior.shape[0], 1))
        )


def test_two_members_run_one_raises(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey, n_ens=2)
    assert jnp.all(
        jnp.isfinite(enkf_analysis(prior, obs_prior, y, noise, key=getkey()))
    )

    single_prior, single_obs = prior[:1], obs_prior[:1]
    with pytest.raises(ValueError, match="Bessel correction requires"):
        enkf_analysis(single_prior, single_obs, y, noise, key=getkey())


def test_jit(getkey):
    prior, obs_prior, y, noise = _small_problem(getkey)
    key = getkey()
    jitted = eqx.filter_jit(enkf_analysis)
    out = jitted(prior, obs_prior, y, noise, key=key)
    assert out.shape == prior.shape
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# The documented limitation: non-Gaussian bias that does not shrink with J
# ---------------------------------------------------------------------------


def _chipilski_problem():
    """The lognormal / logit-normal test case of Chipilski (2025).

    The latent state is ``zeta ~ N(mu, Sigma)``; the physical state is
    ``x = (exp(zeta_0), logistic(zeta_1))``. The first latent coordinate is
    observed with additive Gaussian noise of variance ``r``, i.e. the physical
    observation carries multiplicative lognormal noise.

    ``Sigma`` and ``r`` are recovered exactly from the latent posterior
    covariance the paper reports. ``mu`` and the observation are fixed by its
    reported posterior mean up to one free parameter, taken here as
    ``mu[0] = 0``. The resulting exact posterior mean in physical space
    reproduces the paper's ``[0.548062, 0.353937]`` to seven figures, which is
    what the assertions below anchor on.
    """
    prior_cov = jnp.array([[0.6, 0.25], [0.25, 0.4]])
    obs_var = 0.05
    mu = jnp.array([0.0, -23.0 / 60.0])
    y_latent = jnp.array([-0.6764811])

    gain = prior_cov[:, :1] / (prior_cov[0, 0] + obs_var)  # (2, 1)
    post_mean = mu + gain[:, 0] * (y_latent[0] - mu[0])
    post_cov = prior_cov - gain @ prior_cov[:1, :]

    # Exact posterior mean in physical space: closed form for the lognormal
    # coordinate, fine-grid quadrature for the logit-normal one.
    grid = jnp.linspace(-12.0, 12.0, 200_001)
    weights = jnp.exp(-0.5 * (grid - post_mean[1]) ** 2 / post_cov[1, 1])
    weights = weights / weights.sum()
    exact = jnp.array(
        [
            jnp.exp(post_mean[0] + post_cov[0, 0] / 2),
            (jax.nn.sigmoid(grid) * weights).sum(),
        ]
    )
    return mu, prior_cov, obs_var, y_latent, exact


def _to_physical(latent):
    return jnp.stack([jnp.exp(latent[:, 0]), jax.nn.sigmoid(latent[:, 1])], axis=1)


def test_chipilski_setup_reproduces_the_published_posterior():
    """Guards the test fixture itself against a silent drift in the constants."""
    *_, exact = _chipilski_problem()
    assert jnp.allclose(exact, jnp.array([0.548062, 0.353937]), atol=1e-6)


@pytest.mark.slow
def test_non_gaussian_bias_does_not_shrink_with_ensemble_size():
    """The physical-space update plateaus; the latent-space one converges.

    This is the limitation stated in the `enkf_analysis` docstring, and the
    property the downstream conjugate-transform filter exists to remove. A
    single-`J` mean test would not show it -- the point is the *plateau*.
    """
    mu, prior_cov, obs_var, y_latent, exact = _chipilski_problem()
    noise = lx.DiagonalLinearOperator(jnp.array([obs_var]))
    chol = jnp.linalg.cholesky(prior_cov)

    physical_errors, latent_errors = [], []
    for n_ens in (20_000, 200_000):
        k_prior, k_noise = jr.split(jr.key(0))
        latent = mu + jr.normal(k_prior, (n_ens, 2)) @ chol.T  # (J, 2)
        # One noise realisation, shared by both routes.
        perturbed_latent = y_latent[None, :] + jnp.sqrt(obs_var) * jr.normal(
            k_noise, (n_ens, 1)
        )  # (J, 1)

        # Latent coordinates: the prior is Gaussian there, so this is exact.
        latent_analysis = enkf_analysis(
            latent, latent[:, :1], y_latent, noise, perturbed_obs=perturbed_latent
        )
        latent_errors.append(
            jnp.linalg.norm(_to_physical(latent_analysis).mean(axis=0) - exact)
        )

        # Physical coordinates: the prior is lognormal / logit-normal there.
        physical = _to_physical(latent)  # (J, 2)
        perturbed_physical = jnp.exp(perturbed_latent)  # (J, 1)
        physical_noise = lx.DiagonalLinearOperator(
            jnp.array([jnp.var(perturbed_physical[:, 0])])
        )
        physical_analysis = enkf_analysis(
            physical,
            physical[:, :1],
            jnp.exp(y_latent),
            physical_noise,
            perturbed_obs=perturbed_physical,
        )
        physical_errors.append(jnp.linalg.norm(physical_analysis.mean(axis=0) - exact))

    # The latent-space error shrinks with J; the physical-space error does not.
    assert latent_errors[1] < latent_errors[0]
    assert latent_errors[1] < 0.001
    assert physical_errors[1] > 0.01
    assert physical_errors[1] > 0.5 * physical_errors[0]  # plateau, not decay
    assert physical_errors[1] > 10 * latent_errors[1]
