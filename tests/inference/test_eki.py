"""Tests for the ensemble Kalman inversion step and its two helpers.

Evidence-first: every mathematical claim in the `eki_step` /
`tikhonov_augment` / `discrepancy_step_size` docstrings is a test here, not
a docstring assertion. The load-bearing one is `test_tempering_is_exact`:
the tempered product of Gaussian updates reproduces the exact posterior
only when the schedule sums to one.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import (
    BlockDiag,
    Kronecker,
    ScaledOperator,
    discrepancy_step_size,
    eki_step,
    enkf_analysis,
    etkf_transform,
    localization_matrix,
    tikhonov_augment,
)
from gaussx._primitives._cholesky import DenseFallbackWarning
from gaussx._primitives._sqrt import dense_symmetric_sqrt
from gaussx._testing import random_pd_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exact_ensemble(mean, cov, n_ens):
    """An ensemble whose empirical moments are *exactly* ``(mean, cov)``.

    The population-limit tests need a gain built from the true moments, not a
    sampled approximation, and drawing ``J`` large enough to get there costs
    seconds per test and still only gets to ``1 / sqrt(J)``. Instead build the
    anomalies as ``sqrt(J - 1) * Omega L``, with ``L L^T = cov`` and ``Omega``
    an orthonormal ``(J, N)`` basis of the mean-zero subspace: then
    ``X'^T X' / (J - 1) = cov`` and ``sum_j X'_j = 0`` to round-off, for any
    ``J >= N + 1``.
    """
    n_state = mean.shape[0]
    if n_ens < n_state + 1:
        raise ValueError("need J >= N + 1 for a full-rank exact ensemble")
    # QR of [1, e_1, ..., e_N]: the first column of Q spans the ones-vector, so
    # the next N are orthonormal *and* orthogonal to it.
    basis = jnp.concatenate([jnp.ones((n_ens, 1)), jnp.eye(n_ens)[:, :n_state]], axis=1)
    q, _ = jnp.linalg.qr(basis)
    omega = q[:, 1 : n_state + 1]  # (J, N)
    factor = dense_symmetric_sqrt(cov)  # symmetric, so L L^T = L L = cov
    return mean[None, :] + jnp.sqrt(n_ens - 1.0) * omega @ factor


def _empirical(ensemble):
    """``(mean, covariance)`` with the ``1 / (J - 1)`` divisor."""
    n_ens = ensemble.shape[0]
    mean = jnp.mean(ensemble, axis=0)
    anomalies = ensemble - mean[None, :]
    return mean, anomalies.T @ anomalies / (n_ens - 1)


def _exact_posterior(prior_mean, prior_cov, obs_model, obs_noise, observation):
    """Linear-Gaussian posterior in information form."""
    prior_prec = jnp.linalg.inv(prior_cov)
    noise_prec = jnp.linalg.inv(obs_noise)
    post_prec = prior_prec + obs_model.T @ noise_prec @ obs_model
    post_cov = jnp.linalg.inv(post_prec)
    post_mean = post_cov @ (
        prior_prec @ prior_mean + obs_model.T @ noise_prec @ observation
    )
    return post_mean, post_cov


def _problem(key, n_state=3, n_obs=2, n_ens=8):
    """A small linear-Gaussian inverse problem with an exact-moment ensemble."""
    k_m, k_c, k_r, k_a, k_y = jr.split(key, 5)
    prior_mean = jr.normal(k_m, (n_state,))
    prior_cov = random_pd_matrix(k_c, n_state)
    obs_noise = random_pd_matrix(k_r, n_obs)
    obs_model = jr.normal(k_a, (n_obs, n_state))
    observation = jr.normal(k_y, (n_obs,))
    particles = _exact_ensemble(prior_mean, prior_cov, n_ens)
    return particles, obs_model, obs_noise, observation, prior_mean, prior_cov


# ---------------------------------------------------------------------------
# 1. Reduction to the existing steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dense_innovation", [None, True, False])
def test_reduces_to_enkf_analysis(getkey, dense_innovation):
    """``dt=1``, ``step=None``, same perturbations => bitwise `enkf_analysis`."""
    particles = jr.normal(getkey(), (6, 4))
    obs_particles = jr.normal(getkey(), (6, 5))
    observation = jr.normal(getkey(), (5,))
    obs_noise = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), 5), lx.positive_semidefinite_tag
    )
    perturbed_obs = observation[None, :] + 0.3 * jr.normal(getkey(), (6, 5))

    kwargs = dict(perturbed_obs=perturbed_obs, dense_innovation=dense_innovation)
    got = eki_step(particles, obs_particles, observation, obs_noise, dt=1, **kwargs)
    expected = enkf_analysis(particles, obs_particles, observation, obs_noise, **kwargs)
    assert jnp.array_equal(got, expected)


def test_reduces_to_enkf_analysis_with_localization(getkey):
    """The taper path reduces bitwise too, not just the structured one."""
    particles = jr.normal(getkey(), (6, 4))
    obs_particles = jr.normal(getkey(), (6, 5))
    observation = jr.normal(getkey(), (5,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 5))
    perturbed_obs = observation[None, :] + 0.3 * jr.normal(getkey(), (6, 5))
    taper = localization_matrix(
        jnp.linspace(0, 1, 4)[:, None], jnp.linspace(0, 1, 5)[:, None], c=0.6
    )

    kwargs = dict(perturbed_obs=perturbed_obs, localization=taper)
    got = eki_step(particles, obs_particles, observation, obs_noise, **kwargs)
    expected = enkf_analysis(particles, obs_particles, observation, obs_noise, **kwargs)
    assert jnp.array_equal(got, expected)


def test_reduces_to_etkf_transform(getkey):
    """``deterministic=True``, ``dt=1``, ``step=None`` => plain `etkf_transform`."""
    particles = jr.normal(getkey(), (7, 4))
    obs_particles = jr.normal(getkey(), (7, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), 3), lx.positive_semidefinite_tag
    )

    got = eki_step(particles, obs_particles, observation, obs_noise, deterministic=True)

    w_mean, transform = etkf_transform(obs_particles, observation, obs_noise)
    mean = jnp.mean(particles, axis=0)
    anomalies = particles - mean[None, :]
    expected = (mean + w_mean @ anomalies)[None, :] + transform @ anomalies

    assert jnp.allclose(got, expected, atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# 3. Tempering exactness -- the load-bearing test
# ---------------------------------------------------------------------------


def test_tempering_is_exact(getkey):
    """A schedule summing to one reproduces the exact linear-Gaussian posterior."""
    particles, obs_model, obs_noise, observation, prior_mean, prior_cov = _problem(
        getkey(), n_state=3, n_obs=2, n_ens=8
    )
    noise_op = lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag)

    for dt in (0.25, 0.25, 0.5):
        particles = eki_step(
            particles,
            particles @ obs_model.T,
            observation,
            noise_op,
            dt=dt,
            deterministic=True,
        )

    post_mean, post_cov = _exact_posterior(
        prior_mean, prior_cov, obs_model, obs_noise, observation
    )
    got_mean, got_cov = _empirical(particles)
    assert jnp.allclose(got_mean, post_mean, atol=1e-6, rtol=0.0)
    assert jnp.allclose(got_cov, post_cov, atol=1e-6, rtol=0.0)


def test_schedule_not_summing_to_one_is_not_the_posterior(getkey):
    """The ``sum(dt) = 1`` condition is load-bearing, not decorative."""
    particles, obs_model, obs_noise, observation, prior_mean, prior_cov = _problem(
        getkey(), n_state=3, n_obs=2, n_ens=8
    )
    noise_op = lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag)

    for dt in (0.25, 0.25):  # sums to 0.5
        particles = eki_step(
            particles,
            particles @ obs_model.T,
            observation,
            noise_op,
            dt=dt,
            deterministic=True,
        )

    post_mean, _ = _exact_posterior(
        prior_mean, prior_cov, obs_model, obs_noise, observation
    )
    got_mean, got_cov = _empirical(particles)
    assert not jnp.allclose(got_mean, post_mean, atol=1e-3)

    # It is exactly the posterior of the *half-weighted* likelihood, i.e. 2R.
    half_mean, half_cov = _exact_posterior(
        prior_mean, prior_cov, obs_model, 2.0 * obs_noise, observation
    )
    assert jnp.allclose(got_mean, half_mean, atol=1e-6, rtol=0.0)
    assert jnp.allclose(got_cov, half_cov, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# 4. Structured operators dispatch
# ---------------------------------------------------------------------------


def _structured_noises(key):
    """``(name, operator)`` pairs, each with a structured solve, all (9, 9)."""
    k1, k2 = jr.split(key)
    diagonal = lx.DiagonalLinearOperator(jnp.linspace(0.5, 2.0, 9))
    block = BlockDiag(
        lx.DiagonalLinearOperator(jnp.linspace(0.3, 1.0, 4)),
        lx.MatrixLinearOperator(random_pd_matrix(k1, 5), lx.positive_semidefinite_tag),
    )
    kron = Kronecker(
        lx.MatrixLinearOperator(random_pd_matrix(k2, 3), lx.positive_semidefinite_tag),
        lx.DiagonalLinearOperator(jnp.linspace(0.4, 1.2, 3)),
    )
    return [("diagonal", diagonal), ("block_diag", block), ("kronecker", kron)]


@pytest.mark.parametrize("name", ["diagonal", "block_diag", "kronecker"])
def test_structured_obs_noise_dispatches(getkey, name):
    """A structured ``R / dt`` stays structured and matches the dense answer."""
    key = getkey()
    obs_noise = dict(_structured_noises(key))[name]
    particles = jr.normal(getkey(), (6, 4))
    obs_particles = jr.normal(getkey(), (6, 9))
    observation = jr.normal(getkey(), (9,))
    perturbed_obs = observation[None, :] + 0.3 * jr.normal(getkey(), (6, 9))
    dt = 0.4

    with warnings.catch_warnings():
        warnings.simplefilter("error", DenseFallbackWarning)
        got = eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            dt=dt,
            perturbed_obs=perturbed_obs,
            dense_innovation=False,
        )

    dense = lx.MatrixLinearOperator(obs_noise.as_matrix(), lx.positive_semidefinite_tag)
    expected = eki_step(
        particles,
        obs_particles,
        observation,
        dense,
        dt=dt,
        perturbed_obs=perturbed_obs,
        dense_innovation=False,
    )
    assert jnp.allclose(got, expected, atol=1e-8, rtol=0.0)


def test_kronecker_prior_cov_dispatches(getkey):
    """A `gaussx.Kronecker` ``C0`` survives `tikhonov_augment` into the solve."""
    prior_cov = Kronecker(
        lx.MatrixLinearOperator(
            random_pd_matrix(getkey(), 2), lx.positive_semidefinite_tag
        ),
        lx.DiagonalLinearOperator(jnp.array([0.5, 1.5, 2.5])),
    )
    particles = jr.normal(getkey(), (6, 6))
    obs_particles = jr.normal(getkey(), (6, 4))
    observation = jr.normal(getkey(), (4,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 4))
    prior_mean = jr.normal(getkey(), (6,))
    perturbed_obs = jr.normal(getkey(), (6, 10))

    aug = tikhonov_augment(
        particles, obs_particles, observation, obs_noise, prior_mean, prior_cov
    )
    obs_aug, y_aug, noise_aug = aug
    assert isinstance(noise_aug, BlockDiag)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DenseFallbackWarning)
        got = eki_step(
            particles,
            obs_aug,
            y_aug,
            noise_aug,
            perturbed_obs=perturbed_obs,
            dense_innovation=False,
        )

    dense = lx.MatrixLinearOperator(noise_aug.as_matrix(), lx.positive_semidefinite_tag)
    expected = eki_step(
        particles,
        obs_aug,
        y_aug,
        dense,
        perturbed_obs=perturbed_obs,
        dense_innovation=False,
    )
    assert jnp.allclose(got, expected, atol=1e-8, rtol=0.0)


def test_block_diag_step_dispatches(getkey):
    """A `gaussx.BlockDiag` ``step`` matches its dense materialisation."""
    particles = jr.normal(getkey(), (6, 5))
    obs_particles = jr.normal(getkey(), (6, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 3))
    perturbed_obs = observation[None, :] + 0.3 * jr.normal(getkey(), (6, 3))
    step = BlockDiag(
        ScaledOperator(
            lx.IdentityLinearOperator(jax.eval_shape(lambda: jnp.zeros(2))), 0.3
        ),
        ScaledOperator(
            lx.IdentityLinearOperator(jax.eval_shape(lambda: jnp.zeros(3))), 1.7
        ),
    )

    kwargs = dict(perturbed_obs=perturbed_obs, dt=0.5)
    got = eki_step(
        particles, obs_particles, observation, obs_noise, step=step, **kwargs
    )
    expected = eki_step(
        particles,
        obs_particles,
        observation,
        obs_noise,
        step=lx.MatrixLinearOperator(step.as_matrix()),
        **kwargs,
    )
    assert jnp.allclose(got, expected, atol=1e-8, rtol=0.0)


# ---------------------------------------------------------------------------
# 5. Per-block step rates
# ---------------------------------------------------------------------------


def test_step_scales_each_block_of_the_increment(getkey):
    """One step's increment is linear in ``Lambda``, blockwise."""
    particles = jr.normal(getkey(), (6, 5))
    obs_particles = jr.normal(getkey(), (6, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 3))
    perturbed_obs = observation[None, :] + 0.3 * jr.normal(getkey(), (6, 3))
    rate_a, rate_b = 0.3, 1.7
    step = BlockDiag(
        lx.DiagonalLinearOperator(rate_a * jnp.ones(2)),
        lx.DiagonalLinearOperator(rate_b * jnp.ones(3)),
    )

    kwargs = dict(perturbed_obs=perturbed_obs, dt=0.5)
    base = (
        eki_step(particles, obs_particles, observation, obs_noise, **kwargs) - particles
    )
    scaled = (
        eki_step(particles, obs_particles, observation, obs_noise, step=step, **kwargs)
        - particles
    )

    assert jnp.allclose(scaled[:, :2], rate_a * base[:, :2], atol=1e-12, rtol=0.0)
    assert jnp.allclose(scaled[:, 2:], rate_b * base[:, 2:], atol=1e-12, rtol=0.0)


def test_deterministic_step_scales_the_anomaly_increment(getkey):
    """``Lambda`` acts on the anomaly *increment*, so ``Lambda = I`` is the ETKF."""
    particles = jr.normal(getkey(), (7, 4))
    obs_particles = jr.normal(getkey(), (7, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 3))

    identity = lx.DiagonalLinearOperator(jnp.ones(4))
    kwargs = dict(deterministic=True, dt=0.6)
    with_identity = eki_step(
        particles, obs_particles, observation, obs_noise, step=identity, **kwargs
    )
    without = eki_step(particles, obs_particles, observation, obs_noise, **kwargs)
    assert jnp.allclose(with_identity, without, atol=1e-12, rtol=0.0)

    rate = 0.25
    scaled = eki_step(
        particles,
        obs_particles,
        observation,
        obs_noise,
        step=lx.DiagonalLinearOperator(rate * jnp.ones(4)),
        **kwargs,
    )
    assert jnp.allclose(
        scaled - particles, rate * (without - particles), atol=1e-12, rtol=0.0
    )


# ---------------------------------------------------------------------------
# 6. Fixed-point invariance
# ---------------------------------------------------------------------------


def test_fixed_point_mean_is_invariant_to_step(getkey):
    """At ``K (y - Gbar) = 0`` the mean does not move, for any invertible Lambda."""
    key = getkey()
    n_state, n_obs, n_ens = 3, 2, 8
    prior_mean = jr.normal(key, (n_state,))
    prior_cov = random_pd_matrix(getkey(), n_state)
    obs_model = jr.normal(getkey(), (n_obs, n_state))
    obs_noise = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), n_obs), lx.positive_semidefinite_tag
    )
    particles = _exact_ensemble(prior_mean, prior_cov, n_ens)
    obs_particles = particles @ obs_model.T
    # The fixed point: the observation the ensemble mean already predicts.
    observation = jnp.mean(obs_particles, axis=0)

    reference = jnp.mean(
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            dt=0.4,
        ),
        axis=0,
    )
    assert jnp.allclose(reference, prior_mean, atol=1e-10, rtol=0.0)

    for step in (
        lx.DiagonalLinearOperator(jnp.array([0.1, 5.0, 0.7])),
        BlockDiag(
            lx.DiagonalLinearOperator(jnp.array([3.0])),
            lx.MatrixLinearOperator(random_pd_matrix(getkey(), 2)),
        ),
    ):
        got = eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            dt=0.4,
            step=step,
        )
        assert jnp.allclose(jnp.mean(got, axis=0), prior_mean, atol=1e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# 7. TEKI recovers the prior-weighted MAP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spread", [1e6, 1e8])
def test_tikhonov_recovers_the_regularised_map(getkey, spread):
    """One augmented step from a diffuse ensemble lands on the TEKI objective's MAP.

    The augmented update is the exact posterior mean under the *ensemble*
    prior ``N(ubar, C_ens)``, so it minimises the stated objective plus an
    extra ``1/2 ||u - ubar||^2_{C_ens}``. That term vanishes as
    ``C_ens -> inf``: the error is ``O(1 / spread)``, which is why this is
    parametrised over two decades rather than asserted at one.
    """
    key = getkey()
    n_state, n_obs, n_ens = 3, 2, 8
    obs_model = jr.normal(key, (n_obs, n_state))
    obs_noise = random_pd_matrix(getkey(), n_obs)
    prior_cov = random_pd_matrix(getkey(), n_state)
    prior_mean = jr.normal(getkey(), (n_state,))
    observation = jr.normal(getkey(), (n_obs,))

    particles = _exact_ensemble(jnp.zeros(n_state), spread * jnp.eye(n_state), n_ens)
    obs_aug, y_aug, noise_aug = tikhonov_augment(
        particles,
        particles @ obs_model.T,
        observation,
        lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag),
        prior_mean,
        lx.MatrixLinearOperator(prior_cov, lx.positive_semidefinite_tag),
    )
    updated = eki_step(particles, obs_aug, y_aug, noise_aug, deterministic=True)

    # MAP of 1/2 ||y - A u||^2_R + 1/2 ||u - m0||^2_{C0}.
    noise_prec = jnp.linalg.inv(obs_noise)
    prior_prec = jnp.linalg.inv(prior_cov)
    hessian = obs_model.T @ noise_prec @ obs_model + prior_prec
    expected = jnp.linalg.solve(
        hessian, obs_model.T @ noise_prec @ observation + prior_prec @ prior_mean
    )
    got = jnp.mean(updated, axis=0)
    assert jnp.allclose(got, expected, atol=2e4 / spread, rtol=0.0)


def test_tikhonov_augment_shapes_and_blocks(getkey):
    """The triple is a plain concatenation plus a `gaussx.BlockDiag`."""
    particles = jr.normal(getkey(), (5, 3))
    obs_particles = jr.normal(getkey(), (5, 2))
    observation = jr.normal(getkey(), (2,))
    obs_noise = lx.DiagonalLinearOperator(jnp.array([0.5, 2.0]))
    prior_mean = jr.normal(getkey(), (3,))
    prior_cov = lx.DiagonalLinearOperator(jnp.array([1.0, 3.0, 5.0]))

    obs_aug, y_aug, noise_aug = tikhonov_augment(
        particles, obs_particles, observation, obs_noise, prior_mean, prior_cov
    )
    assert obs_aug.shape == (5, 5)
    assert y_aug.shape == (5,)
    assert jnp.array_equal(obs_aug[:, :2], obs_particles)
    assert jnp.array_equal(obs_aug[:, 2:], particles)
    assert jnp.array_equal(y_aug, jnp.concatenate([observation, prior_mean]))
    assert jnp.allclose(
        noise_aug.as_matrix(),
        jnp.diag(jnp.array([0.5, 2.0, 1.0, 3.0, 5.0])),
    )


# ---------------------------------------------------------------------------
# 8. The data misfit controller
# ---------------------------------------------------------------------------


def test_discrepancy_step_size_matches_the_paper_formula(getkey):
    """Iglesias & Yang (2021) eq. (14), against a direct computation."""
    obs_particles = jr.normal(getkey(), (12, 4))
    observation = jr.normal(getkey(), (4,))
    noise = random_pd_matrix(getkey(), 4)
    obs_noise = lx.MatrixLinearOperator(noise, lx.positive_semidefinite_tag)

    residuals = observation[None, :] - obs_particles
    misfit = 0.5 * jnp.einsum(
        "jm,mn,jn->j", residuals, jnp.linalg.inv(noise), residuals
    )
    expected = jnp.minimum(
        jnp.maximum(
            4 / (2 * jnp.mean(misfit)),
            jnp.sqrt(4 / (2 * jnp.var(misfit, ddof=1))),
        ),
        0.7,
    )
    got = discrepancy_step_size(
        obs_particles, observation, obs_noise, remaining=jnp.asarray(0.7)
    )
    assert jnp.allclose(got, expected, atol=1e-12, rtol=0.0)


def test_discrepancy_step_size_uses_per_particle_misfits():
    """The C2 term is the variance *over particles*, which is what distinguishes
    the paper's rule (eq. 13-14) from the misfit of the ensemble mean.

    Both ensembles below have mean misfit ``Phi = 5`` and so the same accuracy
    candidate ``M / 2 Phi = 0.4``, and both have the same ensemble-mean misfit
    direction. They differ only in the *spread* of ``Phi`` across members, and
    that alone moves the step by more than a factor of two. A rule reading the
    mean misfit twice could not tell them apart.
    """
    n_obs = 4
    obs_noise = lx.DiagonalLinearOperator(jnp.ones(n_obs))
    observation = jnp.zeros(n_obs)

    def ensemble(squared_radii):
        # Phi_j = ||G_j||^2 / 2 with R = I and y = 0, so the radii set the
        # misfits directly. Spread across axes to keep the mean at the origin.
        axes = jnp.eye(n_obs)
        return jnp.stack(
            [
                jnp.sqrt(r) * s * axes[i % n_obs]
                for i, (r, s) in enumerate(
                    zip(squared_radii, [1, -1, 1, -1], strict=True)
                )
            ]
        )

    flat = ensemble([10.0, 10.0, 10.0, 10.0])  # var(Phi) = 0
    spread = ensemble([2.0, 6.0, 14.0, 18.0])  # same mean, var(Phi) = 40 / 3

    remaining = jnp.asarray(1.0)
    for members in (flat, spread):
        misfit = 0.5 * jnp.sum(members**2, axis=-1)
        assert jnp.allclose(jnp.mean(misfit), 5.0)

    # var = 0 => C2 is infinite => the max saturates and the budget is returned.
    assert jnp.allclose(
        discrepancy_step_size(flat, observation, obs_noise, remaining=remaining),
        1.0,
    )
    # var = 40 / 3 => C2 = sqrt(4 / (80 / 3)) = 0.387 < 0.4, so C1 binds.
    assert jnp.allclose(
        discrepancy_step_size(spread, observation, obs_noise, remaining=remaining),
        0.4,
    )


@pytest.mark.parametrize("remaining", [1.0, 0.3, 0.05])
def test_discrepancy_step_size_is_bounded(getkey, remaining):
    """Positive, and never more than what is left of the tempering budget."""
    obs_particles = 30.0 * jr.normal(getkey(), (20, 6))
    observation = jr.normal(getkey(), (6,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.1, 1.0, 6))
    dt = discrepancy_step_size(
        obs_particles, observation, obs_noise, remaining=jnp.asarray(remaining)
    )
    assert 0.0 < float(dt) <= remaining


@pytest.mark.parametrize("remaining", [1.0, 0.4])
def test_discrepancy_step_size_at_the_discrepancy_level(remaining):
    """Mean misfit exactly ``M / 2`` and a spread that keeps C2 below one:
    the accuracy criterion binds at ``dt = 1``, so the step is ``min(1, rem)``."""
    n_obs = 4
    obs_noise = lx.DiagonalLinearOperator(jnp.ones(n_obs))
    observation = jnp.zeros(n_obs)
    # Phi_j = ||G_j||^2 / 2. Put half the members at radius^2 = M - s and half
    # at M + s: the mean of Phi is M / 2 exactly, and the variance is s^2 / 4.
    # s = 2 gives sigma^2 = 1 <= M / 2 = 2, so sqrt(M / 2 sigma^2) >= 1 fails
    # to bind only if it is <= the accuracy term -- pick s large enough.
    spread = 4.0
    radii = jnp.array([n_obs - spread, n_obs + spread])
    obs_particles = jnp.stack(
        [jnp.sqrt(r) * jnp.eye(n_obs)[0] for r in radii]
        + [jnp.sqrt(r) * jnp.eye(n_obs)[1] for r in radii]
    )
    misfit = 0.5 * jnp.sum(obs_particles**2, axis=-1)
    assert jnp.allclose(jnp.mean(misfit), n_obs / 2)
    assert jnp.sqrt(n_obs / (2 * jnp.var(misfit, ddof=1))) <= 1.0

    got = discrepancy_step_size(
        obs_particles, observation, obs_noise, remaining=jnp.asarray(remaining)
    )
    assert jnp.allclose(got, min(1.0, remaining), atol=1e-12, rtol=0.0)


def test_discrepancy_step_size_degenerate_ensemble_saturates(getkey):
    """Zero misfit variance sends C2 to infinity, so the budget is returned."""
    single = jr.normal(getkey(), (4,))
    obs_particles = jnp.tile(single, (5, 1))
    obs_noise = lx.DiagonalLinearOperator(jnp.ones(4))
    got = discrepancy_step_size(
        obs_particles, jnp.zeros(4), obs_noise, remaining=jnp.asarray(0.6)
    )
    assert jnp.allclose(got, 0.6)


def test_discrepancy_schedule_sums_to_one(getkey):
    """Driving `eki_step` with the controller lands exactly on the budget."""
    particles, obs_model, obs_noise, observation, _, _ = _problem(
        getkey(), n_state=3, n_obs=2, n_ens=8
    )
    noise_op = lx.MatrixLinearOperator(obs_noise, lx.positive_semidefinite_tag)

    remaining = jnp.asarray(1.0)
    for _ in range(50):
        obs_particles = particles @ obs_model.T
        dt = discrepancy_step_size(
            obs_particles, observation, noise_op, remaining=remaining
        )
        particles = eki_step(
            particles, obs_particles, observation, noise_op, dt=dt, deterministic=True
        )
        remaining = remaining - dt
        if remaining <= 0.0:
            break
    assert jnp.allclose(remaining, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Transformations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deterministic", [False, True])
def test_jit_and_grad(getkey, deterministic):
    """Both variants trace under ``jit`` and differentiate w.r.t. ``observation``."""
    key = getkey()
    particles = jr.normal(getkey(), (6, 4))
    obs_particles = jr.normal(getkey(), (6, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 3))
    step = lx.DiagonalLinearOperator(jnp.array([0.5, 0.5, 1.0, 2.0]))

    def run(observation, dt):
        return eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            dt=dt,
            step=step,
            deterministic=deterministic,
            key=None if deterministic else key,
        )

    jitted = jax.jit(run)
    out = jitted(observation, jnp.asarray(0.4))
    assert out.shape == (6, 4)
    assert jnp.allclose(out, run(observation, 0.4), atol=1e-12, rtol=0.0)

    grad = jax.grad(lambda y: jnp.sum(run(y, jnp.asarray(0.4))))(observation)
    assert grad.shape == (3,)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_dt_may_be_traced(getkey):
    """An adaptive schedule stays inside ``jit``: ``dt`` is not static."""
    particles = jr.normal(getkey(), (6, 4))
    obs_particles = jr.normal(getkey(), (6, 3))
    observation = jr.normal(getkey(), (3,))
    obs_noise = lx.DiagonalLinearOperator(jnp.linspace(0.2, 1.0, 3))

    @jax.jit
    def run(remaining):
        dt = discrepancy_step_size(
            obs_particles, observation, obs_noise, remaining=remaining
        )
        return eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            dt=dt,
            deterministic=True,
        )

    assert run(jnp.asarray(1.0)).shape == (6, 4)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _base(getkey):
    return (
        jr.normal(getkey(), (6, 4)),
        jr.normal(getkey(), (6, 3)),
        jr.normal(getkey(), (3,)),
        lx.DiagonalLinearOperator(jnp.ones(3)),
    )


def test_stochastic_requires_exactly_one_perturbation_source(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="exactly one"):
        eki_step(particles, obs_particles, observation, obs_noise)
    with pytest.raises(ValueError, match="exactly one"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            key=getkey(),
            perturbed_obs=jnp.zeros((6, 3)),
        )


def test_deterministic_rejects_perturbations(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="draws no perturbations"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            key=getkey(),
        )


def test_deterministic_rejects_localization(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="does not support 'localization'"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            localization=jnp.ones((4, 3)),
        )


def test_deterministic_rejects_maximum_likelihood_divisor(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="requires bessel=True"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            bessel=False,
        )


def test_step_shape_is_checked(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="step must be"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            step=lx.DiagonalLinearOperator(jnp.ones(3)),
        )


def test_dt_must_be_scalar(getkey):
    particles, obs_particles, observation, obs_noise = _base(getkey)
    with pytest.raises(ValueError, match="dt must be a scalar"):
        eki_step(
            particles,
            obs_particles,
            observation,
            obs_noise,
            deterministic=True,
            dt=jnp.ones(2),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prior_mean": jnp.zeros(2)}, "prior_mean must have shape"),
        ({"prior_cov": lx.DiagonalLinearOperator(jnp.ones(2))}, "prior_cov must be"),
        ({"observation": jnp.zeros(5)}, "observation must have shape"),
        ({"obs_noise": lx.DiagonalLinearOperator(jnp.ones(7))}, "obs_noise must be"),
    ],
)
def test_tikhonov_augment_validates_shapes(getkey, kwargs, match):
    defaults = dict(
        particles=jr.normal(getkey(), (5, 3)),
        obs_particles=jr.normal(getkey(), (5, 2)),
        observation=jnp.zeros(2),
        obs_noise=lx.DiagonalLinearOperator(jnp.ones(2)),
        prior_mean=jnp.zeros(3),
        prior_cov=lx.DiagonalLinearOperator(jnp.ones(3)),
    )
    with pytest.raises(ValueError, match=match):
        tikhonov_augment(**{**defaults, **kwargs})


def test_discrepancy_step_size_validates_shapes(getkey):
    obs_particles = jr.normal(getkey(), (5, 3))
    with pytest.raises(ValueError, match="observation must have shape"):
        discrepancy_step_size(
            obs_particles,
            jnp.zeros(2),
            lx.DiagonalLinearOperator(jnp.ones(3)),
            remaining=jnp.asarray(1.0),
        )
    with pytest.raises(ValueError, match="obs_noise must be"):
        discrepancy_step_size(
            obs_particles,
            jnp.zeros(3),
            lx.DiagonalLinearOperator(jnp.ones(4)),
            remaining=jnp.asarray(1.0),
        )
