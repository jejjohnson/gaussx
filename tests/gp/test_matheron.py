"""Tests for Matheron's-rule pathwise conditioning."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsla
import lineax as lx

import gaussx
from gaussx._gp._matheron import matheron_update
from gaussx._operators import low_rank_plus_diag
from gaussx._testing import random_pd_matrix


def _joint_problem(getkey, num_target: int = 4, num_conditioning: int = 3):
    K_mm = random_pd_matrix(getkey(), num_conditioning)
    K_sm = 0.2 * jr.normal(getkey(), (num_target, num_conditioning))
    posterior_cov = random_pd_matrix(getkey(), num_target) + jnp.eye(num_target)
    K_ss = posterior_cov + K_sm @ jnp.linalg.solve(K_mm, K_sm.T)
    joint_cov = jnp.block([[K_ss, K_sm], [K_sm.T, K_mm]])
    observed_value = jr.normal(getkey(), (num_conditioning,))
    posterior_mean = K_sm @ jnp.linalg.solve(K_mm, observed_value)
    return K_mm, K_sm, posterior_mean, posterior_cov, observed_value, joint_cov


def _joint_samples(key, joint_cov, num_samples: int, num_target: int):
    jitter = 10 * jnp.finfo(joint_cov.dtype).eps * joint_cov.shape[0]
    jitter_matrix = jitter * jnp.eye(joint_cov.shape[0], dtype=joint_cov.dtype)
    L = jnp.linalg.cholesky(joint_cov + jitter_matrix)
    standard = jr.normal(key, (num_samples, joint_cov.shape[0]))
    standard = standard - jnp.mean(standard, axis=0)
    standard_cov = standard.T @ standard / (num_samples - 1)
    standard_chol = jnp.linalg.cholesky(standard_cov)
    standard = jsla.solve_triangular(standard_chol, standard.T, lower=True).T
    samples = standard @ L.T
    return samples[:, :num_target], samples[:, num_target:]


def _ks_statistic(x, y):
    x = jnp.sort(x)
    y = jnp.sort(y)
    values = jnp.sort(jnp.concatenate([x, y]))
    cdf_x = jnp.searchsorted(x, values, side="right") / x.shape[0]
    cdf_y = jnp.searchsorted(y, values, side="right") / y.shape[0]
    return jnp.max(jnp.abs(cdf_x - cdf_y))


def test_matheron_update_matches_dense_formula(getkey):
    K_mm, K_sm, _mean, _cov, observed_value, _joint_cov = _joint_problem(getkey)
    prior_target = jr.normal(getkey(), (5, K_sm.shape[0]))
    prior_conditioning = jr.normal(getkey(), (5, K_mm.shape[0]))

    actual = matheron_update(
        prior_target,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_sm),
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
    )

    residuals = observed_value[None, :] - prior_conditioning
    expected = prior_target + residuals @ jnp.linalg.solve(K_mm, K_sm.T)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_matheron_update_accepts_structured_conditioning_covariance(getkey):
    num_target, num_conditioning, num_rank = 3, 5, 2
    diag = jnp.linspace(1.0, 2.0, num_conditioning)
    U = 0.1 * jr.normal(getkey(), (num_conditioning, num_rank))
    d = jnp.array([0.5, 1.5])
    conditioning_covariance = low_rank_plus_diag(diag, U, d)
    K_mm = conditioning_covariance.as_matrix()
    K_sm = jr.normal(getkey(), (num_target, num_conditioning))
    observed_value = jr.normal(getkey(), (num_conditioning,))
    prior_target = jr.normal(getkey(), (4, num_target))
    prior_conditioning = jr.normal(getkey(), (4, num_conditioning))

    actual = matheron_update(
        prior_target,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_sm),
        conditioning_covariance,
    )

    residuals = observed_value[None, :] - prior_conditioning
    expected = prior_target + residuals @ jnp.linalg.solve(K_mm, K_sm.T)
    assert jnp.allclose(actual, expected, atol=1e-10)


def test_matheron_samples_match_schur_posterior_moments(getkey):
    K_mm, K_sm, posterior_mean, posterior_cov, observed_value, joint_cov = (
        _joint_problem(getkey)
    )
    num_samples = 64
    prior_target, prior_conditioning = _joint_samples(
        getkey(), joint_cov, num_samples, K_sm.shape[0]
    )

    samples = matheron_update(
        prior_target,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_sm),
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
    )

    sample_mean = jnp.mean(samples, axis=0)
    centered = samples - sample_mean
    sample_cov = centered.T @ centered / (num_samples - 1)
    assert jnp.allclose(sample_mean, posterior_mean, atol=8e-2)
    assert jnp.allclose(sample_cov, posterior_cov, rtol=5e-2, atol=1e-1)


def test_matheron_marginals_match_dense_posterior_samples(getkey):
    K_mm, K_sm, posterior_mean, posterior_cov, observed_value, joint_cov = (
        _joint_problem(getkey)
    )
    num_samples = 2048
    prior_target, prior_conditioning = _joint_samples(
        getkey(), joint_cov, num_samples, K_sm.shape[0]
    )

    matheron_samples = matheron_update(
        prior_target,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_sm),
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
    )
    L = jnp.linalg.cholesky(posterior_cov)
    dense_samples = posterior_mean + jr.normal(getkey(), matheron_samples.shape) @ L.T

    marginal_ks = jnp.array(
        [
            _ks_statistic(matheron_samples[:, i], dense_samples[:, i])
            for i in range(K_sm.shape[0])
        ]
    )
    assert jnp.all(marginal_ks < 0.08)


def test_matheron_noiseless_conditioning_recovers_observed_values(getkey):
    num_conditioning = 4
    K_mm = random_pd_matrix(getkey(), num_conditioning)
    prior_conditioning = jr.normal(getkey(), (6, num_conditioning))
    observed_value = jr.normal(getkey(), (num_conditioning,))

    samples = matheron_update(
        prior_conditioning,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_mm),
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
    )

    expected = jnp.broadcast_to(observed_value, prior_conditioning.shape)
    assert jnp.allclose(samples, expected, atol=1e-10)


def test_matheron_update_is_public_api(getkey):
    K_mm, K_sm, _mean, _cov, observed_value, _joint_cov = _joint_problem(getkey)
    prior_target = jnp.zeros((2, K_sm.shape[0]))
    prior_conditioning = jnp.zeros((2, K_mm.shape[0]))

    samples = gaussx.matheron_update(
        prior_target,
        prior_conditioning,
        observed_value,
        lx.MatrixLinearOperator(K_sm),
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
    )

    assert samples.shape == prior_target.shape
