"""Tests for inference sugar operations."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._sugar import (
    cavity_distribution,
    gaussian_expected_log_lik,
    log_marginal_likelihood,
    newton_update,
    process_noise_covariance,
    trace_correction,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def test_log_marginal_likelihood_matches_log_prob(getkey):
    """MLL should equal gaussian_log_prob."""
    from gaussx._sugar import gaussian_log_prob

    N = 5
    mat = random_pd_matrix(getkey(), N)
    mu = jr.normal(getkey(), (N,))
    y = jr.normal(getkey(), (N,))
    op = lx.MatrixLinearOperator(mat)

    mll = log_marginal_likelihood(mu, op, y)
    lp = gaussian_log_prob(mu, op, y)
    assert tree_allclose(mll, lp, rtol=1e-6)


def test_log_marginal_likelihood_finite(getkey):
    N = 6
    mat = random_pd_matrix(getkey(), N)
    y = jr.normal(getkey(), (N,))
    op = lx.MatrixLinearOperator(mat)
    assert jnp.isfinite(log_marginal_likelihood(jnp.zeros(N), op, y))


def test_gaussian_expected_log_lik(getkey):
    """When q_cov = 0, expected log-lik should equal regular log-prob."""
    N = 4
    R_mat = random_pd_matrix(getkey(), N)
    R = lx.MatrixLinearOperator(R_mat)
    q_mu = jr.normal(getkey(), (N,))
    y = jr.normal(getkey(), (N,))

    # Zero covariance: trace term vanishes
    q_cov = lx.MatrixLinearOperator(1e-10 * jnp.eye(N))
    ell = gaussian_expected_log_lik(y, q_mu, q_cov, R)

    from gaussx._sugar import gaussian_log_prob

    lp = gaussian_log_prob(q_mu, R, y)
    assert tree_allclose(ell, lp, atol=1e-3)


def test_trace_correction_positive(getkey):
    """Trace correction should be non-negative for valid kernels."""
    N, M = 8, 3
    K_xx_mat = random_pd_matrix(getkey(), N)
    K_zz_mat = random_pd_matrix(getkey(), M)
    K_xz = jr.normal(getkey(), (N, M)) * 0.3

    K_xx = lx.MatrixLinearOperator(K_xx_mat)
    K_zz = lx.MatrixLinearOperator(K_zz_mat)

    tc = trace_correction(K_xx, K_xz, K_zz)
    assert jnp.isfinite(tc)


def test_trace_correction_matches_manual(getkey):
    N, M = 5, 2
    K_xx_mat = random_pd_matrix(getkey(), N)
    K_zz_mat = random_pd_matrix(getkey(), M)
    K_xz = jr.normal(getkey(), (N, M)) * 0.3

    K_xx = lx.MatrixLinearOperator(K_xx_mat)
    K_zz = lx.MatrixLinearOperator(K_zz_mat)

    result = trace_correction(K_xx, K_xz, K_zz)
    # tr(K_xz^T K_zz^{-1} K_xz) where K_xz is (N, M), K_zz is (M, M)
    W = jnp.linalg.solve(K_zz_mat, K_xz.T)  # (M, N)
    expected = jnp.trace(K_xx_mat) - jnp.trace(K_xz @ W)
    assert tree_allclose(result, expected, rtol=1e-4)


def test_cavity_distribution(getkey):
    """Cavity should satisfy: post_prec = cav_prec + site_prec."""
    N = 3
    post_cov_mat = random_pd_matrix(getkey(), N)
    post_mean = jr.normal(getkey(), (N,))
    site_nat2_mat = 0.5 * random_pd_matrix(getkey(), N)
    site_nat1 = jr.normal(getkey(), (N,))

    post_cov = lx.MatrixLinearOperator(post_cov_mat)
    site_nat2 = lx.MatrixLinearOperator(site_nat2_mat)

    _cav_mean, cav_cov = cavity_distribution(post_mean, post_cov, site_nat1, site_nat2)

    # Verify: post_prec = cav_prec + site_prec
    post_prec = jnp.linalg.inv(post_cov_mat)
    cav_prec = jnp.linalg.inv(cav_cov.as_matrix())
    assert tree_allclose(post_prec, cav_prec + site_nat2_mat, rtol=1e-3)


def test_newton_update(getkey):
    N = 4
    mean = jr.normal(getkey(), (N,))
    jac = jr.normal(getkey(), (N,))
    hess = -random_pd_matrix(getkey(), N)  # negative definite

    nat1, nat2 = newton_update(mean, jac, hess)
    assert tree_allclose(nat1, jac - hess @ mean)
    assert tree_allclose(nat2, -hess)


def test_process_noise_covariance(getkey):
    N = 3
    A = 0.9 * jnp.eye(N) + 0.05 * jr.normal(getkey(), (N, N))
    Pinf = random_pd_matrix(getkey(), N)
    Q = process_noise_covariance(A, Pinf)
    expected = Pinf - A @ Pinf @ A.T
    assert tree_allclose(Q, expected, rtol=1e-6)
