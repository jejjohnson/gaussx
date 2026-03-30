"""Tests for GaussianExpFam exponential family module."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._expfam import (
    GaussianExpFam,
    fisher_info,
    kl_divergence,
    log_partition,
    sufficient_stats,
    to_expectation,
    to_natural,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def test_from_mean_cov_roundtrip(getkey):
    """from_mean_cov -> to_expectation should recover (mu, Sigma)."""
    N = 4
    Sigma_mat = random_pd_matrix(getkey(), N)
    Sigma = lx.MatrixLinearOperator(Sigma_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    ef = GaussianExpFam.from_mean_cov(mu, Sigma)
    mu_rec, Sigma_rec = to_expectation(ef)

    assert tree_allclose(mu_rec, mu, rtol=1e-4)
    assert tree_allclose(Sigma_rec.as_matrix(), Sigma_mat, rtol=1e-4)


def test_from_mean_prec(getkey):
    """from_mean_prec should set eta1 = Lambda mu, eta2 = -0.5 Lambda."""
    N = 3
    Lambda_mat = random_pd_matrix(getkey(), N)
    Lambda = lx.MatrixLinearOperator(Lambda_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    ef = GaussianExpFam.from_mean_prec(mu, Lambda)

    expected_eta1 = Lambda_mat @ mu
    expected_eta2 = -0.5 * Lambda_mat

    assert tree_allclose(ef.eta1, expected_eta1, rtol=1e-5)
    assert tree_allclose(ef.eta2.as_matrix(), expected_eta2, rtol=1e-5)


def test_to_natural_roundtrip(getkey):
    """to_natural -> to_expectation from GaussianExpFam should roundtrip."""
    N = 3
    Sigma_mat = random_pd_matrix(getkey(), N)
    Sigma = lx.MatrixLinearOperator(Sigma_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    eta1, eta2 = to_natural(mu, Sigma)
    ef = GaussianExpFam(eta1=eta1, eta2=eta2)
    mu_rec, Sigma_rec = to_expectation(ef)

    assert tree_allclose(mu_rec, mu, rtol=1e-4)
    assert tree_allclose(Sigma_rec.as_matrix(), Sigma_mat, rtol=1e-4)


def test_log_partition_known(getkey):
    """Log-partition for known case: N(0, sigma^2 I)."""
    N = 3
    sigma2 = 2.0
    mu = jnp.zeros(N)
    Sigma = lx.MatrixLinearOperator(sigma2 * jnp.eye(N), lx.positive_semidefinite_tag)
    ef = GaussianExpFam.from_mean_cov(mu, Sigma)

    A = log_partition(ef)

    # For N(0, sigma^2 I): A = N/2 * log(2 pi sigma^2)
    expected = 0.5 * N * jnp.log(2.0 * jnp.pi * sigma2)
    assert tree_allclose(A, expected, rtol=1e-4)


def test_log_partition_finite(getkey):
    N = 4
    Sigma = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), N), lx.positive_semidefinite_tag
    )
    mu = jr.normal(getkey(), (N,))
    ef = GaussianExpFam.from_mean_cov(mu, Sigma)
    assert jnp.isfinite(log_partition(ef))


def test_fisher_info_is_precision(getkey):
    """Fisher information should be the precision matrix."""
    N = 3
    Lambda_mat = random_pd_matrix(getkey(), N)
    Lambda = lx.MatrixLinearOperator(Lambda_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    ef = GaussianExpFam.from_mean_prec(mu, Lambda)
    F = fisher_info(ef)

    assert tree_allclose(F.as_matrix(), Lambda_mat, rtol=1e-5)


def test_sufficient_stats_1d():
    x = jnp.array([1.0, 2.0, 3.0])
    t1, t2 = sufficient_stats(x)
    assert tree_allclose(t1, x)
    assert tree_allclose(t2, jnp.outer(x, x))


def test_sufficient_stats_batched():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    t1, t2 = sufficient_stats(x)
    assert t1.shape == (2, 2)
    assert t2.shape == (2, 2, 2)
    assert tree_allclose(t2[0], jnp.outer(x[0], x[0]))


def test_kl_divergence_self_is_zero(getkey):
    """KL(q || q) = 0."""
    N = 3
    Sigma = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), N), lx.positive_semidefinite_tag
    )
    mu = jr.normal(getkey(), (N,))
    ef = GaussianExpFam.from_mean_cov(mu, Sigma)

    kl = kl_divergence(ef, ef)
    assert tree_allclose(kl, jnp.array(0.0), atol=1e-4)


def test_kl_divergence_positive(getkey):
    """KL(q || p) >= 0 for distinct q, p."""
    N = 3
    mu_q = jr.normal(getkey(), (N,))
    mu_p = jr.normal(getkey(), (N,))
    Sigma_q = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), N), lx.positive_semidefinite_tag
    )
    Sigma_p = lx.MatrixLinearOperator(
        random_pd_matrix(getkey(), N), lx.positive_semidefinite_tag
    )

    q = GaussianExpFam.from_mean_cov(mu_q, Sigma_q)
    p = GaussianExpFam.from_mean_cov(mu_p, Sigma_p)

    kl = kl_divergence(q, p)
    assert kl >= -1e-6  # should be non-negative
