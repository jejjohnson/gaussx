"""Tests for GaussianExpFam exponential family module."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._expfam import (
    GaussianExpFam,
    fisher_info,
    kl_divergence,
    log_partition,
    mean_cov_to_natural,
    natural_to_expectation,
    sufficient_stats,
    to_expectation,
    to_mean_cov,
    to_natural,
)
from gaussx._testing import random_pd_matrix, tree_allclose


@pytest.mark.slow
def test_from_mean_cov_roundtrip(getkey):
    """from_mean_cov -> to_mean_cov should recover (mu, Sigma)."""
    N = 4
    Sigma_mat = random_pd_matrix(getkey(), N)
    Sigma = lx.MatrixLinearOperator(Sigma_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    ef = GaussianExpFam.from_mean_cov(mu, Sigma)
    mu_rec, Sigma_rec = to_mean_cov(ef)

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
    """mean_cov_to_natural -> to_mean_cov should roundtrip."""
    N = 3
    Sigma_mat = random_pd_matrix(getkey(), N)
    Sigma = lx.MatrixLinearOperator(Sigma_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    eta1, eta2 = mean_cov_to_natural(mu, Sigma)
    ef = GaussianExpFam(eta1=eta1, eta2=eta2)
    mu_rec, Sigma_rec = to_mean_cov(ef)

    assert tree_allclose(mu_rec, mu, rtol=1e-4)
    assert tree_allclose(Sigma_rec.as_matrix(), Sigma_mat, rtol=1e-4)


def test_log_partition_known():
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


def test_mean_cov_versus_expectation_parameters():
    """gh-335: to_mean_cov gives (mu, Sigma); natural_to_expectation gives
    the expectation parameters (mu, mu mu^T + Sigma)."""
    k1, k2 = jr.split(jr.key(0))
    mu = jr.normal(k1, (3,))
    a = jr.normal(k2, (3, 3))
    S = a @ a.T + jnp.eye(3)
    Sigma = lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
    eta1, eta2 = mean_cov_to_natural(mu, Sigma)
    m1, m2 = natural_to_expectation(eta1, eta2.as_matrix())
    _, cov = to_mean_cov(GaussianExpFam(eta1=eta1, eta2=eta2))
    assert jnp.allclose(m1, mu, rtol=1e-10)
    assert jnp.allclose(m2, jnp.outer(mu, mu) + S, rtol=1e-10)
    assert jnp.allclose(cov.as_matrix(), S, rtol=1e-10)


def test_misnamed_aliases_are_deprecated():
    """gh-335: same values as their replacements, with a DeprecationWarning."""
    mu = jnp.array([1.0, -0.5])
    Sigma = lx.MatrixLinearOperator(
        jnp.array([[2.0, 0.3], [0.3, 1.0]]), lx.positive_semidefinite_tag
    )
    ef = GaussianExpFam.from_mean_cov(mu, Sigma)
    with pytest.warns(DeprecationWarning, match="to_mean_cov"):
        old = to_expectation(ef)
    new = to_mean_cov(ef)
    assert jnp.allclose(old[0], new[0])
    assert jnp.allclose(old[1].as_matrix(), new[1].as_matrix())
    with pytest.warns(DeprecationWarning, match="mean_cov_to_natural"):
        old_eta = to_natural(mu, Sigma)
    new_eta = mean_cov_to_natural(mu, Sigma)
    assert jnp.allclose(old_eta[0], new_eta[0])
    assert jnp.allclose(old_eta[1].as_matrix(), new_eta[1].as_matrix())


def test_documented_density_matches_scipy():
    """gh-342: with h(x) = 1, eta^T T(x) - A(eta) is the log-density, so the
    docstrings and log_partition cannot drift apart again."""
    import numpy as np
    import scipy.stats

    k1, k2, k3 = jr.split(jr.key(0), 3)
    mu = jr.normal(k1, (3,))
    a = jr.normal(k2, (3, 3))
    S = a @ a.T + jnp.eye(3)
    x = jr.normal(k3, (3,))
    q = GaussianExpFam.from_mean_cov(
        mu, lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
    )
    t1, t2 = sufficient_stats(x)
    eta_T = q.eta1 @ t1 + jnp.sum(q.eta2.as_matrix() * t2)
    log_h = 0.0
    expected = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(S))
    assert jnp.allclose(
        log_h + eta_T - log_partition(q), expected.logpdf(np.asarray(x)), rtol=1e-10
    )


def test_fisher_info_is_the_precision_not_the_natural_hessian():
    """gh-342: fisher_info returns Lambda; d^2 A / d eta1^2 is Sigma."""
    import jax

    a = jr.normal(jr.key(0), (3, 3))
    S = a @ a.T + jnp.eye(3)
    q = GaussianExpFam.from_mean_cov(
        jnp.ones(3), lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
    )
    eta2 = q.eta2

    def A(eta1):
        return log_partition(GaussianExpFam(eta1=eta1, eta2=eta2))

    assert jnp.allclose(fisher_info(q).as_matrix(), jnp.linalg.inv(S), rtol=1e-10)
    assert jnp.allclose(jax.hessian(A)(q.eta1), S, rtol=1e-10)


@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_sufficient_stats_any_batch_rank(shape):
    """gh-347: *batch in the annotation means any number of batch axes."""
    x = jr.normal(jr.key(0), shape)
    t1, t2 = sufficient_stats(x)
    assert jnp.array_equal(t1, x)
    assert t2.shape == (*shape, shape[-1])
    assert jnp.allclose(t2, x[..., :, None] * x[..., None, :])


@pytest.mark.parametrize(
    "Sigma",
    [
        pytest.param(
            lx.MatrixLinearOperator(jnp.array([[2.0]]), lx.positive_semidefinite_tag),
            id="1x1",
        ),
        pytest.param(
            lx.MatrixLinearOperator(
                jnp.diag(jnp.array([1.0, 2.0, 3.0])),
                (lx.diagonal_tag, lx.positive_semidefinite_tag),
            ),
            id="diagonal_tagged",
        ),
    ],
)
def test_expfam_on_diagonal_like_covariance(Sigma):
    """``solve(-0.5 inv(Σ), ·)`` used to reach lineax's Diagonal solver (gh-349)."""
    N = Sigma.in_size()
    mu = jnp.arange(1.0, N + 1.0)
    S = Sigma.as_matrix()
    ef = GaussianExpFam.from_mean_cov(mu, Sigma)
    expected = (
        0.5 * mu @ jnp.linalg.solve(S, mu)
        + 0.5 * jnp.linalg.slogdet(S)[1]
        + 0.5 * N * jnp.log(2.0 * jnp.pi)
    )
    assert tree_allclose(log_partition(ef), expected)
    assert tree_allclose(kl_divergence(ef, ef), jnp.zeros(()), atol=1e-5)
    mu_back, Sigma_back = to_mean_cov(ef)
    assert tree_allclose(mu_back, mu)
    assert tree_allclose(Sigma_back.as_matrix(), S)
