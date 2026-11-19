"""Independent reference values for the Gaussian layer (gh-355).

Every quantity is checked against scipy or a dense closed form,
on random dense SPD covariances with N in {1, 3, 7} and pinned keys.
rtol=1e-10 throughout: float64 closed forms on well-conditioned matrices
(eigenvalues >= 1), so only round-off separates the two sides.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.stats

import gaussx
from gaussx._testing import psd_operator


pytestmark = pytest.mark.x64_only(reason="float64 references at rtol=1e-10")

RTOL = 1e-10
SIZES = [1, 3, 7]


def _problem(n):
    k1, k2, k3, k4, k5 = jr.split(jr.key(n), 5)
    a, b = jr.normal(k1, (n, n)), jr.normal(k2, (n, n))
    S = a @ a.T + jnp.eye(n)
    T = b @ b.T + jnp.eye(n)
    return S, T, jr.normal(k3, (n,)), jr.normal(k4, (n,)), jr.normal(k5, (n,))


def _dense_kl(mu, S, nu, T):
    """KL(𝒩(μ, S) ‖ 𝒩(ν, T)) = ½ (tr(T⁻¹ S) + dᵀ T⁻¹ d − N + log det T − log det S)."""
    d = nu - mu
    return 0.5 * (
        jnp.trace(jnp.linalg.solve(T, S))
        + d @ jnp.linalg.solve(T, d)
        - S.shape[0]
        + jnp.linalg.slogdet(T)[1]
        - jnp.linalg.slogdet(S)[1]
    )


@pytest.mark.parametrize("n", SIZES)
def test_log_prob_and_entropy_match_scipy(n):
    S, _, mu, _, x = _problem(n)
    ref = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(S))
    ref_lp, ref_h = ref.logpdf(np.asarray(x)), ref.entropy()
    precision = psd_operator(jnp.linalg.inv(S))
    lp = gaussx.gaussian_log_prob(mu, psd_operator(S), x)
    assert jnp.allclose(lp, ref_lp, rtol=RTOL)
    field = gaussx.GaussianMRF(mu, precision)
    assert jnp.allclose(field.log_prob(x), ref_lp, rtol=1e-9)  # one more inverse
    assert jnp.allclose(gaussx.gaussian_entropy(psd_operator(S)), ref_h, rtol=RTOL)
    h_prec = gaussx.gaussian_entropy(gaussx.inv(precision))
    assert jnp.allclose(h_prec, ref_h, rtol=1e-9)  # one more inverse


@pytest.mark.parametrize("n", SIZES)
def test_kl_matches_closed_form(n):
    S, T, mu, nu, _ = _problem(n)
    ref = _dense_kl(mu, S, nu, T)
    dist_kl = gaussx.gaussian_kl(mu, psd_operator(S), nu, psd_operator(T))
    expfam_kl = gaussx.kl_divergence(
        gaussx.GaussianExpFam.from_mean_cov(mu, psd_operator(S)),
        gaussx.GaussianExpFam.from_mean_cov(nu, psd_operator(T)),
    )
    assert jnp.allclose(dist_kl, ref, rtol=RTOL)
    assert jnp.allclose(expfam_kl, ref, rtol=1e-9)  # through two inverses


@pytest.mark.parametrize("n", SIZES)
def test_log_partition_matches_closed_form(n):
    """A = 0.5 mu^T Sigma^{-1} mu + 0.5 log|Sigma| + N/2 log 2 pi."""
    S, _, mu, _, _ = _problem(n)
    expected = (
        0.5 * mu @ jnp.linalg.solve(S, mu)
        + 0.5 * jnp.linalg.slogdet(S)[1]
        + 0.5 * n * jnp.log(2 * jnp.pi)
    )
    q = gaussx.GaussianExpFam.from_mean_cov(mu, psd_operator(S))
    assert jnp.allclose(gaussx.log_partition(q), expected, rtol=1e-9)


def _diagonal_tagged(d):
    return lx.MatrixLinearOperator(
        jnp.diag(d), frozenset({lx.diagonal_tag, lx.positive_semidefinite_tag})
    )


@pytest.mark.parametrize(
    "cov",
    [
        pytest.param(lambda: psd_operator(jnp.array([[2.0]])), id="n1"),
        pytest.param(lambda: _diagonal_tagged(jnp.array([2.0, 0.5, 1.5])), id="diag"),
    ],
)
def test_expfam_edge_cases(cov):
    """N = 1 and diagonal-tagged covariances (the InverseOperator paths)."""
    Sigma = cov()
    n = Sigma.in_size()
    S = Sigma.as_matrix()
    mu = jnp.linspace(-1.0, 1.0, n)
    nu = mu + 0.5
    q = gaussx.GaussianExpFam.from_mean_cov(mu, Sigma)
    p = gaussx.GaussianExpFam.from_mean_cov(nu, psd_operator(S + jnp.eye(n)))
    expected_A = (
        0.5 * mu @ jnp.linalg.solve(S, mu)
        + 0.5 * jnp.linalg.slogdet(S)[1]
        + 0.5 * n * jnp.log(2 * jnp.pi)
    )
    expected_kl = _dense_kl(mu, S, nu, S + jnp.eye(n))
    assert jnp.allclose(q.eta1, jnp.linalg.solve(S, mu), rtol=RTOL)
    assert jnp.allclose(gaussx.log_partition(q), expected_A, rtol=1e-9)
    assert jnp.allclose(gaussx.kl_divergence(q, p), expected_kl, rtol=1e-9)
