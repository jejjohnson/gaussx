"""Tests for the shared AbstractMultivariateNormal base (gh-360)."""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as nd

import gaussx
from gaussx._distributions import (
    AbstractMultivariateNormal,
    MultivariateNormal,
    MultivariateNormalPrecision,
)
from gaussx._einx import rearrange
from gaussx._testing import (
    default_tolerances,
    psd_operator,
    random_pd_matrix,
    random_spd_block_tridiag,
)


def _pair(n=5):
    key_mu, key_cov = jr.split(jr.key(0))
    mu = jr.normal(key_mu, (n,))
    Sigma = random_pd_matrix(key_cov, n)
    cov = MultivariateNormal(mu, psd_operator(Sigma))
    prec = MultivariateNormalPrecision(mu, psd_operator(jnp.linalg.inv(Sigma)))
    return mu, Sigma, cov, prec


def _close(a, b, scale=10.0):
    rtol, atol = default_tolerances(a, b)
    return jnp.allclose(a, b, rtol=scale * rtol, atol=scale * atol)


def test_both_classes_share_the_base():
    _, _, cov, prec = _pair()
    assert isinstance(cov, AbstractMultivariateNormal)
    assert isinstance(prec, AbstractMultivariateNormal)
    assert gaussx.AbstractMultivariateNormal is AbstractMultivariateNormal


def test_native_operators_are_the_stored_fields():
    _, _, cov, prec = _pair()
    assert cov.covariance_operator is cov.cov_operator
    assert prec.precision_operator is prec.prec_operator
    # The other one is the lazy inverse, whose inverse is the original.
    assert gaussx.inv(prec.covariance_operator) is prec.prec_operator
    assert gaussx.inv(cov.precision_operator) is cov.cov_operator


@pytest.mark.parametrize("which", ["covariance", "precision"])
def test_parameterisations_agree_with_numpyro(which):
    mu, Sigma, cov, prec = _pair()
    ref = nd.MultivariateNormal(mu, covariance_matrix=Sigma)
    x = mu + 0.5
    for d in (cov if which == "covariance" else prec,):
        assert _close(d.mean, ref.mean)
        assert _close(d.variance, ref.variance)
        assert _close(d.covariance_matrix, ref.covariance_matrix)
        assert _close(d.precision_matrix, ref.precision_matrix, scale=100.0)
        assert _close(d.scale_tril, ref.scale_tril)
        L = d.scale_tril
        assert _close(L @ rearrange(L, "i j -> j i"), Sigma)
        assert _close(d.entropy(), ref.entropy())
        assert _close(d.log_prob(x), ref.log_prob(x))


def test_dense_accessors_broadcast_over_the_batch():
    _, Sigma, _, _ = _pair(3)
    loc = jnp.zeros((2, 3))
    d = MultivariateNormalPrecision(loc, psd_operator(jnp.linalg.inv(Sigma)))
    assert d.covariance_matrix.shape == (2, 3, 3)
    assert d.precision_matrix.shape == (2, 3, 3)
    assert d.scale_tril.shape == (2, 3, 3)


def test_kl_all_four_combinations():
    mu_p, Sigma_p, cov_p, prec_p = _pair()
    key_mu, key_cov = jr.split(jr.key(1))
    mu_q = jr.normal(key_mu, mu_p.shape)
    Sigma_q = random_pd_matrix(key_cov, mu_p.shape[0])
    cov_q = MultivariateNormal(mu_q, psd_operator(Sigma_q))
    prec_q = MultivariateNormalPrecision(mu_q, psd_operator(jnp.linalg.inv(Sigma_q)))
    expected = gaussx.dist_kl_divergence(
        mu_p, psd_operator(Sigma_p), mu_q, psd_operator(Sigma_q)
    )
    for p in (cov_p, prec_p):
        for q in (cov_q, prec_q):
            assert _close(p.kl(q), expected, scale=100.0), (type(p), type(q))
    assert _close(cov_p.kl(cov_p), jnp.zeros(()), scale=100.0)


def test_kl_rejects_mismatched_event_shapes():
    _, _, cov, _ = _pair(5)
    _, _, other, _ = _pair(3)
    with pytest.raises(ValueError, match="same event shape"):
        cov.kl(other)


def test_precision_variance_takes_the_selected_inverse(monkeypatch):
    prec = random_spd_block_tridiag(jr.key(0), num_blocks=25, block_size=4)
    expected = jnp.diag(jnp.linalg.inv(prec.as_matrix()))
    d = MultivariateNormalPrecision(jnp.zeros(100), prec)

    def boom(self):
        raise AssertionError("variance materialised the precision")

    monkeypatch.setattr(type(prec), "as_matrix", boom)
    variance = d.variance
    monkeypatch.undo()
    assert _close(variance, expected, scale=100.0)
