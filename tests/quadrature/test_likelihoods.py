"""Tests for non-Gaussian likelihood functions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import scipy.stats
from jax import random
from numpyro.distributions import NegativeBinomial2

from gaussx import (
    BernoulliLikelihood,
    BinomialLikelihood,
    HeteroscedasticGaussianLikelihood,
    NegativeBinomialLikelihood,
    PoissonLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)


@pytest.fixture
def key():
    return random.PRNGKey(0)


class TestBernoulliLikelihood:
    def test_log_prob_known_values(self):
        y = jnp.array([1.0, 0.0])
        f = jnp.array([0.0, 0.0])
        lik = BernoulliLikelihood(y=y)
        lp = lik.log_prob(f)
        expected = 2 * jnp.log(jnp.array(0.5))
        assert jnp.allclose(lp, expected, atol=1e-6)

    def test_gradient_finite(self, key):
        N = 5
        y = random.bernoulli(key, shape=(N,)).astype(jnp.float32)
        f = random.normal(key, (N,))
        lik = BernoulliLikelihood(y=y)
        grad = jax.grad(lik.log_prob)(f)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self, key):
        N = 5
        y = random.bernoulli(key, shape=(N,)).astype(jnp.float32)
        f = random.normal(key, (N,))
        lik = BernoulliLikelihood(y=y)

        @eqx.filter_jit
        def eval_lp(lik, f):
            return lik.log_prob(f)

        assert jnp.isfinite(eval_lp(lik, f))


class TestPoissonLikelihood:
    def test_log_prob_known_values(self):
        y = jnp.array([0.0])
        f = jnp.array([1.0])
        lik = PoissonLikelihood(y=y)
        lp = lik.log_prob(f)
        assert jnp.allclose(lp, -jnp.exp(1.0), atol=1e-5)


class TestStudentTLikelihood:
    def test_reduces_to_gaussian_large_df(self, key):
        N = 20
        k1, k2 = random.split(key)
        y = random.normal(k1, (N,))
        f = random.normal(k2, (N,))
        lik_t = StudentTLikelihood(y=y, df=1e6, scale=1.0)
        lp_t = lik_t.log_prob(f)
        residual = y - f
        lp_gauss = jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - 0.5 * residual**2)
        assert jnp.allclose(lp_t, lp_gauss, atol=1e-2)


class TestSoftmaxLikelihood:
    def test_latent_dim(self):
        y = jnp.array([0, 1, 2])
        lik = SoftmaxLikelihood(y=y, num_classes=4)
        assert lik.latent_dim == 4

    def test_gradient_finite(self, key):
        N, C = 5, 3
        y = random.randint(key, (N,), 0, C)
        f = random.normal(key, (N * C,))
        lik = SoftmaxLikelihood(y=y, num_classes=C)
        grad = jax.grad(lik.log_prob)(f)
        assert jnp.all(jnp.isfinite(grad))


class TestHeteroscedasticGaussianLikelihood:
    def test_latent_dim(self):
        y = jnp.zeros(5)
        lik = HeteroscedasticGaussianLikelihood(y=y)
        assert lik.latent_dim == 2

    def test_gradient_finite(self, key):
        N = 5
        k1, k2 = random.split(key)
        y = random.normal(k1, (N,))
        f = random.normal(k2, (2 * N,))
        lik = HeteroscedasticGaussianLikelihood(y=y)
        grad = jax.grad(lik.log_prob)(f)
        assert jnp.all(jnp.isfinite(grad))


def _site_reference(lik, f):
    """``(jax.grad, diag(jax.hessian))`` of the summed log-density."""
    return jax.grad(lik.log_prob)(f), jnp.diag(jax.hessian(lik.log_prob)(f))


class TestBinomialLikelihood:
    y = jnp.array([0.0, 3.0, 7.0, 2.0])
    n = jnp.array([4.0, 5.0, 7.0, 10.0])
    f = jnp.array([-1.3, 0.2, 2.5, -0.4])

    def test_log_prob_matches_scipy(self):
        lik = BinomialLikelihood(self.y, self.n)
        p = jax.nn.sigmoid(self.f)
        expected = scipy.stats.binom.logpmf(self.y, self.n, p).sum()
        assert jnp.allclose(lik.log_prob(self.f), expected, atol=1e-10)

    def test_site_derivatives_match_autodiff(self):
        lik = BinomialLikelihood(self.y, self.n)
        grad, hess = lik.site_derivatives(self.f)
        ref_grad, ref_hess = _site_reference(lik, self.f)
        assert jnp.allclose(grad, ref_grad, atol=1e-12)
        assert jnp.allclose(hess, ref_hess, atol=1e-12)

    def test_one_trial_is_bernoulli(self):
        y = jnp.array([1.0, 0.0, 1.0])
        f = jnp.array([0.3, -2.0, 1.1])
        binomial = BinomialLikelihood(y, 1.0)
        assert jnp.allclose(binomial.log_prob(f), BernoulliLikelihood(y).log_prob(f))


class TestNegativeBinomialLikelihood:
    y = jnp.array([0.0, 1.0, 4.0, 12.0])
    f = jnp.array([-0.5, 0.1, 1.2, 2.0])

    @pytest.mark.parametrize("r", [0.5, 3.0, jnp.array([1.0, 2.0, 5.0, 40.0])])
    def test_log_prob_matches_scipy_and_numpyro(self, r):
        lik = NegativeBinomialLikelihood(self.y, r)
        mu = jnp.exp(self.f)
        r_ = jnp.broadcast_to(jnp.asarray(r), mu.shape)
        expected = scipy.stats.nbinom.logpmf(self.y, r_, r_ / (r_ + mu)).sum()
        numpyro_lp = NegativeBinomial2(mu, r_).log_prob(self.y).sum()
        assert jnp.allclose(lik.log_prob(self.f), expected, atol=1e-10)
        assert jnp.allclose(lik.log_prob(self.f), numpyro_lp, atol=1e-10)

    @pytest.mark.parametrize("r", [0.5, 3.0])
    def test_site_derivatives_match_autodiff(self, r):
        lik = NegativeBinomialLikelihood(self.y, r)
        grad, hess = lik.site_derivatives(self.f)
        ref_grad, ref_hess = _site_reference(lik, self.f)
        assert jnp.allclose(grad, ref_grad, atol=1e-12)
        assert jnp.allclose(hess, ref_hess, atol=1e-12)

    def test_large_concentration_is_poisson(self):
        lik = NegativeBinomialLikelihood(self.y, 1e8)
        poisson = PoissonLikelihood(self.y)
        assert jnp.allclose(lik.log_prob(self.f), poisson.log_prob(self.f), atol=1e-5)

    def test_gradient_in_concentration(self):
        def log_prob(log_r):
            return NegativeBinomialLikelihood(self.y, jnp.exp(log_r)).log_prob(self.f)

        h = 1e-6
        fd = (log_prob(0.3 + h) - log_prob(0.3 - h)) / (2 * h)
        assert jnp.allclose(jax.grad(log_prob)(0.3), fd, atol=1e-7)


class TestDefaultSiteDerivatives:
    @pytest.mark.parametrize(
        "lik",
        [
            PoissonLikelihood(jnp.array([0.0, 2.0, 5.0])),
            BernoulliLikelihood(jnp.array([1.0, 0.0, 1.0])),
            StudentTLikelihood(jnp.array([0.3, -1.0, 2.0]), df=4.0, scale=0.7),
        ],
    )
    def test_matches_autodiff(self, lik):
        f = jnp.array([0.4, -0.2, 1.1])
        grad, hess = lik.site_derivatives(f)
        ref_grad, ref_hess = _site_reference(lik, f)
        assert jnp.allclose(grad, ref_grad, atol=1e-12)
        assert jnp.allclose(hess, ref_hess, atol=1e-12)
