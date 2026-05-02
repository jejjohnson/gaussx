"""Tests for non-Gaussian likelihood functions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import random

from gaussx import (
    BernoulliLikelihood,
    HeteroscedasticGaussianLikelihood,
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
