"""Tests for numpyro compatibility of gaussx distributions.

Verifies that MultivariateNormal and MultivariateNormalPrecision work
correctly with numpyro primitives: seed, trace, log_density, NUTS,
SVI, and Predictive.
"""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density

from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._operators import Kronecker
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.1 * jnp.eye(n)


# ------------------------------------------------------------------ #
# Helpers: reusable numpyro models
# ------------------------------------------------------------------ #


def _cov_model(obs=None):
    """Model using MultivariateNormal (covariance form)."""
    mu = numpyro.sample("mu", dist.Normal(0, 5).expand([3]))
    Sigma = jnp.eye(3) + 0.3 * jnp.ones((3, 3))
    op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
    numpyro.sample("x", MultivariateNormal(mu, op), obs=obs)


def _prec_model(obs=None):
    """Model using MultivariateNormalPrecision."""
    mu = numpyro.sample("mu", dist.Normal(0, 5).expand([3]))
    Lambda = 2.0 * jnp.eye(3)
    op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
    numpyro.sample("x", MultivariateNormalPrecision(mu, op), obs=obs)


# ------------------------------------------------------------------ #
# Tests: handlers (seed, trace)
# ------------------------------------------------------------------ #


class TestHandlers:
    def test_seed_and_trace(self):
        trace = handlers.trace(handlers.seed(_cov_model, rng_seed=0)).get_trace()

        assert "mu" in trace
        assert "x" in trace
        assert trace["x"]["value"].shape == (3,)
        assert jnp.all(jnp.isfinite(trace["x"]["value"]))

    def test_trace_with_observations(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        trace = handlers.trace(handlers.seed(_cov_model, rng_seed=0)).get_trace(obs=obs)

        assert trace["x"]["is_observed"]
        assert tree_allclose(trace["x"]["value"], obs)

    def test_log_prob_in_trace(self):
        trace = handlers.trace(handlers.seed(_cov_model, rng_seed=0)).get_trace()

        x_val = trace["x"]["value"]
        lp = trace["x"]["fn"].log_prob(x_val)
        assert jnp.isfinite(lp)

    def test_precision_seed_and_trace(self):
        trace = handlers.trace(handlers.seed(_prec_model, rng_seed=0)).get_trace()

        assert "x" in trace
        assert trace["x"]["value"].shape == (3,)


# ------------------------------------------------------------------ #
# Tests: log_density
# ------------------------------------------------------------------ #


class TestLogDensity:
    def test_log_density_covariance(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        params = {"mu": jnp.zeros(3)}
        ld, _ = log_density(_cov_model, (obs,), {}, params)

        assert jnp.isfinite(ld)

    def test_log_density_precision(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        params = {"mu": jnp.zeros(3)}
        ld, _ = log_density(_prec_model, (obs,), {}, params)

        assert jnp.isfinite(ld)

    def test_log_density_grad(self):
        obs = jnp.array([1.0, 0.5, -0.5])

        def ld_fn(mu):
            ld, _ = log_density(_cov_model, (obs,), {}, {"mu": mu})
            return ld

        g = jax.grad(ld_fn)(jnp.zeros(3))
        assert g.shape == (3,)
        assert jnp.all(jnp.isfinite(g))


# ------------------------------------------------------------------ #
# Tests: MCMC (NUTS)
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestMCMC:
    def test_nuts_covariance(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        kernel = infer.NUTS(_cov_model)
        mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=100, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)
        samples = mcmc.get_samples()

        assert samples["mu"].shape == (100, 3)
        assert jnp.all(jnp.isfinite(samples["mu"]))
        # Posterior mean should be pulled toward obs
        mu_mean = jnp.mean(samples["mu"], axis=0)
        assert jnp.linalg.norm(mu_mean - obs) < 3.0

    def test_nuts_precision(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        kernel = infer.NUTS(_prec_model)
        mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=100, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)
        samples = mcmc.get_samples()

        assert samples["mu"].shape == (100, 3)
        assert jnp.all(jnp.isfinite(samples["mu"]))


# ------------------------------------------------------------------ #
# Tests: SVI
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestSVI:
    def test_svi_converges(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        guide = AutoNormal(_cov_model)
        optimizer = numpyro.optim.Adam(0.01)
        svi = SVI(_cov_model, guide, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(jr.PRNGKey(1), obs=obs)

        losses = []
        for _ in range(100):
            svi_state, loss = svi.update(svi_state, obs=obs)
            losses.append(float(loss))

        assert jnp.isfinite(losses[-1])
        # Loss should decrease
        assert losses[-1] < losses[0]


# ------------------------------------------------------------------ #
# Tests: Predictive
# ------------------------------------------------------------------ #


class TestPredictive:
    def test_prior_predictive(self):
        def model():
            mu = jnp.zeros(3)
            Sigma = jnp.eye(3) + 0.3 * jnp.ones((3, 3))
            op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
            numpyro.sample("x", MultivariateNormal(mu, op))

        predictive = Predictive(model, num_samples=200)
        samples = predictive(jr.PRNGKey(42))

        assert samples["x"].shape == (200, 3)
        assert jnp.all(jnp.isfinite(samples["x"]))
        # Mean should be close to zero
        assert jnp.linalg.norm(jnp.mean(samples["x"], axis=0)) < 0.5

    @pytest.mark.slow
    def test_posterior_predictive(self):
        obs = jnp.array([1.0, 0.5, -0.5])
        kernel = infer.NUTS(_cov_model)
        mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=50, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)

        predictive = Predictive(_cov_model, posterior_samples=mcmc.get_samples())
        pred = predictive(jr.PRNGKey(1))

        assert pred["x"].shape == (50, 3)
        assert jnp.all(jnp.isfinite(pred["x"]))


# ------------------------------------------------------------------ #
# Tests: structured operators in numpyro
# ------------------------------------------------------------------ #


class TestStructuredInNumpyro:
    def test_kronecker_predictive(self):
        def model():
            A = jnp.eye(2) + 0.3 * jnp.ones((2, 2))
            B = jnp.eye(3) + 0.2 * jnp.ones((3, 3))
            A_op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
            B_op = lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
            kron = Kronecker(A_op, B_op)
            numpyro.sample("x", MultivariateNormal(jnp.zeros(6), kron))

        predictive = Predictive(model, num_samples=100)
        samples = predictive(jr.PRNGKey(0))

        assert samples["x"].shape == (100, 6)
        assert jnp.all(jnp.isfinite(samples["x"]))

    @pytest.mark.slow
    def test_diagonal_nuts(self):
        def model(obs=None):
            mu = numpyro.sample("mu", dist.Normal(0, 2).expand([4]))
            d_vals = jnp.array([1.0, 2.0, 0.5, 1.5])
            op = lx.DiagonalLinearOperator(d_vals)
            numpyro.sample("x", MultivariateNormal(mu, op), obs=obs)

        obs = jnp.array([1.0, -1.0, 0.5, 2.0])
        kernel = infer.NUTS(model)
        mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=100, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)
        samples = mcmc.get_samples()

        assert samples["mu"].shape == (100, 4)
        assert jnp.all(jnp.isfinite(samples["mu"]))
