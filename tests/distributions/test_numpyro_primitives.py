"""Tests for gaussx distributions with numpyro primitives.

Verifies that MultivariateNormal and MultivariateNormalPrecision work
with numpyro's plate, scan, condition, substitute, mask, deterministic,
and factor primitives.
"""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
from numpyro import handlers
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density

from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._testing import tree_allclose


def _make_op(n, scale=1.0):
    Sigma = scale * (jnp.eye(n) + 0.3 * jnp.ones((n, n)))
    return lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)


# ------------------------------------------------------------------ #
# plate
# ------------------------------------------------------------------ #


class TestPlate:
    def test_plate_log_prob_shape(self):
        def model(obs=None):
            op = _make_op(3)
            with numpyro.plate("data", 10):
                numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op), obs=obs)

        obs = jr.normal(jr.PRNGKey(0), (10, 3))
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(obs=obs)

        lp = trace["x"]["fn"].log_prob(obs)
        assert lp.shape == (10,)
        assert jnp.all(jnp.isfinite(lp))

    def test_plate_matches_numpyro(self):
        n = 3
        Sigma = jnp.eye(n) + 0.3 * jnp.ones((n, n))
        obs = jr.normal(jr.PRNGKey(0), (8, n))
        mu = jnp.zeros(n)

        # numpyro reference
        d_np = dist.MultivariateNormal(mu, covariance_matrix=Sigma)
        lp_np = d_np.log_prob(obs)

        # ours
        op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
        d_ours = MultivariateNormal(mu, op)
        lp_ours = d_ours.log_prob(obs)

        assert tree_allclose(lp_ours, lp_np, rtol=1e-5)

    def test_plate_nuts(self):
        def model(obs=None):
            mu = numpyro.sample("mu", dist.Normal(0, 2).expand([3]))
            op = _make_op(3)
            with numpyro.plate("data", obs.shape[0]):
                numpyro.sample("x", MultivariateNormal(mu, op), obs=obs)

        obs = jr.normal(jr.PRNGKey(0), (15, 3))
        kernel = infer.NUTS(model)
        mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=50, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)
        samples = mcmc.get_samples()

        assert samples["mu"].shape == (50, 3)
        assert jnp.all(jnp.isfinite(samples["mu"]))

    def test_plate_predictive(self):
        def model():
            op = _make_op(3)
            with numpyro.plate("data", 10):
                numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        pred = Predictive(model, num_samples=5)
        samples = pred(jr.PRNGKey(0))

        assert samples["x"].shape == (5, 10, 3)
        assert jnp.all(jnp.isfinite(samples["x"]))

    def test_plate_svi(self):
        def model(obs=None):
            mu = numpyro.sample("mu", dist.Normal(0, 2).expand([3]))
            op = _make_op(3)
            with numpyro.plate("data", obs.shape[0]):
                numpyro.sample("x", MultivariateNormal(mu, op), obs=obs)

        obs = jr.normal(jr.PRNGKey(0), (20, 3))
        guide = AutoNormal(model)
        svi = SVI(model, guide, numpyro.optim.Adam(0.05), loss=Trace_ELBO())
        svi_state = svi.init(jr.PRNGKey(1), obs=obs)

        for _ in range(30):
            svi_state, loss = svi.update(svi_state, obs=obs)

        assert jnp.isfinite(loss)

    def test_plate_precision(self):
        def model(obs=None):
            Lambda = 2.0 * jnp.eye(3)
            op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
            with numpyro.plate("data", 5):
                numpyro.sample(
                    "x",
                    MultivariateNormalPrecision(jnp.zeros(3), op),
                    obs=obs,
                )

        obs = jr.normal(jr.PRNGKey(0), (5, 3))
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(obs=obs)
        lp = trace["x"]["fn"].log_prob(obs)

        assert lp.shape == (5,)
        assert jnp.all(jnp.isfinite(lp))

    def test_nested_plates(self):
        def model():
            op = _make_op(2)
            with numpyro.plate("batch", 3), numpyro.plate("data", 5):
                numpyro.sample("x", MultivariateNormal(jnp.zeros(2), op))

        pred = Predictive(model, num_samples=4)
        samples = pred(jr.PRNGKey(0))

        assert samples["x"].shape == (4, 5, 3, 2)


# ------------------------------------------------------------------ #
# scan
# ------------------------------------------------------------------ #


class TestScan:
    def test_scan_state_space(self):
        def model(T=5):
            def transition_fn(carry, t):
                x_prev = carry
                Q = 0.1 * jnp.eye(2)
                op = lx.MatrixLinearOperator(Q, lx.positive_semidefinite_tag)
                x = numpyro.sample("x", MultivariateNormal(x_prev, op))
                numpyro.sample("y", dist.Normal(x[0], 0.5))
                return x, None

            numpyro_scan(transition_fn, jnp.zeros(2), jnp.arange(T))

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()

        assert trace["x"]["value"].shape == (5, 2)
        assert trace["y"]["value"].shape == (5,)
        assert jnp.all(jnp.isfinite(trace["x"]["value"]))

    def test_scan_with_observations(self):
        def model(T=5, obs_y=None):
            def transition_fn(carry, t):
                x_prev = carry
                Q = 0.1 * jnp.eye(2)
                op = lx.MatrixLinearOperator(Q, lx.positive_semidefinite_tag)
                x = numpyro.sample("x", MultivariateNormal(x_prev, op))
                numpyro.sample("y", dist.Normal(x[0], 0.5), obs=obs_y)
                return x, None

            numpyro_scan(transition_fn, jnp.zeros(2), jnp.arange(T))

        obs_y = jnp.array([0.1, 0.5, -0.2, 0.8, 0.3])
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(obs_y=obs_y)

        assert trace["y"]["is_observed"]

    def test_scan_predictive(self):
        def model(T=5):
            def transition_fn(carry, t):
                x_prev = carry
                Q = 0.1 * jnp.eye(2)
                op = lx.MatrixLinearOperator(Q, lx.positive_semidefinite_tag)
                x = numpyro.sample("x", MultivariateNormal(x_prev, op))
                return x, x

            numpyro_scan(transition_fn, jnp.zeros(2), jnp.arange(T))

        pred = Predictive(model, num_samples=10)
        samples = pred(jr.PRNGKey(0))

        assert samples["x"].shape == (10, 5, 2)


# ------------------------------------------------------------------ #
# condition, substitute, mask
# ------------------------------------------------------------------ #


class TestHandlerPrimitives:
    def test_condition(self):
        def model():
            op = _make_op(3)
            numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        x_cond = jnp.array([1.0, 2.0, 3.0])
        conditioned = handlers.condition(model, data={"x": x_cond})
        trace = handlers.trace(handlers.seed(conditioned, rng_seed=0)).get_trace()

        assert trace["x"]["is_observed"]
        assert tree_allclose(trace["x"]["value"], x_cond)

    def test_substitute(self):
        def model():
            op = _make_op(3)
            numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        x_sub = jnp.array([4.0, 5.0, 6.0])
        subst = handlers.substitute(model, data={"x": x_sub})
        trace = handlers.trace(handlers.seed(subst, rng_seed=0)).get_trace()

        assert tree_allclose(trace["x"]["value"], x_sub)

    def test_mask(self):
        def model():
            op = _make_op(3)
            with handlers.mask(mask=False):
                numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()

        assert trace["x"]["value"].shape == (3,)
        assert trace["x"]["scale"] is None

    def test_condition_log_density(self):
        def model():
            op = _make_op(3)
            numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        x_val = jnp.ones(3)
        ld, _ = log_density(model, (), {}, {"x": x_val})

        # Should match direct log_prob
        op = _make_op(3)
        lp = MultivariateNormal(jnp.zeros(3), op).log_prob(x_val)
        assert tree_allclose(ld, lp, rtol=1e-5)


# ------------------------------------------------------------------ #
# deterministic, factor
# ------------------------------------------------------------------ #


class TestDeterministicAndFactor:
    def test_deterministic(self):
        def model():
            op = _make_op(3)
            x = numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))
            numpyro.deterministic("x_norm", jnp.linalg.norm(x))

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()

        x = trace["x"]["value"]
        x_norm = trace["x_norm"]["value"]
        assert tree_allclose(x_norm, jnp.linalg.norm(x))

    def test_factor(self):
        def model():
            op = _make_op(3)
            x = numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))
            numpyro.factor("penalty", -0.1 * jnp.sum(x**2))

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()

        assert "penalty" in trace
        assert "x" in trace

    def test_factor_affects_log_density(self):
        def model_no_factor():
            op = _make_op(3)
            numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))

        def model_with_factor():
            op = _make_op(3)
            x = numpyro.sample("x", MultivariateNormal(jnp.zeros(3), op))
            numpyro.factor("penalty", -0.1 * jnp.sum(x**2))

        x_val = jnp.ones(3)
        ld_no, _ = log_density(model_no_factor, (), {}, {"x": x_val})
        ld_with, _ = log_density(model_with_factor, (), {}, {"x": x_val})

        expected_diff = -0.1 * jnp.sum(x_val**2)
        assert tree_allclose(ld_with - ld_no, expected_diff, rtol=1e-5)
