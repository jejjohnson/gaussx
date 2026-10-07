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
import numpyro.distributions as nd
import numpyro.infer as infer
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density

import gaussx
from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._operators import Kronecker
from gaussx._testing import tree_allclose


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
        """Matches the same model written with numpyro's own MVN (gh-324)."""
        obs = jnp.array([1.0, 0.5, -0.5])
        mu = jnp.array([0.2, -0.1, 0.3])
        ld, _ = log_density(_cov_model, (obs,), {}, {"mu": mu})
        Sigma = jnp.eye(3) + 0.3 * jnp.ones((3, 3))
        expected = dist.Normal(0, 5).log_prob(mu).sum() + dist.MultivariateNormal(
            mu, covariance_matrix=Sigma
        ).log_prob(obs)
        assert jnp.allclose(ld, expected, rtol=1e-12)

    def test_log_density_precision(self):
        """Matches the same model written with numpyro's own MVN (gh-324)."""
        obs = jnp.array([1.0, 0.5, -0.5])
        mu = jnp.array([0.2, -0.1, 0.3])
        ld, _ = log_density(_prec_model, (obs,), {}, {"mu": mu})
        expected = dist.Normal(0, 5).log_prob(mu).sum() + dist.MultivariateNormal(
            mu, precision_matrix=2.0 * jnp.eye(3)
        ).log_prob(obs)
        assert jnp.allclose(ld, expected, rtol=1e-12)

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
@pytest.mark.integration
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
@pytest.mark.integration
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
    @pytest.mark.integration
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
    @pytest.mark.integration
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


# ---------------------------------------------------------------------------
# gh-313: numpyro kl_divergence for the gaussx MVN classes
# ---------------------------------------------------------------------------


def _random_spd(key, n):
    a = jr.normal(key, (n, n))
    return a @ a.T + n * jnp.eye(n)


def _as(kind, loc, cov):
    """A Gaussian of the given class with mean loc and covariance cov."""
    psd = lx.positive_semidefinite_tag
    if kind == "gx_mvn":
        return gaussx.MultivariateNormal(loc, lx.MatrixLinearOperator(cov, psd))
    if kind == "gx_prec":
        precision = lx.MatrixLinearOperator(jnp.linalg.inv(cov), psd)
        return gaussx.MultivariateNormalPrecision(loc, precision)
    return nd.MultivariateNormal(loc, covariance_matrix=cov)


_CASES = [
    (p, q, n)
    for p in ("gx_mvn", "gx_prec", "nd_mvn")
    for q in ("gx_mvn", "gx_prec", "nd_mvn")
    for n in (1, 3, 7)
    if (p, q) != ("nd_mvn", "nd_mvn")
]


@pytest.mark.parametrize(("p_kind", "q_kind", "n"), _CASES)
def test_kl_divergence_matches_numpyro_dense(p_kind, q_kind, n):
    """Every gaussx/numpyro pairing agrees with numpyro's dense MVN-MVN KL.

    rtol=1e-10: both sides are float64 closed forms on SPD matrices with
    condition numbers below ~10, so only round-off separates them; the
    precision class adds one dense inverse.
    """
    k1, k2, k3, k4 = jr.split(jr.key(0), 4)
    p_loc, q_loc = jr.normal(k1, (n,)), jr.normal(k2, (n,))
    p_cov, q_cov = _random_spd(k3, n), _random_spd(k4, n)
    expected = nd.kl_divergence(
        nd.MultivariateNormal(p_loc, covariance_matrix=p_cov),
        nd.MultivariateNormal(q_loc, covariance_matrix=q_cov),
    )
    actual = nd.kl_divergence(_as(p_kind, p_loc, p_cov), _as(q_kind, q_loc, q_cov))
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("q_kind", ["gx_mvn", "nd_mvn"])
def test_kl_divergence_broadcasts_batch_shapes(q_kind):
    k1, k2, k3 = jr.split(jr.key(0), 3)
    p_locs = jr.normal(k1, (4, 3))
    cov = _random_spd(k2, 3)
    q_loc = jr.normal(k3, (3,))
    q = _as(q_kind, q_loc, cov)
    batched = nd.kl_divergence(_as("gx_mvn", p_locs, cov), q)
    assert batched.shape == (4,)
    for i in range(4):
        single = nd.kl_divergence(_as("gx_mvn", p_locs[i], cov), q)
        assert jnp.allclose(batched[i], single, rtol=1e-12)


def test_kl_divergence_rejects_mismatched_event_shapes():
    p = _as("gx_mvn", jnp.zeros(3), jnp.eye(3))
    q = _as("gx_mvn", jnp.zeros(2), jnp.eye(2))
    with pytest.raises(ValueError, match="same event shape"):
        nd.kl_divergence(p, q)


def test_trace_mean_field_elbo_uses_the_analytic_kl():
    """With the KL registered, TraceMeanField_ELBO stops sampling it: the
    loss of a model whose only latent is the MVN site is deterministic."""
    import numpyro
    from numpyro.infer import TraceMeanField_ELBO

    cov = lx.MatrixLinearOperator(
        _random_spd(jr.key(1), 3), lx.positive_semidefinite_tag
    )

    def model():
        numpyro.sample("f", gaussx.MultivariateNormal(jnp.zeros(3), cov))

    def guide():
        numpyro.sample("f", gaussx.MultivariateNormal(jnp.ones(3), cov))

    elbo = TraceMeanField_ELBO()
    loss_a = elbo.loss(jr.key(0), {}, model, guide)
    loss_b = elbo.loss(jr.key(1), {}, model, guide)
    expected = gaussx.dist_kl_divergence(jnp.ones(3), cov, jnp.zeros(3), cov)
    assert jnp.allclose(loss_a, loss_b, rtol=1e-12)
    assert jnp.allclose(loss_a, expected, rtol=1e-10)


def test_kl_defers_to_monte_carlo_for_matrix_free_strategies():
    """gh-313 review: a CG strategy asks for matrix-free solves, which the
    closed form would not honour, so numpyro keeps its Monte Carlo KL."""
    from gaussx import CGSolver

    cov = lx.MatrixLinearOperator(
        _random_spd(jr.key(1), 3), lx.positive_semidefinite_tag
    )
    p = gaussx.MultivariateNormal(jnp.zeros(3), cov, solver=CGSolver())
    q = gaussx.MultivariateNormal(jnp.ones(3), cov)
    with pytest.raises(NotImplementedError, match="exact solves"):
        nd.kl_divergence(p, q)


def test_kl_survives_plate_expansion():
    """numpyro's ExpandedDistribution KL delegates to the base pair."""
    cov = lx.MatrixLinearOperator(2 * jnp.eye(3), lx.positive_semidefinite_tag)
    p = gaussx.MultivariateNormal(jnp.zeros(3), cov).expand((4,))
    q = gaussx.MultivariateNormal(jnp.ones(3), cov).expand((4,))
    expected = gaussx.dist_kl_divergence(jnp.zeros(3), cov, jnp.ones(3), cov)
    assert jnp.allclose(nd.kl_divergence(p, q), jnp.full(4, expected), rtol=1e-12)
