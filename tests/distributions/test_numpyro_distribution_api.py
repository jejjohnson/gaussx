"""Distribution-API coverage for gaussx's custom MultivariateNormal.

Fills gaps that ``test_numpyro_compat.py`` and ``test_numpyro_primitives.py``
don't cover:

- Distribution API: ``.expand``, ``.shape``, ``Independent`` wrapping,
  PyTree flatten/unflatten roundtrip, ``validate_args``.
- Primitives: ``plate_stack``, ``subsample``, ``prng_key``, ``get_mask``.
- Handlers: ``block``, ``replay``, ``do``, ``infer_config``, ``uncondition``,
  ``reparam``, ``scale``, ``scope``.
- Inference: ``HMC`` kernel, ``AutoMultivariateNormal`` /
  ``AutoLowRankMultivariateNormal`` / ``AutoDelta`` guides, hand-written
  MVN-as-guide.
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
from numpyro import handlers, primitives
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)
from numpyro.infer.reparam import LocScaleReparam

from gaussx._distributions import MultivariateNormal, MultivariateNormalPrecision
from gaussx._testing import tree_allclose


def _cov_op(n: int = 3, scale: float = 1.0) -> lx.AbstractLinearOperator:
    Sigma = scale * (jnp.eye(n) + 0.3 * jnp.ones((n, n)))
    return lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)


def _prec_op(n: int = 3) -> lx.AbstractLinearOperator:
    Lambda = 2.0 * jnp.eye(n)
    return lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)


def _build_cov(loc=None):
    loc = jnp.zeros(3) if loc is None else loc
    return MultivariateNormal(loc, _cov_op())


def _build_prec(loc=None):
    loc = jnp.zeros(3) if loc is None else loc
    return MultivariateNormalPrecision(loc, _prec_op())


BUILDERS = [
    pytest.param(_build_cov, id="cov"),
    pytest.param(_build_prec, id="prec"),
]


# ------------------------------------------------------------------ #
# Distribution API: expand, shape, Independent, PyTree
# ------------------------------------------------------------------ #


class TestDistributionAPI:
    @pytest.mark.parametrize("build", BUILDERS)
    def test_shape_method(self, build):
        d = build()
        assert d.shape() == (3,)
        assert d.shape((4,)) == (4, 3)
        assert d.shape((2, 4)) == (2, 4, 3)

    @pytest.mark.parametrize("build", BUILDERS)
    def test_expand_batch_shape(self, build):
        d = build().expand([5])
        assert d.batch_shape == (5,)
        assert d.event_shape == (3,)

    @pytest.mark.parametrize("build", BUILDERS)
    def test_expand_then_sample(self, build):
        d = build().expand([4])
        samples = d.sample(jr.PRNGKey(0))
        assert samples.shape == (4, 3)
        assert jnp.all(jnp.isfinite(samples))

    @pytest.mark.parametrize("build", BUILDERS)
    def test_expand_then_log_prob(self, build):
        d = build().expand([4])
        x = jnp.zeros((4, 3))
        lp = d.log_prob(x)
        assert lp.shape == (4,)
        assert jnp.all(jnp.isfinite(lp))

    @pytest.mark.parametrize("build", BUILDERS)
    def test_support_is_real_vector(self, build):
        d = build()
        assert d.support is dist.constraints.real_vector


class TestIndependent:
    def test_independent_wrap_moves_batch_to_event(self):
        d = _build_cov().expand([5])
        di = dist.Independent(d, 1)
        assert di.batch_shape == ()
        assert di.event_shape == (5, 3)

    def test_independent_log_prob_sums_over_batch(self):
        d = _build_cov().expand([5])
        di = dist.Independent(d, 1)
        x = jnp.zeros((5, 3))
        lp_indep = di.log_prob(x)
        lp_raw = d.log_prob(x).sum()
        assert tree_allclose(lp_indep, lp_raw, rtol=1e-5)

    def test_independent_sample_shape(self):
        d = _build_cov().expand([4])
        di = dist.Independent(d, 1)
        samples = di.sample(jr.PRNGKey(0))
        assert samples.shape == (4, 3)


class TestPyTree:
    @pytest.mark.parametrize("build", BUILDERS)
    def test_flatten_unflatten_roundtrip(self, build):
        d = build()
        leaves, treedef = jax.tree_util.tree_flatten(d)
        d2 = jax.tree_util.tree_unflatten(treedef, leaves)
        x = jnp.ones(3)
        assert tree_allclose(d.log_prob(x), d2.log_prob(x), rtol=1e-6)

    def test_tree_map_scales_loc(self):
        d = _build_cov(loc=jnp.ones(3))

        # Multiply only array leaves by 2
        def scale(leaf):
            return leaf * 2 if isinstance(leaf, jax.Array) else leaf

        d2 = jax.tree_util.tree_map(scale, d)
        assert tree_allclose(d2.loc, 2.0 * d.loc)


# ------------------------------------------------------------------ #
# Less-common handlers: block, replay, do, infer_config
# ------------------------------------------------------------------ #


class TestAdditionalHandlers:
    def test_block_hides_site(self):
        def model():
            numpyro.sample("x", _build_cov())
            numpyro.sample("y", dist.Normal(0.0, 1.0))

        blocked = handlers.block(model, hide=["x"])
        trace = handlers.trace(handlers.seed(blocked, rng_seed=0)).get_trace()
        assert "x" not in trace
        assert "y" in trace

    def test_replay_reproduces_samples(self):
        def model():
            numpyro.sample("x", _build_cov())

        trace_a = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        replayed = handlers.replay(model, trace_a)
        trace_b = handlers.trace(handlers.seed(replayed, rng_seed=99)).get_trace()
        assert tree_allclose(trace_a["x"]["value"], trace_b["x"]["value"])

    def test_do_propagates_to_downstream(self):
        """``do`` intervenes on a site — downstream reads see the new value."""

        def model():
            x = numpyro.sample("x", _build_cov())
            numpyro.deterministic("x_sum", x.sum())

        x_intervene = jnp.array([10.0, 20.0, 30.0])
        interv = handlers.do(model, data={"x": x_intervene})
        trace = handlers.trace(handlers.seed(interv, rng_seed=0)).get_trace()
        assert tree_allclose(trace["x_sum"]["value"], jnp.array(60.0))

    def test_infer_config_attaches_metadata(self):
        def model():
            numpyro.sample("x", _build_cov())

        configured = handlers.infer_config(
            model, config_fn=lambda _site: {"enumerate": "sequential"}
        )
        trace = handlers.trace(handlers.seed(configured, rng_seed=0)).get_trace()
        assert trace["x"]["infer"].get("enumerate") == "sequential"


# ------------------------------------------------------------------ #
# Alternative inference: HMC, AutoMVN guides, handwritten MVN guide
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestAdditionalInference:
    @pytest.fixture
    def latent_cov_model(self):
        def model(obs=None):
            mu = numpyro.sample("mu", dist.Normal(0.0, 2.0).expand([3]))
            numpyro.sample("x", MultivariateNormal(mu, _cov_op()), obs=obs)

        return model

    def test_hmc_kernel(self, latent_cov_model):
        obs = jnp.array([1.0, 0.5, -0.5])
        kernel = infer.HMC(latent_cov_model, num_steps=5)
        mcmc = infer.MCMC(kernel, num_warmup=30, num_samples=30, progress_bar=False)
        mcmc.run(jr.PRNGKey(0), obs=obs)
        samples = mcmc.get_samples()
        assert samples["mu"].shape == (30, 3)
        assert jnp.all(jnp.isfinite(samples["mu"]))

    def test_auto_multivariate_normal_guide(self, latent_cov_model):
        obs = jnp.array([1.0, 0.5, -0.5])
        guide = AutoMultivariateNormal(latent_cov_model)
        svi = SVI(
            latent_cov_model,
            guide,
            numpyro.optim.Adam(0.05),
            loss=Trace_ELBO(),
        )
        state = svi.init(jr.PRNGKey(1), obs=obs)
        for _ in range(30):
            state, loss = svi.update(state, obs=obs)
        assert jnp.isfinite(loss)

    def test_auto_low_rank_mvn_guide(self, latent_cov_model):
        obs = jnp.array([1.0, 0.5, -0.5])
        guide = AutoLowRankMultivariateNormal(latent_cov_model, rank=2)
        svi = SVI(
            latent_cov_model,
            guide,
            numpyro.optim.Adam(0.05),
            loss=Trace_ELBO(),
        )
        state = svi.init(jr.PRNGKey(1), obs=obs)
        for _ in range(30):
            state, loss = svi.update(state, obs=obs)
        assert jnp.isfinite(loss)

    def test_auto_delta_guide(self, latent_cov_model):
        obs = jnp.array([1.0, 0.5, -0.5])
        guide = AutoDelta(latent_cov_model)
        svi = SVI(
            latent_cov_model,
            guide,
            numpyro.optim.Adam(0.05),
            loss=Trace_ELBO(),
        )
        state = svi.init(jr.PRNGKey(1), obs=obs)
        for _ in range(30):
            state, loss = svi.update(state, obs=obs)
        assert jnp.isfinite(loss)

    def test_handwritten_mvn_guide(self):
        """gaussx MultivariateNormal as the variational family itself."""

        def model(obs=None):
            mu = numpyro.sample(
                "mu",
                MultivariateNormal(jnp.zeros(3), _cov_op(scale=4.0)),
            )
            numpyro.sample("x", MultivariateNormal(mu, _cov_op()), obs=obs)

        def guide(**_kwargs):
            loc = numpyro.param("q_loc", jnp.zeros(3))
            scale_diag = numpyro.param(
                "q_scale",
                jnp.ones(3),
                constraint=dist.constraints.positive,
            )
            cov = lx.DiagonalLinearOperator(jnp.square(scale_diag))
            numpyro.sample("mu", MultivariateNormal(loc, cov))

        obs = jnp.array([1.0, 0.5, -0.5])
        svi = SVI(model, guide, numpyro.optim.Adam(0.05), loss=Trace_ELBO())
        state = svi.init(jr.PRNGKey(2), obs=obs)
        for _ in range(30):
            state, loss = svi.update(state, obs=obs)
        assert jnp.isfinite(loss)


# ------------------------------------------------------------------ #
# validate_args path
# ------------------------------------------------------------------ #


class TestValidateArgs:
    def test_validate_args_accepts_valid_sample(self):
        d = MultivariateNormal(jnp.zeros(3), _cov_op(), validate_args=True)
        lp = d.log_prob(jnp.ones(3))
        assert jnp.isfinite(lp)

    def test_validate_args_rejects_non_vector_sample(self):
        """``real_vector`` support requires ``ndim >= 1`` — scalar input fails."""
        d = MultivariateNormal(jnp.zeros(3), _cov_op(), validate_args=True)
        with pytest.raises((ValueError, AssertionError)):
            d.log_prob(jnp.array(1.0))


# ------------------------------------------------------------------ #
# Additional primitives: plate_stack, subsample, prng_key, get_mask
# ------------------------------------------------------------------ #


class TestAdditionalPrimitives:
    def test_plate_stack_nests_plates(self):
        def model():
            with primitives.plate_stack("stack", [4, 5]):
                numpyro.sample("x", _build_cov())

        pred = infer.Predictive(model, num_samples=2)
        samples = pred(jr.PRNGKey(0))
        assert samples["x"].shape == (2, 4, 5, 3)

    def test_subsample_inside_plate(self):
        """``subsample`` indexes data within a subsampled plate."""
        N, k = 20, 5
        data = jr.normal(jr.PRNGKey(0), (N, 3))

        def model():
            with numpyro.plate("data", N, subsample_size=k):
                batch = numpyro.subsample(data, event_dim=1)
                numpyro.sample("x", _build_cov(), obs=batch)

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        assert trace["x"]["value"].shape == (k, 3)

    def test_prng_key_inside_model(self):
        def model():
            key = numpyro.prng_key()
            numpyro.deterministic("fresh_sample", jr.normal(key, (3,)))
            numpyro.sample("x", _build_cov())

        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        assert trace["fresh_sample"]["value"].shape == (3,)
        assert jnp.all(jnp.isfinite(trace["fresh_sample"]["value"]))

    def test_get_mask_returns_handler_mask(self):
        captured = {}

        def model():
            captured["mask"] = numpyro.get_mask()
            numpyro.sample("x", _build_cov())

        masked = handlers.mask(model, mask=False)
        handlers.trace(handlers.seed(masked, rng_seed=0)).get_trace()
        assert captured["mask"] is False


# ------------------------------------------------------------------ #
# Additional handlers: uncondition, reparam, scale, scope
# ------------------------------------------------------------------ #


class TestMoreHandlers:
    def test_uncondition_restores_latent(self):
        """``uncondition`` turns observed sites back into latent ones."""

        def model(obs=None):
            numpyro.sample("x", _build_cov(), obs=obs)

        obs = jnp.ones(3)
        trace_obs = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(obs=obs)
        assert trace_obs["x"]["is_observed"]

        unc = handlers.uncondition(model)
        trace_unc = handlers.trace(handlers.seed(unc, rng_seed=0)).get_trace(obs=obs)
        assert not trace_unc["x"]["is_observed"]

    def test_reparam_locscale_on_sibling_leaves_mvn_intact(self):
        """LocScaleReparam applies to scalar Normal sites without breaking MVN."""

        def model():
            numpyro.sample("mu", dist.Normal(0.0, 1.0))
            numpyro.sample("x", _build_cov())

        reparamed = handlers.reparam(model, config={"mu": LocScaleReparam()})
        trace = handlers.trace(handlers.seed(reparamed, rng_seed=0)).get_trace()
        # reparam introduces a mu_decentered latent; x should still sample cleanly
        assert "x" in trace
        assert trace["x"]["value"].shape == (3,)
        assert jnp.all(jnp.isfinite(trace["x"]["value"]))

    def test_scale_multiplies_log_density(self):
        from numpyro.infer.util import log_density

        def model():
            numpyro.sample("x", _build_cov())

        x_val = jnp.ones(3)
        ld_base, _ = log_density(model, (), {}, {"x": x_val})
        scaled = handlers.scale(model, scale=3.0)
        ld_scaled, _ = log_density(scaled, (), {}, {"x": x_val})
        assert tree_allclose(ld_scaled, 3.0 * ld_base, rtol=1e-5)

    def test_scope_prefixes_site_names(self):
        def model():
            numpyro.sample("x", _build_cov())

        scoped = handlers.scope(model, prefix="inner")
        trace = handlers.trace(handlers.seed(scoped, rng_seed=0)).get_trace()
        assert any(k.startswith("inner") for k in trace)
        assert "x" not in trace or "inner" in next(iter(trace))
