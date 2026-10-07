"""Tests for EP tilted moment computation."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    CubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    UnscentedIntegrator,
    cavity_distribution,
    damped_natural_update,
    ep_tilted_moments,
    moment_match,
    site_natural_from_tilted,
)


class TestEPTiltedMoments:
    def test_gaussian_likelihood_exact(self):
        """With Gaussian likelihood, tilted moments match exact posterior."""
        y = jnp.array(2.0)
        R = jnp.array(0.5)

        def log_lik(f):
            return -0.5 * (y - f) ** 2 / R

        cav_mean = jnp.array(1.0)
        cav_var = jnp.array(1.0)
        post_prec = 1.0 / cav_var + 1.0 / R
        post_var = 1.0 / post_prec
        post_mean = post_var * (cav_mean / cav_var + y / R)

        t_mean, t_var = ep_tilted_moments(log_lik, cav_mean, cav_var, order=30)
        assert jnp.allclose(t_mean, post_mean, atol=1e-4)
        assert jnp.allclose(t_var, post_var, atol=1e-4)

    def test_bernoulli_likelihood(self):
        y = 1.0

        def log_lik(f):
            return y * jax.nn.log_sigmoid(f) + (1.0 - y) * jax.nn.log_sigmoid(-f)

        t_mean, t_var = ep_tilted_moments(
            log_lik, jnp.array(0.0), jnp.array(1.0), order=30
        )
        assert t_mean > 0.0
        assert t_var < 1.0
        assert t_var > 0.0

    def test_jit_compatible(self):
        def log_lik(f):
            return -0.5 * (1.0 - f) ** 2

        @jax.jit
        def compute(cav_mean, cav_var):
            return ep_tilted_moments(log_lik, cav_mean, cav_var)

        t_mean, t_var = compute(jnp.array(0.0), jnp.array(1.0))
        assert jnp.isfinite(t_mean)
        assert jnp.isfinite(t_var)

    def test_integration_with_site_naturals(self):
        def log_lik(f):
            return jax.nn.log_sigmoid(f)

        t_mean, t_var = ep_tilted_moments(log_lik, jnp.array(0.0), jnp.array(2.0))
        nat1, nat2 = site_natural_from_tilted(
            t_mean, t_var, jnp.array(0.0), jnp.array(2.0)
        )
        assert nat2 > 0.0
        assert jnp.isfinite(nat1)


# gh-357: a negative cavity variance must not poison the EP sites.


def _gaussian_site(f):
    return -0.5 * (1.0 - f) ** 2 / 0.1


def test_negative_cavity_variance_is_passed_through():
    t_mean, t_var = ep_tilted_moments(
        _gaussian_site, jnp.array([0.0]), jnp.array([-0.5])
    )
    assert jnp.array_equal(t_mean, jnp.array([0.0]))
    assert jnp.array_equal(t_var, jnp.array([-0.5]))


@pytest.mark.slow
def test_mixed_batch_leaves_valid_sites_bit_identical():
    means = jnp.array([0.0, 0.3, -0.2])
    variances = jnp.array([0.5, -1.0, 2.0])
    valid = jnp.array([True, False, True])
    t_mean, t_var = ep_tilted_moments(_gaussian_site, means, variances)
    ref_mean, ref_var = ep_tilted_moments(
        _gaussian_site, means[valid], variances[valid]
    )
    assert jnp.array_equal(t_mean[valid], ref_mean)
    assert jnp.array_equal(t_var[valid], ref_var)
    assert t_mean[1] == 0.3 and t_var[1] == -1.0


@pytest.mark.parametrize("transform", ["eager", "jit", "vmap"])
def test_gradients_finite_with_negative_cavity(transform):
    means = jnp.array([0.0, 0.3])
    variances = jnp.array([0.5, -1.0])

    def loss(m, v):
        t_mean, t_var = ep_tilted_moments(_gaussian_site, m, v)
        return jnp.sum(t_mean) + jnp.sum(t_var)

    grad = jax.grad(loss, argnums=(0, 1))
    if transform == "jit":
        grad = jax.jit(grad)
    if transform == "vmap":
        g_m, g_v = jax.vmap(grad)(means[None], variances[None])
    else:
        g_m, g_v = grad(means, variances)
    assert jnp.all(jnp.isfinite(g_m)) and jnp.all(jnp.isfinite(g_v))


def test_ep_sweep_with_default_cavity_stays_finite():
    q_mean, q_var = jnp.array([0.2]), jnp.array([0.5])  # posterior precision 2
    nat1, nat2 = jnp.array([1.0]), jnp.array([3.0])  # site holds precision 3
    cm, cv = cavity_distribution(q_mean, q_var, nat1, nat2)
    assert cv[0] < 0
    tm, tv = ep_tilted_moments(_gaussian_site, cm, cv)
    new1, new2 = damped_natural_update(
        nat1, nat2, tm / tv - cm / cv, 1 / tv - 1 / cv, lr=0.5
    )
    assert jnp.all(jnp.isfinite(new1)) and jnp.all(jnp.isfinite(new2))


def test_float32_negative_cavity():
    t_mean, t_var = ep_tilted_moments(
        _gaussian_site,
        jnp.array([0.0, 0.1], jnp.float32),
        jnp.array([-0.5, 0.4], jnp.float32),
    )
    assert jnp.all(jnp.isfinite(t_mean)) and jnp.all(jnp.isfinite(t_var))
    assert t_var[0] == jnp.float32(-0.5)


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case")),
    ],
)
def test_tilted_moments_follow_the_cavity_dtype(dtype):
    """gh-400: the Gauss-Hermite nodes were default-float, so a float32
    cavity came back float64 under x64."""
    cav_mean = jnp.array([0.0, 0.5], dtype)
    cav_var = jnp.array([1.0, 2.0], dtype)
    t_mean, t_var = ep_tilted_moments(
        lambda f: -0.5 * (1.0 - f) ** 2, cav_mean, cav_var
    )
    assert t_mean.dtype == dtype
    assert t_var.dtype == dtype


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_tilted_variance_is_scale_invariant(scale):
    """gh-400: an absolute 1e-10 floor returned 200x the exact variance at
    scale 1e-12. Gaussian likelihood (variance s2) times a Gaussian cavity
    (variance v) has the closed-form tilted variance v * s2 / (v + s2)."""
    v = s2 = scale
    # order=30, as in test_gaussian_likelihood_exact: GH-20's own error on
    # this integrand is ~2e-8 relative, at every scale.
    _, t_var = ep_tilted_moments(
        lambda f: -0.5 * f**2 / s2, jnp.array([0.0]), jnp.array([v]), order=30
    )
    assert jnp.allclose(t_var[0], v * s2 / (v + s2), rtol=1e-8, atol=0.0)


# gh-299: `integrator=` / `power=`, one implementation shared with moment_match.


def _bernoulli_logit(f):
    return jax.nn.log_sigmoid(2.0 * f)


def _via_moment_match(log_lik, m, v, integrator, power=1.0):
    state = GaussianState(
        mean=jnp.atleast_1d(m),
        cov=lx.MatrixLinearOperator(jnp.atleast_2d(v), lx.positive_semidefinite_tag),
    )
    r = moment_match(lambda f: log_lik(f[0]), state, integrator, power=power)
    return m + v * r.d_log_Z[0], v + v * r.d2_log_Z[0, 0] * v


_RULES = [
    pytest.param(GaussHermiteIntegrator(order=20), id="gh20"),
    pytest.param(GaussHermiteIntegrator(order=5), id="gh5"),
    pytest.param(UnscentedIntegrator(alpha=1.0), id="unscented"),
    pytest.param(CubatureIntegrator(), id="cubature"),
]


@pytest.mark.x64_only(reason="1e-12 agreement in float64")
class TestEPTiltedMomentsIntegrator:
    def test_default_reproduces_gh20(self):
        """integrator=None is GH-20; values observed on a1f6dec (gh-299)."""
        m, v = jnp.array(0.3), jnp.array(1.7)
        t_mean, t_var = ep_tilted_moments(_bernoulli_logit, m, v)
        assert jnp.allclose(t_mean, 1.0374931256417765, rtol=0, atol=1e-14)
        assert jnp.allclose(t_var, 1.0000996875137864, rtol=0, atol=1e-14)
        t2 = ep_tilted_moments(
            _bernoulli_logit, m, v, integrator=GaussHermiteIntegrator(order=20)
        )
        assert jnp.allclose(t2[0], t_mean, rtol=0, atol=1e-14)
        assert jnp.allclose(t2[1], t_var, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("integrator", _RULES)
    def test_matches_moment_match(self, integrator):
        m, v = jnp.array(0.3), jnp.array(1.7)
        got = ep_tilted_moments(_bernoulli_logit, m, v, integrator=integrator)
        ref = _via_moment_match(_bernoulli_logit, m, v, integrator)
        assert jnp.allclose(got[0], ref[0], rtol=0, atol=1e-12)
        assert jnp.allclose(got[1], ref[1], rtol=0, atol=1e-12)

    def test_unscented_differs_from_gh(self):
        """The rule is honoured, not silently replaced by GH-20."""
        m, v = jnp.array(0.3), jnp.array(1.7)
        ut = ep_tilted_moments(
            _bernoulli_logit, m, v, integrator=UnscentedIntegrator(alpha=1.0)
        )
        assert jnp.allclose(ut[0], 1.3178374042560848, atol=1e-12)
        assert jnp.allclose(ut[1], 0.6640070184972355, atol=1e-12)

    def test_power_matches_moment_match(self):
        m, v = jnp.array(-0.4), jnp.array(0.8)
        integ = CubatureIntegrator()
        got = ep_tilted_moments(_bernoulli_logit, m, v, integrator=integ, power=0.5)
        ref = _via_moment_match(_bernoulli_logit, m, v, integ, power=0.5)
        assert jnp.allclose(got[0], ref[0], atol=1e-12)
        assert jnp.allclose(got[1], ref[1], atol=1e-12)

    def test_gaussian_likelihood_closed_form(self):
        """Cavity N(0, 1), likelihood N(0.5 | f, 1): tilted N(0.25, 0.5)."""

        def log_lik(f):
            return -0.5 * (0.5 - f) ** 2

        m, v = ep_tilted_moments(
            log_lik,
            jnp.array(0.0),
            jnp.array(1.0),
            integrator=GaussHermiteIntegrator(order=30),
        )
        assert jnp.allclose(m, 0.25, rtol=0, atol=1e-12)
        assert jnp.allclose(v, 0.5, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("integrator", _RULES)
    def test_batch_shape_jit_grad(self, integrator):
        cav_mean = jnp.array([[0.3, -1.0, 2.0], [0.0, 0.5, -0.2]])
        cav_var = jnp.array([[1.7, 0.2, 3.0], [1.0, 0.5, 2.0]])
        f = jax.jit(
            lambda m, v: ep_tilted_moments(
                _bernoulli_logit, m, v, integrator=integrator
            )
        )
        t_mean, t_var = f(cav_mean, cav_var)
        assert t_mean.shape == t_var.shape == (2, 3)
        ref = _via_moment_match(
            _bernoulli_logit, cav_mean[0, 1], cav_var[0, 1], integrator
        )
        assert jnp.allclose(t_mean[0, 1], ref[0], atol=1e-12)
        g = jax.grad(lambda m: f(m, cav_var)[0].sum())(cav_mean)
        assert g.shape == (2, 3)
        assert jnp.all(jnp.isfinite(g))


def test_integrator_and_order_conflict():
    with pytest.raises(ValueError, match="not both"):
        ep_tilted_moments(
            _bernoulli_logit,
            jnp.array(0.0),
            jnp.array(1.0),
            order=10,
            integrator=GaussHermiteIntegrator(order=10),
        )


def test_order_keyword_still_works():
    t20 = ep_tilted_moments(_bernoulli_logit, jnp.array(0.0), jnp.array(1.0))
    t20b = ep_tilted_moments(_bernoulli_logit, jnp.array(0.0), jnp.array(1.0), order=20)
    assert jnp.array_equal(t20[0], t20b[0])
    assert jnp.array_equal(t20[1], t20b[1])
