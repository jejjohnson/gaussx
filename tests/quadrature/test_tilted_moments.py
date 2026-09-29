"""Tests for EP tilted moment computation."""

import jax
import jax.numpy as jnp
import pytest

from gaussx import (
    cavity_distribution,
    damped_natural_update,
    ep_tilted_moments,
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
