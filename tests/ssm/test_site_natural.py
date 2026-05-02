"""Tests for per-site natural parameter conversions."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from gaussx import (
    cavity_from_marginal,
    site_mean_var_from_natural,
    site_natural_from_tilted,
)


@pytest.fixture
def key():
    return random.PRNGKey(42)


class TestSiteNaturalFromTilted:
    def test_zero_site_when_tilted_equals_cavity(self, key):
        k1, k2 = random.split(key)
        N = 5
        mean = random.normal(k1, (N,))
        var = jnp.abs(random.normal(k2, (N,))) + 0.1
        nat1, nat2 = site_natural_from_tilted(mean, var, mean, var)
        assert jnp.allclose(nat1, 0.0, atol=1e-6)
        assert jnp.allclose(nat2, 0.0, atol=1e-6)

    def test_jit_compatible(self, key):
        k1, k2, k3, k4 = random.split(key, 4)
        N = 5
        tilted_mean = random.normal(k1, (N,))
        tilted_var = jnp.abs(random.normal(k2, (N,))) + 0.1
        cav_mean = random.normal(k3, (N,))
        cav_var = jnp.abs(random.normal(k4, (N,))) + 0.1
        jit_fn = jax.jit(site_natural_from_tilted)
        nat1, nat2 = jit_fn(tilted_mean, tilted_var, cav_mean, cav_var)
        assert jnp.all(jnp.isfinite(nat1))
        assert jnp.all(jnp.isfinite(nat2))


class TestSiteMeanVarFromNatural:
    def test_basic_inversion(self):
        nat1 = jnp.array([2.0, -1.0, 0.5])
        nat2 = jnp.array([1.0, 2.0, 4.0])
        mean, var = site_mean_var_from_natural(nat1, nat2)
        assert jnp.allclose(mean, jnp.array([2.0, -0.5, 0.125]), atol=1e-6)
        assert jnp.allclose(var, jnp.array([1.0, 0.5, 0.25]), atol=1e-6)


class TestCavityFromMarginal:
    def test_zero_site_returns_marginal(self, key):
        k1, k2 = random.split(key)
        N = 5
        marg_mean = random.normal(k1, (N,))
        marg_var = jnp.abs(random.normal(k2, (N,))) + 0.1
        cav_mean, cav_var = cavity_from_marginal(
            marg_mean, marg_var, jnp.zeros(N), jnp.zeros(N)
        )
        assert jnp.allclose(cav_mean, marg_mean, atol=1e-6)
        assert jnp.allclose(cav_var, marg_var, atol=1e-6)

    def test_batch_dimensions(self, key):
        k1, k2, k3, k4 = random.split(key, 4)
        shape = (3, 4)
        marg_mean = random.normal(k1, shape)
        marg_var = jnp.abs(random.normal(k2, shape)) + 0.1
        site_nat1 = random.normal(k3, shape) * 0.1
        site_nat2 = jnp.abs(random.normal(k4, shape)) * 0.1
        cav_mean, cav_var = cavity_from_marginal(
            marg_mean, marg_var, site_nat1, site_nat2
        )
        assert cav_mean.shape == shape
        assert cav_var.shape == shape
