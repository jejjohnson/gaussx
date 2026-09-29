"""Tests for SDE autocovariance utility."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest

from gaussx import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    QuasiPeriodicSDE,
    sde_autocovariance,
)


class TestSDEAutocovariance:
    def test_zero_lag_equals_variance(self):
        kern = MaternSDE(variance=jnp.array(2.5), lengthscale=jnp.array(1.0), order=1)
        k0 = sde_autocovariance(kern, jnp.array(0.0))
        assert jnp.allclose(k0, 2.5, atol=1e-6)

    def test_symmetry(self):
        kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k_pos = sde_autocovariance(kern, jnp.array(0.5))
        k_neg = sde_autocovariance(kern, jnp.array(-0.5))
        assert jnp.allclose(k_pos, k_neg, atol=1e-6)

    def test_matern12_analytical(self):
        sigma2 = jnp.array(2.0)
        ell = jnp.array(1.5)
        kern = MaternSDE(variance=sigma2, lengthscale=ell, order=0)
        taus = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.exp(-jnp.abs(taus) / ell)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_cosine_kernel(self):
        sigma2 = jnp.array(1.5)
        w = jnp.array(3.0)
        kern = CosineSDE(variance=sigma2, frequency=w)
        taus = jnp.array([0.0, 0.1, 0.5, 1.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.cos(w * taus)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_constant_kernel(self):
        sigma2 = jnp.array(3.0)
        kern = ConstantSDE(variance=sigma2)
        taus = jnp.array([0.0, 1.0, 10.0, 100.0])
        k_vals = sde_autocovariance(kern, taus)
        assert jnp.allclose(k_vals, sigma2, atol=1e-5)

    def test_differentiable(self):
        def loss(variance, lengthscale):
            kern = MaternSDE(variance=variance, lengthscale=lengthscale, order=1)
            return sde_autocovariance(kern, jnp.array(0.5))

        grad_fn = jax.grad(loss, argnums=(0, 1))
        g_var, g_ell = grad_fn(jnp.array(1.0), jnp.array(1.0))
        assert jnp.isfinite(g_var)
        assert jnp.isfinite(g_ell)


class TestPeriodicClosedForm:
    """gh-289: the j = 0 (constant) harmonic was missing, so k(0) != σ²."""

    @staticmethod
    def _periodic(variance, ell, n_harmonics=10):
        return PeriodicSDE(
            variance=jnp.array(variance),
            lengthscale=jnp.array(ell),
            period=jnp.array(1.0),
            n_harmonics=n_harmonics,
        )

    @pytest.mark.parametrize("ell", [0.5, 1.0, 2.0])
    def test_matches_mackay_kernel(self, ell):
        # The truncation tail 2σ² Σ_{j>10} I_j(x) e^{-x} is 3e-6·σ² at ell = 0.5.
        variance = 2.0
        taus = jnp.linspace(0.0, 1.0, 21)
        k_sde = sde_autocovariance(self._periodic(variance, ell), taus)
        expected = variance * jnp.exp(-2.0 * jnp.sin(jnp.pi * taus) ** 2 / ell**2)
        assert jnp.allclose(k_sde, expected, rtol=0.0, atol=1e-5 * variance)
        assert jnp.allclose(k_sde[0], variance, rtol=1e-5)

    def test_quasi_periodic_zero_lag_is_product_of_variances(self):
        matern = MaternSDE(
            variance=jnp.array(1.5), lengthscale=jnp.array(10.0), order=0
        )
        kern = QuasiPeriodicSDE(kernel1=matern, kernel2=self._periodic(2.0, 1.0))
        assert jnp.allclose(sde_autocovariance(kern, jnp.array([0.0])), 3.0, rtol=1e-6)

    def test_sde_params_and_discretise_agree(self):
        kern = self._periodic(2.0, 0.7, n_harmonics=4)
        params = kern.sde_params()
        A, Q = kern.discretise(jnp.array(0.13))
        assert jnp.allclose(A, jsl.expm(params.F * 0.13), atol=1e-10)
        assert jnp.allclose(A @ params.P_inf @ A.T + Q, params.P_inf, atol=1e-12)
