"""Tests for SDE kernel implementations."""

import jax
import jax.numpy as jnp
import pytest

from gaussx import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SDEParams,
    SumSDE,
)


class TestMaternSDE:
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_state_dim(self, order):
        kern = MaternSDE(
            variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=order
        )
        assert kern.state_dim == order + 1

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_sde_params_shapes(self, order):
        kern = MaternSDE(
            variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=order
        )
        params = kern.sde_params()
        d = order + 1
        assert params.F.shape == (d, d)
        assert params.H.shape == (1, d)
        assert params.P_inf.shape == (d, d)
        assert isinstance(params, SDEParams)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_stationary_condition(self, order):
        kern = MaternSDE(
            variance=jnp.array(1.5), lengthscale=jnp.array(0.8), order=order
        )
        dt = jnp.array(0.1)
        A, Q = kern.discretise(dt)
        params = kern.sde_params()
        reconstructed = A @ params.P_inf @ A.T + Q
        assert jnp.allclose(reconstructed, params.P_inf, atol=1e-5)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_jit_compatible(self, order):
        kern = MaternSDE(
            variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=order
        )

        @jax.jit
        def get_AQ(dt):
            return kern.discretise(dt)

        A, Q = get_AQ(jnp.array(0.1))
        assert jnp.all(jnp.isfinite(A))
        assert jnp.all(jnp.isfinite(Q))


class TestConstantSDE:
    def test_discretise_identity(self):
        kern = ConstantSDE(variance=jnp.array(2.0))
        A, Q = kern.discretise(jnp.array(0.5))
        assert jnp.allclose(A, jnp.eye(1))
        assert jnp.allclose(Q, jnp.zeros((1, 1)))


class TestCosineSDE:
    def test_discretise_is_rotation(self):
        kern = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(1.5))
        A, Q = kern.discretise(jnp.array(0.3))
        assert jnp.allclose(A @ A.T, jnp.eye(2), atol=1e-6)
        assert jnp.allclose(Q, jnp.zeros((2, 2)), atol=1e-10)


class TestPeriodicSDE:
    def test_state_dim(self):
        kern = PeriodicSDE(
            variance=jnp.array(1.0),
            lengthscale=jnp.array(1.0),
            period=jnp.array(1.0),
            n_harmonics=4,
        )
        assert kern.state_dim == 8

    def test_stationary_condition(self):
        kern = PeriodicSDE(
            variance=jnp.array(1.0),
            lengthscale=jnp.array(1.0),
            period=jnp.array(1.0),
            n_harmonics=3,
        )
        A, Q = kern.discretise(jnp.array(0.1))
        params = kern.sde_params()
        reconstructed = A @ params.P_inf @ A.T + Q
        assert jnp.allclose(reconstructed, params.P_inf, atol=1e-5)


class TestSumSDE:
    def test_state_dim(self):
        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k2 = ConstantSDE(variance=jnp.array(0.5))
        kern = SumSDE(kernels=(k1, k2))
        assert kern.state_dim == 3


class TestProductSDE:
    def test_state_dim(self):
        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=0)
        k2 = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(1.0))
        kern = ProductSDE(kernel1=k1, kernel2=k2)
        assert kern.state_dim == 2

    def test_discretise_matches_dense_expm(self):
        """ProductSDE.discretise via Kronecker expm equals dense expm(F*dt)."""
        import jax.scipy.linalg as jsl

        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(2.0), order=1)
        k2 = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(0.5))
        kern = ProductSDE(kernel1=k1, kernel2=k2)
        dt = jnp.array(0.3)

        A, Q = kern.discretise(dt)

        # Reference: dense expm of the full F.
        params = kern.sde_params()
        A_ref = jsl.expm(params.F * dt)
        Q_ref = params.P_inf - A_ref @ params.P_inf @ A_ref.T
        Q_ref = 0.5 * (Q_ref + Q_ref.T)

        assert jnp.allclose(A, A_ref, atol=1e-6)
        assert jnp.allclose(Q, Q_ref, atol=1e-6)

    def test_discretise_kronecker_structure(self):
        """A = expm(F1*dt) ⊗ expm(F2*dt) is exactly Kronecker-structured."""
        import jax.scipy.linalg as jsl

        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(2.0), order=1)
        k2 = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(0.5))
        kern = ProductSDE(kernel1=k1, kernel2=k2)
        dt = jnp.array(0.25)

        A, _ = kern.discretise(dt)
        A1 = jsl.expm(k1.sde_params().F * dt)
        A2 = jsl.expm(k2.sde_params().F * dt)
        assert jnp.allclose(A, jnp.kron(A1, A2), atol=1e-10)


class TestQuasiPeriodicSDE:
    def test_is_product(self):
        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(2.0), order=1)
        k2 = PeriodicSDE(
            variance=jnp.array(1.0),
            lengthscale=jnp.array(1.0),
            period=jnp.array(1.0),
            n_harmonics=2,
        )
        kern = QuasiPeriodicSDE(kernel1=k1, kernel2=k2)
        assert kern.state_dim == 2 * 4
        assert isinstance(kern, ProductSDE)


class TestDiscretiseSequence:
    def test_vmap_discretise(self):
        kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        dts = jnp.array([0.1, 0.2, 0.5])
        A_seq, Q_seq = kern.discretise_sequence(dts)
        assert A_seq.shape == (3, 2, 2)
        assert Q_seq.shape == (3, 2, 2)
