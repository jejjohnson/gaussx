"""Tests for SDE kernel implementations."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    ConstantSDE,
    CosineSDE,
    Kronecker,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SDEParams,
    SumSDE,
    discrete_lyapunov_solve,
    process_noise_covariance,
    symmetrize,
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


class TestDiscretiseSharesProcessNoiseHelper:
    """`SDEKernel.discretise` must not re-inline the process-noise formula.

    Two copies of ``Q = P_inf - A P_inf A^T`` had already drifted once: the
    kernel path symmetrised the result and the standalone helper did not.
    These tests pin the base implementation to the shared helper so a future
    change to the numerics lands in both places (gh-151).
    """

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_matches_shared_helper(self, order):
        kern = MaternSDE(
            variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=order
        )
        A, Q = kern.discretise(jnp.array(0.1))
        expected = symmetrize(process_noise_covariance(A, kern.sde_params().P_inf))
        assert jnp.allclose(Q, expected, atol=1e-12)

    def test_helper_is_reachable_from_both_import_paths(self):
        """The pre-move `_inference` path keeps working."""
        from gaussx._inference import process_noise_covariance as legacy
        from gaussx._ssm import process_noise_covariance as current

        assert legacy is current is process_noise_covariance

    def test_discretised_covariance_is_symmetric(self):
        """`discretise` symmetrises; the bare helper deliberately does not."""
        kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=2)
        _, Q = kern.discretise(jnp.array(0.3))
        assert jnp.allclose(Q, Q.T, atol=1e-12)


class TestProcessNoiseCovarianceReuse:
    """`process_noise_covariance` delegates its congruence to `cov_transform`.

    That keeps one implementation of ``A P A^T`` in the package and lets
    structured operands stay factorised instead of being materialised.
    """

    def _factors(self):
        A1 = jnp.array([[0.9, 0.0], [0.0, 0.7]])
        A2 = jnp.array([[0.5, 0.1], [0.1, 0.6]])
        P1 = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        P2 = jnp.array([[1.5, 0.2], [0.2, 1.1]])
        return A1, A2, P1, P2

    def test_array_inputs_return_arrays(self):
        A1, _, P1, _ = self._factors()
        Q = process_noise_covariance(A1, P1)
        assert isinstance(Q, jnp.ndarray)
        assert jnp.allclose(Q, P1 - A1 @ P1 @ A1.T)

    def test_operator_inputs_stay_lazy(self):
        """Operator operands return an operator, not a materialised array."""
        A1, _, P1, _ = self._factors()
        Q = process_noise_covariance(
            lx.MatrixLinearOperator(A1),
            lx.MatrixLinearOperator(P1, lx.symmetric_tag),
        )
        assert isinstance(Q, lx.AbstractLinearOperator)
        assert jnp.allclose(Q.as_matrix(), P1 - A1 @ P1 @ A1.T, atol=1e-6)

    def test_kronecker_operands_keep_kronecker_structure(self):
        """The congruence term must not collapse to a dense block."""
        A1, A2, P1, P2 = self._factors()
        A_op = Kronecker(lx.MatrixLinearOperator(A1), lx.MatrixLinearOperator(A2))
        P_op = Kronecker(
            lx.MatrixLinearOperator(P1, lx.symmetric_tag),
            lx.MatrixLinearOperator(P2, lx.symmetric_tag),
        )
        Q = process_noise_covariance(A_op, P_op)

        # Q = P - (A P A^T); the subtracted term is the sandwich.
        sandwiched = Q.operator2.operator
        assert isinstance(sandwiched, Kronecker)

        A_dense, P_dense = A_op.as_matrix(), P_op.as_matrix()
        expected = P_dense - A_dense @ P_dense @ A_dense.T
        assert jnp.allclose(Q.as_matrix(), expected, atol=1e-5)

    def test_diagonal_covariance_matches_dense(self):
        A1, _, _, _ = self._factors()
        d = jnp.array([2.0, 3.0])
        Q = process_noise_covariance(A1, lx.DiagonalLinearOperator(d))
        P_dense = jnp.diag(d)
        assert jnp.allclose(Q.as_matrix(), P_dense - A1 @ P_dense @ A1.T, atol=1e-6)

    def test_inverts_discrete_lyapunov_solve(self):
        """Q -> P_inf -> Q is a round trip; the two are inverse maps."""
        A1, _, P1, _ = self._factors()
        Q = process_noise_covariance(A1, P1)
        P_recovered = discrete_lyapunov_solve(A1, Q)
        assert jnp.allclose(P_recovered, P1, atol=1e-5)


class TestProductSDEKroneckerDiscretisation:
    """`ProductSDE.discretise` contracts per factor, not on the full matrix."""

    def _kernel(self):
        return QuasiPeriodicSDE(
            kernel1=MaternSDE(
                variance=jnp.array(1.3), lengthscale=jnp.array(0.7), order=2
            ),
            kernel2=PeriodicSDE(
                variance=jnp.array(1.0),
                lengthscale=jnp.array(1.0),
                period=jnp.array(2.0),
                n_harmonics=3,
            ),
        )

    def test_matches_dense_triple_product(self):
        """Per-factor congruence agrees with the full (d1*d2)-square form."""
        kern = self._kernel()
        dt = jnp.array(0.37)
        A, Q = kern.discretise(dt)
        params = kern.sde_params()
        Q_dense = symmetrize(process_noise_covariance(A, params.P_inf))
        assert jnp.allclose(Q, Q_dense, atol=1e-5)

    def test_result_is_symmetric(self):
        kern = self._kernel()
        _, Q = kern.discretise(jnp.array(0.2))
        assert jnp.allclose(Q, Q.T, atol=1e-12)

    def test_jit_compatible(self):
        kern = self._kernel()

        @jax.jit
        def get_AQ(dt):
            return kern.discretise(dt)

        A, Q = get_AQ(jnp.array(0.37))
        A_ref, Q_ref = kern.discretise(jnp.array(0.37))
        assert jnp.allclose(A, A_ref, atol=1e-6)
        assert jnp.allclose(Q, Q_ref, atol=1e-6)
