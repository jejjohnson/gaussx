"""Tests for SDE kernel implementations."""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl
import lineax as lx
import pytest

from gaussx import (
    ConstantSDE,
    CosineSDE,
    IntegratedWienerSDE,
    Kronecker,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SDEParams,
    SumSDE,
    discrete_lyapunov_solve,
    discretise_mfd,
    kalman_filter,
    process_noise_covariance,
    sde_autocovariance,
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


def _kernel_zoo():
    """Every kernel with a closed-form stationary covariance, composites included."""
    m0 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=0)
    m1 = MaternSDE(variance=jnp.array(1.3), lengthscale=jnp.array(2.0), order=1)
    m2 = MaternSDE(variance=jnp.array(0.7), lengthscale=jnp.array(0.8), order=2)
    cos = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(1.4))
    const = ConstantSDE(variance=jnp.array(2.0))
    per = PeriodicSDE(
        variance=jnp.array(1.0),
        lengthscale=jnp.array(1.0),
        period=jnp.array(2.0),
        n_harmonics=2,
    )
    return {
        "matern0": m0,
        "matern1": m1,
        "matern2": m2,
        "cosine": cos,
        "constant": const,
        "periodic": per,
        "sum(m1, cos)": SumSDE(kernels=(m1, cos)),
        "sum(m0, m2, const)": SumSDE(kernels=(m0, m2, const)),
        # The Cosine factor is the pointed case: Q_c = 0, so the old
        # B1 (x) B2 diffusion was identically zero.
        "product(m1, cos)": ProductSDE(kernel1=m1, kernel2=cos),
        "product(m1, m2)": ProductSDE(kernel1=m1, kernel2=m2),
        "product(m0, per)": ProductSDE(kernel1=m0, kernel2=per),
        "quasiperiodic(m1, per)": QuasiPeriodicSDE(kernel1=m1, kernel2=per),
        # Nested composite: the corrected diffusion must survive one more level.
        "product(product(m1, cos), m0)": ProductSDE(
            kernel1=ProductSDE(kernel1=m1, kernel2=cos), kernel2=m0
        ),
    }


class TestLyapunovConsistency:
    """Every reported ``(F, L, Q_c, P_inf)`` must satisfy its own Lyapunov equation.

    For a stationary linear SDE, ``F P + P Fᵀ + L Q_c Lᵀ = 0``. ``ProductSDE``
    used to report ``L = L1 ⊗ L2`` / ``Q_c = Q_c1 ⊗ Q_c2``, implying a
    diffusion ``B1 ⊗ B2`` — but the drift is the Kronecker *sum* ``F1 ⊕ F2``
    and the stationary covariance the Kronecker *product* ``P1 ⊗ P2``, which
    force ``B = B1 ⊗ P2 + P1 ⊗ B2`` instead. Regression for gh-219.
    """

    @pytest.mark.parametrize("name", list(_kernel_zoo()))
    def test_stationary_lyapunov_residual_is_zero(self, name):
        params = _kernel_zoo()[name].sde_params()
        assert params.P_inf is not None, f"{name} has no stationary covariance"

        B = params.L @ params.Q_c @ params.L.T
        residual = params.F @ params.P_inf + params.P_inf @ params.F.T + B

        scale = jnp.maximum(jnp.max(jnp.abs(B)), 1.0)
        assert jnp.max(jnp.abs(residual)) < 1e-8 * scale

    @pytest.mark.parametrize("name", list(_kernel_zoo()))
    def test_reported_shapes_are_consistent(self, name):
        kern = _kernel_zoo()[name]
        params = kern.sde_params()
        d = kern.state_dim

        assert params.F.shape == (d, d)
        assert params.H.shape == (1, d)
        assert params.P_inf.shape == (d, d)
        # L is (d, s) and Q_c is (s, s) for whatever noise dimension s the
        # kernel needs — products widen it to s1*d2 + d1*s2.
        assert params.L.shape[0] == d
        s = params.L.shape[1]
        assert params.Q_c.shape == (s, s)


class TestProductSDEDiffusionConsistency:
    def test_composite_diffusion_matches_closed_form(self):
        """``L Q_c Lᵀ`` reproduces ``B1 ⊗ P2 + P1 ⊗ B2`` exactly."""
        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k2 = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(1.4))
        p1, p2 = k1.sde_params(), k2.sde_params()

        params = ProductSDE(kernel1=k1, kernel2=k2).sde_params()
        reported = params.L @ params.Q_c @ params.L.T

        B1 = p1.L @ p1.Q_c @ p1.L.T
        B2 = p2.L @ p2.Q_c @ p2.L.T
        expected = jnp.kron(B1, p2.P_inf) + jnp.kron(p1.P_inf, B2)

        assert jnp.allclose(reported, expected, atol=1e-10)
        # The old B1 ⊗ B2 was identically zero here, so this is the
        # assertion that actually pins the bug.
        assert jnp.max(jnp.abs(reported)) > 1.0

    def test_sde_params_rejects_missing_stationary_covariance(self):
        """A factor without ``P_inf`` makes the composite diffusion unbuildable."""

        class _NoPInfSDE(MaternSDE):
            def sde_params(self):
                params = super().sde_params()
                return SDEParams(
                    F=params.F, L=params.L, H=params.H, Q_c=params.Q_c, P_inf=None
                )

        k1 = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k2 = _NoPInfSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        kern = ProductSDE(kernel1=k1, kernel2=k2)

        with pytest.raises(NotImplementedError, match="B1 \\(x\\) P2"):
            kern.sde_params()
        with pytest.raises(NotImplementedError, match="B1 \\(x\\) P2"):
            kern.discretise(jnp.array(0.1))


class TestProductSDEParamsRobustness:
    """Cases the square-root formulation would have got wrong (gh-219 review)."""

    def test_semidefinite_factor_stays_exact(self):
        """A zero-variance factor must not pick up a jitter term.

        ``ConstantSDE(variance=0)`` has ``P_inf = 0``, so the product's
        stationary covariance and diffusion are both exactly zero. A
        jittered Cholesky would report ``eps * B_other`` instead and break
        the Lyapunov equation.
        """
        k1 = ConstantSDE(variance=jnp.array(0.0))
        k2 = MaternSDE(variance=jnp.array(1.3), lengthscale=jnp.array(0.8), order=1)
        params = ProductSDE(kernel1=k1, kernel2=k2).sde_params()

        B = params.L @ params.Q_c @ params.L.T
        assert jnp.max(jnp.abs(params.P_inf)) == 0.0
        assert jnp.max(jnp.abs(B)) == 0.0

        residual = params.F @ params.P_inf + params.P_inf @ params.F.T + B
        assert jnp.max(jnp.abs(residual)) == 0.0

    def test_composition_adds_no_dtype_promotion(self):
        """The identity blocks must follow the factors, not JAX's default.

        An untyped ``jnp.eye`` is float64 under x64 and would promote the
        composite past what its factors carry. Since gh-224 the leaf
        kernels report all-float32 params from float32 hyperparameters,
        so the composite must be float32 throughout.
        """
        f32 = jnp.float32
        k1 = MaternSDE(
            variance=jnp.array(1.0, dtype=f32),
            lengthscale=jnp.array(1.0, dtype=f32),
            order=1,
        )
        k2 = CosineSDE(
            variance=jnp.array(1.0, dtype=f32), frequency=jnp.array(1.4, dtype=f32)
        )
        params = ProductSDE(kernel1=k1, kernel2=k2).sde_params()

        assert params.F.dtype == f32
        assert params.L.dtype == f32
        assert params.H.dtype == f32
        assert params.Q_c.dtype == f32
        assert params.P_inf.dtype == f32

    def test_sde_params_is_reverse_mode_differentiable(self):
        """``safe_cholesky``'s ``lax.while_loop`` has no reverse-mode rule."""

        def loss(variance):
            k1 = MaternSDE(variance=variance, lengthscale=jnp.array(0.8), order=1)
            k2 = CosineSDE(variance=jnp.array(1.0), frequency=jnp.array(1.4))
            params = ProductSDE(kernel1=k1, kernel2=k2).sde_params()
            return jnp.sum(params.L @ params.Q_c @ params.L.T)

        grad = jax.grad(loss)(jnp.array(1.3))
        assert jnp.isfinite(grad)
        assert jnp.abs(grad) > 0.0


def _leaf_kernels_f32():
    """One float32-hyperparameter instance of every leaf SDE kernel."""
    f32 = jnp.float32
    v = jnp.array(1.0, dtype=f32)
    ell = jnp.array(1.0, dtype=f32)
    return [
        pytest.param(
            MaternSDE(variance=v, lengthscale=ell, order=order), id=f"matern{order}"
        )
        for order in (0, 1, 2)
    ] + [
        pytest.param(
            CosineSDE(variance=v, frequency=jnp.array(1.4, dtype=f32)), id="cosine"
        ),
        pytest.param(
            PeriodicSDE(variance=v, lengthscale=ell, period=jnp.array(2.0, dtype=f32)),
            id="periodic",
        ),
        pytest.param(ConstantSDE(variance=v), id="constant"),
    ]


class TestLeafKernelDtypes:
    """Float32 hyperparameters must survive x64 mode end to end (gh-224).

    The suite runs with ``jax_enable_x64`` on (see ``tests/conftest.py``),
    so any untyped ``jnp.array``/``jnp.zeros``/``jnp.eye`` in a kernel
    would surface here as a float64 field.
    """

    @pytest.mark.parametrize("kernel", _leaf_kernels_f32())
    def test_sde_params_keep_float32(self, kernel):
        params = kernel.sde_params()
        for field in ("F", "L", "H", "Q_c", "P_inf"):
            assert getattr(params, field).dtype == jnp.float32, field

    @pytest.mark.parametrize("kernel", _leaf_kernels_f32())
    def test_discretise_keeps_float32(self, kernel):
        A, Q = kernel.discretise(jnp.array(0.1, dtype=jnp.float32))
        assert A.dtype == jnp.float32
        assert Q.dtype == jnp.float32


class TestIntegratedWienerSDE:
    """The non-stationary local-linear-trend prior (gh-234)."""

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_state_dim_and_params(self, order):
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.7), order=order)
        params = kern.sde_params()
        d = order + 1

        assert kern.state_dim == d
        assert params.F.shape == (d, d)
        assert params.L.shape == (d, 1)
        assert params.H.shape == (1, d)
        assert params.Q_c.shape == (1, 1)
        # No stationary covariance exists — that is the whole point.
        assert params.P_inf is None
        assert kern.stationary is False
        # Drift is the nilpotent shift: f' feeds f, and the top
        # derivative is driven by the noise alone.
        assert jnp.allclose(params.F, jnp.eye(d, k=1))
        assert jnp.allclose(jnp.linalg.matrix_power(params.F, d), 0.0)

    def test_local_linear_trend_closed_form(self):
        """``order=1`` is the textbook local linear trend pair."""
        q, dt = 0.7, jnp.array(0.37)
        A, Q = IntegratedWienerSDE(diffusion=jnp.array(q)).discretise(dt)

        assert jnp.allclose(A, jnp.array([[1.0, dt], [0.0, 1.0]]))
        assert jnp.allclose(
            Q,
            q * jnp.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]),
        )

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_closed_form_matches_matrix_fraction(self, order):
        """The closed form is exact, so MFD must agree to round-off."""
        kern = IntegratedWienerSDE(diffusion=jnp.array(1.3), order=order)
        params = kern.sde_params()
        dt = jnp.array(0.37)

        A, Q = kern.discretise(dt)
        A_mfd, Q_mfd = discretise_mfd(params.F, params.L @ params.Q_c @ params.L.T, dt)

        assert jnp.allclose(A, A_mfd, atol=1e-12)
        assert jnp.allclose(Q, Q_mfd, atol=1e-12)

    def test_recursion_reproduces_exact_covariance(self):
        """Marginals of the discretised recursion match the IWP covariance.

        For ``order=1`` started at ``f(0) = f'(0) = 0`` with no
        uncertainty, ``Var f(t) = q t^3 / 3``, ``Var f'(t) = q t``, and
        ``Cov(f(t), f'(t)) = q t^2 / 2`` — the covariance grows without
        bound, unlike any stationary kernel in the zoo.
        """
        q = 0.6
        kern = IntegratedWienerSDE(diffusion=jnp.array(q))
        times = jnp.array([0.0, 0.4, 1.1, 2.3, 3.0])

        cov = jnp.zeros((2, 2))
        for start, end in itertools.pairwise(times):
            A, Q = kern.discretise(end - start)
            cov = A @ cov @ A.T + Q
            t = end
            expected = q * jnp.array([[t**3 / 3, t**2 / 2], [t**2 / 2, t]])
            assert jnp.allclose(cov, expected, atol=1e-10)

    def test_zero_step_is_the_identity(self):
        """``dt = 0`` must not raise a masked entry to a negative power."""
        A, Q = IntegratedWienerSDE(diffusion=jnp.array(0.5), order=2).discretise(
            jnp.array(0.0)
        )
        assert jnp.allclose(A, jnp.eye(3))
        assert jnp.allclose(Q, 0.0)
        assert jnp.all(jnp.isfinite(A)) and jnp.all(jnp.isfinite(Q))

    def test_initial_covariance_defaults_to_diffuse(self):
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.5))
        P_0 = kern.initial_covariance()
        kappa = P_0[0, 0]

        assert P_0.shape == (2, 2)
        assert jnp.allclose(P_0, kappa * jnp.eye(2))
        # Diffuse: vague by many orders of magnitude against a noise
        # variance of order one, which is what "diffuse" has to mean
        # for a prior that cannot see the data scale.
        assert kappa > 1e6

    def test_default_diffuse_prior_survives_float32(self):
        """A vaguer default would cancel the first update to zero variance.

        The Kalman update forms ``P - K S K^T``, so once ``P_0 / R``
        exceeds the dtype's precision the subtraction gives exactly
        zero: the filter reports a *noiseless* first observation and
        stays overconfident. In float32 a ``1e6`` default does that for
        any ``R`` below about 0.1 — here ``R = 0.05^2``.
        """
        f32 = jnp.float32
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.5, dtype=f32))
        kappa = kern.initial_covariance()[0, 0]
        R = jnp.array(0.05**2, dtype=f32)

        gain = kappa / (kappa + R)
        posterior = kappa - gain * (kappa + R) * gain
        exact = kappa * R / (kappa + R)

        assert gain < 1.0
        assert posterior > 0.0
        assert jnp.abs(posterior - exact) < 0.1 * exact

    def test_initial_covariance_is_caller_supplied(self):
        """The initial covariance is a modelling choice, not a derived one."""
        P_0 = jnp.diag(jnp.array([1e4, 0.25]))
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.5), P_0=P_0)

        assert jnp.allclose(kern.initial_covariance(), P_0)

    def test_negative_step_is_rejected(self):
        """The closed form has no guard of its own, unlike ``expm``.

        At ``order=1`` a negative step returns a negative-definite
        ``Q`` — not a covariance — which would propagate silently into
        filtering. ``discretise_mfd`` rejects the same input.
        """
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.5))

        with pytest.raises(eqx.EquinoxRuntimeError, match="requires dt >= 0"):
            kern.discretise(jnp.array(-0.3))

    def test_zero_step_gradient_is_finite(self):
        """``diff(times, prepend=times[0])`` puts a ``dt = 0`` in every sequence.

        A traced exponent would differentiate ``dt ** 0`` through the
        generic power rule and give ``0 * inf = NaN``, poisoning the
        gradient of any objective differentiated through the timestamps.
        """
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.6))

        grad_A = jax.grad(lambda t: kern.discretise(t)[0].sum())(jnp.array(0.0))
        grad_Q = jax.grad(lambda t: kern.discretise(t)[1].sum())(jnp.array(0.0))

        # dA/ddt at 0 is the drift's superdiagonal; dQ/ddt at 0 is q,
        # from the single linear-in-dt entry.
        assert jnp.allclose(grad_A, 1.0)
        assert jnp.allclose(grad_Q, 0.6)

    @pytest.mark.parametrize("order", [13, 20])
    def test_high_orders_keep_valid_coefficients(self, order):
        """Factorials must not overflow into negative coefficients.

        ``(p-i)! (p-j)! (2p+1-i-j)`` exceeds int64 from order 13 up, so
        an integer denominator would wrap and hand back a ``Q`` that is
        not PSD even for positive ``diffusion`` and ``dt``.
        """
        kern = IntegratedWienerSDE(diffusion=jnp.array(1.0), order=order)
        A, Q = kern.discretise(jnp.array(0.5))

        assert jnp.all(jnp.isfinite(A)) and jnp.all(jnp.isfinite(Q))
        assert jnp.all(jnp.diag(Q) > 0.0)
        # Round-off only: the matrix spans many orders of magnitude at
        # these orders, so PSD is asserted to float64 round-off.
        assert jnp.linalg.eigvalsh(Q).min() > -1e-14

    def test_integer_hyperparameters_promote(self):
        """An integer ``diffusion`` or ``dt`` must not truncate the fractions.

        Every entry is a fraction of a power of the step, so an integer
        promotion would floor the coefficients to zero — ``Q`` came back
        as ``[[0, 0], [0, 1]]`` — and ``jnp.finfo`` rejects the dtype
        the diffuse default needs.
        """
        kern = IntegratedWienerSDE(diffusion=jnp.array(1))
        A, Q = kern.discretise(jnp.array(1))

        assert jnp.issubdtype(Q.dtype, jnp.inexact)
        assert jnp.allclose(A, jnp.array([[1.0, 1.0], [0.0, 1.0]]))
        assert jnp.allclose(Q, jnp.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]))
        assert jnp.issubdtype(kern.initial_covariance().dtype, jnp.inexact)
        assert jnp.issubdtype(kern.sde_params().F.dtype, jnp.inexact)

    def test_negative_order_is_rejected_at_construction(self):
        """``order=-1`` gives an empty state and an unhelpful late failure."""
        with pytest.raises(ValueError, match="must be non-negative"):
            IntegratedWienerSDE(diffusion=jnp.array(1.0), order=-1)

    @pytest.mark.slow
    def test_extreme_order_does_not_overflow_python_floats(self):
        """``1.0 / n`` would raise OverflowError from order 98 up.

        The denominator ``(p-i)! (p-j)! (2p+1-i-j)`` exceeds the float
        range there, even though its reciprocal is representable. Integer
        division keeps both operands unbounded Python ints and rounds
        once, at the end. Marked slow: the big-integer coefficient table
        is ~10 s to build at this order.
        """
        kern = IntegratedWienerSDE(diffusion=jnp.array(1.0), order=98)
        A, Q = kern.discretise(jnp.array(10.0))

        assert jnp.all(jnp.isfinite(A)) and jnp.all(jnp.isfinite(Q))
        # The far corner underflows (see the discretise docstring); the
        # point of the test is that it neither raises nor returns NaN.
        assert Q[-1, -1] > 0.0

    def test_autocovariance_is_rejected(self):
        """``K(tau)`` is meaningless for a non-stationary process."""
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.5))

        with pytest.raises(ValueError, match="not stationary"):
            sde_autocovariance(kern, jnp.array([0.1, 0.5]))

    def test_jit_and_grad(self):
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.6))

        @jax.jit
        def get_AQ(dt):
            return kern.discretise(dt)

        A, Q = get_AQ(jnp.array(0.5))
        assert jnp.all(jnp.isfinite(A)) and jnp.all(jnp.isfinite(Q))

        def loss(diffusion):
            _A, Q = IntegratedWienerSDE(diffusion=diffusion).discretise(jnp.array(0.5))
            return jnp.sum(Q)

        grad = jax.grad(loss)(jnp.array(0.6))
        # Q is linear in q, so the gradient is the sum of its coefficients.
        assert jnp.allclose(grad, 0.5**3 / 3 + 0.5**2 + 0.5)

    def test_keeps_float32(self):
        """Float32 hyperparameters must survive x64 mode (gh-224)."""
        f32 = jnp.float32
        kern = IntegratedWienerSDE(diffusion=jnp.array(0.6, dtype=f32))

        params = kern.sde_params()
        for field in ("F", "L", "H", "Q_c"):
            assert getattr(params, field).dtype == f32, field

        A, Q = kern.discretise(jnp.array(0.1, dtype=f32))
        assert A.dtype == f32
        assert Q.dtype == f32
        assert kern.initial_covariance().dtype == f32

    def test_filters_a_linear_trend(self):
        """End to end: the prior it exists for, through the Kalman filter."""
        times = jnp.linspace(0.0, 20.0, 40)
        noise_std = 0.05
        observations = (
            0.5
            + 0.3 * times
            + noise_std * jr.normal(jr.key(0), times.shape)  # key pinned: the
            # test is about the filter, not about a particular draw.
        )

        kern = IntegratedWienerSDE(diffusion=jnp.array(1e-4))
        A_seq, Q_seq = kern.discretise_sequence(jnp.diff(times, prepend=times[0]))
        state = kalman_filter(
            transition=A_seq,
            obs_model=kern.sde_params().H,
            process_noise=Q_seq,
            obs_noise=jnp.array([[noise_std**2]]),
            observations=observations[:, None],
            init_mean=jnp.zeros(2),
            init_cov=kern.initial_covariance(),
        )

        level, slope = state.filtered_means[-1]
        assert jnp.allclose(slope, 0.3, atol=0.02)
        assert jnp.allclose(level, observations[-1], atol=0.1)


class TestSumWithNonStationaryComponent:
    """A trend plus a stationary component — the mixed case (gh-234)."""

    def _kernel(self):
        trend = IntegratedWienerSDE(diffusion=jnp.array(0.2))
        matern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(2.0), order=1)
        return trend, matern, SumSDE(kernels=(trend, matern))

    def test_sum_is_non_stationary_without_p_inf(self):
        _, _, kern = self._kernel()

        assert kern.stationary is False
        assert kern.sde_params().P_inf is None

    def test_initial_covariance_is_block_diagonal(self):
        trend, matern, kern = self._kernel()
        P_0 = kern.initial_covariance()

        assert P_0.shape == (4, 4)
        assert jnp.allclose(P_0[:2, :2], trend.initial_covariance())
        assert jnp.allclose(P_0[2:, 2:], matern.sde_params().P_inf)
        assert jnp.allclose(P_0[:2, 2:], 0.0)

    def test_discretises_through_the_matrix_fraction_fallback(self):
        """No ``P_inf``, so the sum takes the MFD route — block by block."""
        trend, matern, kern = self._kernel()
        dt = jnp.array(0.3)

        A, Q = kern.discretise(dt)
        A_trend, Q_trend = trend.discretise(dt)
        A_matern, Q_matern = matern.discretise(dt)

        assert jnp.allclose(A, jsl.block_diag(A_trend, A_matern), atol=1e-9)
        assert jnp.allclose(Q, jsl.block_diag(Q_trend, Q_matern), atol=1e-9)

    def test_product_with_a_trend_factor_is_rejected(self):
        """A Kronecker product needs both factors' ``P_inf``."""
        trend, matern, _ = self._kernel()
        kern = ProductSDE(kernel1=matern, kernel2=trend)

        with pytest.raises(NotImplementedError, match="B1 \\(x\\) P2"):
            kern.sde_params()


class TestStationaryKernelContract:
    """The additive half of gh-234: existing kernels keep working."""

    @pytest.mark.parametrize("kernel", _leaf_kernels_f32())
    def test_stationary_kernels_start_from_p_inf(self, kernel):
        assert kernel.stationary is True
        assert jnp.allclose(kernel.initial_covariance(), kernel.sde_params().P_inf)

    def test_missing_p_inf_reports_what_to_override(self):
        """``P_inf=None`` on a stationary kernel leaves nothing to start from."""

        class _NoPInfSDE(MaternSDE):
            def sde_params(self):
                params = super().sde_params()
                return SDEParams(
                    F=params.F, L=params.L, H=params.H, Q_c=params.Q_c, P_inf=None
                )

        kern = _NoPInfSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)

        with pytest.raises(ValueError, match="initial_covariance"):
            kern.initial_covariance()
