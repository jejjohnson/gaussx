"""Tests for natural gradient primitives."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum, rearrange
from gaussx._inference._natural_gradient import (
    damped_natural_update,
    gauss_newton_precision,
    riemannian_psd_correction,
)
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._testing import (
    default_tolerances,
    key_sequence,
    random_pd_matrix,
    random_pd_operator,
)


def _make_block_tridiag(getkey, N=3, d=2):
    """Helper to build a test BlockTriDiag."""
    diag_raw = jax.random.normal(getkey(), (N, d, d))
    diag = jax.vmap(lambda A: A @ A.T + 2.0 * jnp.eye(d))(diag_raw)
    sub = 0.1 * jax.random.normal(getkey(), (N - 1, d, d))
    return BlockTriDiag(diag, sub)


class TestDampedNaturalUpdate:
    def test_lr_zero_returns_old(self, getkey):
        """lr=0 should return old parameters unchanged."""
        nat1_old = jax.random.normal(getkey(), (4,))
        nat2_old = jax.random.normal(getkey(), (4, 4))
        nat1_target = jax.random.normal(getkey(), (4,))
        nat2_target = jax.random.normal(getkey(), (4, 4))

        nat1_new, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=0.0
        )
        assert jnp.allclose(nat1_new, nat1_old)
        assert jnp.allclose(nat2_new, nat2_old)

    def test_lr_one_returns_target(self, getkey):
        """lr=1 should return target parameters."""
        nat1_old = jax.random.normal(getkey(), (4,))
        nat2_old = jax.random.normal(getkey(), (4, 4))
        nat1_target = jax.random.normal(getkey(), (4,))
        nat2_target = jax.random.normal(getkey(), (4, 4))

        nat1_new, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=1.0
        )
        assert jnp.allclose(nat1_new, nat1_target)
        assert jnp.allclose(nat2_new, nat2_target)

    @pytest.mark.slow
    def test_block_tridiag_preserved(self, getkey):
        """BlockTriDiag structure should be preserved."""
        nat1_old = jax.random.normal(getkey(), (6,))
        nat2_old = _make_block_tridiag(getkey)
        nat1_target = jax.random.normal(getkey(), (6,))
        nat2_target = _make_block_tridiag(getkey)

        _, nat2_new = damped_natural_update(
            nat1_old, nat2_old, nat1_target, nat2_target, lr=0.5
        )
        assert isinstance(nat2_new, BlockTriDiag)

    def test_block_tridiag_interpolation(self, getkey):
        """BlockTriDiag interpolation should match dense interpolation."""
        nat2_old = _make_block_tridiag(getkey)
        nat2_target = _make_block_tridiag(getkey)
        nat1 = jnp.zeros(6)

        _, nat2_new = damped_natural_update(nat1, nat2_old, nat1, nat2_target, lr=0.3)

        expected = 0.7 * nat2_old.as_matrix() + 0.3 * nat2_target.as_matrix()
        assert jnp.allclose(nat2_new.as_matrix(), expected, atol=1e-10)

    def test_generic_operator(self):
        """Generic operators should materialize to MatrixLinearOperator."""
        nat1 = jnp.zeros(3)
        nat2_old = lx.MatrixLinearOperator(jnp.eye(3))
        nat2_target = lx.MatrixLinearOperator(2.0 * jnp.eye(3))

        _, nat2_new = damped_natural_update(nat1, nat2_old, nat1, nat2_target, lr=0.5)
        assert isinstance(nat2_new, lx.MatrixLinearOperator)
        assert jnp.allclose(nat2_new.as_matrix(), 1.5 * jnp.eye(3))

    def test_numpy_arrays_are_accepted(self):
        """gh-394: NumPy arrays dispatch as arrays, not as a type mismatch."""
        nat1, nat2 = damped_natural_update(
            np.zeros(3), np.eye(3), np.ones(3), 2.0 * np.eye(3), 0.5
        )
        assert isinstance(nat2, jax.Array)
        assert jnp.allclose(nat2, 1.5 * jnp.eye(3))
        assert jnp.allclose(nat1, 0.5 * jnp.ones(3))

    def test_mixed_types_error_names_both(self):
        """gh-394: an array mixed with an operator names both types."""
        with pytest.raises(
            TypeError,
            match=r"both be arrays or both be linear operators.*"
            r"ArrayImpl and MatrixLinearOperator",
        ):
            damped_natural_update(
                jnp.zeros(3),
                jnp.eye(3),
                jnp.zeros(3),
                lx.MatrixLinearOperator(jnp.eye(3)),
                0.5,
            )

    def test_traced_learning_rate(self):
        """gh-394: ``lr`` may be a traced scalar (``jax.grad`` through it)."""

        def loss(lr):
            _, nat2 = damped_natural_update(
                jnp.zeros(2), jnp.eye(2), jnp.zeros(2), 3.0 * jnp.eye(2), lr
            )
            return jnp.trace(nat2)

        assert jnp.allclose(jax.grad(loss)(jnp.asarray(0.5)), 4.0)


class TestRiemannianPSDCorrection:
    def test_shape(self, getkey):
        """Output should match input shape."""
        d = 4
        H = jax.random.normal(getkey(), (d, d))
        S_prec = jax.random.normal(getkey(), (d, d))
        S_prec = S_prec @ S_prec.T
        S_cov = jnp.linalg.inv(S_prec)

        result = riemannian_psd_correction(H, S_prec, S_cov, lr=1.0)
        assert result.shape == (d, d)

    def test_zero_lr_returns_hessian(self, getkey):
        """lr=0 should return the original Hessian."""
        d = 3
        H = jax.random.normal(getkey(), (d, d))
        S_prec = jnp.eye(d)
        S_cov = jnp.eye(d)

        result = riemannian_psd_correction(H, S_prec, S_cov, lr=0.0)
        assert jnp.allclose(result, H)

    # gh-401: what the correction guarantees is not that the corrected Hessian
    # is negative semi-definite (for H = diag(-1, 2), Lambda = I, lr = 0.1 it
    # has eigenvalues [-1, 1.55]) but that the damped precision update
    # (1 - lr) Lambda - lr H_psd is positive definite: with Sigma = Lambda^-1
    # and G = Lambda + H it equals
    # 1/2 Lambda + 1/2 (I - lr G Sigma) Lambda (I - lr Sigma G),
    # PD plus PSD. The 1e-12 tolerance is float64 round-off on a 2 x 2 or
    # 3 x 3 problem.
    @pytest.mark.x64_only(reason="identity checked to float64 round-off")
    @pytest.mark.parametrize("lr", [0.1, 0.5, 1.0])
    def test_damped_precision_update_is_positive_definite(self, lr):
        H = jnp.diag(jnp.array([-1.0, 2.0]))
        Lam = jnp.eye(2)

        H_psd = riemannian_psd_correction(H, Lam, jnp.linalg.inv(Lam), lr=lr)
        updated = (1.0 - lr) * Lam - lr * H_psd

        assert jnp.min(jnp.linalg.eigvalsh(updated)) > 0.0
        if lr == 0.1:
            # The corrected Hessian itself stays indefinite.
            assert jnp.allclose(
                jnp.linalg.eigvalsh(H_psd), jnp.array([-1.0, 1.55]), atol=1e-12
            )

    @pytest.mark.x64_only(reason="identity checked to float64 round-off")
    @pytest.mark.parametrize("lr", [0.1, 0.5, 1.0])
    def test_damped_precision_update_identity(self, lr):
        nextkey = key_sequence(0)
        d = 3
        A = jax.random.normal(nextkey(), (d, d))
        H = 0.5 * (A + rearrange(A, "i j -> j i"))  # symmetric, indefinite
        Lam = random_pd_matrix(nextkey(), d, jitter=1.0)
        Sigma = jnp.linalg.inv(Lam)
        G = Lam + H

        H_psd = riemannian_psd_correction(H, Lam, Sigma, lr=lr)
        updated = (1.0 - lr) * Lam - lr * H_psd

        left = jnp.eye(d) - lr * einsum(G, Sigma, "i k, k j -> i j")
        right = jnp.eye(d) - lr * einsum(Sigma, G, "i k, k j -> i j")
        expected = 0.5 * Lam + 0.5 * einsum(left, Lam, right, "i a, a b, b j -> i j")
        assert jnp.min(jnp.linalg.eigvalsh(H)) < 0.0
        assert jnp.allclose(updated, expected, atol=1e-12, rtol=0.0)
        assert jnp.min(jnp.linalg.eigvalsh(updated)) > 0.0


class TestGaussNewtonPrecision:
    def test_low_rank_when_obs_lt_latent(self, getkey):
        """Should return LowRankUpdate when D_obs < D_latent."""
        from gaussx._operators._low_rank_update import LowRankUpdate

        J = jax.random.normal(getkey(), (3, 10))
        op = gauss_newton_precision(J)
        assert isinstance(op, LowRankUpdate)

    def test_dense_when_obs_ge_latent(self, getkey):
        """Should return MatrixLinearOperator when D_obs >= D_latent."""
        J = jax.random.normal(getkey(), (10, 3))
        op = gauss_newton_precision(J)
        assert isinstance(op, lx.MatrixLinearOperator)

    def test_matches_dense(self, getkey):
        """Should match J^T J."""
        J = jax.random.normal(getkey(), (5, 8))
        op = gauss_newton_precision(J)
        expected = J.T @ J
        assert jnp.allclose(op.as_matrix(), expected, atol=1e-6)

    def test_psd(self, getkey):
        """Result should always be PSD."""
        J = jax.random.normal(getkey(), (3, 6))
        op = gauss_newton_precision(J)
        eigvals = jnp.linalg.eigvalsh(op.as_matrix())
        # J has rank 3, so its zero eigenvalues are computed to rounding
        rounding = 10 * jnp.finfo(eigvals.dtype).eps * jnp.max(eigvals)
        assert jnp.all(eigvals >= -rounding)


class TestGaussNewtonPrecisionWithBase:
    """gh-334: a prior precision makes J^T J invertible and keeps Woodbury."""

    @pytest.mark.parametrize("shape", [(2, 4), (5, 3)], ids=["wide", "tall"])
    def test_matches_dense_posterior_precision(self, shape):
        from gaussx._operators._low_rank_update import LowRankUpdate

        J = jax.random.normal(jax.random.key(0), shape)
        D_latent = shape[1]
        prior = lx.DiagonalLinearOperator(2.0 * jnp.ones(D_latent))
        op = gauss_newton_precision(J, base=prior)
        assert isinstance(op, LowRankUpdate)
        assert op.base is prior
        assert lx.is_symmetric(op)
        M = J.T @ J + 2.0 * jnp.eye(D_latent)
        b = jnp.ones(D_latent)
        assert jnp.allclose(op.as_matrix(), M, atol=1e-12)
        assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(M, b), atol=1e-12)
        assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(M)[1], atol=1e-12)

    def test_psd_tag_follows_the_base(self):
        J = jax.random.normal(jax.random.key(0), (2, 4))
        tagged = lx.MatrixLinearOperator(2.0 * jnp.eye(4), lx.positive_semidefinite_tag)
        assert lx.is_positive_semidefinite(gauss_newton_precision(J, base=tagged))
        untagged = lx.MatrixLinearOperator(2.0 * jnp.eye(4))
        assert not lx.is_positive_semidefinite(gauss_newton_precision(J, base=untagged))

    def test_base_shape_mismatch_raises(self):
        J = jax.random.normal(jax.random.key(0), (2, 4))
        with pytest.raises(ValueError, match="base must be"):
            gauss_newton_precision(J, base=lx.DiagonalLinearOperator(jnp.ones(3)))

    def test_without_base_is_documented_singular(self):
        # Pinned: rank D_obs < D_latent, so there is no inverse to compute.
        J = jax.random.normal(jax.random.key(0), (2, 4))
        op = gauss_newton_precision(J)
        assert jnp.linalg.matrix_rank(op.as_matrix()) == 2
        assert not jnp.all(jnp.isfinite(gaussx.solve(op, jnp.ones(4))))


# ---------------------------------------------------------------------------
# gauss_newton_operator: A = P + Jᵀ W J, matrix-free
# ---------------------------------------------------------------------------


def _nonlinear_residual(theta):
    """r: ℝ³ → ℝ⁴, nonlinear in every entry of θ."""
    return jnp.stack(
        [
            jnp.sin(theta[0]) + theta[1] ** 2,
            theta[0] * theta[1],
            jnp.exp(0.5 * theta[2]),
            theta[0] + theta[2],
        ]
    )


def _gn_point():
    return jnp.array([0.3, -0.7, 0.2], dtype=jnp.result_type(float))


def _gn_noise():
    return random_pd_operator(jr.key(0), 4)


def _gn_prior():
    return random_pd_operator(jr.key(1), 3)


def _gn_dense(point):
    jacobian = jax.jacfwd(_nonlinear_residual)(point)
    weighted = _gn_noise().as_matrix() @ jacobian
    return _gn_prior().as_matrix() + einsum(jacobian, weighted, "m i, m j -> i j")


class TestGaussNewtonOperator:
    @pytest.mark.parametrize("jac", ["fwd", "bwd"])
    def test_matches_the_dense_jacobian(self, jac):
        point = _gn_point()
        operator = gaussx.gauss_newton_operator(
            _nonlinear_residual,
            point,
            noise_precision=_gn_noise(),
            prior_precision=_gn_prior(),
            jac=jac,
        )
        expected = _gn_dense(point)
        rtol, atol = default_tolerances(expected)
        assert lx.is_positive_semidefinite(operator)
        assert jnp.allclose(operator.as_matrix(), expected, rtol=rtol, atol=atol)
        matrix = operator.as_matrix()
        assert jnp.allclose(
            matrix, rearrange(matrix, "i j -> j i"), rtol=rtol, atol=atol
        )

    def test_defaults_are_identity_noise_and_no_prior(self):
        point = _gn_point()
        operator = gaussx.gauss_newton_operator(_nonlinear_residual, point)
        jacobian = jax.jacfwd(_nonlinear_residual)(point)
        expected = einsum(jacobian, jacobian, "m i, m j -> i j")
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(operator.as_matrix(), expected, rtol=rtol, atol=atol)

    def test_callable_precisions_match_operators(self):
        point = _gn_point()
        noise, prior = _gn_noise(), _gn_prior()
        by_operator = gaussx.gauss_newton_operator(
            _nonlinear_residual, point, noise_precision=noise, prior_precision=prior
        )
        by_callable = gaussx.gauss_newton_operator(
            _nonlinear_residual,
            point,
            noise_precision=noise.mv,
            prior_precision=prior.mv,
        )
        assert jnp.array_equal(by_operator.as_matrix(), by_callable.as_matrix())

    def test_linear_residual_is_exact(self):
        H = jr.normal(jr.key(2), (5, 3), dtype=jnp.result_type(float))
        operator = gaussx.gauss_newton_operator(
            lambda theta: H @ theta, jnp.zeros(3, H.dtype)
        )
        expected = einsum(H, H, "m i, m j -> i j")
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(operator.as_matrix(), expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("jac", ["fwd", "bwd"])
    def test_pytree_control_and_shaped_residual(self, jac):
        """θ = {a, b}, r of shape (2, 2), W a 2-D diagonal, P added leaf by leaf."""
        dtype = jnp.result_type(float)
        point = {"a": jnp.array([0.5, -0.5], dtype), "b": jnp.array(0.25, dtype)}

        def residual(theta):
            a, b = theta["a"], theta["b"]
            return jnp.array([[a[0] ** 2, a[1] * b], [jnp.sin(b), a[0] + a[1]]])

        weights = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype)
        operator = gaussx.gauss_newton_operator(
            residual,
            point,
            noise_precision=lx.DiagonalLinearOperator(weights),
            prior_precision=lambda theta: jax.tree.map(lambda v: 2.0 * v, theta),
            jac=jac,
        )
        assert jax.tree.map(lambda s: s.shape, operator.in_structure()) == {
            "a": (2,),
            "b": (),
        }
        direction = {"a": jnp.array([1.0, 2.0], dtype), "b": jnp.array(-1.0, dtype)}
        applied = operator.mv(direction)

        flat_point, unravel = jax.flatten_util.ravel_pytree(point)
        flat_direction, _ = jax.flatten_util.ravel_pytree(direction)
        jacobian = jax.jacfwd(lambda flat: jnp.ravel(residual(unravel(flat))))(
            flat_point
        )
        weighted = jnp.ravel(weights) * (jacobian @ flat_direction)
        expected = 2.0 * flat_direction + einsum(jacobian, weighted, "m d, m -> d")
        flat_applied, _ = jax.flatten_util.ravel_pytree(applied)
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(flat_applied, expected, rtol=rtol, atol=atol)

    def test_bwd_takes_a_residual_with_only_a_vjp(self):
        @jax.custom_vjp
        def square(x):
            return x**2

        def square_fwd(x):
            return x**2, x

        def square_bwd(x, cotangent):
            return (2.0 * x * cotangent,)

        square.defvjp(square_fwd, square_bwd)
        point = jnp.array([1.0, 2.0], dtype=jnp.result_type(float))
        with pytest.raises(TypeError, match="forward-mode"):
            gaussx.gauss_newton_operator(square, point, jac="fwd").mv(point)
        operator = gaussx.gauss_newton_operator(square, point, jac="bwd")
        assert jnp.allclose(operator.as_matrix(), jnp.diag(jnp.array([4.0, 16.0])))

    def test_rejects_an_unknown_jac(self):
        with pytest.raises(ValueError, match="jac must be"):
            gaussx.gauss_newton_operator(_nonlinear_residual, _gn_point(), jac="rev")

    def test_jit_and_vmap_over_points(self):
        vector = jnp.ones(3, dtype=jnp.result_type(float))

        def apply(point):
            operator = gaussx.gauss_newton_operator(
                _nonlinear_residual,
                point,
                noise_precision=_gn_noise(),
                prior_precision=_gn_prior(),
            )
            return operator.mv(vector)

        points = jnp.stack([_gn_point(), 2.0 * _gn_point()])
        expected = jnp.stack([_gn_dense(point) @ vector for point in points])
        rtol, atol = default_tolerances(expected)
        batched = jax.jit(jax.vmap(apply))(points)
        assert jnp.allclose(batched, expected, rtol=10 * rtol, atol=10 * atol)

    def test_gradient_through_a_cg_solve(self):
        """∂/∂b of ½ bᵀ A⁻¹ b is A⁻¹ b, and ∂/∂y flows through the operator."""
        point = _gn_point()
        right_hand_side = jnp.array([1.0, -1.0, 0.5], dtype=point.dtype)
        solver = gaussx.CGSolver(rtol=1e-5, atol=1e-5)

        def energy(scale, rhs):
            def residual(theta):
                return scale * _nonlinear_residual(theta)

            operator = gaussx.gauss_newton_operator(
                residual, point, prior_precision=_gn_prior()
            )
            return 0.5 * rhs @ solver.solve(operator, rhs)

        def dense_energy(scale, rhs):
            jacobian = scale * jax.jacfwd(_nonlinear_residual)(point)
            matrix = _gn_prior().as_matrix() + einsum(
                jacobian, jacobian, "m i, m j -> i j"
            )
            return 0.5 * rhs @ jnp.linalg.solve(matrix, rhs)

        one = jnp.asarray(1.0, dtype=point.dtype)
        by_scale = jax.grad(energy)(one, right_hand_side)
        expected_scale = jax.grad(dense_energy)(one, right_hand_side)
        by_rhs = jax.grad(energy, argnums=1)(one, right_hand_side)
        expected_rhs = jax.grad(dense_energy, argnums=1)(one, right_hand_side)
        assert jnp.allclose(by_scale, expected_scale, rtol=1e-2, atol=1e-3)
        assert jnp.allclose(by_rhs, expected_rhs, rtol=1e-2, atol=1e-3)
