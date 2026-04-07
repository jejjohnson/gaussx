"""Tests for ImplicitCrossKernelOperator with explicit params + custom JVP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from gaussx._operators import ImplicitCrossKernelOperator, implicit_cross_kernel
from gaussx._testing import tree_allclose


def _rbf_params(params, x, y):
    """RBF kernel with explicit params."""
    diff = x - y
    sq_dist = jnp.sum(diff**2) / params["lengthscale"] ** 2
    return params["variance"] * jnp.exp(-0.5 * sq_dist)


def _rbf_no_params(x, y):
    """RBF kernel without params."""
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2))


def _make_params(key):
    k1, k2 = jr.split(key)
    return {
        "variance": jnp.abs(jr.normal(k1, ())) + 0.5,
        "lengthscale": jnp.abs(jr.normal(k2, ())) + 0.5,
    }


def _build_dense(kernel_fn, params, X, Z):
    return jax.vmap(lambda x_i: jax.vmap(lambda z_j: kernel_fn(params, x_i, z_j))(Z))(X)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_params_still_works(self, getkey):
        X = jr.normal(getkey(), (12, 3))
        Z = jr.normal(getkey(), (6, 3))
        v = jr.normal(getkey(), (6,))
        op = ImplicitCrossKernelOperator(_rbf_no_params, X, Z, batch_size=4)
        K = jax.vmap(lambda x_i: jax.vmap(lambda z_j: _rbf_no_params(x_i, z_j))(Z))(X)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_convenience_no_params(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (5, 3))
        op = implicit_cross_kernel(_rbf_no_params, X, Z, batch_size=4)
        assert isinstance(op, ImplicitCrossKernelOperator)


# ---------------------------------------------------------------------------
# With params — mv correctness
# ---------------------------------------------------------------------------


class TestParamsMv:
    def test_mv_matches_dense(self, getkey):
        X = jr.normal(getkey(), (12, 3))
        Z = jr.normal(getkey(), (6, 3))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (6,))
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, batch_size=4, params=params)
        K = _build_dense(_rbf_params, params, X, Z)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_mv_batch_size_1(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (5,))
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, batch_size=1, params=params)
        K = _build_dense(_rbf_params, params, X, Z)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_as_matrix(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        params = _make_params(getkey())
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, params=params)
        K = _build_dense(_rbf_params, params, X, Z)
        assert tree_allclose(op.as_matrix(), K, rtol=1e-5)

    def test_convenience_with_params(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (5, 3))
        params = _make_params(getkey())
        op = implicit_cross_kernel(_rbf_params, X, Z, batch_size=4, params=params)
        K = _build_dense(_rbf_params, params, X, Z)
        v = jr.normal(getkey(), (5,))
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_mv_respects_batch_size_with_params(self, getkey):
        X = jr.normal(getkey(), (5, 3))
        Z = jr.normal(getkey(), (4, 3))
        params = _make_params(getkey())
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, batch_size=2, params=params)
        v = jr.normal(getkey(), (4,))

        closed_jaxpr = jax.make_jaxpr(op.mv)(v)
        custom_jvp_eqn = closed_jaxpr.jaxpr.eqns[0]
        inner_eqns = custom_jvp_eqn.params["call_jaxpr"].eqns
        scan_eqn = next(eqn for eqn in inner_eqns if eqn.primitive.name == "scan")

        assert scan_eqn.params["length"] == 3


# ---------------------------------------------------------------------------
# Transpose with params
# ---------------------------------------------------------------------------


class TestTransposeParams:
    def test_transpose_matrix(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        params = _make_params(getkey())
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, batch_size=4, params=params)
        assert tree_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-5)

    def test_transpose_mv(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        params = _make_params(getkey())
        op = ImplicitCrossKernelOperator(_rbf_params, X, Z, batch_size=4, params=params)
        u = jr.normal(getkey(), (10,))
        K = _build_dense(_rbf_params, params, X, Z)
        assert tree_allclose(op.T.mv(u), K.T @ u, rtol=1e-5)


# ---------------------------------------------------------------------------
# Gradients with params
# ---------------------------------------------------------------------------


class TestParamsGradients:
    def test_grad_params(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        Z = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (5,))
        u = jr.normal(getkey(), (8,))

        def loss_custom(params):
            op = ImplicitCrossKernelOperator(
                _rbf_params, X, Z, batch_size=4, params=params
            )
            return jnp.dot(u, op.mv(v))

        def loss_dense(params):
            K = _build_dense(_rbf_params, params, X, Z)
            return jnp.dot(u, K @ v)

        assert tree_allclose(
            jax.grad(loss_custom)(params),
            jax.grad(loss_dense)(params),
            rtol=1e-4,
        )

    def test_grad_x_data(self, getkey):
        Z = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (5,))
        u = jr.normal(getkey(), (8,))

        def loss_custom(X):
            op = ImplicitCrossKernelOperator(
                _rbf_params, X, Z, batch_size=4, params=params
            )
            return jnp.dot(u, op.mv(v))

        def loss_dense(X):
            K = _build_dense(_rbf_params, params, X, Z)
            return jnp.dot(u, K @ v)

        X = jr.normal(getkey(), (8, 2))
        assert tree_allclose(
            jax.grad(loss_custom)(X),
            jax.grad(loss_dense)(X),
            rtol=1e-4,
        )

    def test_grad_x_inducing(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (5,))
        u = jr.normal(getkey(), (8,))

        def loss_custom(Z):
            op = ImplicitCrossKernelOperator(
                _rbf_params, X, Z, batch_size=4, params=params
            )
            return jnp.dot(u, op.mv(v))

        def loss_dense(Z):
            K = _build_dense(_rbf_params, params, X, Z)
            return jnp.dot(u, K @ v)

        Z = jr.normal(getkey(), (5, 2))
        assert tree_allclose(
            jax.grad(loss_custom)(Z),
            jax.grad(loss_dense)(Z),
            rtol=1e-4,
        )
