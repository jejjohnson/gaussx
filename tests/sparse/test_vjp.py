"""Gradients through the sparse factor (G4), per stored value.

The references differentiate dense linear algebra on the matrix the operator
represents: ``as_matrix()`` mirrors a symmetric (lower-triangle) pattern, so
an off-diagonal stored value moves ``Q_ij`` and ``Q_ji`` and its gradient is
doubled; general storage is factored as ``½(Q + Qᵀ)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import gaussx
from gaussx import SparseOperator
from gaussx._einx import rearrange


LAYOUTS = [("rcm", True), ("rcm", False), ("natural", False)]


def _dense(op: SparseOperator, values: jax.Array) -> jax.Array:
    Q = SparseOperator(values, op.pattern).as_matrix()
    return 0.5 * (Q + rearrange(Q, "i j -> j i"))


def _finite_differences(f, values: jax.Array, eps: float = 1e-6) -> np.ndarray:
    """Central differences, one stored value at a time."""
    out = np.zeros(values.shape[0])
    for k in range(values.shape[0]):
        step = jnp.zeros_like(values).at[k].set(eps)
        out[k] = (f(values + step) - f(values - step)) / (2 * eps)
    return out


def _functions(op, sym):
    b = jr.normal(jr.key(0), (op.in_size(),))
    w = jr.normal(jr.key(1), (op.in_size(),))

    def logdet(values):
        return gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym).logdet()

    def logdet_dense(values):
        return jnp.linalg.slogdet(_dense(op, values))[1]

    def solve(values):
        factor = gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym)
        return jnp.dot(w, factor.solve(b))

    def solve_dense(values):
        return jnp.dot(w, jnp.linalg.solve(_dense(op, values), b))

    return {"logdet": (logdet, logdet_dense), "solve": (solve, solve_dense)}


@pytest.mark.parametrize("quantity", ["logdet", "solve"])
@pytest.mark.parametrize(("ordering", "banded"), LAYOUTS)
@pytest.mark.parametrize("symmetric", [True, False])
def test_vjp_matches_dense_and_finite_differences(
    grid, symbolic_for, quantity, ordering, banded, symmetric
):
    op = grid(3, 4, symmetric=symmetric)
    sym = symbolic_for(op, ordering, banded=banded)
    f, f_dense = _functions(op, sym)[quantity]
    grad = jax.grad(f)(op.values)
    np.testing.assert_allclose(grad, jax.grad(f_dense)(op.values), atol=1e-10)
    np.testing.assert_allclose(
        grad, _finite_differences(jax.jit(f), op.values), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("symmetric", [True, False])
def test_logdet_gradient_is_selected_inverse_with_storage_factor(grid, symmetric):
    # Symmetric storage: 2 Z_ij off the diagonal, Z_ii on it. General: Z_ij.
    op = grid(3, 3, symmetric=symmetric)
    grad = jax.grad(
        lambda v: gaussx.sparse_cholesky(SparseOperator(v, op.pattern)).logdet()
    )(op.values)
    Z = np.linalg.inv(np.asarray(op.as_matrix()))
    rows, cols = op.pattern.rows, op.pattern.cols
    factor = np.where(symmetric & (rows != cols), 2.0, 1.0)
    np.testing.assert_allclose(grad, factor * Z[rows, cols], atol=1e-12)


def test_solve_gradient_wrt_rhs(grid):
    op = grid(3, 4)
    factor = gaussx.sparse_cholesky(op)
    w = jnp.arange(12.0)
    grad = jax.grad(lambda b: jnp.dot(w, factor.solve(b)))(jnp.ones(12))
    np.testing.assert_allclose(grad, np.linalg.solve(np.asarray(op.as_matrix()), w))


@pytest.mark.slow
@pytest.mark.parametrize("banded", [True, False])
def test_autodiff_through_factorisation(grid, symbolic_for, banded):
    # diag_inv and sampling are differentiated by JAX through the factor.
    op = grid(3, 3, fixed_effect=True)
    sym = symbolic_for(op, "rcm", banded=banded)
    z = jr.normal(jr.key(2), (op.in_size(),))

    def variances(values):
        factor = gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym)
        return jnp.sum(jnp.sin(factor.diag_inv()))

    def variances_dense(values):
        return jnp.sum(jnp.sin(jnp.diag(jnp.linalg.inv(_dense(op, values)))))

    def sample(values):
        factor = gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym)
        return jnp.sum(factor.solve_lower_transpose(z) ** 2)

    def sample_dense(values):
        # Same draw: x = Pᵀ L⁻ᵀ z, with L the factor of the permuted matrix.
        Q = _dense(op, values)[sym.perm][:, sym.perm]
        L = jnp.linalg.cholesky(Q)
        x = jax.scipy.linalg.solve_triangular(L, z, lower=True, trans="T")
        return jnp.sum(x**2)

    for f, f_dense in [(variances, variances_dense), (sample, sample_dense)]:
        np.testing.assert_allclose(
            jax.grad(f)(op.values), jax.grad(f_dense)(op.values), atol=1e-10
        )


def test_vmap_of_grad(grid):
    op = grid(3, 3)
    sym = gaussx.symbolic_cholesky(op.pattern)

    def logdet(scale):
        values = scale * op.values
        return gaussx.sparse_cholesky(SparseOperator(values, op.pattern), sym).logdet()

    # log|s Q| = n log s + log|Q|, so d/ds = n / s.
    scales = jnp.array([0.5, 1.0, 3.0])
    np.testing.assert_allclose(jax.vmap(jax.grad(logdet))(scales), 9 / scales)


@pytest.mark.parametrize("banded", [True, False])
def test_reverse_over_reverse_hessian_matches_dense(grid, symbolic_for, banded):
    # A θ Hessian (INLA's θ-design) differentiates the custom backward passes.
    # It was silently wrong while L carried stop_gradient.
    op = grid(3, 3)
    sym = symbolic_for(op, "rcm", banded=banded)
    on_diagonal = jnp.asarray(op.pattern.rows == op.pattern.cols, op.values.dtype)
    b = jr.normal(jr.key(3), (op.in_size(),))

    def values(theta):
        return jnp.exp(theta[0]) * op.values + jnp.exp(theta[1]) * on_diagonal

    def f(theta):
        factor = gaussx.sparse_cholesky(SparseOperator(values(theta), op.pattern), sym)
        return factor.logdet() + jnp.dot(b, factor.solve(b))

    def f_dense(theta):
        Q = _dense(op, values(theta))
        return jnp.linalg.slogdet(Q)[1] + jnp.dot(b, jnp.linalg.solve(Q, b))

    theta = jnp.array([0.3, -0.7])
    np.testing.assert_allclose(
        jax.jacrev(jax.jacrev(f))(theta), jax.hessian(f_dense)(theta), atol=1e-9
    )
