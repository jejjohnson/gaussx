"""Tests for gaussx.row_kron, the row-wise Kronecker product."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gaussx


def _sparse(matrix, symmetric=False):
    rows, cols = np.nonzero(np.tril(matrix) if symmetric else matrix)
    values = jnp.asarray(matrix[rows, cols])
    return gaussx.SparseOperator.from_coo(
        rows, cols, values, matrix.shape, symmetric=symmetric
    )


def _random(rng, shape, density=0.5):
    return rng.normal(size=shape) * (rng.random(shape) < density)


def _dense(left, right):
    return np.stack([np.kron(a, b) for a, b in zip(left, right, strict=True)])


@pytest.mark.parametrize(
    "shapes", [((6, 3), (6, 4)), ((6, 2), (6, 3)), ((1, 4), (1, 1))]
)
def test_matches_dense_rows(shapes):
    # (6, 2) • (6, 3) is square, where a pattern stores its diagonal.
    rng = np.random.default_rng(0)
    left, right = _random(rng, shapes[0]), _random(rng, shapes[1])
    left[0] = 0.0  # an empty row
    product = gaussx.row_kron(_sparse(left), _sparse(right))
    np.testing.assert_allclose(product.as_matrix(), _dense(left, right), rtol=1e-6)
    assert not product.pattern.symmetric


def test_symmetric_storage():
    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    other = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 5.0]])
    product = gaussx.row_kron(_sparse(matrix, symmetric=True), _sparse(other))
    np.testing.assert_allclose(product.as_matrix(), _dense(matrix, other))


def test_acts_as_the_product_of_the_factors():
    rng = np.random.default_rng(1)
    left, right = _sparse(_random(rng, (5, 3))), _sparse(_random(rng, (5, 4)))
    x, y = jnp.arange(1.0, 4.0), jnp.array([0.5, -1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        gaussx.row_kron(left, right).mv(jnp.kron(x, y)),
        left.mv(x) * right.mv(y),
        rtol=1e-5,
    )


def test_no_entries():
    product = gaussx.row_kron(_sparse(np.zeros((3, 2))), _sparse(np.ones((3, 2))))
    np.testing.assert_array_equal(product.as_matrix(), np.zeros((3, 4)))


def test_grad_jit_and_vmap_over_values():
    rng = np.random.default_rng(2)
    left, right = _sparse(_random(rng, (5, 3))), _sparse(_random(rng, (5, 4)))
    x = jnp.asarray(rng.normal(size=12))

    def loss(values_left, values_right):
        a = eqx.tree_at(lambda op: op.values, left, values_left)
        b = eqx.tree_at(lambda op: op.values, right, values_right)
        return jnp.sum(gaussx.row_kron(a, b).mv(x) ** 2)

    def dense_loss(values_left, values_right):
        a = eqx.tree_at(lambda op: op.values, left, values_left).as_matrix()
        b = eqx.tree_at(lambda op: op.values, right, values_right).as_matrix()
        return jnp.sum((jax.vmap(jnp.kron)(a, b) @ x) ** 2)

    point = (left.values, right.values)
    grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(*point)
    expected = jax.grad(dense_loss, argnums=(0, 1))(*point)
    for got, want in zip(grads, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-4)
    batch = jnp.stack([left.values, 2.0 * left.values])
    batched = jax.vmap(loss, in_axes=(0, None))(batch, right.values)
    np.testing.assert_allclose(batched[1], 4.0 * batched[0], rtol=1e-5)


def test_dtype_follows_the_operands():
    one = gaussx.SparseOperator.from_coo([0], [0], jnp.ones(1, jnp.float32), (1, 2))
    assert gaussx.row_kron(one, one).values.dtype == jnp.float32


def test_row_mismatch_raises():
    with pytest.raises(ValueError, match="same number of rows"):
        gaussx.row_kron(_sparse(np.ones((3, 2))), _sparse(np.ones((2, 2))))


def test_too_many_columns_raises():
    wide = gaussx.SparseOperator.from_coo([0], [0], jnp.ones(1), (1, 2**16))
    with pytest.raises(ValueError, match="32-bit"):
        gaussx.row_kron(wide, wide)
