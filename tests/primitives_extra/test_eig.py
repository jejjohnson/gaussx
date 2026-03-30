"""Tests for eig, eigvals primitives."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker
from gaussx._primitives._eig import eig, eigvals
from gaussx._testing import random_pd_matrix, tree_allclose


def test_eig_diagonal(getkey):
    d = jnp.array([3.0, 1.0, 2.0])
    op = lx.DiagonalLinearOperator(d)
    vals, vecs = eig(op)
    assert tree_allclose(vals, d)
    assert tree_allclose(vecs, jnp.eye(3))


def test_eigvals_diagonal(getkey):
    d = jnp.array([5.0, 2.0, 7.0])
    op = lx.DiagonalLinearOperator(d)
    assert tree_allclose(eigvals(op), d)


def test_eig_symmetric(getkey):
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals, vecs = eig(op)
    # Reconstruct: A = V diag(lam) V^T
    reconstructed = vecs @ jnp.diag(vals) @ vecs.T
    assert tree_allclose(reconstructed, mat, rtol=1e-4)


def test_eig_dense(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    vals, vecs = eig(op)
    # Check A @ v = lam * v for each eigenpair
    for i in range(3):
        lhs = mat @ vecs[:, i]
        rhs = vals[i] * vecs[:, i]
        assert tree_allclose(lhs, rhs, atol=1e-4)


def test_eigvals_symmetric(getkey):
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals = eigvals(op)
    expected = jnp.linalg.eigvalsh(mat)
    assert tree_allclose(jnp.sort(vals), jnp.sort(expected), rtol=1e-5)


def test_eig_block_diag(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    A_op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    B_op = lx.MatrixLinearOperator(B, lx.symmetric_tag)
    bd = BlockDiag(A_op, B_op)

    vals, vecs = eig(bd)
    reconstructed = vecs @ jnp.diag(vals) @ vecs.T
    assert tree_allclose(reconstructed, bd.as_matrix(), rtol=1e-4)


def test_eigvals_block_diag(getkey):
    d1 = jnp.array([1.0, 2.0])
    d2 = jnp.array([3.0, 4.0, 5.0])
    bd = BlockDiag(
        lx.DiagonalLinearOperator(d1),
        lx.DiagonalLinearOperator(d2),
    )
    expected = jnp.concatenate([d1, d2])
    assert tree_allclose(eigvals(bd), expected)


def test_eig_kronecker(getkey):
    A = random_pd_matrix(getkey(), 2)
    B = random_pd_matrix(getkey(), 3)
    A_op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    B_op = lx.MatrixLinearOperator(B, lx.symmetric_tag)
    K = Kronecker(A_op, B_op)

    vals, _vecs = eig(K)
    # Check eigenvalue product property
    vals_A = jnp.linalg.eigvalsh(A)
    vals_B = jnp.linalg.eigvalsh(B)
    expected_vals = jnp.kron(vals_A, vals_B)
    assert tree_allclose(jnp.sort(vals), jnp.sort(expected_vals), rtol=1e-4)


def test_eigvals_kronecker(getkey):
    d1 = jnp.array([2.0, 3.0])
    d2 = jnp.array([4.0, 5.0])
    K = Kronecker(
        lx.DiagonalLinearOperator(d1),
        lx.DiagonalLinearOperator(d2),
    )
    expected = jnp.kron(d1, d2)
    assert tree_allclose(eigvals(K), expected)
