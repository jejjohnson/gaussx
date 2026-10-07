"""Tests for svd primitive."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import BlockDiag, Kronecker, svd
from gaussx._einx import einsum
from gaussx._testing import random_pd_matrix, tree_allclose


def test_svd_diagonal():
    d = jnp.array([3.0, -1.0, 2.0])
    op = lx.DiagonalLinearOperator(d)
    U, s, Vt = svd(op)
    # s should be abs(d)
    assert tree_allclose(s, jnp.abs(d))
    # Reconstruct
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, jnp.diag(d), rtol=1e-5)


def test_svd_dense(getkey):
    mat = jr.normal(getkey(), (4, 4)) + 2 * jnp.eye(4)
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op)
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, mat, rtol=1e-4)


def test_svd_rectangular(getkey):
    mat = jr.normal(getkey(), (3, 5))
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op)
    reconstructed = U @ jnp.diag(s) @ Vt
    assert tree_allclose(reconstructed, mat, rtol=1e-4)


# ---------------------------------------------------------------------------
# rank: exact structured decomposition first, then the top-k (gh-383)
# ---------------------------------------------------------------------------


def _psd(key, n):
    return lx.MatrixLinearOperator(
        random_pd_matrix(key, n), lx.positive_semidefinite_tag
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: Kronecker(_psd(jr.key(0), 3), _psd(jr.key(1), 4)), id="kronecker"
        ),
        pytest.param(
            lambda: BlockDiag(
                lx.DiagonalLinearOperator(jnp.arange(1.0, 7.0)),
                lx.DiagonalLinearOperator(-jnp.arange(1.0, 7.0)),
            ),
            id="block_diag",
        ),
        pytest.param(
            lambda: lx.DiagonalLinearOperator(jnp.array([3.0, -1.0, 6.0, 2.0, -5.0])),
            id="diagonal",
        ),
        pytest.param(
            lambda: lx.TaggedLinearOperator(
                Kronecker(_psd(jr.key(0), 3), _psd(jr.key(1), 4)),
                lx.positive_semidefinite_tag,
            ),
            id="tagged_kronecker",
        ),
    ],
)
def test_svd_rank_structured_is_exact_top_k(build):
    op = build()
    k = 2
    U, s, Vt = svd(op, rank=k)
    assert U.shape == (op.out_size(), k)
    assert s.shape == (k,)
    assert Vt.shape == (k, op.in_size())
    top = jnp.linalg.svd(op.as_matrix(), compute_uv=False)[:k]
    assert tree_allclose(s, top)
    # The truncated factors reproduce the top-k action exactly.
    dense_k = einsum(U * s, Vt, "m k, k n -> m n")
    U_ref, s_ref, Vt_ref = jnp.linalg.svd(op.as_matrix())
    expected = einsum(U_ref[:, :k] * s_ref[:k], Vt_ref[:k], "m k, k n -> m n")
    assert tree_allclose(dense_k, expected)


def test_svd_rank_dense_shape():
    U, s, Vt = svd(_psd(jr.key(2), 12), rank=2)
    assert (U.shape, s.shape, Vt.shape) == ((12, 2), (2,), (2, 12))
