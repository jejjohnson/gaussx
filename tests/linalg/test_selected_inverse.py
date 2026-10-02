"""Tests for the block Takahashi selected inverse of a BlockTriDiag.

Keys are pinned: the assertions are about an identity, not about sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gaussx
from gaussx._einx import rearrange


def _spd_block_tridiag(N, d, key):
    """Band of a random SPD matrix, made SPD as a band by diagonal dominance."""
    k1, k2 = jr.split(key)
    sub = 0.3 * jr.normal(k1, (N - 1, d, d))
    raw = jr.normal(k2, (N, d, d))
    diag = raw @ rearrange(raw, "N i j -> N j i") + 5.0 * jnp.eye(d)
    return gaussx.BlockTriDiag(diag, sub)


def _dense_band(matrix, N, d):
    blocks = rearrange(matrix, "(N i) (M j) -> N M i j", N=N, M=N)
    diagonal = jnp.stack([blocks[k, k] for k in range(N)])
    sub = jnp.stack([blocks[k + 1, k] for k in range(N - 1)])
    return diagonal, sub


@pytest.mark.parametrize("d", [1, 2, 3])
def test_matches_dense_inverse_band(d):
    N = 6
    op = _spd_block_tridiag(N, d, jr.key(0))
    band = gaussx.selected_inverse(op)
    diagonal, sub = _dense_band(jnp.linalg.inv(op.as_matrix()), N, d)
    assert isinstance(band, gaussx.BlockTriDiag)
    assert jnp.allclose(band.diagonal, diagonal, atol=1e-12)
    assert jnp.allclose(band.sub_diagonal, sub, atol=1e-12)


def test_single_block():
    op = _spd_block_tridiag(1, 2, jr.key(1))
    band = gaussx.selected_inverse(op)
    assert band.sub_diagonal.shape == (0, 2, 2)
    assert jnp.allclose(band.diagonal[0], jnp.linalg.inv(op.diagonal[0]), atol=1e-12)


def test_jit_and_float32():
    op = _spd_block_tridiag(5, 2, jr.key(2))
    op32 = gaussx.BlockTriDiag(
        op.diagonal.astype(jnp.float32), op.sub_diagonal.astype(jnp.float32)
    )
    band = jax.jit(gaussx.selected_inverse)(op32)
    assert band.diagonal.dtype == jnp.float32
    assert band.sub_diagonal.dtype == jnp.float32
    diagonal, _ = _dense_band(jnp.linalg.inv(op.as_matrix()), 5, 2)
    assert jnp.allclose(band.diagonal, diagonal, atol=1e-5)


def test_rejects_unsupported_operators():
    op = _spd_block_tridiag(3, 2, jr.key(3))
    with pytest.raises(TypeError, match="BlockTriDiag"):
        gaussx.selected_inverse(gaussx.Kronecker(op, op))
    nonsym = gaussx.BlockTriDiag(op.diagonal, op.sub_diagonal, symmetric=False)
    with pytest.raises(ValueError, match="symmetric"):
        gaussx.selected_inverse(nonsym)
