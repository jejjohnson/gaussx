"""Tests for the block Takahashi selected inverse of a BlockTriDiag.

Keys are pinned: the assertions are about an identity, not about sampling.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import gaussx
from gaussx._einx import rearrange
from gaussx._testing import random_spd_block_tridiag


def _dense_band(matrix, N, d):
    blocks = rearrange(matrix, "(N i) (M j) -> N M i j", N=N, M=N)
    diagonal = jnp.stack([blocks[k, k] for k in range(N)])
    sub = jnp.stack([blocks[k + 1, k] for k in range(N - 1)])
    return diagonal, sub


@pytest.mark.slow
@pytest.mark.parametrize("d", [1, 2, 3])
def test_matches_dense_inverse_band(d):
    N = 6
    op = random_spd_block_tridiag(jr.key(0), N, d)
    band = gaussx.selected_inverse(op)
    diagonal, sub = _dense_band(jnp.linalg.inv(op.as_matrix()), N, d)
    assert isinstance(band, gaussx.BlockTriDiag)
    assert jnp.allclose(band.diagonal, diagonal, atol=1e-12)
    assert jnp.allclose(band.sub_diagonal, sub, atol=1e-12)


def test_single_block():
    op = random_spd_block_tridiag(jr.key(1), 1, 2)
    band = gaussx.selected_inverse(op)
    assert band.sub_diagonal.shape == (0, 2, 2)
    assert jnp.allclose(band.diagonal[0], jnp.linalg.inv(op.diagonal[0]), atol=1e-12)


@pytest.mark.slow
def test_jit_and_float32():
    op = random_spd_block_tridiag(jr.key(2), 5, 2)
    op32 = gaussx.BlockTriDiag(
        op.diagonal.astype(jnp.float32), op.sub_diagonal.astype(jnp.float32)
    )
    band = jax.jit(gaussx.selected_inverse)(op32)
    assert band.diagonal.dtype == jnp.float32
    assert band.sub_diagonal.dtype == jnp.float32
    diagonal, _ = _dense_band(jnp.linalg.inv(op.as_matrix()), 5, 2)
    assert jnp.allclose(band.diagonal, diagonal, atol=1e-5)


def test_rejects_unsupported_operators():
    op = random_spd_block_tridiag(jr.key(3), 3, 2)
    with pytest.raises(TypeError, match="BlockTriDiag"):
        gaussx.selected_inverse(gaussx.Kronecker(op, op))
    nonsym = gaussx.BlockTriDiag(op.diagonal, op.sub_diagonal, symmetric=False)
    with pytest.raises(ValueError, match="symmetric"):
        gaussx.selected_inverse(nonsym)
