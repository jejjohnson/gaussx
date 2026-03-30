"""Tests for low_rank_plus_identity convenience constructor."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from gaussx._operators import LowRankUpdate, low_rank_plus_identity
from gaussx._testing import tree_allclose


def test_low_rank_plus_identity_basic(getkey):
    N, k = 5, 2
    U = jr.normal(getkey(), (N, k)) * 0.3
    op = low_rank_plus_identity(U)
    assert isinstance(op, LowRankUpdate)

    expected = jnp.eye(N) + U @ U.T
    assert tree_allclose(op.as_matrix(), expected, rtol=1e-5)


def test_low_rank_plus_identity_scaled(getkey):
    N, k = 4, 2
    U = jr.normal(getkey(), (N, k)) * 0.3
    scale = 2.5
    op = low_rank_plus_identity(U, scale=scale)

    expected = scale * jnp.eye(N) + U @ U.T
    assert tree_allclose(op.as_matrix(), expected, rtol=1e-5)


def test_low_rank_plus_identity_with_d(getkey):
    N, k = 4, 2
    U = jr.normal(getkey(), (N, k)) * 0.3
    d = jnp.array([2.0, 3.0])
    op = low_rank_plus_identity(U, d=d)

    expected = jnp.eye(N) + U @ jnp.diag(d) @ U.T
    assert tree_allclose(op.as_matrix(), expected, rtol=1e-5)
