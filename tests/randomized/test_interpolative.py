"""Tests for column_id and cur (G17, gh-486).

Every key is pinned; the draws are incidental. The accuracy test is
Voronin & Martinsson's (2017) headline: on matrices with a known
spectrum, the ID error is within the strong rank-revealing-QR bound
(Halko, Martinsson & Tropp, 2011, §5.2) of the sketch's own QB error, and
so a small multiple of the optimal sigma_{k+1}.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx as gx
from gaussx._testing import default_tolerances


def _known_spectrum(m, n, sigma):
    """``U diag(sigma) Vᵀ`` with Haar-random orthonormal ``U``, ``V``."""
    r = sigma.shape[0]
    U, _ = jnp.linalg.qr(jr.normal(jr.key(0), (m, r)))
    V, _ = jnp.linalg.qr(jr.normal(jr.key(1), (n, r)))
    return einx.dot("m r, n r -> m n", einx.multiply("m r, r -> m r", U, sigma), V)


def _spectral_norm(M):
    return float(np.linalg.norm(np.asarray(M), 2))


def _rank_k_system():
    m, n, k = 40, 30, 4
    A = _known_spectrum(m, n, jnp.arange(k, 0.0, -1.0))
    rtol, atol = default_tolerances(A)
    tol = {"rtol": 100 * rtol, "atol": 100 * atol}  # pivoted-QR solves, size k
    return A, k, tol


def test_column_id_exact_on_a_rank_k_matrix():
    A, k, tol = _rank_k_system()
    cols, X = eqx.filter_jit(gx.column_id)(lx.MatrixLinearOperator(A), k)
    assert jnp.allclose(A[:, cols] @ X, A, **tol)
    assert jnp.array_equal(X[:, cols], jnp.eye(k, dtype=X.dtype))
    assert len(set(np.asarray(cols))) == k


def test_cur_exact_and_actual_on_a_rank_k_matrix():
    A, k, tol = _rank_k_system()
    d = eqx.filter_jit(gx.cur)(lx.MatrixLinearOperator(A), k)
    assert jnp.allclose(d.C @ d.U @ d.R, A, **tol)
    # Actual columns and rows of A, k distinct each.
    assert jnp.array_equal(d.C, A[:, d.columns])
    assert jnp.array_equal(d.R, A[d.rows])
    assert len(set(np.asarray(d.columns))) == len(set(np.asarray(d.rows))) == k


@pytest.mark.slow
@pytest.mark.parametrize("decay", ["geometric", "polynomial"])
def test_error_within_the_rank_revealing_bound(decay):
    """Voronin & Martinsson (2017): near-optimal ID and CUR errors."""
    m, n, k = 300, 200, 20
    i = jnp.arange(n, dtype=float)
    sigma = 0.7**i if decay == "geometric" else 1.0 / (1.0 + i) ** 2
    A = _known_spectrum(m, n, sigma)
    op = lx.MatrixLinearOperator(A)
    key = jr.key(2)
    cols, X = gx.column_id(op, k, key=key)
    d = gx.cur(op, k, key=key)
    Q, _ = gx.qb(op, k, key=key)  # the same sketch column_id uses
    qb_error = _spectral_norm(A - Q @ einx.dot("m l, m n -> l n", Q, A))
    id_error = _spectral_norm(A - A[:, cols] @ X)
    cur_error = _spectral_norm(A - d.C @ d.U @ d.R)
    # HMT (2011) §5.2: ||A - A_J X|| <= (1 + sqrt(1 + 4k(n - k))) ||A - QQᵀA||;
    # measured ID / sigma_{k+1} = 1.5 (geometric) and 1.7 (polynomial).
    assert sigma[k] <= id_error <= (1 + math.sqrt(1 + 4 * k * (n - k))) * qb_error
    # The row ID of C adds the same factor over m (measured CUR / ID < 1.2).
    row_factor = 1 + math.sqrt(1 + 4 * k * (m - k))
    assert sigma[k] <= cur_error <= row_factor * id_error


def test_matrix_free_equals_dense():
    m, n, k = 40, 30, 5
    A = _known_spectrum(m, n, 0.5 ** jnp.arange(n, dtype=float))
    function_op = lx.FunctionLinearOperator(
        lambda v: einx.dot("m n, n -> m", A, v), jax.ShapeDtypeStruct((n,), A.dtype)
    )
    dense, matrix_free = eqx.filter_jit(
        lambda A, f: (gx.cur(lx.MatrixLinearOperator(A), k), gx.cur(f, k))
    )(A, function_op)
    assert jnp.array_equal(dense.columns, matrix_free.columns)
    assert jnp.array_equal(dense.rows, matrix_free.rows)
    rtol, atol = default_tolerances(A)
    assert jnp.allclose(dense.U, matrix_free.U, rtol=100 * rtol, atol=100 * atol)


def test_key_none_means_prngkey_zero():
    A = _known_spectrum(20, 15, 0.5 ** jnp.arange(15, dtype=float))
    op = lx.MatrixLinearOperator(A)
    default, seeded = eqx.filter_jit(
        lambda op: (gx.column_id(op, 3), gx.column_id(op, 3, key=jr.PRNGKey(0)))
    )(op)
    assert jnp.array_equal(default.columns, seeded.columns)


@pytest.mark.parametrize("rank", [0, 16])
def test_rejects_rank_outside_one_to_min_m_n(rank):
    op = lx.MatrixLinearOperator(jnp.ones((20, 15)))
    with pytest.raises(ValueError, match=r"rank must be in \[1, min\(m, n\)\]"):
        gx.column_id(op, rank)
    with pytest.raises(ValueError, match=r"rank must be in \[1, min\(m, n\)\]"):
        gx.cur(op, rank)
