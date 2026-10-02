"""Tests for randomized_nystrom."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import LowRankUpdate, randomized_nystrom, svd_low_rank_plus_diag
from gaussx._testing import random_pd_matrix, tree_allclose


_PSD = lx.positive_semidefinite_tag


def _low_rank_psd(n, r):
    W = jr.normal(jr.key(0), (n, r))
    return einx.dot("i r, j r -> i j", W, W)


@pytest.mark.parametrize(("rank", "oversample"), [(5, 0), (8, 0), (5, 3)])
def test_exact_for_a_low_rank_psd_matrix(rank, oversample):
    # A rank-5 PSD matrix is recovered exactly once the sketch has l >= 5
    # columns; nu only stabilises the Cholesky and is subtracted again.
    A = _low_rank_psd(60, 5)
    approx = randomized_nystrom(
        lx.MatrixLinearOperator(A, _PSD), rank, oversample=oversample, key=jr.key(1)
    )
    assert approx.U.shape == (60, rank)
    assert tree_allclose(approx.as_matrix(), A, rtol=1e-8, atol=1e-8)


def test_matrix_free_operator_matches_dense():
    A = random_pd_matrix(jr.key(0), 30)
    dense = randomized_nystrom(lx.MatrixLinearOperator(A, _PSD), 10, key=jr.key(1))
    func = lx.FunctionLinearOperator(
        lambda v: A @ v, jax.ShapeDtypeStruct((30,), A.dtype), _PSD
    )
    free = randomized_nystrom(func, 10, key=jr.key(1))
    assert tree_allclose(free.as_matrix(), dense.as_matrix(), rtol=1e-10)


def test_returns_an_orthonormal_psd_low_rank_update():
    A = random_pd_matrix(jr.key(0), 30)
    approx = randomized_nystrom(lx.MatrixLinearOperator(A, _PSD), 10, key=jr.key(1))
    assert isinstance(approx, LowRankUpdate)
    assert approx.orthonormal
    assert approx.V is approx.U
    assert lx.is_symmetric(approx) and lx.is_positive_semidefinite(approx)
    assert tree_allclose(approx.base.as_matrix(), jnp.zeros((30, 30)))
    gram = einx.dot("n i, n j -> i j", approx.U, approx.U)
    assert tree_allclose(gram, jnp.eye(10), atol=1e-10)
    assert bool(jnp.all(jnp.diff(approx.d) <= 0)) and bool(jnp.all(approx.d >= 0))


def test_is_dominated_by_the_operator():
    # 0 ⪯ Â ⪯ A (Tropp et al., 2017): A − Â is PSD up to round-off.
    A = random_pd_matrix(jr.key(0), 40)
    approx = randomized_nystrom(lx.MatrixLinearOperator(A, _PSD), 12, key=jr.key(1))
    gap = jnp.linalg.eigvalsh(A - approx.as_matrix())
    assert float(jnp.min(gap)) > -1e-10 * float(jnp.max(jnp.abs(A)))


def test_factors_plug_into_low_rank_update_solve_and_logdet():
    # The same factors on a σ²I base dispatch through the Woodbury rules.
    A = random_pd_matrix(jr.key(0), 40)
    approx = randomized_nystrom(lx.MatrixLinearOperator(A, _PSD), 12, key=jr.key(1))
    noise = 0.3
    noisy = svd_low_rank_plus_diag(
        jnp.full(40, noise), approx.U, approx.d, approx.U, psd=True
    )
    dense = approx.as_matrix() + noise * jnp.eye(40)
    b = jr.normal(jr.key(2), (40,))
    assert tree_allclose(gaussx.solve(noisy, b), jnp.linalg.solve(dense, b))
    assert tree_allclose(gaussx.logdet(noisy), jnp.linalg.slogdet(dense)[1])


def test_jit_and_float32_stay_finite():
    # The stabilising shift keeps the small Cholesky finite in float32.
    x = jnp.linspace(0.0, 10.0, 300, dtype=jnp.float32)
    r = jnp.sqrt(jnp.float32(3.0)) * jnp.abs(einx.subtract("i, j -> i j", x, x))
    K = (1 + r) * jnp.exp(-r)
    op = lx.MatrixLinearOperator(K, _PSD)
    approx = jax.jit(lambda o: randomized_nystrom(o, 60, key=jr.key(0)))(op)
    assert approx.d.dtype == jnp.float32 and approx.U.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(approx.d))) and bool(
        jnp.all(jnp.isfinite(approx.U))
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"rank": 0}, "rank"), ({"rank": 3, "oversample": -1}, "oversample")],
)
def test_rejects_bad_arguments(kwargs, match):
    op = lx.MatrixLinearOperator(jnp.eye(4), _PSD)
    with pytest.raises(ValueError, match=match):
        randomized_nystrom(op, **kwargs)


def test_rejects_a_non_square_operator():
    with pytest.raises(ValueError, match="square"):
        randomized_nystrom(lx.MatrixLinearOperator(jnp.ones((4, 3))), 2)
