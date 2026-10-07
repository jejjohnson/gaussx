"""Tests for the sketching operators (G11).

Every test pins its key: the draws are incidental to the property checked.
The subspace-embedding test bounds the distortion by a concentration bound,
stated where it is used.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

import gaussx as gx
from gaussx._einx import einsum, reduce


M, D = 60, 24


def _sketches(d=D, m=M, probabilities=None):
    key = jr.key(0)
    return {
        "gaussian": gx.GaussianSketch.sample(key, d, m),
        "orthonormal": gx.OrthonormalSketch.sample(key, d, m),
        "sparse_sign": gx.SparseSignSketch.sample(key, d, m, nnz=4),
        "srht": gx.SRHTSketch.sample(key, d, m),
        "row_sampling": gx.RowSamplingSketch.sample(
            key, d, m, probabilities=probabilities
        ),
    }


SKETCH_NAMES = list(_sketches())

# Tracing SRHT's jitted apply builds an einx graph per butterfly pass, ~1-5 s
# per distinct input shape in CI, so its fixture cases run in the slow lane;
# test_srht_small_adjoint keeps apply and apply_transpose in the fast lane.
_SLOW_SRHT = pytest.param("srht", marks=pytest.mark.slow)


@pytest.fixture(
    params=[_SLOW_SRHT if name == "srht" else name for name in SKETCH_NAMES]
)
def sketch(request):
    return _sketches()[request.param]


def _dense(S):
    return S.apply(jnp.eye(S.in_size))


# --- shapes and the operator view -------------------------------------------


@pytest.mark.parametrize("rest", [(), (3,), (2, 3)])
def test_apply_shapes(sketch, rest):
    A = jr.normal(jr.key(1), (M, *rest))
    assert sketch.apply(A).shape == (D, *rest)
    assert sketch.apply_transpose(jr.normal(jr.key(2), (D, *rest))).shape == (
        M,
        *rest,
    )
    assert (sketch.in_size, sketch.out_size) == (M, D)


def test_apply_is_linear_on_columns(sketch):
    A = jr.normal(jr.key(1), (M, 3))
    SA = sketch.apply(A)
    assert jnp.allclose(SA, einsum(_dense(sketch), A, "d m, m n -> d n"), atol=1e-10)


def test_as_operator_matches_apply(sketch):
    op = sketch.as_operator()
    assert (op.out_size(), op.in_size()) == (D, M)
    assert jnp.allclose(op.as_matrix(), _dense(sketch), atol=1e-12)


# --- adjoint consistency -----------------------------------------------------


@pytest.mark.parametrize("rest", [(), (3,)])
def test_adjoint_consistency(sketch, rest):
    """<S x, y> = <x, S^T y>."""
    x = jr.normal(jr.key(1), (M, *rest))
    y = jr.normal(jr.key(2), (D, *rest))
    lhs = jnp.sum(sketch.apply(x) * y)
    rhs = jnp.sum(x * sketch.apply_transpose(y))
    assert jnp.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


# --- matrix-free agreement ---------------------------------------------------


def test_sketch_operator_matrix_free_matches_dense(sketch):
    A = jr.normal(jr.key(1), (M, 5))
    op = lx.FunctionLinearOperator(lambda v: einsum(A, v, "m n, n -> m"), jnp.zeros(5))
    expected = sketch.apply(op.as_matrix())
    assert jnp.allclose(sketch.sketch_operator(op), expected, atol=1e-10)
    assert jnp.allclose(
        sketch.sketch_operator(lx.MatrixLinearOperator(A)), expected, atol=1e-12
    )


def test_sketch_operator_rejects_wrong_size(sketch):
    with pytest.raises(ValueError, match="in_size"):
        sketch.sketch_operator(lx.MatrixLinearOperator(jnp.ones((M + 1, 2))))


# --- subspace embedding ------------------------------------------------------


@pytest.mark.slow
def test_subspace_embedding():
    m, n = 2000, 20
    d = 4 * n
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(1), (m, n)))
    leverage = reduce(Q**2, "m n -> m", "sum")
    # Davidson-Szarek: for a d x n Gaussian G / sqrt(d), the singular values
    # lie in 1 -+ (sqrt(n/d) + t/sqrt(d)) with probability >= 1 - 2 exp(-t^2/2);
    # t = 3 gives eps = 0.5 + 3/sqrt(80) ~= 0.84 at failure probability ~2%.
    # The SJLT / SRHT / leverage-sampling bounds (Cohen 2016; Tropp 2011; matrix
    # Chernoff) carry unspecified constants, so the Gaussian bound is the
    # reference for all of them; on this incoherent (Gaussian) subspace they
    # concentrate like the Gaussian sketch.
    eps = math.sqrt(n / d) + 3 / math.sqrt(d)
    for name, S in _sketches(d, m, probabilities=leverage).items():
        SQ = S.apply(Q)
        if name == "orthonormal":  # orthonormal rows: E[S^T S] = (d/m) I
            SQ = math.sqrt(m / d) * SQ
        sv = jnp.linalg.svd(SQ, compute_uv=False)
        assert 1 - eps <= sv.min() and sv.max() <= 1 + eps, (name, sv)


# --- per-sketch structure ----------------------------------------------------


def test_gaussian_sketch_entries_scaled_by_d():
    S = gx.GaussianSketch.sample(jr.key(0), 50, 400)
    # The mean square of the 20000 N(0, 1/d) entries is (1/d) (1 +- sqrt(2/N));
    # 6 sigma of that chi-square spread.
    ms = jnp.mean(S.matrix**2)
    assert abs(ms * 50 - 1) < 6 * math.sqrt(2 / S.matrix.size)


def test_orthonormal_sketch_rows_are_orthonormal():
    S = gx.OrthonormalSketch.sample(jr.key(0), D, M)
    assert jnp.allclose(
        einsum(S.matrix, S.matrix, "i m, j m -> i j"), jnp.eye(D), atol=1e-12
    )


def test_orthonormal_sketch_rejects_d_above_m():
    with pytest.raises(ValueError, match="d <= m"):
        gx.OrthonormalSketch.sample(jr.key(0), 5, 4)


@pytest.mark.parametrize(("d", "nnz"), [(24, 4), (24, 8), (5, 8), (3, 3)])
def test_sparse_sign_columns(d, nnz):
    S = gx.SparseSignSketch.sample(jr.key(0), d, M, nnz=nnz)
    k = min(nnz, d)
    assert S.rows.shape == S.signs.shape == (k, M)
    assert bool(jnp.all((S.rows >= 0) & (S.rows < d)))
    # nnz distinct rows per column, so exactly nnz non-zeros of size 1/sqrt(nnz).
    dense = np.asarray(_dense(S))
    assert (reduce(dense != 0, "d m -> m", "sum") == k).all()
    np.testing.assert_allclose(np.abs(dense[dense != 0]), 1 / math.sqrt(k))


@pytest.mark.slow
def test_srht_equals_dense_construction():
    m, d = 100, 16
    S = gx.SRHTSketch.sample(jr.key(0), d, m)
    m2 = 128
    P = np.eye(m)[np.asarray(S.permutation)]  # (P x)_i = x[perm_i]
    Dg = np.diag(np.asarray(S.signs))
    pad = np.eye(m2, m)
    H = scipy.linalg.hadamard(m2) / math.sqrt(m2)
    R = np.eye(m2)[np.asarray(S.rows)]
    dense = math.sqrt(m2 / d) * R @ H @ pad @ Dg @ P
    np.testing.assert_allclose(_dense(S), dense, atol=1e-12)
    assert len(np.unique(np.asarray(S.rows))) == d


def test_srht_small_adjoint():
    """<S x, y> = <x, S^T y> at m = 6 (three butterfly passes)."""
    S = gx.SRHTSketch.sample(jr.key(0), 4, 6)
    x = jr.normal(jr.key(1), (6,))
    y = jr.normal(jr.key(2), (4,))
    lhs = jnp.sum(S.apply(x) * y)
    rhs = jnp.sum(x * S.apply_transpose(y))
    assert jnp.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_srht_rejects_d_above_padded_size():
    with pytest.raises(ValueError, match="d <= 8"):
        gx.SRHTSketch.sample(jr.key(0), 9, 5)


def test_row_sampling_weights():
    p = jnp.arange(1.0, M + 1.0)
    S = gx.RowSamplingSketch.sample(jr.key(0), D, M, probabilities=p)
    p = p / jnp.sum(p)
    assert jnp.allclose(S.weights, 1 / jnp.sqrt(D * p[S.rows]))
    uniform = gx.RowSamplingSketch.sample(jr.key(0), D, M)
    assert jnp.allclose(uniform.weights, math.sqrt(M / D))


# --- dtype and transformations -----------------------------------------------


def test_float32_input_stays_float32(sketch):
    A = jnp.ones((M, 2), dtype=jnp.float32)
    assert sketch.apply(A).dtype == jnp.float32
    Y = jnp.ones((D, 2), dtype=jnp.float32)
    assert sketch.apply_transpose(Y).dtype == jnp.float32


def test_sketch_is_a_pytree_under_jit(sketch):
    A = jr.normal(jr.key(1), (M, 3))
    out = eqx.filter_jit(lambda S, A: S.apply(A))(sketch, A)
    assert jnp.allclose(out, sketch.apply(A), atol=1e-12)


def test_key_none_means_prngkey_zero():
    a = gx.SparseSignSketch.sample(None, D, M)
    b = gx.SparseSignSketch.sample(jax.random.PRNGKey(0), D, M)
    assert jnp.array_equal(a.rows, b.rows) and jnp.array_equal(a.signs, b.signs)
