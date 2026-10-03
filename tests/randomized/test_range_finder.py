"""Tests for range_finder, qb, randomized_svd and randomized_eigh (G12)."""

from __future__ import annotations

import itertools
import math

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx as gx
from gaussx._einx import einsum


def _known_spectrum(m, n, sigma, seed=0):
    """``U diag(sigma) Vᵀ`` with Haar-random orthonormal ``U``, ``V``."""
    r = sigma.shape[0]
    U, _ = jnp.linalg.qr(jr.normal(jr.key(seed), (m, r)))
    V, _ = jnp.linalg.qr(jr.normal(jr.key(seed + 1), (n, r)))
    return einx.dot("m r, n r -> m n", einx.multiply("m r, r -> m r", U, sigma), V)


def _sym(lam, seed=0):
    """Symmetric ``Q diag(lam) Qᵀ``."""
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(seed), (lam.shape[0], lam.shape[0])))
    return einx.dot("i k, j k -> i j", einx.multiply("i k, k -> i k", Q, lam), Q)


def _function_op(A):
    """``A`` as a matrix-free operator (only ``mv`` and its transpose)."""
    return lx.FunctionLinearOperator(
        lambda x: A @ x, jax.ShapeDtypeStruct((A.shape[1],), A.dtype)
    )


def _projection_error(Q, A):
    return jnp.linalg.norm(A - Q @ einsum(Q, A, "m l, m n -> l n"), ord=2)


@pytest.mark.slow
def test_range_finder_within_hmt_expected_bound():
    # HMT (2011) Thm 10.6 bounds E‖A − QQᵀA‖₂ for a Gaussian Ω and q = 0.
    # The bound is on the expectation, so compare the mean over 16 draws.
    # Here the bound is ~5x the observed mean, and the spread across draws
    # (max/mean ~1.3) is far smaller than that slack.
    m, n, k, p = 80, 60, 10, 5
    sigma = 1.0 / jnp.arange(1, 51, dtype=jnp.float64)
    A = _known_spectrum(m, n, sigma)
    op = lx.MatrixLinearOperator(A)
    tail = sigma[k:]
    bound = (1 + math.sqrt(k / (p - 1))) * tail[0] + math.e * math.sqrt(
        k + p
    ) / p * jnp.sqrt(jnp.sum(tail**2))

    def err(key):
        Q = gx.range_finder(op, k, oversample=p, n_power_iter=0, key=key)
        return _projection_error(Q, A)

    errs = jax.vmap(err)(jr.split(jr.key(0), 16))
    assert float(jnp.mean(errs)) <= float(bound)
    # And no draw beats the optimal rank-(k+p) error σ_{k+p+1}.
    assert float(jnp.min(errs)) >= float(sigma[k + p]) * (1 - 1e-8)


@pytest.mark.slow
def test_range_finder_orthonormal_and_exact_on_low_rank():
    A = _known_spectrum(50, 40, jnp.array([5.0, 3.0, 1.0, 0.5]))
    Q = gx.range_finder(_function_op(A), 4, oversample=2, key=jr.key(1))
    assert Q.shape == (50, 6)
    assert jnp.allclose(einsum(Q, Q, "m a, m b -> a b"), jnp.eye(6), atol=1e-12)
    assert _projection_error(Q, A) < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["gaussian", "sparse_sign", "srht"])
def test_range_finder_with_sketch(kind):
    n, ell = 64, 12
    A = _known_spectrum(80, n, jnp.array([4.0, 2.0, 1.0, 0.5, 0.25]))
    key = jr.key(3)
    if kind == "gaussian":
        S = gx.GaussianSketch.sample(key, d=ell, m=n)
    elif kind == "sparse_sign":
        S = gx.SparseSignSketch.sample(key, d=ell, m=n, nnz=4)
    else:
        S = gx.SRHTSketch.sample(key, d=ell, m=n)
    Q = gx.range_finder(lx.MatrixLinearOperator(A), 5, sketch=S, n_power_iter=1)
    assert Q.shape == (80, ell)
    assert _projection_error(Q, A) < 1e-10


def test_range_finder_sketch_validation():
    op = lx.MatrixLinearOperator(jnp.ones((10, 8)))
    with pytest.raises(ValueError, match="in_size"):
        gx.range_finder(op, 2, sketch=gx.GaussianSketch.sample(None, d=4, m=10))
    with pytest.raises(ValueError, match="at least rank"):
        gx.range_finder(op, 5, sketch=gx.GaussianSketch.sample(None, d=4, m=8))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rank": 0}, "rank"),
        ({"rank": 2, "oversample": -1}, "oversample"),
        ({"rank": 2, "n_power_iter": -1}, "n_power_iter"),
    ],
)
def test_range_finder_argument_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gx.range_finder(lx.MatrixLinearOperator(jnp.eye(5)), **kwargs)


def test_range_finder_caps_ell_and_defaults_key():
    A = jr.normal(jr.key(0), (12, 7))
    op = lx.MatrixLinearOperator(A)
    Q = gx.range_finder(op, 5)  # l = 15 is capped at min(m, n) = 7
    assert Q.shape == (12, 7)
    assert jnp.array_equal(Q, gx.range_finder(op, 5, key=jr.PRNGKey(0)))


@pytest.mark.slow
def test_qb_matrix_free_matches_dense():
    A = _known_spectrum(40, 30, jnp.array([3.0, 2.0, 1.0]))
    Q, B = gx.qb(_function_op(A), 3, oversample=3, key=jr.key(2))
    assert Q.shape == (40, 6) and B.shape == (6, 30)
    assert jnp.allclose(B, einsum(Q, A, "m l, m n -> l n"), atol=1e-12)
    assert jnp.allclose(Q @ B, A, atol=1e-10)


@pytest.mark.slow
def test_randomized_svd_equals_dense_on_fast_decay():
    sigma = 2.0 ** -jnp.arange(40, dtype=jnp.float64)
    A = _known_spectrum(100, 70, sigma)
    k = 8
    U, s, Vt = gx.randomized_svd(lx.MatrixLinearOperator(A), k, key=jr.key(0))
    U0, s0, Vt0 = jnp.linalg.svd(A, full_matrices=False)
    assert U.shape == (100, k) and s.shape == (k,) and Vt.shape == (k, 70)
    assert jnp.allclose(s, s0[:k], rtol=1e-10)
    # Singular vectors agree up to sign.
    assert jnp.allclose(jnp.abs(einsum(U, U0[:, :k], "m a, m a -> a")), 1.0)
    assert jnp.allclose(jnp.abs(einsum(Vt, Vt0[:k], "a n, a n -> a")), 1.0)
    assert jnp.allclose(
        U @ einx.multiply("k, k n -> k n", s, Vt),
        U0[:, :k] @ einx.multiply("k, k n -> k n", s0[:k], Vt0[:k]),
        atol=1e-10,
    )


def test_randomized_svd_jit_and_float32():
    A = _known_spectrum(30, 20, jnp.array([3.0, 2.0, 1.0])).astype(jnp.float32)
    f = jax.jit(lambda M: gx.randomized_svd(lx.MatrixLinearOperator(M), 3))
    U, s, Vt = f(A)
    assert U.dtype == s.dtype == Vt.dtype == jnp.float32
    assert jnp.allclose(s, jnp.array([3.0, 2.0, 1.0]), atol=1e-4)


@pytest.mark.slow
def test_randomized_eigh_indefinite_which():
    lam = jnp.array([-10.0, 6.0, 3.0, -2.0, 1.0] + [1e-3] * 25)
    A = lx.MatrixLinearOperator(_sym(lam), lx.symmetric_tag)
    vals, vecs = gx.randomized_eigh(A, 3, oversample=5, which="magnitude")
    assert jnp.allclose(vals, jnp.array([-10.0, 3.0, 6.0]), atol=1e-8)
    assert jnp.allclose(A.as_matrix() @ vecs, vecs * vals, atol=1e-6)
    vals, _ = gx.randomized_eigh(A, 3, oversample=5, which="largest")
    assert jnp.allclose(vals, jnp.array([1.0, 3.0, 6.0]), atol=1e-6)


def test_randomized_eigh_zero_power_iter_is_rayleigh_ritz():
    # n_power_iter=0: Q = orth(AΩ), then the eigenpairs of QᵀAQ.
    lam = jnp.linspace(1.0, 0.01, 40)
    M = _sym(lam)
    op = lx.MatrixLinearOperator(M, lx.symmetric_tag)
    key = jr.key(4)
    vals, vecs = gx.randomized_eigh(op, 6, oversample=0, n_power_iter=0, key=key)
    omega = jr.normal(key, (40, 6))
    Q, _ = jnp.linalg.qr(M @ omega)
    ref_vals, W = jnp.linalg.eigh(einsum(Q, M, Q, "i a, i j, j b -> a b"))
    assert jnp.allclose(vals, ref_vals, atol=1e-12)
    assert jnp.allclose(jnp.abs(einsum(vecs, Q @ W, "n a, n a -> a")), 1.0)


def test_randomized_eigh_validation():
    with pytest.raises(ValueError, match="which"):
        gx.randomized_eigh(lx.MatrixLinearOperator(jnp.eye(4)), 2, which="smallest")
    with pytest.raises(ValueError, match="square"):
        gx.randomized_eigh(lx.MatrixLinearOperator(jnp.ones((4, 3))), 2)


@pytest.mark.slow
def test_svd_method_randomized_dispatch():
    A = _known_spectrum(50, 40, 2.0 ** -jnp.arange(20, dtype=jnp.float64))
    op = _function_op(A)
    U, s, Vt = gx.svd(op, rank=5, method="randomized", key=jr.key(1))
    ref = gx.randomized_svd(op, 5, key=jr.key(1))
    for a, b in zip((U, s, Vt), ref, strict=True):
        assert jnp.array_equal(a, b)
    assert jnp.allclose(s, jnp.linalg.svd(A, compute_uv=False)[:5], rtol=1e-10)


def test_eig_method_randomized_dispatch():
    lam = 2.0 ** -jnp.arange(30, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(_sym(lam), lx.symmetric_tag)
    vals, vecs = gx.eig(op, rank=4, method="randomized", key=jr.key(1))
    ref_vals, ref_vecs = gx.randomized_eigh(op, 4, key=jr.key(1))
    assert jnp.array_equal(vals, ref_vals) and jnp.array_equal(vecs, ref_vecs)
    assert jnp.allclose(vals, lam[:4][::-1], rtol=1e-10)


@pytest.mark.parametrize("fn", [gx.svd, gx.eig])
def test_primitive_method_validation(fn):
    op = lx.MatrixLinearOperator(jnp.eye(4), lx.symmetric_tag)
    with pytest.raises(ValueError, match="method"):
        fn(op, rank=2, method="qr")
    with pytest.raises(ValueError, match="needs a rank"):
        fn(op, method="randomized")


@pytest.mark.slow
def test_power_iterations_monotonically_improve_matern12():
    # Matérn-½ (exponential) Gram matrix: a slowly decaying spectrum where the
    # HMT tail term dominates; each power step should shrink the error.
    x = jnp.linspace(0.0, 10.0, 600)
    K = jnp.exp(-jnp.abs(einx.subtract("i, j -> i j", x, x)))
    op = lx.MatrixLinearOperator(K, lx.symmetric_tag)
    k = 20
    errs = []
    for q in range(4):
        U, s, Vt = gx.randomized_svd(op, k, n_power_iter=q, key=jr.key(0))
        errs.append(
            float(jnp.linalg.norm(K - U @ einx.multiply("k, k n -> k n", s, Vt), 2))
        )
    s_true = jnp.linalg.svd(K, compute_uv=False)
    assert all(a > b for a, b in itertools.pairwise(errs))
    # With q = 3 the truncated SVD is within 10% of the optimal error σ_{k+1}.
    assert errs[-1] <= 1.1 * float(s_true[k])
