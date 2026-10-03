"""Tests for gaussx.eigh_generalized (A v = λ B v, dispatch on B)."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

from gaussx import eigh_generalized
from gaussx._einx import einsum


def _spd(key, n, shift=1.0):
    M = jr.normal(key, (n, n))
    return einsum(M, M, "i k, j k -> i j") / n + shift * jnp.eye(n)


def _gram(V, B):
    """``Vᵀ B V``."""
    return einsum(V, B, V, "i a, i j, j b -> a b")


def _residual(A, B, lam, V):
    """``max |A v − λ B v|`` over all returned pairs."""
    AV = A @ V
    BV = einx.multiply("i k, k -> i k", B @ V, lam)
    return float(jnp.max(jnp.abs(AV - BV)))


def _align_signs(V, V_ref):
    """Flip columns of ``V`` to match the signs of ``V_ref``."""
    signs = jnp.sign(einsum(V, V_ref, "i k, i k -> k"))
    return einx.multiply("i k, k -> i k", V, signs)


# -- positive definite B -----------------------------------------------------


@pytest.mark.parametrize("tagged", [True, False])
def test_pd_b_matches_scipy_eigh(tagged):
    n = 7
    A = _spd(jr.key(0), n, shift=-0.5)  # symmetric, indefinite
    B = _spd(jr.key(1), n)
    B_op = (
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
        if tagged
        else lx.MatrixLinearOperator(B)
    )
    lam, V = eigh_generalized(lx.MatrixLinearOperator(A), B_op)
    lam_ref, V_ref = scipy.linalg.eigh(np.asarray(A), np.asarray(B))
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_align_signs(V, V_ref), V_ref, atol=1e-8)
    np.testing.assert_allclose(_gram(V, B), jnp.eye(n), atol=1e-10)


@pytest.mark.parametrize("which", ["smallest", "largest"])
def test_pd_b_rank_and_which(which):
    n, k = 7, 3
    A = _spd(jr.key(2), n, shift=-0.5)
    B = _spd(jr.key(3), n)
    lam, V = eigh_generalized(
        lx.MatrixLinearOperator(A),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
        rank=k,
        which=which,
    )
    lam_ref = scipy.linalg.eigh(np.asarray(A), np.asarray(B), eigvals_only=True)
    expected = lam_ref[:k] if which == "smallest" else lam_ref[n - k :]
    assert V.shape == (n, k)
    np.testing.assert_allclose(lam, expected, rtol=1e-10)
    assert _residual(A, B, lam, V) < 1e-10


# -- diagonal B --------------------------------------------------------------


def _kernellib_smallest_generalized(A, degree, n_components, drop_first):
    """Port of kernellib's ``_decomposition._eigenmaps._smallest_generalized``
    (the dense degree-constraint path that K5 moves onto G2), einx-ified."""
    scale = 1.0 / jnp.sqrt(jnp.where(degree > 0, degree, 1.0))
    S = einx.multiply("i, i j -> i j", scale, einx.multiply("i j, j -> i j", A, scale))
    lam, U = jnp.linalg.eigh(S)
    start = 1 if drop_first else 0
    stop = start + n_components
    return lam[start:stop], einx.multiply("i, i k -> i k", scale, U[:, start:stop])


def _random_graph_laplacian(key, n):
    W = jr.uniform(key, (n, n))
    W = (W + einx.id("i j -> j i", W)) * (1 - jnp.eye(n))
    degree = W @ jnp.ones(n)
    return jnp.diag(degree) - W, degree


@pytest.mark.parametrize("use_rank", [False, True])
@pytest.mark.parametrize("drop_first", [False, True])
def test_diagonal_b_matches_kernellib_smallest_generalized(drop_first, use_rank):
    n, n_components = 12, 3
    L, degree = _random_graph_laplacian(jr.key(4), n)
    lam_ref, Y_ref = _kernellib_smallest_generalized(
        L, degree, n_components, drop_first
    )
    lam, Y = eigh_generalized(
        lx.MatrixLinearOperator(L, lx.symmetric_tag),
        lx.DiagonalLinearOperator(degree),
        rank=n_components + int(drop_first) if use_rank else None,
    )
    start = int(drop_first)
    lam, Y = lam[start : start + n_components], Y[:, start : start + n_components]
    np.testing.assert_allclose(lam, lam_ref, atol=1e-10)
    np.testing.assert_allclose(_align_signs(Y, Y_ref), Y_ref, atol=1e-8)


@pytest.mark.parametrize("which", ["smallest", "largest"])
def test_diagonal_b_lanczos_rank_matches_dense(which):
    # S = B^{-1/2} A B^{-1/2} has a designed spectrum with well-separated
    # extremes, so a 23-step Krylov space on N = 80 converges them tightly.
    n, k = 80, 3
    d = jr.uniform(jr.key(5), (n,), minval=0.5, maxval=2.0)
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(6), (n, n)))
    spectrum = jnp.concatenate(
        [
            jnp.array([0.1, 0.2, 0.3]),
            jnp.linspace(1.0, 2.0, n - 6),
            jnp.array([5.0, 6.0, 7.0]),
        ]
    )
    S = einsum(einx.multiply("i k, k -> i k", Q, spectrum), Q, "i k, j k -> i j")
    sq = jnp.sqrt(d)
    A = einx.multiply("i, i j -> i j", sq, einx.multiply("i j, j -> i j", S, sq))
    A_op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    B_op = lx.DiagonalLinearOperator(d)

    lam_dense, V_dense = eigh_generalized(A_op, B_op, which=which)
    lam_dense, V_dense = (
        (lam_dense[:k], V_dense[:, :k])
        if which == "smallest"
        else (lam_dense[n - k :], V_dense[:, n - k :])
    )
    lam, V = eigh_generalized(A_op, B_op, rank=k, which=which, key=jr.key(7))
    np.testing.assert_allclose(lam, lam_dense, rtol=1e-8)
    np.testing.assert_allclose(_align_signs(V, V_dense), V_dense, atol=1e-6)
    np.testing.assert_allclose(_gram(V, jnp.diag(d)), jnp.eye(k), atol=1e-8)


def test_diagonal_b_lanczos_jits_and_keeps_float32():
    n = 10
    L, degree = _random_graph_laplacian(jr.key(8), n)
    L, degree = L.astype(jnp.float32), degree.astype(jnp.float32)

    @jax.jit
    def f(L, degree):
        return eigh_generalized(
            lx.MatrixLinearOperator(L, lx.symmetric_tag),
            lx.DiagonalLinearOperator(degree),
            rank=2,
        )

    lam, Y = f(L, degree)
    assert lam.dtype == jnp.float32
    assert Y.dtype == jnp.float32
    np.testing.assert_allclose(lam[0], 0.0, atol=1e-4)  # constant solution


@pytest.mark.slow
def test_diagonal_b_with_zero_entry_uses_singular_path():
    A = _spd(jr.key(9), 5)
    d = jnp.array([1.0, 2.0, 0.0, 3.0, 0.5])
    lam, V = eigh_generalized(lx.MatrixLinearOperator(A), lx.DiagonalLinearOperator(d))
    lam_ref, _ = eigh_generalized(
        lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(jnp.diag(d))
    )
    assert lam.shape == (4,)
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-10)
    assert _residual(A, jnp.diag(d), lam, V) < 1e-10


# -- singular B --------------------------------------------------------------


def _singular_pencil(n=8, r=5):
    """SPD ``A`` (so ``A₀₀`` is PD) and a rank-``r`` PSD ``B`` in a random basis;
    a dense random ``A`` couples ``range(B)`` and ``ker B``."""
    A = _spd(jr.key(10), n)
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(11), (n, n)))
    s = jnp.concatenate(
        [jr.uniform(jr.key(12), (r,), minval=0.5, maxval=2.0), jnp.zeros(n - r)]
    )
    B = einsum(einx.multiply("i k, k -> i k", Q, s), Q, "i k, j k -> i j")
    return A, B, Q, r


def _finite_qz_eigenvalues(A, B):
    """Finite eigenvalues of the pencil from scipy's QZ (``eig(a, b)``)."""
    w = scipy.linalg.eig(np.asarray(A), np.asarray(B), homogeneous_eigvals=True)[0]
    alpha, beta = w
    finite = np.abs(beta) > 1e-8 * np.abs(alpha)
    return np.sort((alpha[finite] / beta[finite]).real)


@pytest.mark.slow
def test_singular_b_with_coupling_matches_qz():
    A, B, Q, r = _singular_pencil()
    # The coupling block A₀₊ is non-zero, so dropping ker B would be wrong.
    U0, Up = Q[:, r:], Q[:, :r]
    assert float(jnp.max(jnp.abs(einsum(U0, A, Up, "i a, i j, j b -> a b")))) > 1e-2

    lam, V = eigh_generalized(lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(B))
    assert lam.shape == (r,)
    assert _residual(A, B, lam, V) < 1e-10
    np.testing.assert_allclose(_gram(V, B), jnp.eye(r), atol=1e-10)
    np.testing.assert_allclose(lam, _finite_qz_eigenvalues(A, B), rtol=1e-8)

    # Naively dropping ker B gives different (wrong) eigenvalues.
    App = einsum(Up, A, Up, "i a, i j, j b -> a b")
    Bpp = einsum(Up, B, Up, "i a, i j, j b -> a b")
    naive = scipy.linalg.eigh(np.asarray(App), np.asarray(Bpp), eigvals_only=True)
    assert not np.allclose(naive, lam, rtol=1e-3)


def test_singular_b_largest_and_rank():
    A, B, _, r = _singular_pencil()
    lam, V = eigh_generalized(
        lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(B), rank=2, which="largest"
    )
    np.testing.assert_allclose(lam, _finite_qz_eigenvalues(A, B)[r - 2 :], rtol=1e-8)
    assert _residual(A, B, lam, V) < 1e-10


def test_rank_warning_fires_and_k_shrinks():
    A, B, _, r = _singular_pencil()
    with pytest.warns(UserWarning, match="numerical rank 5"):
        lam, V = eigh_generalized(
            lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(B), rank=r + 2
        )
    assert lam.shape == (r,)
    assert V.shape == (A.shape[0], r)


def test_indefinite_a00_raises():
    # ker B = span(e₃) and A₃₃ = −1: tr(YᵀAY) is unbounded below along e₃.
    A = jnp.array([[2.0, 0.0, 0.3], [0.0, 1.0, 0.0], [0.3, 0.0, -1.0]])
    B = jnp.diag(jnp.array([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="not positive definite"):
        eigh_generalized(lx.MatrixLinearOperator(A), lx.MatrixLinearOperator(B))


def test_invalid_arguments_raise():
    A = lx.MatrixLinearOperator(jnp.eye(3))
    B = lx.DiagonalLinearOperator(jnp.ones(3))
    with pytest.raises(ValueError, match="which"):
        eigh_generalized(A, B, which="middle")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="rank"):
        eigh_generalized(A, B, rank=0)
    with pytest.raises(ValueError, match="numerically zero"):
        eigh_generalized(A, lx.MatrixLinearOperator(jnp.zeros((3, 3))))
