"""Tests for `EigenFactorization` and `kronecker_sum_solve`.

All matrices are built from pinned keys (the randomness is incidental — any
diagonalizable matrix would do), so tolerances are round-off bounds for
well-conditioned systems rather than sampling bounds.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx


def _nonsymmetric(key, n: int, lo: float = -5.0, hi: float = -1.0):
    """A = V diag(λ) V⁻¹ with real λ in [lo, hi] and well-conditioned V.

    V = I + 0.3·G keeps cond(V) small, so solves are accurate to ~1e-12.
    """
    k1, k2 = jr.split(key)
    V = jnp.eye(n) + 0.3 * jr.normal(k1, (n, n)) / jnp.sqrt(n)
    lam = jr.uniform(k2, (n,), minval=lo, maxval=hi)
    return (V * lam) @ jnp.linalg.inv(V)


@pytest.fixture
def mat_a():
    return _nonsymmetric(jr.key(0), 6)


@pytest.fixture
def mat_b():
    return _nonsymmetric(jr.key(1), 5)


def test_from_matrix_reconstructs(mat_a):
    fac = gaussx.EigenFactorization.from_matrix(mat_a)
    assert jnp.allclose(fac.as_matrix(), mat_a, atol=1e-12)


def test_from_matrix_symmetric_uses_orthonormal_basis():
    G = jr.normal(jr.key(2), (5, 5))
    S = G @ G.T
    fac = gaussx.EigenFactorization.from_matrix(
        lx.MatrixLinearOperator(S, lx.symmetric_tag)
    )
    assert jnp.allclose(fac.eigenvectors_inv, fac.eigenvectors.T)
    assert jnp.allclose(fac.as_matrix(), S, atol=1e-10)


def test_from_matrix_rejects_complex_spectrum():
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])  # eigenvalues ±i
    with pytest.raises(ValueError, match="real spectra"):
        gaussx.EigenFactorization.from_matrix(rotation)


def test_from_matrix_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        gaussx.EigenFactorization.from_matrix(jnp.ones((2, 3)))


def test_from_matrix_rejects_tracer(mat_a):
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(gaussx.EigenFactorization.from_matrix)(mat_a)


@pytest.mark.parametrize("shift", [0.0, 0.7, 3.0])
def test_solve_shifted_matches_dense(mat_a, shift):
    fac = gaussx.EigenFactorization.from_matrix(mat_a)
    b = jr.normal(jr.key(3), (6,))
    x = fac.solve_shifted(b, shift)
    expected = jnp.linalg.solve(mat_a - shift * jnp.eye(6), b)
    assert jnp.allclose(x, expected, atol=1e-11)


def test_solve_shifted_batched_rhs(mat_a):
    fac = gaussx.EigenFactorization.from_matrix(mat_a)
    B = jr.normal(jr.key(4), (6, 3))
    assert jnp.allclose(
        fac.solve_shifted(B, 1.0), jnp.linalg.solve(mat_a - jnp.eye(6), B)
    )


def test_solve_shifted_traced_shift_jit_and_grad(mat_a):
    """The shift may be traced; d/dσ (A − σI)⁻¹b = (A − σI)⁻² b."""
    fac = gaussx.EigenFactorization.from_matrix(mat_a)
    b = jr.normal(jr.key(5), (6,))
    solve = eqx.filter_jit(lambda s: fac.solve_shifted(b, s))
    assert jnp.allclose(solve(0.5), fac.solve_shifted(b, 0.5))
    M = mat_a - 0.5 * jnp.eye(6)
    expected = jnp.linalg.solve(M, jnp.linalg.solve(M, b))
    got = jax.jacfwd(lambda s: fac.solve_shifted(b, s))(0.5)
    assert jnp.allclose(got, expected, atol=1e-10)


def test_solve_shifted_drop_projects_out_null_mode():
    """Singular A with null vector 1: dropping that mode gives a solution
    of A x = b whenever b is compatible (no component along the null mode).
    """
    fac0 = gaussx.EigenFactorization.from_matrix(_nonsymmetric(jr.key(6), 4))
    lam = fac0.eigenvalues.at[0].set(0.0)
    fac = gaussx.EigenFactorization(lam, fac0.eigenvectors, fac0.eigenvectors_inv)
    A = fac.as_matrix()
    b = A @ jr.normal(jr.key(7), (4,))  # in range(A) ⇒ compatible
    x = fac.solve_shifted(b, 0.0, drop=jnp.arange(4) == 0)
    assert jnp.all(jnp.isfinite(x))
    assert jnp.allclose(A @ x, b, atol=1e-11)


def test_kronecker_sum_solve_2d_is_sylvester(mat_a, mat_b):
    """A X + X Bᵀ − σX = R with non-symmetric A, B."""
    fa = gaussx.EigenFactorization.from_matrix(mat_a)
    fb = gaussx.EigenFactorization.from_matrix(mat_b)
    R = jr.normal(jr.key(8), (6, 5))
    X = gaussx.kronecker_sum_solve((fa, fb), R, 0.4)
    residual = mat_a @ X + X @ mat_b.T - 0.4 * X - R
    assert jnp.max(jnp.abs(residual)) < 1e-11


def test_kronecker_sum_solve_matches_dense_kron(mat_a, mat_b):
    """Row-major vec: (A ⊗ I + I ⊗ B − σI) vec(X) = vec(R)."""
    fa = gaussx.EigenFactorization.from_matrix(mat_a)
    fb = gaussx.EigenFactorization.from_matrix(mat_b)
    R = jr.normal(jr.key(9), (6, 5))
    X = gaussx.kronecker_sum_solve((fa, fb), R, 1.5)
    K = jnp.kron(mat_a, jnp.eye(5)) + jnp.kron(jnp.eye(6), mat_b) - 1.5 * jnp.eye(30)
    assert jnp.allclose(X.ravel(), jnp.linalg.solve(K, R.ravel()), atol=1e-11)


def test_kronecker_sum_solve_3d():
    mats = [_nonsymmetric(jr.key(10 + i), n) for i, n in enumerate((3, 4, 5))]
    facs = tuple(gaussx.EigenFactorization.from_matrix(m) for m in mats)
    R = jr.normal(jr.key(13), (3, 4, 5))
    X = gaussx.kronecker_sum_solve(facs, R, 0.2)
    AX = (
        jnp.einsum("ij,jbc->ibc", mats[0], X)
        + jnp.einsum("ij,ajc->aic", mats[1], X)
        + jnp.einsum("ij,abj->abi", mats[2], X)
    )
    assert jnp.max(jnp.abs(AX - 0.2 * X - R)) < 1e-11


def test_kronecker_sum_solve_trailing_batch(mat_a, mat_b):
    fa = gaussx.EigenFactorization.from_matrix(mat_a)
    fb = gaussx.EigenFactorization.from_matrix(mat_b)
    R = jr.normal(jr.key(14), (6, 5, 2))
    X = gaussx.kronecker_sum_solve((fa, fb), R)
    for i in range(2):
        assert jnp.allclose(X[..., i], gaussx.kronecker_sum_solve((fa, fb), R[..., i]))


def test_kronecker_sum_solve_shape_mismatch_raises(mat_a, mat_b):
    fa = gaussx.EigenFactorization.from_matrix(mat_a)
    fb = gaussx.EigenFactorization.from_matrix(mat_b)
    with pytest.raises(ValueError, match="does not match"):
        gaussx.kronecker_sum_solve((fa, fb), jnp.zeros((5, 6)))
