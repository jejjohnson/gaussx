"""Tests for the Falkon Nyström KRR recipe."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import random_pd_matrix


def _rbf(x, z):
    return jnp.exp(-0.5 * jnp.sum((x - z) ** 2))


def _gram(a, b):
    return jax.vmap(lambda x: jax.vmap(lambda z: _rbf(x, z))(b))(a)


def _krr_problem(n: int = 100, m: int = 20, seed: int = 0):
    """RBF Nyström KRR data: X, Z = first m rows of X, K_nm, K_mm."""
    X = jr.normal(jr.key(seed), (n, 2))
    Z = X[:m]
    return X, Z, _gram(X, Z), _gram(Z, Z)


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------


def test_factors_are_upper_triangular_choleskys() -> None:
    m, lam = 12, 0.05
    K_mm = random_pd_matrix(jr.key(1), m)

    pre = gaussx.falkon_preconditioner(K_mm, lam, jitter=0.0)

    assert jnp.allclose(pre.T, jnp.triu(pre.T))
    assert jnp.allclose(pre.A, jnp.triu(pre.A))
    assert jnp.allclose(pre.T.T @ pre.T, K_mm, atol=1e-10)
    expected = pre.T @ pre.T.T / m + lam * jnp.eye(m)
    assert jnp.allclose(pre.A.T @ pre.A, expected, atol=1e-10)


def test_preconditioner_inverts_the_nystrom_approximation() -> None:
    # P Pᵀ = n ((n/m) K² + λ n K)⁻¹: the defining property, and the reason
    # both factors must be upper triangular.
    n, m, lam = 500, 12, 0.05
    K_mm = random_pd_matrix(jr.key(2), m)

    pre = gaussx.falkon_preconditioner(K_mm, lam, jitter=0.0)
    P = pre.precondition(jnp.eye(m))

    expected = n * jnp.linalg.inv((n / m) * K_mm @ K_mm + lam * n * K_mm)
    assert jnp.allclose(P @ P.T, expected, rtol=1e-8, atol=1e-10)


def test_k_mm_cancels_in_the_preconditioned_system() -> None:
    # Pᵀ (K_nmᵀ K_nm + λ n K_mm) P = A⁻ᵀ [T⁻ᵀ K_nmᵀ K_nm T⁻¹ + λ n I] A⁻¹.
    n, lam = 100, 1e-3
    _, _, K_nm, K_mm = _krr_problem(n)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    m = K_mm.shape[0]
    K_mm_jittered = pre.T.T @ pre.T

    P = pre.precondition(jnp.eye(m))
    system = K_nm.T @ K_nm + lam * n * K_mm_jittered
    T_inv = jnp.linalg.inv(pre.T)
    A_inv = jnp.linalg.inv(pre.A)
    cancelled = (
        A_inv.T @ (T_inv.T @ K_nm.T @ K_nm @ T_inv + lam * n * jnp.eye(m)) @ A_inv
    )

    scale = jnp.max(jnp.abs(cancelled))
    assert jnp.max(jnp.abs(P.T @ system @ P - cancelled)) < 1e-8 * scale


def test_preconditioning_collapses_the_condition_number() -> None:
    # Measured: cond 6e8 -> 40 on this RBF problem.
    n, lam = 100, 1e-3
    _, _, K_nm, K_mm = _krr_problem(n)
    pre = gaussx.falkon_preconditioner(K_mm, lam)
    P = pre.precondition(jnp.eye(K_mm.shape[0]))
    system = K_nm.T @ K_nm + lam * n * K_mm

    assert jnp.linalg.cond(P.T @ system @ P) < 1e-4 * jnp.linalg.cond(system)
    assert jnp.linalg.cond(P.T @ system @ P) < 1e3


def test_transpose_application_is_the_transpose() -> None:
    m = 10
    pre = gaussx.falkon_preconditioner(random_pd_matrix(jr.key(3), m), 0.1)
    P = pre.precondition(jnp.eye(m))

    block = jr.normal(jr.key(4), (m, 3))
    assert jnp.allclose(pre.precondition_transpose(block), P.T @ block, atol=1e-10)
    assert jnp.allclose(pre.precondition(block[:, 0]), P @ block[:, 0], atol=1e-10)


def test_default_jitter_handles_duplicated_inducing_points() -> None:
    # Two identical inducing points make K_mm exactly singular.
    Z = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.5], [-0.3, 2.0]])
    pre = gaussx.falkon_preconditioner(_gram(Z, Z), 1e-3)

    assert jnp.all(jnp.isfinite(pre.T))
    assert jnp.all(jnp.isfinite(pre.A))


def test_default_jitter_is_positive_for_a_zero_kernel() -> None:
    # A linear kernel at all-zero inducing points: max|diag| = 0, so a jitter
    # scaled by it alone would leave the Cholesky of the zero matrix.
    pre = gaussx.falkon_preconditioner(jnp.zeros((4, 4)), 1e-3)

    assert jnp.all(jnp.isfinite(pre.T))
    assert jnp.all(jnp.isfinite(pre.A))


def test_accepts_an_operator() -> None:
    K_mm = random_pd_matrix(jr.key(5), 6)
    from_array = gaussx.falkon_preconditioner(K_mm, 0.1)
    from_operator = gaussx.falkon_preconditioner(
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag), 0.1
    )

    assert jnp.allclose(from_array.T, from_operator.T)
    assert jnp.allclose(from_array.A, from_operator.A)


def test_preconditioner_is_jittable() -> None:
    K_mm = random_pd_matrix(jr.key(6), 6)
    eager = gaussx.falkon_preconditioner(K_mm, 0.1)
    jitted = jax.jit(gaussx.falkon_preconditioner)(K_mm, 0.1)

    assert jnp.allclose(eager.A, jitted.A)


def test_factors_share_one_promoted_dtype() -> None:
    # A float64 regularization with a float32 K_mm used to give a float32 T
    # and a float64 A.
    K_mm = random_pd_matrix(jr.key(19), 6).astype(jnp.float32)

    pre = gaussx.falkon_preconditioner(K_mm, jnp.asarray(1e-3, dtype=jnp.float64))

    assert pre.T.dtype == pre.A.dtype == jnp.float64


def test_rejects_a_non_square_k_mm() -> None:
    with pytest.raises(ValueError, match="square"):
        gaussx.falkon_preconditioner(jnp.ones((3, 4)), 0.1)
