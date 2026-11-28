"""Tests for the Nyström-plus-diagonal FITC preconditioner (gh-824).

P = U Uᵀ + diag(r + λ), U = K_XZ L_ZZ⁻ᵀ, r = max(diag(K_XX − U Uᵀ), 0). Every
check compares against P built densely from the same kernel matrices.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import rearrange
from gaussx._testing import assert_sample_moments, dense_inv, dense_logdet


_N, _R = 40, 8
_PSD = lx.positive_semidefinite_tag


def _matern32(lengthscale, variance=1.0):
    """Matérn-3/2 Gram function k(X₁, X₂)."""

    def kernel(left, right):
        gaps = einx.subtract("n d, m d -> n m d", left, right)
        # r², with a floor so that √ has a finite gradient at r = 0
        squared = einx.sum("n m [d]", gaps**2) + 1e-12
        # √3 r / ℓ
        scaled = jnp.sqrt(3.0) * jnp.sqrt(squared) / lengthscale
        return variance * (1.0 + scaled) * jnp.exp(-scaled)

    return kernel


def _inputs(n=_N, r=_R):
    """Inputs X on [0, 1] and every (n / r)-th point as Z."""
    X = rearrange(jnp.linspace(0.0, 1.0, n), "n -> n 1")
    return X, X[:: n // r]


def _system(kernel, X, diagonal):
    """A = K_XX + diag(λ), split-able by `gaussx.split_diagonal`."""
    return lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel(X, X)) + lx.DiagonalLinearOperator(diagonal),
        _PSD,
    )


def _dense_p(kernel, X, Z, diagonal, jitter=1e-6):
    """P = Q + diag(max(diag(K − Q), 0) + λ) from dense matrices."""
    inducing = kernel(Z, Z) + jitter * jnp.eye(Z.shape[0], dtype=X.dtype)
    cross = kernel(X, Z)
    # Q = K_XZ (K_ZZ + β I)⁻¹ K_ZX
    nystrom = cross @ jnp.linalg.solve(inducing, cross.T)
    residual = jnp.maximum(jnp.diag(kernel(X, X)) - jnp.diag(nystrom), 0.0)
    return nystrom + jnp.diag(residual + diagonal)


def _rtol(dtype):
    # Woodbury against a dense inverse of a matrix with κ ~ 1e2: ~1e-12 in
    # float64; float32 round-off on the k × k capacitance is ~1e-5.
    return 1e-5 if dtype == jnp.float64 else 1e-3


def test_inverse_matches_the_dense_inverse(getkey):
    kernel = _matern32(0.3)
    X, Z = _inputs()
    diagonal = jnp.linspace(0.1, 0.5, _N, dtype=X.dtype)
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
    system = _system(kernel, X, diagonal)
    dense = _dense_p(kernel, X, Z, diagonal)
    vector = jr.normal(getkey(), (_N,), dtype=X.dtype)
    got = pre.as_operator(system).mv(vector)
    expected = dense_inv(lx.MatrixLinearOperator(dense)) @ vector
    assert jnp.allclose(got, expected, rtol=_rtol(X.dtype), atol=_rtol(X.dtype))
    assert lx.is_positive_semidefinite(pre.as_operator(system))


def test_logdet_matches_both_dense_forms():
    kernel = _matern32(0.3)
    X, Z = _inputs()
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z, diagonal_shift=0.2)
    dense = _dense_p(kernel, X, Z, jnp.full(_N, 0.2, dtype=X.dtype))
    tolerance = _rtol(X.dtype)
    expected = dense_logdet(lx.MatrixLinearOperator(dense))
    assert jnp.allclose(pre.logdet(), expected, rtol=tolerance)
    # log det P = −log det M⁻¹
    inverse_logdet = dense_logdet(
        lx.MatrixLinearOperator(pre.as_operator().as_matrix())
    )
    assert jnp.allclose(pre.logdet(), -inverse_logdet, rtol=tolerance)
    assert jnp.allclose(pre.approximation().as_matrix(), dense, rtol=tolerance)


@pytest.mark.x64_only(reason="rtol=1e-6 round trip needs float64")
def test_full_rank_at_the_inputs_reproduces_the_system():
    """r = n, Z = X: Q = K_XX up to the jitter, r ≈ 0, so P = K + Λ."""
    kernel = _matern32(0.3)
    X, _ = _inputs(n=12)
    diagonal = jnp.linspace(0.1, 0.5, 12)
    pre = gaussx.FITCPreconditioner.from_kernel(
        kernel, X, X, diagonal_shift=diagonal, jitter=1e-12
    )
    expected = kernel(X, X) + jnp.diag(diagonal)
    assert jnp.allclose(pre.approximation().as_matrix(), expected, rtol=1e-6)


def test_shift_none_reads_each_system():
    """One object, two diagonals, two preconditioners, also under jit."""
    kernel = _matern32(0.3)
    X, Z = _inputs()
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
    vector = jnp.ones(_N, dtype=X.dtype)

    def apply(diagonal):
        return pre.as_operator(_system(kernel, X, diagonal)).mv(vector)

    small = jnp.full(_N, 0.1, dtype=X.dtype)
    large = jnp.full(_N, 2.0, dtype=X.dtype)
    eager_small, eager_large = apply(small), apply(large)
    assert not jnp.allclose(eager_small, eager_large)
    tolerance = _rtol(X.dtype)
    for diagonal, eager in ((small, eager_small), (large, eager_large)):
        expected = jnp.linalg.solve(_dense_p(kernel, X, Z, diagonal), vector)
        assert jnp.allclose(eager, expected, rtol=tolerance, atol=tolerance)
        # the diagonal traced through jit
        assert jnp.allclose(
            jax.jit(apply)(diagonal), eager, rtol=tolerance, atol=tolerance
        )


def test_shift_none_without_a_splittable_system_raises():
    kernel = _matern32(0.3)
    X, Z = _inputs()
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
    with pytest.raises(ValueError, match="needs the system operator"):
        pre.as_operator()
    unsplittable = lx.MatrixLinearOperator(kernel(X, X), _PSD)
    with pytest.raises(ValueError, match="split_diagonal"):
        pre.logdet(unsplittable)


def test_residual_is_clamped_at_zero():
    """A Q larger than K on the diagonal gives r = 0, not a negative D."""
    U = jnp.ones((4, 2))
    pre = gaussx.FITCPreconditioner.from_factors(U, jnp.ones(4), diagonal_shift=0.1)
    assert jnp.array_equal(pre.prior_residual, jnp.zeros(4))
    draws = pre.sample(jr.key(0), shape=(3,))
    assert draws.shape == (3, 4)
    assert jnp.all(jnp.isfinite(draws))


@pytest.mark.slow
def test_supports_slq_and_works_inside_bbmm(getkey):
    """BBMM's shared pass takes P's probes, log det P and P⁻¹."""
    kernel = _matern32(0.3)
    X, Z = _inputs()
    diagonal = jnp.full(_N, 0.2, dtype=X.dtype)
    system = _system(kernel, X, diagonal)
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
    assert pre.supports_slq()
    rhs = jr.normal(getkey(), (_N, 1), dtype=X.dtype)
    quadratic, _ = gaussx.inv_quad_logdet(
        system, rhs, strategy=gaussx.BBMMSolver(preconditioner=pre), key=getkey()
    )
    dense = kernel(X, X) + jnp.diag(diagonal)
    expected = rhs[:, 0] @ jnp.linalg.solve(dense, rhs[:, 0])
    assert jnp.allclose(quadratic, expected, rtol=1e-3)


@pytest.mark.slow
def test_grad_jit_vmap_over_the_lengthscale(getkey):
    """vᵀ P⁻¹ v + log det P, differentiated in ℓ, matches the dense P."""
    X, Z = _inputs()
    diagonal = jnp.full(_N, 0.2, dtype=X.dtype)
    vector = jr.normal(getkey(), (_N,), dtype=X.dtype)

    def loss(lengthscale):
        kernel = _matern32(lengthscale)
        pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
        system = _system(kernel, X, diagonal)
        return vector @ pre.as_operator(system).mv(vector) + pre.logdet(system)

    def dense_loss(lengthscale):
        dense = _dense_p(_matern32(lengthscale), X, Z, diagonal)
        return vector @ jnp.linalg.solve(dense, vector) + jnp.linalg.slogdet(dense)[1]

    lengthscale = jnp.asarray(0.3, dtype=X.dtype)
    tolerance = _rtol(X.dtype)
    got = jax.jit(jax.grad(loss))(lengthscale)
    expected = jax.grad(dense_loss)(lengthscale)
    assert jnp.isfinite(got)
    assert jnp.allclose(got, expected, rtol=tolerance)
    lengthscales = jnp.asarray([0.2, 0.3, 0.5], dtype=X.dtype)
    batched = jax.vmap(loss)(lengthscales)
    looped = jnp.stack([dense_loss(value) for value in lengthscales])
    assert jnp.allclose(batched, looped, rtol=tolerance)


@pytest.mark.slow
def test_sample_moments_match_p():
    kernel = _matern32(0.3)
    X, Z = _inputs(n=12, r=4)
    diagonal = jnp.linspace(0.1, 0.5, 12, dtype=X.dtype)
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z, diagonal_shift=diagonal)
    draws = pre.sample(jr.key(0), shape=(20000,))
    # 7σ of each moment estimator's own sampling distribution (gh-220)
    assert_sample_moments(
        draws, jnp.zeros(12, dtype=X.dtype), _dense_p(kernel, X, Z, diagonal)
    )


@pytest.mark.slow
def test_fitc_beats_unpreconditioned_cg():
    """Matérn-3/2, n = 512, small noise: the iteration count is the signal."""
    n, rank = 512, 64
    X = jr.uniform(jr.key(0), (n, 1), dtype=jnp.zeros(()).dtype)
    Z = X[jr.choice(jr.key(1), n, (rank,), replace=False)]
    kernel = _matern32(0.2)
    diagonal = jnp.full(n, 1e-2, dtype=X.dtype)
    system = _system(kernel, X, diagonal)
    pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, Z)
    vector = jr.normal(jr.key(2), (n,), dtype=X.dtype)
    dense = lx.MatrixLinearOperator(kernel(X, X) + jnp.diag(diagonal), _PSD)

    def steps(options):
        solution = lx.linear_solve(
            dense,
            vector,
            lx.CG(rtol=1e-6, atol=1e-6, max_steps=20 * n),
            options=options,
            throw=False,
        )
        return int(solution.stats["num_steps"])

    plain = steps({})
    preconditioned = steps({"preconditioner": pre.as_operator(system)})
    assert preconditioned * 2 <= plain, (preconditioned, plain)
