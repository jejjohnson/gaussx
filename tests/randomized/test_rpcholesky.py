"""Tests for rp_cholesky and its greedy wrapper guarded_pivoted_cholesky."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import rp_cholesky
from gaussx._einx import einsum
from gaussx._primitives._root import guarded_pivoted_cholesky
from gaussx._testing import random_pd_matrix, tree_allclose


def _reference_greedy(diagonal, column, rank):
    """``guarded_pivoted_cholesky`` as it was before G14, verbatim."""
    tol = diagonal.shape[0] * jnp.finfo(diagonal.dtype).eps * jnp.max(jnp.abs(diagonal))

    def body(i, L):
        residual = diagonal - jnp.sum(L * L, axis=1)
        k = jnp.argmax(residual)
        pivot = residual[k]
        ok = pivot > tol
        denom = jnp.sqrt(jnp.where(ok, pivot, 1.0))
        col = (column(k) - L @ L[k, :]) / denom
        return L.at[:, i].set(jnp.where(ok, col, 0.0))

    L0 = jnp.zeros((diagonal.shape[0], rank), dtype=diagonal.dtype)
    return jax.lax.fori_loop(0, rank, body, L0)


def _low_rank(seed, n, r, jitter=0.0):
    w = jr.normal(jr.key(seed), (n, r))
    return einsum(w, w, "i r, j r -> i j") + jitter * jnp.eye(n)


def _gram(F):
    return einsum(F, F, "i k, j k -> i j")


# gh-236: rank-3 draws at keys 0 and 27 (27 produced inf surplus columns);
# gh-237: a rank-3 kernel factored at rank 8, with and without a tiny jitter;
# plus a full-rank PD matrix factored below and at full rank.
_CASES = {
    "gh236-seed0": (_low_rank(0, 10, 3), 6),
    "gh236-seed27": (_low_rank(27, 10, 3), 6),
    "gh237-nojitter": (_low_rank(1, 12, 3), 8),
    "gh237-jitter": (_low_rank(1, 12, 3, 1e-12), 8),
    "pd-partial": (random_pd_matrix(jr.key(2), 15), 7),
    "pd-full": (random_pd_matrix(jr.key(3), 9), 9),
}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("case", list(_CASES))
def test_greedy_is_bit_identical_to_the_old_guarded_pivoted_cholesky(case, dtype):
    mat, rank = _CASES[case]
    mat = mat.astype(dtype)
    diagonal = jnp.diag(mat)

    def column(k):
        return mat[:, k]

    expected = _reference_greedy(diagonal, column, rank)
    F, _ = rp_cholesky(diagonal, column, rank, pivoting="greedy")
    assert jnp.array_equal(F, expected)
    assert jnp.array_equal(guarded_pivoted_cholesky(diagonal, column, rank), expected)
    jitted = jax.jit(guarded_pivoted_cholesky, static_argnums=(1, 2))
    assert jnp.array_equal(jitted(diagonal, column, rank), expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_pivots_are_distinct_and_valid(seed):
    n, rank = 30, 15
    mat = random_pd_matrix(jr.key(10), n)
    F, pivots = rp_cholesky(jnp.diag(mat), lambda k: mat[:, k], rank, key=jr.key(seed))
    assert F.shape == (n, rank)
    assert pivots.shape == (rank,)
    assert bool(jnp.all((pivots >= 0) & (pivots < n)))
    assert len(set(pivots.tolist())) == rank


@pytest.mark.parametrize("pivoting", ["random", "greedy"])
def test_pivots_give_the_column_nystrom_approximation(pivoting):
    # F Fᵀ = A[:, S] A[S, S]⁻¹ A[S, :] on the returned pivots S.
    n, rank = 20, 6
    mat = random_pd_matrix(jr.key(4), n)
    F, S = rp_cholesky(
        jnp.diag(mat), lambda k: mat[:, k], rank, pivoting=pivoting, key=jr.key(5)
    )
    C = mat[:, S]
    nystrom = einsum(C, jnp.linalg.solve(mat[S][:, S], C.T), "i s, s j -> i j")
    assert tree_allclose(_gram(F), nystrom, rtol=1e-8, atol=1e-10)


def test_random_past_the_numerical_rank_is_exact_and_zero():
    # A rank-3 matrix at rank 8: three distinct pivots recover it exactly;
    # the rest are -1 with exactly zero columns (the guard, gh-237).
    mat = _low_rank(1, 12, 3)
    F, pivots = rp_cholesky(jnp.diag(mat), lambda k: mat[:, k], 8, key=jr.key(0))
    assert bool(jnp.all(jnp.isfinite(F)))
    assert len(set(pivots[:3].tolist())) == 3
    assert bool(jnp.all(pivots[:3] >= 0))
    assert bool(jnp.all(pivots[3:] == -1))
    assert bool(jnp.all(F[:, 3:] == 0.0))
    assert tree_allclose(_gram(F), mat, atol=1e-8)


def test_random_with_a_tiny_jitter_stays_finite():
    # gh-237 with a 1e-12 jitter: the surplus pivots are rounding-scale but
    # above the guard, so they must give finite (tiny) columns, not huge ones.
    mat = _low_rank(1, 12, 3, 1e-12)
    F, pivots = rp_cholesky(jnp.diag(mat), lambda k: mat[:, k], 8, key=jr.key(0))
    assert bool(jnp.all(jnp.isfinite(F)))
    valid = pivots[pivots >= 0]
    assert len(set(valid.tolist())) == valid.shape[0]
    assert tree_allclose(_gram(F), mat, atol=1e-8)


def test_key_none_is_prng_key_zero_and_keys_matter():
    mat = random_pd_matrix(jr.key(6), 40)
    d, col = jnp.diag(mat), lambda k: mat[:, k]
    _, p_none = rp_cholesky(d, col, 10)
    _, p_zero = rp_cholesky(d, col, 10, key=jax.random.PRNGKey(0))
    _, p_other = rp_cholesky(d, col, 10, key=jr.key(7))
    assert jnp.array_equal(p_none, p_zero)
    assert not jnp.array_equal(p_none, p_other)


def test_float32_stays_float32():
    mat = random_pd_matrix(jr.key(8), 10).astype(jnp.float32)
    F, pivots = rp_cholesky(jnp.diag(mat), lambda k: mat[:, k], 5)
    assert F.dtype == jnp.float32
    assert pivots.dtype == jnp.int32


def test_jit_and_grad():
    mat = random_pd_matrix(jr.key(9), 8)

    @jax.jit
    def loss(m):
        F, _ = rp_cholesky(jnp.diag(m), lambda k: m[:, k], 4, key=jr.key(1))
        return jnp.sum(F * F)

    g = jax.grad(loss)(mat)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_bad_arguments_raise():
    d, col = jnp.ones(3), lambda k: jnp.eye(3)[:, k]
    with pytest.raises(ValueError, match="pivoting"):
        rp_cholesky(d, col, 2, pivoting="uniform")  # ty: ignore[invalid-argument-type]
    with pytest.raises(NotImplementedError, match="block_size"):
        rp_cholesky(d, col, 2, block_size=2)


@pytest.mark.slow
def test_random_beats_greedy_on_clustered_data_in_expectation():
    # Three clusters, each a few lengthscales wide, plus 25 isolated
    # outliers. Greedy pivoting spends its 20 pivots on outliers (each
    # explains only itself); RPCholesky samples ∝ residual variance and
    # mostly lands in the clusters.
    k_clusters = jr.key(0)
    centers = jnp.array([0.0, 50.0, 100.0])
    clusters = einx.add(
        "c, c p -> (c p)", centers, 4.0 * jr.uniform(k_clusters, (3, 150))
    )
    outliers = 200.0 + 30.0 * jnp.arange(25.0)
    X = jnp.concatenate([clusters, outliers])
    K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", X, X) ** 2)
    diagonal, rank = jnp.diag(K), 20

    def trace_error(F):
        return jnp.trace(K) - jnp.sum(F * F)

    greedy = trace_error(
        rp_cholesky(diagonal, lambda k: K[:, k], rank, pivoting="greedy")[0]
    )

    @jax.jit
    @jax.vmap
    def random_error(key):
        return trace_error(rp_cholesky(diagonal, lambda k: K[:, k], rank, key=key)[0])

    errors = random_error(jr.split(jr.key(1), 64))
    # Bound the sample mean by its own sampling distribution: mean + 7 SEM.
    sem = jnp.std(errors) / jnp.sqrt(errors.shape[0])
    assert float(jnp.mean(errors) + 7.0 * sem) < float(greedy)
