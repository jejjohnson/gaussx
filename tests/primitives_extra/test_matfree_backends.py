"""Tests for matfree-backed primitives."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._primitives._diag import diag
from gaussx._primitives._eig import eig, eigvals
from gaussx._primitives._sqrt import SqrtOperator, sqrt
from gaussx._primitives._svd import svd
from gaussx._primitives._trace import trace
from gaussx._testing import random_pd_matrix, tree_allclose


class LazyPSD(lx.MatrixLinearOperator):
    def as_matrix(self):
        raise NotImplementedError("dense materialization unavailable")


# --- Partial SVD ---


def test_svd_partial_singular_values(getkey):
    """Partial SVD should recover some singular values accurately."""
    mat = random_pd_matrix(getkey(), 10)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    k = 5

    _, s_partial, _ = svd(op, rank=k, key=getkey())

    # Each found singular value should be close to some true one
    _, s_full, _ = jnp.linalg.svd(mat, full_matrices=False)
    for si in s_partial:
        min_dist = jnp.min(jnp.abs(s_full - si))
        assert min_dist < 0.2 * jnp.max(s_full)


def test_svd_partial_shapes(getkey):
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op, rank=3, key=getkey())
    assert U.shape == (8, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 8)


# --- Partial Eig ---


def test_eig_partial_eigenvalues(getkey):
    """Partial eig should recover some eigenvalues accurately."""
    mat = random_pd_matrix(getkey(), 10)
    # Add diagonal shift for better conditioning (Lanczos stability)
    mat = mat + 2.0 * jnp.eye(10)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    k = 5

    vals_partial, _ = eig(op, rank=k, key=getkey())
    vals_full = jnp.linalg.eigvalsh(mat)

    # Each found eigenvalue should be close to some true one
    for vi in vals_partial:
        min_dist = jnp.min(jnp.abs(vals_full - vi))
        assert min_dist < 0.3 * jnp.max(jnp.abs(vals_full))


def test_eig_partial_shapes(getkey):
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals, vecs = eig(op, rank=4, key=getkey())
    assert vals.shape == (4,)
    assert vecs.shape == (8, 4)


def test_eigvals_partial(getkey):
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals = eigvals(op, rank=3, key=getkey())
    assert vals.shape == (3,)


def test_eig_partial_rank_clipped(getkey):
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
    vals, vecs = eig(op, rank=10, key=getkey())
    assert vals.shape == (4,)
    assert vecs.shape == (4, 4)


# --- Matrix-free Sqrt ---


def test_sqrt_lanczos_matvec(getkey):
    """SqrtOperator.mv should approximate sqrt(A) @ v."""
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (8,))

    S = sqrt(op, lanczos_order=8)
    assert isinstance(S, SqrtOperator)

    result = S.mv(v)

    # Dense reference
    vals, vecs = jnp.linalg.eigh(mat)
    S_dense = vecs @ jnp.diag(jnp.sqrt(vals)) @ vecs.T
    expected = S_dense @ v

    assert tree_allclose(result, expected, rtol=0.1)


def test_sqrt_lanczos_lazy_operator(getkey):
    """Lanczos sqrt should not require dense materialization of the input op."""
    mat = random_pd_matrix(getkey(), 6)
    op = LazyPSD(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (6,))
    S = sqrt(op, lanczos_order=6)
    result = S.mv(v)

    vals, vecs = jnp.linalg.eigh(mat)
    expected = vecs @ jnp.diag(jnp.sqrt(vals)) @ vecs.T @ v
    assert tree_allclose(result, expected, rtol=0.1)


def test_sqrt_lanczos_as_matrix(getkey):
    """SqrtOperator.as_matrix should fall back to dense."""
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    S = sqrt(op, lanczos_order=5)
    S_mat = S.as_matrix()
    assert S_mat.shape == (5, 5)
    assert tree_allclose(S_mat @ S_mat, mat, rtol=1e-4)


def test_sqrt_lanczos_tags(getkey):
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    S = sqrt(op, lanczos_order=4)
    assert lx.is_symmetric(S)
    assert lx.is_positive_semidefinite(S)


# --- Stochastic Trace ---


def test_trace_stochastic(getkey):
    """Stochastic trace should approximate true trace."""
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)

    estimated = trace(op, stochastic=True, num_probes=200, key=getkey())
    true_trace = jnp.trace(mat)

    assert jnp.abs(estimated - true_trace) < 0.2 * jnp.abs(true_trace) + 1.0


def test_trace_stochastic_diagonal(getkey):
    """Stochastic trace of diagonal should be accurate."""
    d = jnp.abs(jr.normal(getkey(), (15,))) + 1.0
    op = lx.MatrixLinearOperator(jnp.diag(d))

    estimated = trace(op, stochastic=True, num_probes=200, key=getkey())
    assert jnp.abs(estimated - jnp.sum(d)) < 0.1 * jnp.sum(d) + 1.0


# --- Stochastic Diag ---


def test_diag_stochastic(getkey):
    """Stochastic diag should approximate true diagonal."""
    mat = random_pd_matrix(getkey(), 10)
    op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)

    estimated = diag(op, stochastic=True, num_probes=200, key=getkey())
    true_diag = jnp.diag(mat)

    # Stochastic — generous tolerance
    assert jnp.max(jnp.abs(estimated - true_diag)) < 0.3 * jnp.max(true_diag) + 1.0


def test_svd_partial_rank_clipped(getkey):
    mat = jr.normal(getkey(), (3, 5))
    op = lx.MatrixLinearOperator(mat)
    U, s, Vt = svd(op, rank=10, key=getkey())
    assert U.shape == (3, 3)
    assert s.shape == (3,)
    assert Vt.shape == (3, 5)


def test_sqrt_lanczos_filter_jit(getkey):
    mat = random_pd_matrix(getkey(), 5)
    op = LazyPSD(mat, lx.positive_semidefinite_tag)
    vec = jr.normal(getkey(), (5,))
    sqrt_op = sqrt(op, lanczos_order=5)

    @eqx.filter_jit
    def apply(operator, vector):
        return operator.mv(vector)

    result = apply(sqrt_op, vec)
    vals, vecs = jnp.linalg.eigh(mat)
    expected = vecs @ jnp.diag(jnp.sqrt(vals)) @ vecs.T @ vec
    assert tree_allclose(result, expected, rtol=0.1)
