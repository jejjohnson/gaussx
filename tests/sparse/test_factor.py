"""Tests for the numeric sparse Cholesky and SparseCholeskyFactor (G4)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx import SparseCholeskyFactor, SparseOperator
from gaussx._einx import rearrange


# (ordering, layout): the banded block kernels and the gathered windows each
# see a narrow (RCM / natural) and, through the arrow problem, a wide pattern.
LAYOUTS = [("natural", True), ("natural", False), ("rcm", True), ("rcm", False)]


@pytest.mark.parametrize(("ordering", "banded"), LAYOUTS)
@pytest.mark.parametrize("symmetric", [True, False])
def test_factor_equals_dense_cholesky_of_permuted(
    grid, symbolic_for, to_dense_factor, ordering, banded, symmetric
):
    op = grid(4, 5, symmetric=symmetric)
    sym = symbolic_for(op, ordering, banded=banded)
    factor = gaussx.sparse_cholesky(op, sym)
    dense = np.asarray(op.as_matrix())[np.ix_(sym.perm, sym.perm)]
    np.testing.assert_allclose(
        to_dense_factor(sym, factor.values), np.linalg.cholesky(dense), atol=1e-12
    )


@pytest.mark.parametrize("banded", [True, False])
def test_arrow_pattern(grid, symbolic_for, to_dense_factor, banded):
    op = grid(3, 4, fixed_effect=True)
    sym = symbolic_for(op, "rcm", banded=banded)
    factor = gaussx.sparse_cholesky(op, sym)
    dense = np.asarray(op.as_matrix())
    np.testing.assert_allclose(
        to_dense_factor(sym, factor.values),
        np.linalg.cholesky(dense[np.ix_(sym.perm, sym.perm)]),
        atol=1e-12,
    )
    np.testing.assert_allclose(factor.logdet(), np.linalg.slogdet(dense)[1])


@pytest.mark.parametrize(("ordering", "banded"), LAYOUTS)
def test_solve_logdet_against_dense(grid, symbolic_for, ordering, banded):
    op = grid(4, 5)
    factor = gaussx.sparse_cholesky(op, symbolic_for(op, ordering, banded=banded))
    dense = np.asarray(op.as_matrix())
    b = np.asarray(jr.normal(jr.key(0), (20,)))
    np.testing.assert_allclose(factor.solve(jnp.asarray(b)), np.linalg.solve(dense, b))
    np.testing.assert_allclose(factor.logdet(), np.linalg.slogdet(dense)[1])


@pytest.mark.parametrize("banded", [True, False])
def test_solve_lower_transpose_has_covariance_q_inverse(grid, symbolic_for, banded):
    # x = Pᵀ L⁻ᵀ z, so stacking x over z = e_1..e_n gives M with M Mᵀ = Q⁻¹.
    op = grid(3, 4)
    factor = gaussx.sparse_cholesky(op, symbolic_for(op, "rcm", banded=banded))
    M = jax.vmap(factor.solve_lower_transpose, out_axes=1)(jnp.eye(12))
    cov = M @ rearrange(M, "i j -> j i")
    np.testing.assert_allclose(
        cov, np.linalg.inv(np.asarray(op.as_matrix())), atol=1e-12
    )


def test_cholesky_primitive_returns_sparse_factor(grid):
    op = grid(3, 3)
    factor = gaussx.cholesky(op)
    assert isinstance(factor, SparseCholeskyFactor)
    assert factor.symbolic is gaussx.symbolic_cholesky(op.pattern)
    np.testing.assert_allclose(
        factor.logdet(), np.linalg.slogdet(np.asarray(op.as_matrix()))[1]
    )


def test_jit_and_vmap_over_values(grid):
    op = grid(4, 4)
    sym = gaussx.symbolic_cholesky(op.pattern)
    scales = jnp.array([0.5, 1.0, 2.0])

    @jax.jit
    def logdet(scale):
        Q = eqx.tree_at(lambda o: o.values, op, scale * op.values)
        return gaussx.sparse_cholesky(Q, sym).logdet()

    batched = jax.vmap(logdet)(scales)
    dense = [
        np.linalg.slogdet(float(s) * np.asarray(op.as_matrix()))[1] for s in scales
    ]
    np.testing.assert_allclose(batched, dense)

    b = jnp.ones(16)
    solves = jax.vmap(
        lambda v: gaussx.sparse_cholesky(SparseOperator(v, op.pattern), sym).solve(b)
    )(rearrange(scales, "s -> s 1") * op.values)
    np.testing.assert_allclose(
        solves[2], np.linalg.solve(2.0 * np.asarray(op.as_matrix()), np.ones(16))
    )


def test_float32_stays_float32(grid):
    op = grid(3, 3)
    op32 = SparseOperator(op.values.astype(jnp.float32), op.pattern)
    factor = gaussx.sparse_cholesky(op32)
    assert factor.values.dtype == jnp.float32
    assert factor.logdet().dtype == jnp.float32
    assert factor.solve(jnp.ones(9, jnp.float32)).dtype == jnp.float32
    assert factor.diag_inv().dtype == jnp.float32


@pytest.mark.parametrize("banded", [True, False])
def test_not_positive_definite_is_nan(grid, symbolic_for, banded):
    op = grid(3, 3, shift=-5.0)
    factor = gaussx.sparse_cholesky(op, symbolic_for(op, "rcm", banded=banded))
    assert jnp.isnan(factor.logdet())


@pytest.mark.parametrize("n", [1, 4])
def test_diagonal_operator(n):
    d = jnp.arange(1.0, n + 1)
    op = SparseOperator.from_coo(np.arange(n), np.arange(n), d, (n, n), symmetric=True)
    factor = gaussx.sparse_cholesky(op)
    assert factor.symbolic.max_col == 1
    np.testing.assert_allclose(factor.logdet(), jnp.sum(jnp.log(d)))
    np.testing.assert_allclose(factor.solve(jnp.ones(n)), 1 / d)
    np.testing.assert_allclose(factor.diag_inv(), 1 / d)


def test_errors(grid):
    with pytest.raises(TypeError, match="SparseOperator"):
        gaussx.sparse_cholesky(lx.MatrixLinearOperator(jnp.eye(3)))  # ty: ignore[invalid-argument-type]
    sym = gaussx.symbolic_cholesky(grid(3, 3).pattern)
    with pytest.raises(ValueError, match="different sparsity pattern"):
        gaussx.sparse_cholesky(grid(3, 4), sym)
