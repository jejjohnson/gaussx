"""Tests for gaussx.inv with structural dispatch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._primitives import diag as gaussx_diag, inv
from gaussx._primitives._inv import InverseOperator
from gaussx._testing import dense_inv, tree_allclose


def test_inv_diagonal(getkey):
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    inv_op = inv(op)
    assert isinstance(inv_op, lx.DiagonalLinearOperator)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(op))


def test_inv_block_diag(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    bd = BlockDiag(A, B)
    inv_op = inv(bd)
    assert isinstance(inv_op, BlockDiag)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(bd), rtol=1e-4)


def test_inv_kronecker(getkey):
    A = lx.MatrixLinearOperator(jr.normal(getkey(), (2, 2)) + 3 * jnp.eye(2))
    B = lx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3))
    K = Kronecker(A, B)
    inv_op = inv(K)
    assert isinstance(inv_op, Kronecker)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(K), rtol=1e-4)


def test_inv_dense_returns_inverse_operator(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    assert isinstance(inv_op, InverseOperator)


def test_inv_dense_mv(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    v = jr.normal(getkey(), (3,))
    # inv_op.mv(v) should equal A^{-1} v
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(inv_op.mv(v), expected)


def test_inv_dense_as_matrix(getkey):
    mat = jr.normal(getkey(), (3, 3)) + 3 * jnp.eye(3)
    op = lx.MatrixLinearOperator(mat)
    inv_op = inv(op)
    assert tree_allclose(inv_op.as_matrix(), dense_inv(op))


def test_inv_roundtrip(getkey):
    """inv(A) @ A @ v should give back v."""
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.DiagonalLinearOperator(d)
    inv_op = inv(op)
    v = jr.normal(getkey(), (4,))
    assert tree_allclose(inv_op.mv(op.mv(v)), v)


# ---------------------------------------------------------------------------
# gh-328: inv(LowRankUpdate) is decided from static structure
# ---------------------------------------------------------------------------


def _symmetric_low_rank(n=50, k=4):
    U = jr.normal(jr.key(0), (n, k))
    return LowRankUpdate(lx.DiagonalLinearOperator(jnp.full(n, 2.0)), U)


@pytest.mark.parametrize("transform", ["eager", "filter_jit", "jax_jit"])
def test_inv_low_rank_type_is_stable_under_tracing(transform):
    op = _symmetric_low_rank()
    seen = {}

    def f(o):
        result = inv(o)
        seen["type"] = type(result)
        return result.as_matrix()

    if transform == "eager":
        out = f(op)
    elif transform == "filter_jit":
        out = eqx.filter_jit(f)(op)
    else:
        out = jax.jit(f)(op)
    assert seen["type"] is LowRankUpdate
    assert tree_allclose(out, jnp.linalg.inv(op.as_matrix()))


def test_jitted_diag_inv_low_rank_has_no_dense_factorisation():
    # Passed in as an argument, the operator used to fall to InverseOperator,
    # whose diag materialises and factorises the n x n matrix.
    n = 50
    op = _symmetric_low_rank(n)
    jaxpr = str(jax.make_jaxpr(lambda o: gaussx_diag(inv(o)))(op))
    assert f"f64[{n},{n}]" not in jaxpr and f"f32[{n},{n}]" not in jaxpr


@pytest.mark.slow
def test_inv_low_rank_general_factors():
    n, k = 6, 2
    keys = jr.split(jr.key(0), 4)
    base = lx.MatrixLinearOperator(jr.normal(keys[0], (n, n)) + n * jnp.eye(n))
    U = jr.normal(keys[1], (n, k))
    V = jr.normal(keys[2], (n, k))
    d = jnp.array([0.7, -0.3])
    op = LowRankUpdate(base, U, d, V)
    result = inv(op)
    assert isinstance(result, LowRankUpdate)
    assert jnp.allclose(
        result.as_matrix(), jnp.linalg.inv(op.as_matrix()), rtol=1e-10, atol=1e-10
    )
    # A value-equal but distinct V is general too, and still exact.
    copy = LowRankUpdate(lx.DiagonalLinearOperator(jnp.full(n, 2.0)), U, V=U.copy())
    assert isinstance(inv(copy), LowRankUpdate)
    assert jnp.allclose(
        inv(copy).as_matrix(), jnp.linalg.inv(copy.as_matrix()), atol=1e-10
    )


def test_inv_low_rank_general_zero_weight():
    # The scaled capacitance I + D Vᵀ L⁻¹ U needs no D⁻¹, so a zero weight
    # (here the whole update vanishes: L + 0 = I) stays finite and exact.
    n = 4
    U = jr.normal(jr.key(0), (n, 2))
    op = LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.ones(n)), U, jnp.array([0.0, 1.5]), U.copy()
    )
    result = inv(op)
    assert isinstance(result, LowRankUpdate)
    assert jnp.allclose(result.as_matrix(), jnp.linalg.inv(op.as_matrix()), atol=1e-10)


def test_inv_low_rank_keeps_symmetry_and_definiteness_tags():
    n = 5
    U = jr.normal(jr.key(0), (n, 2))
    psd_base = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(jnp.full(n, 2.0)), lx.positive_semidefinite_tag
    )
    # Symmetry claimed by the caller for distinct factors (general branch).
    claimed = LowRankUpdate(psd_base, U, V=U.copy(), tags=lx.symmetric_tag)
    assert lx.is_symmetric(inv(claimed))
    # Shared factors, PSD base, unit weights: PSD (symmetric branch).
    shared = LowRankUpdate(psd_base, U)
    assert lx.is_positive_semidefinite(shared)
    assert lx.is_positive_semidefinite(inv(shared))
    assert lx.is_symmetric(inv(shared))
