"""Tests for gaussx structural tags and query helpers."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx

from gaussx._tags import (
    block_diagonal_tag,
    diagonal_tag,
    is_block_diagonal,
    is_diagonal,
    is_kronecker,
    is_low_rank,
    is_positive_semidefinite,
    is_symmetric,
    kronecker_tag,
    low_rank_tag,
    positive_semidefinite_tag,
    symmetric_tag,
)


# ---------------------------------------------------------------------------
# Tag identity — gaussx tags are distinct singletons
# ---------------------------------------------------------------------------


def test_tag_reprs():
    assert repr(kronecker_tag) == "kronecker_tag"
    assert repr(block_diagonal_tag) == "block_diagonal_tag"
    assert repr(low_rank_tag) == "low_rank_tag"


def test_tags_are_distinct():
    tags = {kronecker_tag, block_diagonal_tag, low_rank_tag}
    assert len(tags) == 3


# ---------------------------------------------------------------------------
# Lineax re-exports work
# ---------------------------------------------------------------------------


def test_lineax_tag_reexports():
    assert symmetric_tag is lx.symmetric_tag
    assert diagonal_tag is lx.diagonal_tag
    assert positive_semidefinite_tag is lx.positive_semidefinite_tag


def test_lineax_query_reexports():
    diag_op = lx.DiagonalLinearOperator(jnp.ones(3))
    assert is_symmetric(diag_op) is True
    assert is_diagonal(diag_op) is True


def test_is_positive_semidefinite_with_tag():
    mat = jnp.eye(3)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    assert is_positive_semidefinite(op) is True


# ---------------------------------------------------------------------------
# GaussX queries — default fallback returns False
# ---------------------------------------------------------------------------


def test_is_kronecker_default_false():
    op = lx.DiagonalLinearOperator(jnp.ones(3))
    assert is_kronecker(op) is False


def test_is_block_diagonal_default_false():
    op = lx.DiagonalLinearOperator(jnp.ones(3))
    assert is_block_diagonal(op) is False


def test_is_low_rank_default_false():
    op = lx.DiagonalLinearOperator(jnp.ones(3))
    assert is_low_rank(op) is False
