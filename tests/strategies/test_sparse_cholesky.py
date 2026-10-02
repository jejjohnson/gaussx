"""Tests for SparseCholeskySolver and the sparse dispatch of diag_inv (G4)."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx import SparseCholeskySolver, SparseOperator


def _path_precision(n: int) -> SparseOperator:
    diagonal = jnp.full(n, 3.0).at[0].set(2.0).at[-1].set(2.0)
    return SparseOperator.from_coo(
        np.r_[np.arange(n), np.arange(1, n)],
        np.r_[np.arange(n), np.arange(n - 1)],
        jnp.r_[diagonal, -jnp.ones(n - 1)],
        (n, n),
        symmetric=True,
        tags=lx.positive_semidefinite_tag,
    )


def test_solve_logdet_diag_inv_against_dense():
    Q = _path_precision(7)
    dense = np.asarray(Q.as_matrix())
    solver = SparseCholeskySolver()
    b = np.arange(7.0)
    np.testing.assert_allclose(
        solver.solve(Q, jnp.asarray(b)), np.linalg.solve(dense, b)
    )
    np.testing.assert_allclose(solver.logdet(Q), np.linalg.slogdet(dense)[1])
    np.testing.assert_allclose(solver.diag_inv(Q), np.diag(np.linalg.inv(dense)))


def test_ordering_reaches_the_symbolic_analysis():
    Q = _path_precision(5)
    factor = SparseCholeskySolver(ordering="natural").factor(Q)
    assert factor.symbolic.ordering == "natural"
    assert factor.symbolic is gaussx.symbolic_cholesky(Q.pattern, ordering="natural")


def test_unwraps_tagged_and_rejects_dense():
    Q = _path_precision(4)
    tagged = lx.TaggedLinearOperator(Q, lx.positive_semidefinite_tag)
    solver = SparseCholeskySolver()
    np.testing.assert_allclose(solver.logdet(tagged), solver.logdet(Q))
    with pytest.raises(TypeError, match="SparseOperator"):
        solver.logdet(lx.MatrixLinearOperator(jnp.eye(3)))


def test_diag_inv_dispatch():
    Q = _path_precision(6)
    expected = np.diag(np.linalg.inv(np.asarray(Q.as_matrix())))
    natural = SparseCholeskySolver(ordering="natural")
    # The strategy routes "auto" to Takahashi, ...
    np.testing.assert_allclose(gaussx.diag_inv(Q, solver=natural), expected)
    # ... and "cholesky" on a SparseOperator is the sparse factor's sweep.
    np.testing.assert_allclose(gaussx.diag_inv(Q, method="cholesky"), expected)
    with pytest.raises(ValueError, match="pinv"):
        gaussx.diag_inv(Q, solver=natural, pinv=True)


def test_strategy_through_dispatch_helpers():
    from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve

    Q = _path_precision(5)
    solver = SparseCholeskySolver()
    dense = np.asarray(Q.as_matrix())
    np.testing.assert_allclose(dispatch_logdet(Q, solver), np.linalg.slogdet(dense)[1])
    np.testing.assert_allclose(
        dispatch_solve(Q, jnp.ones(5), solver), np.linalg.solve(dense, np.ones(5))
    )
