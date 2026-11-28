"""Strategy configuration is static pytree data (gh-301).

Every strategy scalar (thresholds, step counts, probe counts, tolerances,
seeds, sampler names) must live in the treedef, not in the leaves, so that
``jax.jit`` can take a strategy -- or a distribution that holds one -- as an
argument, and ``tree_map`` over a distribution cannot rewrite it.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._testing import random_pd_operator


def _exported_strategy_classes() -> list[type]:
    bases = (gaussx.AbstractSolveStrategy, gaussx.AbstractLogdetStrategy)
    return sorted(
        (
            obj
            for obj in vars(gaussx).values()
            if inspect.isclass(obj)
            and issubclass(obj, bases)
            and not inspect.isabstract(obj)
            and not obj.__name__.startswith("Abstract")
        ),
        key=lambda c: c.__name__,
    )


def _default(cls: type):
    if cls is gaussx.ComposedSolver:
        return cls(gaussx.CGSolver(), gaussx.SLQLogdet())
    if cls is gaussx.LineaxSolver:
        return cls(lx.CG(rtol=1e-6, atol=1e-6))
    return cls()


# KeyedSolver's key is deliberately a leaf, so that it can change under jit
# (gh-384); see test_keyed_solver_key_is_its_only_leaf. NystromLogdet's shift
# is the model's noise variance, a learned parameter, so it is a leaf too
# (gh-486); see test_nystrom_logdet_shift_is_its_only_leaf.
# MultigridSolver holds its level hierarchy (operators, transfers, smoother
# diagonals), which are arrays by nature (gh-856); see
# test_multigrid_solver_leaves_are_its_levels. PreconditionedCGSolver's and
# PartialCholeskyPreconditioner's shift is the noise variance, or the
# per-point site variances d of a non-Gaussian GP that change at every
# Newton step, so it is a leaf (gh-878); see
# test_partial_cholesky_shift_is_its_only_leaf. With shift=None (read from
# the system) they carry no leaf, so they stay in the checks below that way.
_LEAF_CARRYING = (
    gaussx.KeyedSolver,
    gaussx.NystromLogdet,
    gaussx.MultigridSolver,
    gaussx.PreconditionedCGSolver,
)
_STRATEGIES = [
    _default(cls) for cls in _exported_strategy_classes() if cls not in _LEAF_CARRYING
] + [
    gaussx.ComposedSolver(gaussx.DenseSolver(), gaussx.IndefiniteSLQLogdet()),
    gaussx.CGSolver(
        preconditioner=gaussx.PartialCholeskyPreconditioner(rank=2, shift=None)
    ),
    gaussx.PartialCholeskyPreconditioner(shift=None),
    gaussx.PreconditionedCGSolver(shift=None),
]


def _id(s) -> str:
    return type(s).__name__


def test_every_exported_strategy_is_covered():
    # Guard the discovery helper itself: a rename must not silently empty
    # the parametrisation below.
    names = {c.__name__ for c in _exported_strategy_classes()}
    assert {"AutoSolver", "CGSolver", "ComposedSolver", "SLQLogdet"} <= names


def test_keyed_solver_key_is_its_only_leaf():
    key = jr.key(0)
    keyed = gaussx.KeyedSolver(gaussx.CGSolver(), key)
    assert jax.tree_util.tree_leaves(keyed) == [key]


def test_nystrom_logdet_shift_is_its_only_leaf():
    shift = jnp.asarray(0.1)
    strategy = gaussx.NystromLogdet(shift=shift, rank=3)
    assert jax.tree_util.tree_leaves(strategy) == [shift]


@pytest.mark.parametrize(
    "build",
    [
        lambda shift: gaussx.PartialCholeskyPreconditioner(rank=3, shift=shift),
        lambda shift: gaussx.PreconditionedCGSolver(shift=shift),
    ],
    ids=["PartialCholeskyPreconditioner", "PreconditionedCGSolver"],
)
def test_partial_cholesky_shift_is_its_only_leaf(build):
    for shift in (1.0, jnp.asarray(0.1), jnp.ones(3)):
        assert jax.tree_util.tree_leaves(build(shift)) == [shift]


def test_multigrid_solver_leaves_are_its_levels():
    a = jnp.asarray(2 * np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1))
    operator = lx.MatrixLinearOperator(a, lx.positive_semidefinite_tag)
    transfer = lx.MatrixLinearOperator(jnp.asarray([[0.5], [1.0], [0.5]]))
    levels = gaussx.galerkin_levels(operator, [transfer], restriction_scale=0.5)
    solver = gaussx.MultigridSolver(levels, rtol=1e-6, max_steps=7)
    leaves = jax.tree_util.tree_leaves(solver)
    assert len(leaves) == len(jax.tree_util.tree_leaves(levels))
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=_id)
def test_default_strategy_has_no_leaves(strategy):
    assert jax.tree_util.tree_leaves(strategy) == []


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=_id)
def test_tree_map_leaves_strategy_config_unchanged(strategy):
    assert jax.tree_util.tree_map(lambda leaf: 2 * leaf, strategy) == strategy


# shift=None needs a system that splits into K + diag(d); the jitted
# log-density below takes a dense one, so it gets the default (a traced
# float leaf) instead.
_SOLVERS = [
    s
    for s in _STRATEGIES
    if isinstance(s, gaussx.AbstractSolverStrategy)
    and not isinstance(s, gaussx.PreconditionedCGSolver)
] + [gaussx.PreconditionedCGSolver()]


def _pytest_ids(objs) -> list[str]:
    """pytest's ids: the class name, plus its occurrence index when repeated."""
    names = [_id(o) for o in objs]
    return [
        f"{name}{names[:i].count(name)}" if names.count(name) > 1 else name
        for i, name in enumerate(names)
    ]


_SOLVER_IDS = _pytest_ids(_SOLVERS)


def _log_prob(form, op, strategy, x):
    """log 𝒩(x; 0, Σ), with ``op`` the covariance Σ or the precision Σ⁻¹."""
    if form == "covariance":
        return gaussx.gaussian_log_prob(jnp.zeros(3), op, x, solver=strategy)
    return gaussx.GaussianMRF(jnp.zeros(3), op, solver=strategy).log_prob(x)


# The covariance form with an iterative strategy traces its Krylov solve and
# logdet both jitted and eagerly: ~2-3.5 s in CI per strategy, so those cases
# are slow. The precision form keeps every strategy in the fast lane.
@pytest.mark.parametrize(
    ("form", "strategy"),
    [
        pytest.param(
            form,
            strategy,
            id=f"{sid}-{form}",
            marks=pytest.mark.slow
            if form == "covariance"
            and not isinstance(strategy, gaussx.DenseSolver | gaussx.AutoSolver)
            else (),
        )
        for form in ["covariance", "precision"]
        for strategy, sid in zip(_SOLVERS, _SOLVER_IDS, strict=True)
    ],
)
def test_jax_jit_log_prob_with_strategy_argument(form, strategy):
    op = random_pd_operator(jr.key(0), 3)
    if isinstance(strategy, gaussx.SparseCholeskySolver):
        # A sparse factorisation needs a SparseOperator: the same matrix,
        # stored as its (dense) lower triangle.
        rows, cols = np.tril_indices(3)
        op = gaussx.SparseOperator.from_coo(
            rows,
            cols,
            op.as_matrix()[rows, cols],
            (3, 3),
            symmetric=True,
            tags=lx.positive_semidefinite_tag,
        )
    x = jnp.ones(3)
    jitted = jax.jit(_log_prob, static_argnums=0)(form, op, strategy, x)
    eager = _log_prob(form, op, strategy, x)
    assert jnp.allclose(jitted, eager, rtol=1e-10, atol=1e-10)
