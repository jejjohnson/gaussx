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
import pytest

import gaussx
from gaussx import MultivariateNormal, MultivariateNormalPrecision
from gaussx._testing import random_pd_matrix


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
    return cls()


_STRATEGIES = [_default(cls) for cls in _exported_strategy_classes()] + [
    gaussx.ComposedSolver(gaussx.DenseSolver(), gaussx.IndefiniteSLQLogdet()),
    gaussx.CGSolver(preconditioner=gaussx.PartialCholeskyPreconditioner(rank=2)),
    gaussx.PartialCholeskyPreconditioner(),
]


def _id(s) -> str:
    return type(s).__name__


def test_every_exported_strategy_is_covered():
    # Guard the discovery helper itself: a rename must not silently empty
    # the parametrisation below.
    names = {c.__name__ for c in _exported_strategy_classes()}
    assert {"AutoSolver", "CGSolver", "ComposedSolver", "SLQLogdet"} <= names


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=_id)
def test_default_strategy_has_no_leaves(strategy):
    assert jax.tree_util.tree_leaves(strategy) == []


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=_id)
def test_tree_map_leaves_strategy_config_unchanged(strategy):
    assert jax.tree_util.tree_map(lambda leaf: 2 * leaf, strategy) == strategy


def _pd_operator() -> lx.AbstractLinearOperator:
    return lx.MatrixLinearOperator(
        random_pd_matrix(jr.key(0), 3), lx.positive_semidefinite_tag
    )


def test_distribution_leaves_are_only_arrays():
    op = _pd_operator()
    dist = MultivariateNormal(jnp.zeros(3), op)
    leaves = jax.tree_util.tree_leaves(dist)
    assert len(leaves) == 2
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    doubled = jax.tree_util.tree_map(lambda leaf: 2 * leaf, dist)
    assert doubled.solver == dist.solver


@pytest.mark.parametrize("cls", [MultivariateNormal, MultivariateNormalPrecision])
@pytest.mark.parametrize(
    "strategy",
    [s for s in _STRATEGIES if isinstance(s, gaussx.AbstractSolverStrategy)],
    ids=_id,
)
def test_jax_jit_log_prob_with_distribution_argument(cls, strategy):
    dist = cls(jnp.zeros(3), _pd_operator(), solver=strategy)
    x = jnp.ones(3)
    jitted = jax.jit(lambda d, x: d.log_prob(x))(dist, x)
    assert jnp.allclose(jitted, dist.log_prob(x), rtol=1e-10, atol=1e-10)
