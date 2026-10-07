"""Private utilities shared by operator implementations."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array


def vmap_over_batch_dims(fn: Callable, num_batch_dims: int) -> Callable:
    """Apply ``jax.vmap`` repeatedly over the leading batch dimensions."""
    for _ in range(num_batch_dims):
        fn = jax.vmap(fn)
    return fn


def lineax_diagonal(operator: lx.AbstractLinearOperator) -> Array:
    """``lineax.diagonal`` for a gaussx operator, via `gaussx.diag`.

    `gaussx.diag` is structured where a structured diagonal exists and
    dense otherwise (imported lazily: the primitives import the operators).
    """
    from gaussx._primitives._diag import diag

    return diag(operator)


def lineax_conj(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """``lineax.conj`` for a gaussx operator.

    The identity for a real operator; for a complex one (e.g. a complex
    ``Circulant``) the matrix-free ``v ↦ conj(A conj(v))``.
    """
    structures = (operator.in_structure(), operator.out_structure())
    if not any(
        jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        for leaf in jax.tree_util.tree_leaves(structures)
    ):
        return operator
    return lx.FunctionLinearOperator(
        lambda vector: jnp.conj(operator.mv(jnp.conj(vector))),
        operator.in_structure(),
    )


def register_lineax_structure_functions(*classes: type) -> None:
    """Register ``lineax.linearise``/``materialise``/``diagonal``/``conj``.

    lineax registers these only for its own classes, so lineax's solvers
    raise ``NotImplementedError`` on any operator that lacks them — in
    particular ``AutoLinearSolver`` picks its ``Diagonal`` solver whenever
    ``is_diagonal`` is ``True`` and then calls ``lineax.diagonal`` (gh-410).
    A gaussx operator is already linear and concrete, so ``linearise`` and
    ``materialise`` are the identity.
    """
    for cls in classes:
        lx.linearise.register(cls)(_identity)
        lx.materialise.register(cls)(_identity)
        lx.diagonal.register(cls)(lineax_diagonal)
        lx.conj.register(cls)(lineax_conj)


def _identity(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    return operator
