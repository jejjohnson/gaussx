"""Lineax structure functions shared by every gaussx operator A.

lineax dispatches ``linearise``, ``materialise``, ``diagonal`` and ``conj``
on the operator class and registers them only for its own classes, so each
gaussx operator registers them here through
`register_lineax_structure_functions`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate


def lineax_diagonal(operator: lx.AbstractLinearOperator) -> Array:
    """``lineax.diagonal`` for a gaussx operator, via `gaussx.diag`.

        diag(A)_i = [A]_ii

    `gaussx.diag` is structured where a structured diagonal exists and
    dense otherwise (imported lazily: the primitives import the operators).
    Its structured rules assume square components, so an operator with a
    rectangular block, factor or base takes the dense diagonal instead.

    Args:
        operator: The operator A.

    Returns:
        The diagonal of A.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._operators._utils import lineax_diagonal
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> B = lx.DiagonalLinearOperator(jnp.array([3.0, 4.0]))
        >>> lineax_diagonal(gaussx.Kronecker(A, B)).tolist()
        [3.0, 4.0, 6.0, 8.0]
    """
    # lazy import, cycle: _primitives._diag -> _operators -> _operators._utils
    from gaussx._primitives._diag import diag

    if not _square_components(operator):
        # diag(A) of the dense A, since a rectangular component has no
        # structured diagonal rule
        return jnp.diag(operator.as_matrix())
    return diag(operator)


def _square_components(operator: lx.AbstractLinearOperator) -> bool:
    """Whether A and every block, factor or base inside it is square.

    Args:
        operator: The operator A.

    Returns:
        ``True`` if `gaussx.diag`'s structured rules apply to A.
    """
    if operator.in_size() != operator.out_size():
        return False
    if isinstance(operator, BlockDiag | Kronecker):
        return all(_square_components(op) for op in operator.operators)
    if isinstance(operator, LowRankUpdate):
        return _square_components(operator.base)
    return True


_TAG_PREDICATES = (
    (lx.is_symmetric, lx.symmetric_tag),
    (lx.is_diagonal, lx.diagonal_tag),
    (lx.is_lower_triangular, lx.lower_triangular_tag),
    (lx.is_upper_triangular, lx.upper_triangular_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    (lx.is_tridiagonal, lx.tridiagonal_tag),
    (lx.has_unit_diagonal, lx.unit_diagonal_tag),
)


def lineax_conj(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """``lineax.conj`` for a gaussx operator, the entrywise conjugate Ā.

        Ā x = conj(A conj(x))

    The identity for a real operator; for a complex one (a complex
    ``circulant``, say) the matrix-free x ↦ conj(A conj(x)), carrying the
    structural tags of A: conjugation preserves all of them.

    Args:
        operator: The operator A.

    Returns:
        A itself when it is real, otherwise a matrix-free Ā.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._operators._utils import lineax_conj
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> lineax_conj(A) is A
        True
        >>> B = lx.DiagonalLinearOperator(jnp.array([1.0j, 2.0]))
        >>> x = jnp.array([1.0 + 1.0j, 1.0])
        >>> expected = jnp.conj(B.as_matrix()) @ x
        >>> bool(jnp.allclose(lineax_conj(B).mv(x), expected))
        True
    """
    structures = (operator.in_structure(), operator.out_structure())
    if not any(
        jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        for leaf in jax.tree_util.tree_leaves(structures)
    ):
        return operator
    tags = tuple(tag for check, tag in _TAG_PREDICATES if check(operator))
    return lx.FunctionLinearOperator(
        # Ā x = conj(A conj(x))
        lambda vector: jnp.conj(operator.mv(jnp.conj(vector))),
        operator.in_structure(),
        tags=tags,
    )


def register_lineax_structure_functions(*classes: type) -> None:
    """Register ``lineax.linearise``/``materialise``/``diagonal``/``conj``.

    lineax registers these only for its own classes, so lineax's solvers
    raise ``NotImplementedError`` on any operator that lacks them; in
    particular ``AutoLinearSolver`` picks its ``Diagonal`` solver whenever
    ``is_diagonal`` is ``True`` and then calls ``lineax.diagonal`` (gh-410).
    A gaussx operator is already linear and concrete, so ``linearise`` and
    ``materialise`` are the identity; ``diagonal`` is `lineax_diagonal` and
    ``conj`` is `lineax_conj`.

    Args:
        *classes: The operator classes to register.

    Examples:
        Every gaussx operator is registered on import, so lineax's
        ``diagonal`` works on it:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = gaussx.Toeplitz(jnp.array([2.0, 0.5]))
        >>> lx.diagonal(A).tolist()
        [2.0, 2.0]
        >>> lx.materialise(A) is A
        True
    """
    for cls in classes:
        lx.linearise.register(cls)(_identity)
        lx.materialise.register(cls)(_identity)
        lx.diagonal.register(cls)(lineax_diagonal)
        lx.conj.register(cls)(lineax_conj)


def _identity(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """``linearise`` and ``materialise`` of a gaussx operator: A itself.

    Args:
        operator: The operator A.

    Returns:
        A, unchanged.
    """
    return operator
