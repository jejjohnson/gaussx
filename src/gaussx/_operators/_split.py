"""Read a system A = K + diag(d) back into its kernel part K and its diagonal d.

The split matches on operator types only (lineax sums, scalar multiples and
quotients of diagonals and identities, tags), never on values, so it is the
same eagerly and under ``jit``, and it never densifies an operator.
"""

from __future__ import annotations

import functools as ft
import operator as _op

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


class DiagonalSplit(eqx.Module):
    """A system A = base + diag(diagonal), read back into its two parts.

    Attributes:
        base: The kernel part K, an operator of the system's shape.
        diagonal: The diagonal d, in the system's dtype.
    """

    base: lx.AbstractLinearOperator
    diagonal: Float[Array, " N"]


_INHERITED_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def _summands(operator: lx.AbstractLinearOperator) -> list[lx.AbstractLinearOperator]:
    """Flatten a (nested) lineax sum into its summands, in order.

    A tagged inner sum stays one summand, so its tags are not lost.

    Args:
        operator: An operator, a sum or a nested sum.

    Returns:
        The summands, left to right.
    """
    if isinstance(operator, lx.AddLinearOperator):
        return _summands(operator.operator1) + _summands(operator.operator2)
    return [operator]


def _summand_diagonal(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, " N"] | None:
    """The diagonal of a summand that is diagonal by type, else ``None``.

    Recognised: ``DiagonalLinearOperator(d)`` (d), ``IdentityLinearOperator``
    (𝟏), and a scalar multiple ``s * D`` (s d) or quotient ``D / s`` (d / s)
    of either, through any tags.

    Args:
        operator: One summand of the system.

    Returns:
        The summand's diagonal, or ``None`` if its type is not diagonal.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        return _summand_diagonal(operator.operator)
    if isinstance(operator, lx.DiagonalLinearOperator):
        diagonal = operator.diagonal
        if eqx.is_array(diagonal) and diagonal.ndim == 1:
            return diagonal
        return None
    if isinstance(operator, lx.IdentityLinearOperator):
        structure = operator.in_structure()
        if operator.out_structure() != structure:
            return None
        if not isinstance(structure, jax.ShapeDtypeStruct) or len(structure.shape) != 1:
            return None
        return jnp.ones(structure.shape, dtype=structure.dtype)
    if isinstance(operator, lx.MulLinearOperator):
        inner = _summand_diagonal(operator.operator)
        if inner is None:
            return None
        # s · d, the same product MulLinearOperator.as_matrix forms
        return inner * operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        inner = _summand_diagonal(operator.operator)
        if inner is None:
            return None
        # d / s, the same quotient DivLinearOperator.as_matrix forms
        return inner / operator.scalar
    return None


def split_diagonal(
    operator: lx.AbstractLinearOperator,
) -> DiagonalSplit | None:
    """Read A = K + diag(d) back into its kernel part K and its diagonal d.

        A = base + diag(diagonal)

    A preconditioner or log-determinant estimator for a GP approximates the
    kernel part K and adds the diagonal d exactly, so it needs the two
    apart. In a non-Gaussian GP d holds the site variances and changes at
    every Newton step, so it is read from the system at solve time.

    Recognised forms, with the summands in any order:

    | System | ``base`` | ``diagonal`` |
    | --- | --- | --- |
    | ``K + DiagonalLinearOperator(d)`` | K | d |
    | ``K + s * IdentityLinearOperator(...)`` | K | s 𝟏 |
    | ``K + s * DiagonalLinearOperator(d)``, ``K + D / s`` | K | s d, d / s |
    | ``TaggedLinearOperator(<any of these>)`` | as inside | as inside |
    | nested sums with several diagonals | the other summands | their sum |

    Everything else returns ``None``: a ``LowRankUpdate`` (it already has an
    exact solve and log-determinant, so no preconditioner is wanted), a
    single dense matrix (its diagonal cannot be attributed to a kernel part
    or a noise part), a sum with no diagonal summand, and a sum of diagonals
    only (no kernel part). A tagged inner sum counts as one summand.

    The split looks at operator types only, never at values, so it is the
    same eagerly and under ``jit``, and nothing is densified. ``base`` keeps
    its own tags. When ``operator`` itself carries the symmetric or PSD tag,
    ``base`` gets it too: a PSD-tagged system is read as a PSD kernel plus
    non-negative variances, the contract of every GP system (d = 1 / λ with
    λ > 0). A caller whose diagonal can be negative tags ``base`` itself.
    ``diagonal`` has the system's dtype.

    Args:
        operator: The system A, a square lineax operator on a flat vector.

    Returns:
        The ``DiagonalSplit`` (``base`` K, ``diagonal`` d), or ``None`` when
        the system is not a kernel part plus a diagonal by type.

    Examples:
        A dense kernel plus a diagonal of site variances, PSD-tagged:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> kernel = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]))
        >>> variance = lx.DiagonalLinearOperator(jnp.array([0.5, 0.25]))
        >>> system = lx.TaggedLinearOperator(kernel + variance, psd)
        >>> split = gaussx.split_diagonal(system)
        >>> split.diagonal.tolist()
        [0.5, 0.25]
        >>> lx.is_positive_semidefinite(split.base)
        True

        A scaled identity, on either side, gives a constant diagonal:

        >>> identity = lx.IdentityLinearOperator(kernel.in_structure())
        >>> gaussx.split_diagonal(0.1 * identity + kernel).diagonal.round(2).tolist()
        [0.1, 0.1]

        A single dense matrix has no split:

        >>> gaussx.split_diagonal(kernel) is None
        True
    """
    tags: frozenset[object] = frozenset()
    while isinstance(operator, lx.TaggedLinearOperator):
        tags = tags | operator.tags
        operator = operator.operator
    if not isinstance(operator, lx.AddLinearOperator):
        return None
    structure = operator.in_structure()
    if not isinstance(structure, jax.ShapeDtypeStruct) or len(structure.shape) != 1:
        return None
    if operator.out_structure() != structure:
        return None

    base_parts: list[lx.AbstractLinearOperator] = []
    diagonal_parts: list[Float[Array, " N"]] = []
    for summand in _summands(operator):
        summand_diagonal = _summand_diagonal(summand)
        if summand_diagonal is None:
            base_parts.append(summand)
        else:
            diagonal_parts.append(summand_diagonal)
    if not base_parts or not diagonal_parts:
        return None

    # K = the non-diagonal summands, in their order
    base = ft.reduce(_op.add, base_parts)
    # d = Σ dᵢ over the diagonal summands, in the system's dtype
    diagonal = ft.reduce(_op.add, diagonal_parts).astype(structure.dtype)

    inherited = tags & _INHERITED_TAGS
    missing = frozenset(
        tag
        for tag in inherited
        if not (
            lx.is_positive_semidefinite(base)
            if tag is lx.positive_semidefinite_tag
            else lx.is_symmetric(base)
        )
    )
    if missing:
        base = lx.TaggedLinearOperator(base, missing)
    return DiagonalSplit(base=base, diagonal=diagonal)
