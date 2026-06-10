"""Typed einx adapters for tensor reshaping and contraction.

einx (https://github.com/fferflo/einx) is the tensor-manipulation backend
for gaussx. These thin wrappers expose einops-compatible signatures with
permissive typing, so the rest of the package can reshape and contract
arrays without tripping the type checker on einx's internal ``Tensor``
annotations. Import these helpers instead of using ``einx`` (or ``einops``)
directly.
"""

from __future__ import annotations

from typing import Any

import einx
from jaxtyping import Array


def rearrange(tensor: Array, pattern: str, **axes: int) -> Array:
    """Reorder/reshape ``tensor`` according to an einx pattern.

    einops-compatible wrapper around `einx.id`.

    Args:
        tensor: Input array.
        pattern: einx rearrange pattern, e.g. ``"a b -> b a"``.
        **axes: Named axis lengths referenced in ``pattern``.

    Returns:
        The rearranged array.
    """
    return einx.id(pattern, tensor, **axes)  # ty: ignore[invalid-argument-type]


def repeat(tensor: Array, pattern: str, **axes: int) -> Array:
    """Broadcast/repeat ``tensor`` along new axes via an einx pattern.

    einops-compatible wrapper around `einx.id`.

    Args:
        tensor: Input array.
        pattern: einx pattern introducing new axes, e.g. ``"a -> a r"``.
        **axes: Named lengths for the introduced axes.

    Returns:
        The expanded array.
    """
    return einx.id(pattern, tensor, **axes)  # ty: ignore[invalid-argument-type]


def reduce(tensor: Array, pattern: str, reduction: str, **axes: int) -> Array:
    """Reduce ``tensor`` along axes dropped by an einx pattern.

    einops-compatible wrapper that dispatches to the matching einx reduction
    (``einx.sum``, ``einx.prod``, ``einx.mean``, ...).

    Args:
        tensor: Input array.
        pattern: einx reduction pattern, e.g. ``"a b -> a"``.
        reduction: Reduction name (``"sum"``, ``"prod"``, ``"mean"``, ...).
        **axes: Named axis lengths referenced in ``pattern``.

    Returns:
        The reduced array.
    """
    op = getattr(einx, reduction)
    return op(pattern, tensor, **axes)


def einsum(*operands: Any) -> Array:
    """Contract tensors via an einx pattern (``einops.einsum``-compatible).

    The trailing positional argument is the pattern; all preceding arguments
    are the input tensors. Wraps `einx.dot`.

    Args:
        *operands: Input tensors followed by the einx pattern string, e.g.
            ``einsum(a, b, "i j, j k -> i k")``.

    Returns:
        The contracted array.
    """
    *tensors, pattern = operands
    return einx.dot(pattern, *tensors)
