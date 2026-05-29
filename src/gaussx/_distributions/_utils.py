"""Shared helpers for batched distribution reshaping."""

from __future__ import annotations

from jaxtyping import Array, Float

from gaussx._einx import rearrange


def _axis_names(count: int) -> tuple[str, ...]:
    """Generate ``count`` spreadsheet-style axis names for rearrange patterns.

    Produces ``("a", "b", ..., "z", "aa", "ab", ...)`` so that an arbitrary
    number of batch axes can be referenced by name in an einx pattern.

    Args:
        count: Number of axis names to generate.

    Returns:
        Tuple of ``count`` unique lowercase axis names.
    """
    names = []
    for index in range(count):
        value = index
        chars = []
        while True:
            value, remainder = divmod(value, 26)
            chars.append(chr(ord("a") + remainder))
            if value == 0:
                break
            value -= 1
        names.append("".join(reversed(chars)))
    return tuple(names)


def _reshape_batch(
    values: Float[Array, " flat"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch"]:
    """Reshape a flat batch vector back to its original batch shape.

    Args:
        values: Flat values of shape ``(prod(batch_shape),)``.
        batch_shape: Target batch shape. When empty, the single scalar entry
            is returned unwrapped.

    Returns:
        Array reshaped to ``batch_shape``.
    """
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(values, f"({batch_pattern}) -> {batch_pattern}", **axis_lengths)


def _reshape_samples(
    values: Float[Array, "flat N"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch N"]:
    """Reshape flat samples back to their original batch shape.

    Args:
        values: Flat samples of shape ``(prod(batch_shape), N)``.
        batch_shape: Target batch shape. When empty, the single sample is
            returned unwrapped.

    Returns:
        Array reshaped to ``(*batch_shape, N)``.
    """
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    pattern = f"({batch_pattern}) N -> {batch_pattern} N"
    return rearrange(values, pattern, **axis_lengths)
