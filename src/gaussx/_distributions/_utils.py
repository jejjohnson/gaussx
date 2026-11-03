"""Reshaping helpers that unflatten the batch and sample axes of distributions.

Distributions flatten their batch shape into one axis, vmap a single-point
routine over it, then restore the shape with these helpers. They are pure
bookkeeping: no arithmetic touches the values.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from gaussx._einx import rearrange


def _axis_names(count: int) -> tuple[str, ...]:
    """Spreadsheet-style axis names for an einx pattern over ``count`` axes.

    Produces ``("a", "b", …, "z", "aa", "ab", …)``, so a pattern can name
    an arbitrary number of batch axes.

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
    """Unflatten one value per batch element back to the batch shape.

    Args:
        values: One value per batch element, flattened in row-major order.
        batch_shape: Target batch shape. When empty, the single entry is
            returned unwrapped.

    Returns:
        The values arranged in ``batch_shape``.
    """
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    return rearrange(values, f"({batch_pattern}) -> {batch_pattern}", **axis_lengths)


def _reshape_series(
    values: Float[Array, "flat T M"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch T M"]:
    """Unflatten one series y_1:T per batch element back to the batch shape.

    The rank-2-event counterpart of `_reshape_samples`, for distributions
    whose ``event_shape`` is a matrix.

    Args:
        values: One series per batch element, flattened in row-major order.
        batch_shape: Target batch shape. When empty, the single series is
            returned unwrapped.

    Returns:
        The series arranged in ``batch_shape``.
    """
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    pattern = f"({batch_pattern}) T M -> {batch_pattern} T M"
    return rearrange(values, pattern, **axis_lengths)


def _reshape_samples(
    values: Float[Array, "flat N"],
    batch_shape: tuple[int, ...],
) -> Float[Array, "*batch N"]:
    """Unflatten one vector x per batch element back to the batch shape.

    Args:
        values: One vector per batch element, flattened in row-major order.
        batch_shape: Target batch shape. When empty, the single vector is
            returned unwrapped.

    Returns:
        The vectors arranged in ``batch_shape``.
    """
    if not batch_shape:
        return values[0]
    batch_axes = _axis_names(len(batch_shape))
    axis_lengths = dict(zip(batch_axes, batch_shape, strict=True))
    batch_pattern = " ".join(batch_axes)
    pattern = f"({batch_pattern}) N -> {batch_pattern} N"
    return rearrange(values, pattern, **axis_lengths)


def _unflatten_sample_axis(
    values: Float[Array, "S ..."],
    sample_shape: tuple[int, ...],
) -> Float[Array, "*sample ..."]:
    """Split a leading axis of ``prod(sample_shape)`` draws into ``sample_shape``.

    Args:
        values: Draws stacked along the leading axis, flattened in row-major
            order.
        sample_shape: Target sample shape. When empty, the leading axis must
            have length one and is dropped.

    Returns:
        The draws with ``sample_shape`` in place of the leading axis.
    """
    if not sample_shape:
        return values[0]
    sample_axes = _axis_names(len(sample_shape))
    axis_lengths = dict(zip(sample_axes, sample_shape, strict=True))
    sample_pattern = " ".join(sample_axes)
    return rearrange(
        values, f"({sample_pattern}) ... -> {sample_pattern} ...", **axis_lengths
    )
