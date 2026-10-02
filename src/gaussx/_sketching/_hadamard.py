"""Fast Walsh-Hadamard transform (moved unchanged from kernellib's FastFood)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import rearrange


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def hadamard_transform(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    r"""Unnormalized Walsh-Hadamard transform along the last axis.

    Computes $H_d x$ for the Sylvester-ordered Hadamard matrix
    $H_{2m} = \begin{pmatrix} H_m & H_m \\ H_m & -H_m \end{pmatrix}$,
    $H_1 = 1$, with $\log_2 d$ butterfly passes: $O(d \log d)$ work, no
    $d \times d$ matrix. Applying it twice returns $d\,x$.

    Args:
        x: Array whose last axis has a power-of-two length ``d``. Leading axes
            are batch axes.

    Returns:
        $H_d x$, same shape as ``x``.

    Raises:
        ValueError: If the last axis is not a power of two.

    Examples:
        >>> import jax.numpy as jnp
        >>> from gaussx import hadamard_transform
        >>> hadamard_transform(jnp.array([1.0, 0.0, 0.0, 0.0])).tolist()
        [1.0, 1.0, 1.0, 1.0]
        >>> hadamard_transform(jnp.array([1.0, 2.0])).tolist()
        [3.0, -1.0]
    """
    d = x.shape[-1]
    if not _is_power_of_two(d):
        raise ValueError(f"hadamard_transform needs a power-of-two last axis, got {d}.")
    h = 1
    while h < d:
        # Pair entry i with entry i + h inside each block of 2h.
        y = rearrange(x, "... (m two h) -> ... m two h", two=2, h=h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = rearrange(
            jnp.stack([a + b, a - b], axis=-2), "... m two h -> ... (m two h)"
        )
        h *= 2
    return x
