"""Fast Walsh-Hadamard transform (from kernellib's FastFood, bit-identical)."""

from __future__ import annotations

import einx
import jax
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

    The passes use the constant-geometry (Pease) ordering: every pass reads
    adjacent pairs and writes their sums to the first half, their
    differences to the second,

    $$
    y_j = x_{2j} + x_{2j+1}, \qquad y_{j + d/2} = x_{2j} - x_{2j+1},
    \qquad j = 0, \dots, d/2 - 1,
    $$

    i.e. one pass is $(H_2 \otimes I_{d/2})\,\Pi$ with $\Pi$ the perfect
    unshuffle. Pass $k$ butterflies bit $k$ of the original index, the same
    bit the in-place Sylvester pass with stride $2^k$ uses, and $\log_2 d$
    unshuffles rotate the index bits back to the identity, so the result is
    $H_d x$ with the same floating-point additions in the same order. As each
    pass has the same shape, the loop is a `jax.lax.fori_loop` whose body is
    traced once, so tracing cost does not grow with $\log_2 d$.

    ```text
    for k in 0 .. log2(d) - 1:          # same reshape every pass
        a, b = x[..., 0::2], x[..., 1::2]
        x = concat(a + b, a - b)
    return x
    ```

    References:
        Pease, M. C. (1968). An adaptation of the fast Fourier transform for
        parallel processing. *Journal of the ACM*, 15(2), 252-264.

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
    if d == 1:
        return x
    # (+1, -1): a + (-1) b equals a - b exactly in IEEE arithmetic.
    sign = jnp.asarray([1, -1], dtype=x.dtype)

    def butterfly(_: int, x: Array) -> Array:
        a, b = rearrange(x, "... (h two) -> two ... h", two=2)
        signed_b = einx.multiply("two, ... h -> ... two h", sign, b)
        return einx.add("... h, ... two h -> ... (two h)", a, signed_b)

    return jax.lax.fori_loop(0, d.bit_length() - 1, butterfly, x)
