"""Fast Walsh-Hadamard transform H_d (from kernellib's FastFood, bit-identical).

d is the transform length, a power of two, and H_d the Sylvester-ordered Hadamard
matrix with entries ±1. A sketch of size d in the sibling modules is a different d:
`SRHTSketch` runs this transform at length M₂, the padded input size.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import rearrange


def _is_power_of_two(n: int) -> bool:
    """Whether n = 2^k for an integer k ≥ 0.

    Args:
        n: The integer to test.

    Returns:
        True when n is a power of two.
    """
    return n >= 1 and (n & (n - 1)) == 0


def hadamard_transform(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    """Unnormalized Walsh-Hadamard transform along the last axis.

        H_d x,   H_2h = [H_h, H_h; H_h, −H_h],   H₁ = 1

    Uses log₂ d butterfly passes: O(d log d) work, no d × d matrix. Applying it twice
    returns d x.

    The passes use the constant-geometry (Pease) ordering: every pass reads adjacent
    pairs and writes their sums to the first half, their differences to the second,

        y_j = x_2j + x_2j+1,   y_j+d/2 = x_2j − x_2j+1,   j = 0, …, d/2 − 1

    that is, one pass is (H₂ ⊗ I_d/2) Π with Π the perfect unshuffle. Pass k butterflies
    bit k of the original index, the same bit the in-place Sylvester pass with stride
    2^k uses, and log₂ d unshuffles rotate the index bits back to the identity, so the
    result is H_d x with the same floating-point additions in the same order. As each
    pass has the same shape, the loop is a `jax.lax.fori_loop` whose body is traced
    once, so tracing cost does not grow with log₂ d.

    Algorithm:
        1. Return x when d = 1, since H₁ = 1.
        2. Split x into its even and odd entries (the same reshape every pass).
        3. Write the sums to the first half and the differences to the second.
        4. Repeat steps 2 and 3 for log₂ d passes in total.

    Args:
        x: Array x whose last axis has a power-of-two length d. Leading axes are batch
            axes.

    Returns:
        The transform H_d x, the same shape as ``x``.

    Raises:
        ValueError: If the last axis is not a power of two.

    Examples:
        The first column of H₄, and the 2 × 2 butterfly:

        >>> import jax.numpy as jnp
        >>> from gaussx import hadamard_transform
        >>> hadamard_transform(jnp.array([1.0, 0.0, 0.0, 0.0])).tolist()
        [1.0, 1.0, 1.0, 1.0]
        >>> hadamard_transform(jnp.array([1.0, 2.0])).tolist()
        [3.0, -1.0]

        Leading axes are batch axes, and applying it twice returns d x:

        >>> batch = jnp.stack([jnp.arange(8.0), jnp.ones(8)])
        >>> twice = hadamard_transform(hadamard_transform(batch))
        >>> bool(jnp.allclose(twice, 8.0 * batch))
        True

        A last axis that is not a power of two is rejected:

        >>> hadamard_transform(jnp.ones(3))
        Traceback (most recent call last):
        ...
        ValueError: hadamard_transform needs a power-of-two last axis, got 3.

    References:
        Pease, M. C. (1968). An adaptation of the fast Fourier transform for
        parallel processing. *Journal of the ACM*, 15(2), 252-264.
    """
    size = x.shape[-1]
    if not _is_power_of_two(size):
        raise ValueError(
            f"hadamard_transform needs a power-of-two last axis, got {size}."
        )
    if size == 1:
        # H₁ x = x
        return x
    # (+1, −1): a + (−1) b equals a − b exactly in IEEE arithmetic.
    sign = jnp.asarray([1, -1], dtype=x.dtype)

    def butterfly(_: int, values: Array) -> Array:
        # (x_2j, x_2j+1)
        even_entries, odd_entries = rearrange(values, "... (h two) -> two ... h", two=2)
        # (x_2j+1, −x_2j+1)
        signed_odd = einx.multiply("two, ... h -> ... two h", sign, odd_entries)
        # y_j = x_2j + x_2j+1,   y_j+d/2 = x_2j − x_2j+1
        return einx.add("... h, ... two h -> ... (two h)", even_entries, signed_odd)

    # log₂ d
    num_passes = size.bit_length() - 1
    return jax.lax.fori_loop(0, num_passes, butterfly, x)
