"""Mixed-precision stable squared distances."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from gaussx._einx import einsum, reduce


def stable_squared_distances(
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    *,
    compute_dtype: DTypeLike = jnp.float32,
    accumulate_dtype: DTypeLike | None = None,
) -> Float[Array, "N M"]:
    r"""Squared Euclidean distances with mixed-precision stability.

    The expansion ``||x - z||^2 = ||x||^2 + ||z||^2 - 2 x^T z`` suffers
    catastrophic cancellation when the points are far from the origin
    relative to their separation, producing negative distances and
    non-PSD kernels. Two measures keep it accurate:

    1. Both point sets are centred on their common mean first. Distances
       are translation-invariant, and centring removes the large shared
       offset that the expansion would otherwise cancel. This is what
       protects near-duplicate points, in any dtype.
    2. The norms, the cross term and the subtraction are all computed in
       ``accumulate_dtype``. Widening only the subtraction would not help:
       the digits are already lost in the rounded norms and cross term.

    The result is cast back to ``compute_dtype``.

    Args:
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        compute_dtype: Dtype of the inputs and the result (default
            float32).
        accumulate_dtype: Dtype for the centred norms, cross term and
            subtraction. ``None`` (the default) means the widest float JAX
            has enabled: float64 with x64, float32 without.

    Returns:
        Squared distances, shape ``(N, M)``, guaranteed non-negative.

    Raises:
        ValueError: If ``accumulate_dtype`` is explicitly requested but not
            available, e.g. float64 while ``jax_enable_x64`` is off. JAX
            would otherwise truncate it to float32 with only a warning.
    """
    if accumulate_dtype is None:
        accumulate_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    elif jax.dtypes.canonicalize_dtype(accumulate_dtype) != jnp.dtype(accumulate_dtype):
        raise ValueError(
            f"accumulate_dtype={jnp.dtype(accumulate_dtype)} is not available "
            "(is jax_enable_x64 off?). Pass accumulate_dtype=None to use the "
            "widest available float."
        )

    X_c = X.astype(compute_dtype)
    Z_c = Z.astype(compute_dtype)

    # Centre on the common mean (translation-invariant), then widen.
    centre = reduce(jnp.concatenate([X_c, Z_c]), "N D -> D", "mean")
    X_a = einx.subtract("N D, D -> N D", X_c, centre).astype(accumulate_dtype)
    Z_a = einx.subtract("M D, D -> M D", Z_c, centre).astype(accumulate_dtype)

    X_sq = reduce(X_a**2, "N D -> N", "sum")
    Z_sq = reduce(Z_a**2, "M D -> M", "sum")
    cross = einsum(X_a, Z_a, "N D, M D -> N M")
    dist_sq = einx.add("N, M -> N M", X_sq, Z_sq) - 2.0 * cross

    # Clamp and cast back
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return dist_sq.astype(compute_dtype)
