"""Mixed-precision stable squared distances."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import reduce


def stable_squared_distances(
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    accumulate_dtype: jnp.dtype = jnp.float64,
) -> Float[Array, "N M"]:
    r"""Squared Euclidean distances with mixed-precision stability.

    The expansion ``||x - z||^2 = ||x||^2 + ||z||^2 - 2 x^T z`` suffers
    catastrophic cancellation in float32 for high-D data, producing
    negative distances and non-PSD kernels.

    This function computes dot products in ``compute_dtype`` (fast) and
    performs the subtraction in ``accumulate_dtype`` (stable), then casts
    the result back to ``compute_dtype``.

    Args:
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        compute_dtype: Dtype for dot products (default float32).
        accumulate_dtype: Dtype for subtraction (default float64).

    Returns:
        Squared distances, shape ``(N, M)``, guaranteed non-negative.
    """
    X_c = X.astype(compute_dtype)
    Z_c = Z.astype(compute_dtype)

    # Squared norms — computed in compute_dtype
    X_sq = reduce(X_c**2, "N D -> N", "sum")
    Z_sq = reduce(Z_c**2, "M D -> M", "sum")

    # Cross term — computed in compute_dtype
    cross = X_c @ Z_c.T  # (N, M)

    # Subtraction in accumulate_dtype for stability
    dist_sq = (
        X_sq[:, None].astype(accumulate_dtype)
        + Z_sq[None, :].astype(accumulate_dtype)
        - 2.0 * cross.astype(accumulate_dtype)
    )

    # Clamp and cast back
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return dist_sq.astype(compute_dtype)
