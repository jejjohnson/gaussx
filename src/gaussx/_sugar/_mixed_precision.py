"""Mixed-precision stable squared distances and RBF kernel."""

from __future__ import annotations

import jax.numpy as jnp
from einops import reduce
from jaxtyping import Array, Float


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


def stable_rbf_kernel(
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    lengthscale: float | Float[Array, ""],
    variance: float | Float[Array, ""] = 1.0,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    accumulate_dtype: jnp.dtype = jnp.float64,
) -> Float[Array, "N M"]:
    r"""RBF (squared exponential) kernel with mixed-precision stability.

    Computes ``variance * exp(-0.5 * ||x - z||^2 / lengthscale^2)``
    using :func:`stable_squared_distances` for the distance computation.

    Args:
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        lengthscale: Kernel lengthscale.
        variance: Kernel signal variance (default 1.0).
        compute_dtype: Dtype for dot products (default float32).
        accumulate_dtype: Dtype for subtraction (default float64).

    Returns:
        Kernel matrix, shape ``(N, M)``.
    """
    dist_sq = stable_squared_distances(
        X, Z, compute_dtype=compute_dtype, accumulate_dtype=accumulate_dtype
    )
    return variance * jnp.exp(-0.5 * dist_sq / lengthscale**2)
