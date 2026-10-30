"""Squared Euclidean distances ‖x_n − x′_m‖² that stay accurate far from 0.

X holds N points x_n as rows and X′ holds M points x′_m, each of length D;
x̄ is the mean of all N + M points.
"""

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
    """Squared Euclidean distances between two point sets, centred and widened.

        d²_nm = ‖x_n − x′_m‖² = ‖x_n − x̄‖² + ‖x′_m − x̄‖² − 2 (x_n − x̄)ᵀ (x′_m − x̄)

    The expansion ‖x‖² + ‖x′‖² − 2 xᵀ x′ cancels catastrophically when the
    points sit far from the origin relative to their separation, giving
    negative distances and kernels that are not positive semi-definite.
    Two measures keep it accurate:

    1. Both sets are centred on their common mean x̄ first. Distances do not
       change under translation, and centring removes the large shared
       offset that the expansion would otherwise cancel. This is what
       protects near-duplicate points, in any dtype.
    2. The norms, the cross term and the subtraction all run in
       ``accumulate_dtype``. Widening only the subtraction would not help:
       the digits are already lost in the rounded norms and cross term.

    Algorithm:
        1. Cast both sets to ``compute_dtype`` and centre them on x̄.
        2. Widen to ``accumulate_dtype`` and form ‖x_n − x̄‖², ‖x′_m − x̄‖²
           and the cross term (x_n − x̄)ᵀ (x′_m − x̄).
        3. Combine, clamp at 0 and cast back to ``compute_dtype``.

    Args:
        X: First point set X, rows x_n.
        Z: Second point set X′, rows x′_m.
        compute_dtype: Dtype of the inputs and the result (default
            float32).
        accumulate_dtype: Dtype for the centred norms, cross term and
            subtraction. ``None`` (the default) means the widest float JAX
            has enabled: float64 with x64, float32 without.

    Returns:
        The squared distances d²_nm, non-negative, in ``compute_dtype``.

    Raises:
        ValueError: If ``accumulate_dtype`` is explicitly requested but not
            available, e.g. float64 while ``jax_enable_x64`` is off. JAX
            would otherwise truncate it to float32 with only a warning.

    Examples:
        Two points against three, far from the origin:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> X = 1e4 + jnp.array([[0.0, 0.0], [1.0, 0.0]])
        >>> Z = 1e4 + jnp.array([[0.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        >>> distances = gaussx.stable_squared_distances(X, Z)
        >>> distances.shape
        (2, 3)
        >>> bool(jnp.allclose(distances, jnp.array([[0, 4, 2], [1, 5, 1]])))
        True
    """
    if accumulate_dtype is None:
        accumulate_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
    elif jax.dtypes.canonicalize_dtype(accumulate_dtype) != jnp.dtype(accumulate_dtype):
        raise ValueError(
            f"accumulate_dtype={jnp.dtype(accumulate_dtype)} is not available "
            "(is jax_enable_x64 off?). Pass accumulate_dtype=None to use the "
            "widest available float."
        )

    points = X.astype(compute_dtype)
    other_points = Z.astype(compute_dtype)

    # --- centre, then widen ---
    # x̄ = mean over the rows of [X; X′]
    all_points = jnp.concatenate([points, other_points])
    centre = reduce(all_points, "sample feature -> feature", "mean")
    # x_n − x̄
    centred = einx.subtract("sample feature, feature -> sample feature", points, centre)
    centred = centred.astype(accumulate_dtype)
    # x′_m − x̄
    other_centred = einx.subtract(
        "other feature, feature -> other feature", other_points, centre
    )
    other_centred = other_centred.astype(accumulate_dtype)

    # --- expand the squared distance ---
    # ‖x_n − x̄‖²
    squared_norms = reduce(centred**2, "sample feature -> sample", "sum")
    # ‖x′_m − x̄‖²
    other_squared_norms = reduce(other_centred**2, "other feature -> other", "sum")
    # (x_n − x̄)ᵀ (x′_m − x̄)
    cross = einsum(
        centred, other_centred, "sample feature, other feature -> sample other"
    )
    # ‖x_n − x̄‖² + ‖x′_m − x̄‖²
    norm_sum = einx.add(
        "sample, other -> sample other", squared_norms, other_squared_norms
    )
    # d²_nm = ‖x_n − x̄‖² + ‖x′_m − x̄‖² − 2 (x_n − x̄)ᵀ (x′_m − x̄)
    squared_distances = norm_sum - 2.0 * cross

    # Round-off can still leave tiny negatives; clamp, then cast back.
    squared_distances = jnp.maximum(squared_distances, 0.0)
    return squared_distances.astype(compute_dtype)
