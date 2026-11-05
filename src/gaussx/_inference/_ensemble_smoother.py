"""Backward smoothing of an ensemble in ensemble space."""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._inference._ensemble_stats import _check_ensemble_size


def ensemble_rts_step(
    smoothed_next: Float[Array, "J N"],
    analysis: Float[Array, "J N"],
    forecast_next: Float[Array, "J N"],
    *,
    rcond: float | None = None,
) -> Float[Array, "J N"]:
    """One backward step of the ensemble Rauch-Tung-Striebel smoother.

        X_t^s = X_t^a + D Fᵀ (F Fᵀ)⁺ A = X_t^a + D F⁺ A

    where A and F are the anomalies of X_t^a and X_t+1^f, with members as rows,
    and D = X_t+1^s − X_t+1^f. This applies the smoother gain Σ_af Σ_ff⁺ of
    `rts_smoother` to each member, and no square matrix larger than the smaller
    of |ℰ| and D_x is formed.

    F e = 0 for the constant vector e, so F has a singular value at rounding level
    that the pseudo-inverse has to drop. F⁺ comes from the singular values of F, not
    from F Fᵀ, so that a direction only 10⁻⁴ of the largest stays resolved in
    float32. Its derivative has the closed form of Golub & Pereyra (1973), which
    divides by no difference of singular values, so repeated or zero singular values
    keep finite gradients. It holds while the rank is fixed, and a singular value
    that crosses the cutoff changes X_t^s discontinuously.

    Algorithm:
        1. Centre X_t^a to get A, and X_t+1^f twice to get F.
        2. Form D = X_t+1^s − X_t+1^f.
        3. Contract D F⁺, then apply the result to A.

    Args:
        smoothed_next: Smoothed ensemble X_t+1^s, shape (|ℰ|, D_x).
        analysis: Analysis ensemble X_t^a, shape (|ℰ|, D_x).
        forecast_next: Forecast ensemble X_t+1^f propagated from ``analysis``,
            shape (|ℰ|, D_x). Members must correspond across all three ensembles.
        rcond: Singular values of F at or below ``rcond`` times the largest are
            dropped. ``None`` means |ℰ| times the machine epsilon of the dtype.

    Returns:
        The smoothed ensemble X_t^s, shape (|ℰ|, D_x).

    Raises:
        ValueError: If |ℰ| < 2, or if the three ensembles differ in shape.

    Examples:
        A smoothed ensemble that equals the forecast leaves the analysis alone:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> analysis = jr.normal(jr.key(0), (6, 2))
        >>> forecast = 0.9 * analysis + 0.1 * jr.normal(jr.key(1), (6, 2))
        >>> same = gaussx.ensemble_rts_step(forecast, analysis, forecast)
        >>> bool(jnp.allclose(same, analysis))
        True

        A shift in the smoothed ensemble reaches the earlier time through the
        ensemble's covariances:

        >>> smoothed = forecast + jnp.array([0.5, 0.0])
        >>> gaussx.ensemble_rts_step(smoothed, analysis, forecast).shape
        (6, 2)

    References:
        Evensen & van Leeuwen (2000), *Mon. Wea. Rev.* 128, 1852-1867.
        Golub & Pereyra (1973), *SIAM J. Numer. Anal.* 10, 413-432.
    """
    ensemble_size = smoothed_next.shape[0]
    _check_ensemble_size(ensemble_size, True)
    if not smoothed_next.shape == analysis.shape == forecast_next.shape:
        raise ValueError(
            "smoothed_next, analysis and forecast_next must have the same shape, "
            f"got {smoothed_next.shape}, {analysis.shape} and {forecast_next.shape}."
        )
    # A = X_t^a − x̄_t^a
    analysis_anomalies = analysis - einx.mean("[member] state", analysis)
    # X_t+1^f − x̄_t+1^f
    forecast_anomalies = forecast_next - einx.mean("[member] state", forecast_next)
    # F. Rounding in a mean far from zero leaves F e large enough to pass the
    # cutoff and come back through F⁺, so take the anomalies' own mean out as well.
    forecast_anomalies = forecast_anomalies - einx.mean(
        "[member] state", forecast_anomalies
    )
    # D = X_t+1^s − X_t+1^f
    discrepancy = smoothed_next - forecast_next
    tolerance = (
        ensemble_size * jnp.finfo(forecast_anomalies.dtype).eps
        if rcond is None
        else rcond
    )
    # F⁺
    forecast_pinv = jnp.linalg.pinv(forecast_anomalies, tolerance)
    # D F⁺
    member_weights = einsum(
        discrepancy, forecast_pinv, "member state, state other -> member other"
    )
    # D F⁺ A
    increment = einsum(
        member_weights, analysis_anomalies, "member other, other state -> member state"
    )
    # X_t^s = X_t^a + D F⁺ A
    return analysis + increment
