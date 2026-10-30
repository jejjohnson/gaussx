"""Autocovariance k(τ) of a stationary SDE kernel, from its state-space form.

For dx(t) = F_c x(t) dt + L_c db(t), f(t) = H x(t), with drift matrix F_c,
diffusion (noise-effect) matrix L_c, white-noise spectral density Q_c,
stationary covariance Σ_x,∞ and measurement row H, the kernel is the
covariance of f at lag τ: the measured stationary covariance propagated by
the transition F_t = exp(F_c |τ|).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._ssm._sde_kernel import SDEKernel


def sde_autocovariance(
    kernel: SDEKernel,
    tau: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    """Stationary autocovariance k(τ) of an SDE kernel at each lag τ.

        k(τ) = H exp(F_c |τ|) Σ_x,∞ Hᵀ

    Args:
        kernel: A stationary `gaussx.SDEKernel` with a closed-form Σ_x,∞.
        tau: Lags τ, of any shape.

    Returns:
        Autocovariance k(τ), shaped like ``tau``.

    Raises:
        ValueError: If the kernel is not stationary, or reports no
            stationary covariance (``sde_params().P_inf is None``).

    Examples:
        A Matérn-1/2 kernel, k(τ) = σ_k² exp(−|τ|/ℓ):

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.MaternSDE(
        ...     variance=jnp.asarray(2.0), lengthscale=jnp.asarray(0.5), order=0
        ... )
        >>> lags = jnp.array([[0.0, 0.5], [-0.5, 1.0]])
        >>> got = gaussx.sde_autocovariance(kernel, lags)
        >>> got.shape
        (2, 2)
        >>> bool(jnp.allclose(got, 2.0 * jnp.exp(-jnp.abs(lags) / 0.5), atol=1e-5))
        True

        A non-stationary kernel has no autocovariance:

        >>> trend = gaussx.IntegratedWienerSDE(diffusion=jnp.asarray(1.0))
        >>> gaussx.sde_autocovariance(
        ...     trend, lags
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: IntegratedWienerSDE is not stationary
    """
    if not kernel.stationary:
        msg = (
            f"{type(kernel).__name__} is not stationary, so it has no "
            f"autocovariance K(tau): its covariance depends on both times "
            f"rather than on their difference. Evaluate the covariance "
            f"through the discretised recursion instead."
        )
        raise ValueError(msg)

    params = kernel.sde_params()
    if params.P_inf is None:
        msg = (
            f"{type(kernel).__name__} does not supply a stationary "
            f"covariance (sde_params().P_inf is None), which this "
            f"autocovariance is defined in terms of. Note P_inf=None means "
            f"'no closed form', not necessarily 'not stationary': a Hurwitz "
            f"drift does have a stationary covariance, recoverable as the "
            f"solution of F P + P F^T + L Q_c L^T = 0, but that Lyapunov "
            f"solve is exactly the fragile step gaussx.discretize_mfd "
            f"exists to avoid, so it is not done implicitly here. Supply "
            f"P_inf on the kernel if you have it; to discretise rather than "
            f"evaluate the autocovariance, use gaussx.discretize_mfd, which "
            f"needs no P_inf."
        )
        raise ValueError(msg)
    stationary_covariance = params.P_inf

    def _single_autocov(lag: Float[Array, ""]) -> Float[Array, ""]:
        # |τ|
        abs_lag = jnp.abs(lag)
        # F_t = exp(F_c |τ|)
        transition = jsl.expm(params.F * abs_lag)
        # H F_t
        measured_transition = einsum(
            params.H, transition, "obs state, state col -> obs col"
        )
        # H F_t Σ_x,∞
        measured_propagated = einsum(
            measured_transition,
            stationary_covariance,
            "obs state, state col -> obs col",
        )
        # k(τ) = H F_t Σ_x,∞ Hᵀ
        covariance = einsum(
            measured_propagated, params.H, "obs col, other col -> obs other"
        )
        return covariance.squeeze()

    batch_shape = tau.shape
    # ravel/reshape, not einx: einx cannot infer the size of an empty τ axis.
    flat_tau = tau.ravel()
    flat_autocovariance = jax.vmap(_single_autocov)(flat_tau)
    return flat_autocovariance.reshape(batch_shape)
