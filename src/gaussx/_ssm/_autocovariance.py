"""SDE autocovariance utility."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel


def sde_autocovariance(
    kernel: SDEKernel,
    tau: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    r"""Compute the stationary autocovariance of an SDE kernel.

    Evaluates:

        K(\tau) = H \, \exp(F |\tau|) \, P_\infty \, H^T

    Args:
        kernel: An SDE kernel with ``sde_params()`` method.
        tau: Lag values, shape ``(*batch,)``.

    Returns:
        Autocovariance values ``K(tau)``, shape ``(*batch,)``.
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
            f"solve is exactly the fragile step gaussx.discretise_mfd "
            f"exists to avoid, so it is not done implicitly here. Supply "
            f"P_inf on the kernel if you have it; to discretise rather than "
            f"evaluate the autocovariance, use gaussx.discretise_mfd, which "
            f"needs no P_inf."
        )
        raise ValueError(msg)

    def _single_autocov(t: Float[Array, ""]) -> Float[Array, ""]:
        abs_t = jnp.abs(t)
        eF = jsl.expm(params.F * abs_t)
        cov_matrix = params.H @ eF @ params.P_inf @ params.H.T
        return cov_matrix.squeeze()

    orig_shape = tau.shape
    flat_tau = tau.ravel()
    flat_result = jax.vmap(_single_autocov)(flat_tau)
    return flat_result.reshape(orig_shape)
