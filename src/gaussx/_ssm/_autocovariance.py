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
    params = kernel.sde_params()
    if params.P_inf is None:
        msg = (
            f"{type(kernel).__name__} has no stationary covariance "
            f"(sde_params().P_inf is None), so its stationary "
            f"autocovariance is undefined. This is expected for a kernel "
            f"with a learned drift; discretise it with "
            f"gaussx.discretise_mfd instead."
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
