"""EP tilted moment computation via Gauss-Hermite quadrature."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def ep_tilted_moments(
    log_lik_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    cav_mean: Float[Array, " *batch"],
    cav_var: Float[Array, " *batch"],
    *,
    order: int = 20,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    r"""Compute tilted distribution moments via Gauss-Hermite quadrature.

    Args:
        log_lik_fn: Scalar function mapping latent value ``f`` to scalar
            log-likelihood ``log p(y|f)``.
        cav_mean: Cavity means, shape ``(*batch,)``.
        cav_var: Cavity variances (positive), shape ``(*batch,)``.
        order: Number of Gauss-Hermite quadrature points. Default 20.

    Returns:
        Tuple ``(tilted_mean, tilted_var)``.
    """
    from gaussx._quadrature._quadrature import gauss_hermite_points

    z, w = gauss_hermite_points(order, dim=1)
    z = z.squeeze(-1)
    log_w = jnp.log(w)

    def _compute_moments(mean_i: Float[Array, ""], var_i: Float[Array, ""]):
        std_i = jnp.sqrt(var_i)
        f_nodes = mean_i + std_i * z

        log_lik_vals = jax.vmap(log_lik_fn)(f_nodes)
        log_joint = log_w + log_lik_vals

        log_Z = jax.scipy.special.logsumexp(log_joint)
        weights = jnp.exp(log_joint - log_Z)

        t_mean = jnp.sum(weights * f_nodes)
        t_var = jnp.sum(weights * (f_nodes - t_mean) ** 2)
        t_var = jnp.maximum(t_var, 1e-10)
        return t_mean, t_var

    orig_shape = cav_mean.shape
    flat_mean = cav_mean.ravel()
    flat_var = cav_var.ravel()

    flat_t_mean, flat_t_var = jax.vmap(_compute_moments)(flat_mean, flat_var)

    tilted_mean = flat_t_mean.reshape(orig_shape)
    tilted_var = flat_t_var.reshape(orig_shape)
    return tilted_mean, tilted_var
