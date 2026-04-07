"""Quadrature rules: Gauss-Hermite, sigma points, cubature points."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
from einops import rearrange, reduce
from jaxtyping import Array, Float

from gaussx._primitives._sqrt import sqrt


def gauss_hermite_points(
    order: int,
    dim: int,
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    r"""Gauss-Hermite quadrature points and weights.

    Generates tensor-product Gauss-Hermite quadrature points for
    integrating functions against a standard Gaussian measure
    (probabilists' Hermite polynomials).
    ``P = order^dim`` total points.

    Args:
        order: Number of quadrature points per dimension.
        dim: Dimensionality of the integration domain.

    Returns:
        Tuple ``(points, weights)`` where points has shape
        ``(order^dim, dim)`` and weights has shape ``(order^dim,)``.
    """
    # 1D probabilists' Gauss-Hermite (weight = exp(-x^2/2))
    x1d_np, w1d_np = np.polynomial.hermite_e.hermegauss(order)
    x1d = jnp.array(x1d_np)
    w1d = jnp.array(w1d_np)

    if dim == 1:
        return x1d[:, None], w1d

    # Tensor product via meshgrid
    grids = jnp.meshgrid(*([x1d] * dim), indexing="ij")
    stacked = jnp.stack(grids, axis=0)  # (dim, *grid_shape)
    points = rearrange(stacked, "D ... -> (...) D")

    weight_grids = jnp.meshgrid(*([w1d] * dim), indexing="ij")
    weight_stack = jnp.stack(weight_grids, axis=0)  # (dim, *grid_shape)
    weights = reduce(weight_stack, "D ... -> (...)", "prod")

    return points, weights


def sigma_points(
    mean: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
    r"""Unscented transform sigma points and weights.

    Generates ``2N+1`` deterministic sigma points for a Gaussian with
    the given mean and covariance, using the scaled unscented transform.

    Uses ``gaussx.sqrt(cov)`` for structured square root dispatch.

    Args:
        mean: Mean vector, shape ``(N,)``.
        cov: Covariance operator, shape ``(N, N)``.
        alpha: Spread parameter. Controls how far sigma points are
            from the mean. Default ``1e-3``.
        beta: Prior distribution parameter. ``beta=2`` is optimal for
            Gaussians. Default ``2.0``.
        kappa: Secondary scaling parameter. Default ``0.0``.

    Returns:
        Tuple ``(chi, w_m, w_c)`` where:
        - ``chi``: Sigma points, shape ``(2N+1, N)``.
        - ``w_m``: Mean weights, shape ``(2N+1,)``.
        - ``w_c``: Covariance weights, shape ``(2N+1,)``.
    """
    N = mean.shape[0]
    lam = alpha**2 * (N + kappa) - N
    c = N + lam

    # Matrix square root: S where cov = S S^T
    S = sqrt(cov)
    S_mat = S.as_matrix()  # (N, N)
    S_scaled = jnp.sqrt(c) * S_mat

    # Sigma points: chi_0 = mu, chi_i = mu + S_i, chi_{N+i} = mu - S_i
    chi_0 = mean[None, :]  # (1, N)
    chi_plus = mean[None, :] + S_scaled.T  # (N, N) — each row is a point
    chi_minus = mean[None, :] - S_scaled.T  # (N, N)
    chi = jnp.concatenate([chi_0, chi_plus, chi_minus], axis=0)  # (2N+1, N)

    # Mean weights
    w_m_0 = lam / c
    w_m_i = 1.0 / (2.0 * c)
    w_m = jnp.concatenate(
        [
            jnp.array([w_m_0]),
            jnp.full(2 * N, w_m_i),
        ]
    )

    # Covariance weights
    w_c_0 = lam / c + (1.0 - alpha**2 + beta)
    w_c = jnp.concatenate(
        [
            jnp.array([w_c_0]),
            jnp.full(2 * N, w_m_i),
        ]
    )

    return chi, w_m, w_c


def cubature_points(
    mean: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "P N"], Float[Array, " P"]]:
    r"""Spherical-radial cubature points and weights.

    Generates ``2N`` cubature points with equal weights ``1/(2N)``.
    This is the cubature Kalman filter (CKF) point set.

    Uses ``gaussx.sqrt(cov)`` for structured square root dispatch.

    Args:
        mean: Mean vector, shape ``(N,)``.
        cov: Covariance operator, shape ``(N, N)``.

    Returns:
        Tuple ``(chi, weights)`` where:
        - ``chi``: Cubature points, shape ``(2N, N)``.
        - ``weights``: Equal weights, shape ``(2N,)``.
    """
    N = mean.shape[0]

    S = sqrt(cov)
    S_mat = S.as_matrix()
    S_scaled = jnp.sqrt(N) * S_mat  # (N, N)

    chi_plus = mean[None, :] + S_scaled.T  # (N, N)
    chi_minus = mean[None, :] - S_scaled.T  # (N, N)
    chi = jnp.concatenate([chi_plus, chi_minus], axis=0)  # (2N, N)

    weights = jnp.full(2 * N, 1.0 / (2.0 * N))

    return chi, weights
