"""Exact zero-mean Gaussian draws L z for the factors two layers share.

`gaussx.sample_mvn` (Layer 2) and the preconditioners' ``sample`` (Layer 1.5)
both draw from a diagonal Σ = diag(σ²) and from a low-rank update
Σ = A₀ + U D Uᵀ. The draws live here, below both, so the preconditioners need
not import the distribution layer, which imports the strategies that import
them. z ~ 𝒩(0, I) throughout, one row per draw.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._primitives._cholesky import cholesky


def factor_draws(
    factor: lx.AbstractLinearOperator,
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "draw N"]:
    """Draws L z for ``num_draws`` standard-normal z.

        L z ~ 𝒩(0, L Lᵀ)

    Args:
        factor: Factor L, with L Lᵀ = Σ.
        covariance: Covariance Σ, which sets the size and dtype of z.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws L z, one row per draw.
    """
    dtype = covariance.in_structure().dtype
    # z ~ 𝒩(0, I), one row per draw
    noise = jr.normal(key, (num_draws, covariance.in_size()), dtype=dtype)
    # L z
    return jax.vmap(factor.mv)(noise)


def diagonal_draws(
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "draw N"]:
    """Draws σ ⊙ z from a diagonal Σ = diag(σ²).

        L = diag(σ) = Σ^½

    Args:
        covariance: A diagonal or identity operator, possibly tagged or
            scaled, that `gaussx.cholesky` factors as a diagonal.
        key: PRNG key.
        num_draws: Number of draws.

    Returns:
        The draws σ ⊙ z, one row per draw.
    """
    # L = diag(σ)
    factor = cholesky(covariance)
    return factor_draws(factor, covariance, key, num_draws)


def low_rank_update_draws(
    covariance: LowRankUpdate,
    key: jax.Array,
    num_draws: int,
    base_draws: Callable[
        [lx.AbstractLinearOperator, jax.Array, int], Float[Array, "draw N"]
    ],
) -> Float[Array, "draw N"]:
    """Sample Σ = A₀ + U D Uᵀ as a sum of two independent draws.

        L_A₀ z₁ + U D^½ z₂ ~ 𝒩(0, A₀ + U D Uᵀ)

    A weight d_j = 0 contributes nothing, and its root has a zero gradient
    rather than an infinite one.

    Args:
        covariance: Low-rank update Σ = A₀ + U D Uᵀ with V = U and D ≥ 0.
        key: PRNG key.
        num_draws: Number of draws.
        base_draws: ``(A₀, key, num_draws) -> L_A₀ z₁``, the sampler of the
            base, so the caller decides which base structures it handles.

    Returns:
        The draws, one row per draw.
    """
    base_key, update_key = jr.split(key)
    # L_A₀ z₁
    base = base_draws(covariance.base, base_key, num_draws)
    # z₂ ~ 𝒩(0, I_k), one row per draw
    noise = jr.normal(
        update_key, (num_draws, covariance.rank), dtype=covariance.U.dtype
    )
    weights = covariance.d
    # Double where: √0 has an infinite derivative, so the root of a zero
    # weight is taken at 1 and replaced by 0.
    positive = weights > 0
    safe_weights = jnp.where(positive, weights, 1.0)
    # D^½
    root_weights = jnp.where(positive, jnp.sqrt(safe_weights), 0.0)
    # D^½ z₂
    scaled = einx.multiply("draw rank, rank -> draw rank", noise, root_weights)
    # U D^½ z₂
    update = einsum(scaled, covariance.U, "draw rank, row rank -> draw row")
    # L_A₀ z₁ + U D^½ z₂
    return base + update
