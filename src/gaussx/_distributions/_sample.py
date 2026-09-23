r"""Structure-aware sampling from multivariate normal distributions.

`sample_mvn` draws $x = \mu + L\varepsilon$ with $LL^{\top} = K$, choosing the
factor $L$ from the structure of the covariance operator so a sample never
costs more than the structure demands.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._kronecker_sum import KroneckerSum, kronecker_sum_sample
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers, sumkronecker_sample
from gaussx._operators._toeplitz import Toeplitz, toeplitz_sample
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._sqrt import _DEFAULT_LANCZOS_ORDER


def sample_mvn(
    mean: Float[Array, "*batch N"],
    covariance: lx.AbstractLinearOperator,
    *,
    key: jax.Array,
    num_samples: int = 1,
) -> Float[Array, "num_samples *batch N"]:
    r"""Draw samples from $\mathcal{N}(\mu, K)$, dispatching on $K$'s structure.

    Every branch draws $x = \mu + L\varepsilon$ with $\varepsilon \sim
    \mathcal{N}(0, I)$ and $LL^{\top} = K$; only the factor $L$ changes:

    | Covariance $K$ | Factor | Cost per sample |
    |---|---|---|
    | `lineax.DiagonalLinearOperator` | $\sqrt{d} \odot z$ | $O(N)$ |
    | `gaussx.Kronecker` | per-factor Cholesky | $O(\sum_i N_i^3)$ once |
    | `gaussx.BlockDiag` | per-block Cholesky | $O(\sum_i B_i^3)$ once |
    | `gaussx.BlockTriDiag` | block-banded Cholesky | $O(T d^3)$ once |
    | `gaussx.Toeplitz` | FFT circulant embedding | $O(N \log N)$ |
    | `gaussx.KroneckerSum` | per-factor eigendecomposition | $O(n_A^3 + n_B^3)$ once |
    | `gaussx.SumOfKroneckers` | Lanczos square root | $\min(50, N)$ matvecs |
    | `gaussx.LowRankUpdate` | $L_B z_1 + U\sqrt{D} z_2$ | $B$'s cost $+ O(Nr)$ |
    | anything else | dense Cholesky | $O(N^3)$ once |

    `lineax.TaggedLinearOperator` wrappers are looked through, and a scalar
    multiple $cK$ (with $c > 0$) samples $K$ and rescales by $\sqrt{c}$, so
    wrapping never costs the structure underneath.

    The Toeplitz, Kronecker-sum and sum-of-Kroneckers branches use a square
    root other than the Cholesky factor, and the sum-of-Kroneckers one is a
    truncated Lanczos approximation (see `gaussx.sumkronecker_sample`). All
    of them target the same distribution; they differ only in which
    $\varepsilon$ maps to which draw.

    A `gaussx.LowRankUpdate` takes the low-rank branch when it is symmetric
    ($V = U$) with non-negative weights $D$. Weights known to be negative -- a
    Woodbury downdate -- fall back to dense Cholesky; weights that are only
    known at run time (under `jax.jit`) take the low-rank branch and raise if
    any is negative.

    Args:
        mean: Mean $\mu$ with shape ``(*batch, N)``. Each batch element gets
            independent noise; the covariance is shared.
        covariance: Positive-definite covariance operator $K$ of size
            ``(N, N)``.
        key: PRNG key.
        num_samples: Number of independent draws.

    Returns:
        Samples of shape ``(num_samples, *batch, N)``.

    Raises:
        ValueError: If ``covariance`` is not square, does not match the last
            axis of ``mean``, or ``num_samples`` is below one.
    """
    mean = jnp.asarray(mean)
    if covariance.in_size() != covariance.out_size():
        raise ValueError(
            "sample_mvn requires a square covariance, got "
            f"{covariance.out_size()}x{covariance.in_size()}."
        )
    if mean.ndim < 1 or mean.shape[-1] != covariance.in_size():
        raise ValueError(
            f"mean must have shape (*batch, {covariance.in_size()}), got {mean.shape}."
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    batch_shape = mean.shape[:-1]
    num_draws = num_samples * math.prod(batch_shape)
    draws = _zero_mean_draws(covariance, key, num_draws)

    axes = [f"b{index}" for index in range(len(batch_shape))]
    pattern = f"(s {' '.join(axes)}) n -> s {' '.join(axes)} n"
    draws = rearrange(draws, pattern, **dict(zip(axes, batch_shape, strict=True)))
    return mean.astype(draws.dtype) + draws


def _zero_mean_draws(
    covariance: lx.AbstractLinearOperator,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    """``num_draws`` samples from ``N(0, covariance)``, stacked row-wise."""
    if isinstance(covariance, lx.TaggedLinearOperator):
        return _zero_mean_draws(covariance.operator, key, num_draws)
    if isinstance(covariance, lx.MulLinearOperator):
        inner = _zero_mean_draws(covariance.operator, key, num_draws)
        return jnp.sqrt(covariance.scalar) * inner
    if isinstance(covariance, lx.DivLinearOperator):
        inner = _zero_mean_draws(covariance.operator, key, num_draws)
        return inner / jnp.sqrt(covariance.scalar)
    if isinstance(covariance, Toeplitz):
        return toeplitz_sample(covariance.column, key=key, num_samples=num_draws)
    if isinstance(covariance, KroneckerSum):
        draws = kronecker_sum_sample(
            covariance.A, covariance.B, key=key, num_samples=num_draws
        )
        return rearrange(draws, "s a b -> s (a b)")
    if isinstance(covariance, SumOfKroneckers):
        # A Krylov space never has more dimensions than the operator.
        order = min(_DEFAULT_LANCZOS_ORDER, covariance.in_size())
        return sumkronecker_sample(
            covariance, key=key, num_samples=num_draws, lanczos_order=order
        )
    if isinstance(covariance, LowRankUpdate) and _has_sampleable_update(covariance):
        return _low_rank_draws(covariance, key, num_draws)

    factor = cholesky(covariance)
    dtype = covariance.in_structure().dtype
    noise = jr.normal(key, (num_draws, covariance.in_size()), dtype=dtype)
    return jax.vmap(factor.mv)(noise)


def _has_sampleable_update(covariance: LowRankUpdate) -> bool:
    """Whether ``B + U D Vᵀ`` can be sampled as ``L_B z₁ + U √D z₂``.

    Needs ``V = U`` -- recorded statically as the symmetric tag -- and weights
    not *known* to be negative. Traced weights are checked at run time in
    `_low_rank_draws`.
    """
    if lx.symmetric_tag not in covariance.tags:
        return False
    try:
        return bool(jnp.all(covariance.d >= 0))
    except jax.errors.TracerBoolConversionError:
        return True


def _low_rank_draws(
    covariance: LowRankUpdate,
    key: jax.Array,
    num_draws: int,
) -> Float[Array, "S N"]:
    r"""Sample ``B + U D Uᵀ`` as a sum of two independent draws.

    $L_B z_1 + U\sqrt{D} z_2$ has covariance $B + UDU^{\top}$, and the base
    $B$ is sampled through `_zero_mean_draws` so its own structure is kept.
    """
    base_key, update_key = jr.split(key)
    base = _zero_mean_draws(covariance.base, base_key, num_draws)
    weights = eqx.error_if(
        covariance.d,
        jnp.any(covariance.d < 0),
        "sample_mvn: LowRankUpdate weights must be non-negative for low-rank "
        "sampling; build the covariance as a dense operator instead.",
    )
    noise = jr.normal(
        update_key, (num_draws, covariance.rank), dtype=covariance.U.dtype
    )
    return base + einsum(noise * jnp.sqrt(weights), covariance.U, "s k, n k -> s n")
