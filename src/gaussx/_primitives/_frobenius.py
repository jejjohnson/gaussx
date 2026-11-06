"""Structured Frobenius norm ‖A‖_F, dispatched on the operator type of A.

A is an operator with entries A_ij and ‖A‖_F = √(∑_ij |A_ij|²) its
Frobenius norm, √tr(Aᵀ A) for a real A. A Kronecker product is
A = A_1 ⊗ … ⊗ A_D with factors A_d, d = 1, …, D, and a block-diagonal
operator is blockdiag(A_1, …, A_D). A low-rank update is A = A₀ + U D Vᵀ, with
U M × k, V N × k and D = diag(d). The stochastic estimator averages
‖A z^(i)‖₂² over probes z^(i), i = 1, …, N_probe.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys


def frobenius_norm(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> Float[Array, ""]:
    """Frobenius norm ‖A‖_F of an operator, with structural dispatch.

        ‖A‖_F = √(∑_ij |A_ij|²)

    Structured operators avoid materialisation:

    - Diagonal: the vector 2-norm of the diagonal, ‖diag(a)‖_F = ‖a‖₂.
    - `gaussx.BlockDiag`: the root of the sum of squared block norms.
    - `gaussx.Kronecker`: ‖A_1 ⊗ … ⊗ A_D‖_F = ∏_d ‖A_d‖_F.
    - `gaussx.LowRankUpdate`: the norm of A₀ plus a cross term from k matvecs
      with A₀ and the k × k Gram matrices of U and V, so an ensemble covariance
      costs O(k² N) and no N × N matrix is formed. When A₀ nearly cancels the
      update, ‖A‖_F is much smaller than ‖A₀‖_F and this sum of squares loses
      accuracy against the dense norm.
    - Scaled, negated and tagged operators delegate to the wrapped operator.

    With ``stochastic=True`` any other operator gets Hutchinson's estimate
    of ‖A‖_F² = tr(Aᵀ A) through matfree, from matvecs only:

        ‖A‖_F² ≈ (1 / N_probe) ∑_i ‖A z^(i)‖₂²

    Args:
        operator: The operator A.
        stochastic: If ``True``, estimate the norm of any operator that has
            no structural rule from probes instead of materialising it.
        num_probes: Number of probes N_probe for the estimate.
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution for the estimate (``"signs"``,
            ``"normal"``, ``"sphere"``).

    Returns:
        The Frobenius norm ‖A‖_F, exact or estimated.

    Examples:
        A Kronecker product multiplies its factors' norms:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([3.0, 4.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([1.0, 0.0]))
        >>> float(gaussx.frobenius_norm(gaussx.Kronecker(A_1, A_2)))  # 5 · 1
        5.0

        A scalar multiple scales the norm by |c|:

        >>> float(gaussx.frobenius_norm(-2.0 * A_1))
        10.0

        An ensemble covariance never forms its N × N matrix:

        >>> particles = jr.normal(jr.key(1), (5, 2000))
        >>> covariance = gaussx.ensemble_covariance(particles, bessel=True)
        >>> gaussx.frobenius_norm(covariance).shape
        ()

        Sign probes recover the norm of a stored diagonal matrix exactly:

        >>> A = lx.MatrixLinearOperator(jnp.diag(jnp.array([3.0, 4.0])))
        >>> estimate = gaussx.frobenius_norm(
        ...     A, stochastic=True, num_probes=4, key=jr.key(0)
        ... )
        >>> round(float(estimate), 4)
        5.0
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def recurse(
        part: lx.AbstractLinearOperator, part_key: jax.Array | None = key
    ) -> Array:
        return frobenius_norm(
            part,
            stochastic=stochastic,
            num_probes=num_probes,
            key=part_key,
            sampler=sampler,
        )

    def recurse_each(parts) -> list[Array]:
        parts = tuple(parts)
        part_keys = split_keys(key, len(parts))
        return [
            recurse(part, part_key)
            for part, part_key in zip(parts, part_keys, strict=True)
        ]

    if isinstance(operator, lx.IdentityLinearOperator):
        # ‖I‖_F = √N
        return jnp.sqrt(jnp.asarray(float(operator.in_size())))
    if isinstance(operator, lx.DiagonalLinearOperator):
        diagonal = lx.diagonal(operator)
        # |a_i|², so a complex diagonal gets a real norm
        squared_magnitudes = jnp.abs(diagonal) ** 2
        # ‖diag(a)‖_F = √(∑_i |a_i|²)
        return jnp.sqrt(jnp.sum(squared_magnitudes))
    if isinstance(operator, BlockDiag):
        # ‖A_d‖_F, per block
        block_norms = jnp.stack(recurse_each(operator.operators))
        # ‖A_d‖_F²
        squared_norms = block_norms * block_norms
        # ‖blockdiag(A_1, …, A_D)‖_F = √(∑_d ‖A_d‖_F²)
        return jnp.sqrt(jnp.sum(squared_norms))
    if isinstance(operator, Kronecker):
        # ‖A_1 ⊗ … ⊗ A_D‖_F = ∏_d ‖A_d‖_F
        return ft.reduce(jnp.multiply, recurse_each(operator.operators))
    if isinstance(operator, LowRankUpdate):
        return jnp.sqrt(
            _low_rank_squared_norm(operator, _squared_norm(operator.base, recurse))
        )
    if isinstance(operator, lx.TaggedLinearOperator):
        return recurse(operator.operator)
    if isinstance(operator, lx.MulLinearOperator):
        # ‖c A‖_F = |c| ‖A‖_F
        return jnp.abs(operator.scalar) * recurse(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        # ‖A / c‖_F = ‖A‖_F / |c|
        return recurse(operator.operator) / jnp.abs(operator.scalar)
    if isinstance(operator, lx.NegLinearOperator):
        # ‖−A‖_F = ‖A‖_F
        return recurse(operator.operator)
    if stochastic:
        return _frobenius_stochastic(operator, num_probes, key, sampler)
    matrix = operator.as_matrix()
    # |A_ij|², so a complex operator gets a real norm
    squared_magnitudes = jnp.abs(matrix) ** 2
    # ‖A‖_F = √(∑_ij |A_ij|²), dense
    return jnp.sqrt(jnp.sum(squared_magnitudes))


def _squared_norm(
    operator: lx.AbstractLinearOperator,
    norm: Callable[[lx.AbstractLinearOperator], Array],
) -> Float[Array, ""]:
    """‖A‖_F², with no root where the common operators don't need one.

    The root of a zero norm has an infinite slope, so squaring ‖A‖_F gives a NaN
    gradient at A = 0, the base of an ensemble covariance, even when the operator
    A is added to has a nonzero norm.

    Args:
        operator: The operator A.
        norm: The Frobenius norm, for any operator without a rule here.

    Returns:
        ‖A‖_F².
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return jnp.sum(jnp.abs(lx.diagonal(operator)) ** 2)
    if isinstance(operator, lx.MatrixLinearOperator):
        return jnp.sum(jnp.abs(operator.as_matrix()) ** 2)
    if isinstance(operator, BlockDiag):
        squared = [_squared_norm(block, norm) for block in operator.operators]
        return ft.reduce(jnp.add, squared)
    if isinstance(operator, Kronecker):
        squared = [_squared_norm(factor, norm) for factor in operator.operators]
        return ft.reduce(jnp.multiply, squared)
    if isinstance(operator, LowRankUpdate):
        return _low_rank_squared_norm(operator, _squared_norm(operator.base, norm))
    if isinstance(operator, lx.TaggedLinearOperator | lx.NegLinearOperator):
        return _squared_norm(operator.operator, norm)
    if isinstance(operator, lx.MulLinearOperator):
        return jnp.abs(operator.scalar) ** 2 * _squared_norm(operator.operator, norm)
    if isinstance(operator, lx.DivLinearOperator):
        return _squared_norm(operator.operator, norm) / jnp.abs(operator.scalar) ** 2
    root = norm(operator)
    return root * root


def _low_rank_squared_norm(
    operator: LowRankUpdate, base_squared: Float[Array, ""]
) -> Float[Array, ""]:
    """‖A₀ + U D Vᵀ‖_F² from the squared norm of A₀, without forming the update.

        ‖A‖_F² = ‖A₀‖_F² + 2 Re ∑_k d̄_k u_kᴴ A₀ v̄_k + ∑_kl d_k d̄_l (u_lᴴ u_k) (v_lᴴ v_k)

    The Gram matrices are formed after moving each mode's scale into its weight,
    since U and V can have scales whose squares leave the floating-point range
    while u_k d_k v_kᵀ does not. A scale changes nothing about the update, so it
    is a constant to the derivative.

    Args:
        operator: The update A = A₀ + U D Vᵀ.
        base_squared: ‖A₀‖_F².

    Returns:
        ‖A‖_F², clamped at zero.
    """
    # An update with no modes, rows or columns is its base, and einx rejects a
    # zero-length axis.
    if 0 in operator.U.shape or 0 in operator.V.shape:
        return base_squared
    # s_k = max_i |U_ik|, and t_k likewise for V
    left_scale = _mode_scale(operator.U)
    right_scale = _mode_scale(operator.V)
    # U S⁻¹
    left = einx.divide("row mode, mode -> row mode", operator.U, left_scale)
    # V T⁻¹
    right = einx.divide("col mode, mode -> col mode", operator.V, right_scale)
    # D S T
    weights = _scale_in_range(operator.d, left_scale, right_scale)
    # U D
    weighted_left = einx.multiply("row mode, mode -> row mode", left, weights)
    # V D
    weighted_right = einx.multiply("col mode, mode -> col mode", right, weights)
    # Uᴴ U D
    left_gram = einsum(
        jnp.conj(left), weighted_left, "row mode, row other -> mode other"
    )
    # (V D)ᴴ V
    right_gram = einsum(
        jnp.conj(weighted_right), right, "col mode, col other -> mode other"
    )
    # ∑_kl d_k d̄_l (u_lᴴ u_k) (v_lᴴ v_k)
    update_squared = jnp.real(
        einsum(left_gram, right_gram, "mode other, mode other ->")
    )
    # A₀ v̄_k, one column per mode
    base_right = jax.vmap(operator.base.mv, in_axes=1, out_axes=1)(jnp.conj(right))
    # Re ∑_k d̄_k u_kᴴ A₀ v̄_k
    cross = jnp.real(
        einsum(jnp.conj(weighted_left), base_right, "row mode, row mode ->")
    )
    # ‖A‖_F²
    norm_squared = base_squared + 2.0 * cross + update_squared
    # max(‖A‖_F², 0), which rounding could undershoot when A₀ cancels the update
    return jnp.maximum(norm_squared, 0.0)


def _scale_in_range(
    weights: Float[Array, " modes"],
    left_scale: Float[Array, " modes"],
    right_scale: Float[Array, " modes"],
) -> Float[Array, " modes"]:
    """d s t, in an order where a partial product leaves the range only if d s t does.

    Multiplying a large |d| by the smaller scale first, or a small |d| by the larger
    scale first, keeps every partial product between the result and its factors.

    Args:
        weights: The weights d.
        left_scale: The scales s.
        right_scale: The scales t.

    Returns:
        The product d s t.
    """
    smaller = jnp.minimum(left_scale, right_scale)
    larger = jnp.maximum(left_scale, right_scale)
    large_weight = jnp.abs(weights) >= 1.0
    first = jnp.where(large_weight, smaller, larger)
    second = jnp.where(large_weight, larger, smaller)
    return weights * first * second


def _mode_scale(factor: Float[Array, "rows modes"]) -> Float[Array, " modes"]:
    """The largest magnitude in each mode of a factor, with a zero mode as 1.

    Args:
        factor: The low-rank factor, one column per mode.

    Returns:
        A positive scale per mode, constant to the derivative.
    """
    largest = jax.lax.stop_gradient(einx.max("[row] mode", jnp.abs(factor)))
    return jnp.where(largest > 0.0, largest, 1.0)


def _frobenius_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName,
) -> Float[Array, ""]:
    """Hutchinson's estimate of ‖A‖_F = √tr(Aᵀ A) through matfree.

        ‖A‖_F² ≈ (1 / N_probe) ∑_i ‖A z^(i)‖₂²

    Args:
        operator: The operator A, used through matvecs only.
        num_probes: Number of probes N_probe.
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    Returns:
        The estimated norm, clamped at zero before the root.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    size = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, size, num_probes, dtype=operator.in_structure().dtype
    )
    # ‖A z^(i)‖₂²
    integrand = matfree.stochtrace.monte_carlo_frobeniusnorm_squared()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    # ‖A‖_F² ≈ (1 / N_probe) ∑_i ‖A z^(i)‖₂²
    norm_squared = estimate(operator.mv, key)
    # max(‖A‖_F², 0), which a noisy estimate could undershoot
    clamped_norm_squared = jnp.maximum(norm_squared, 0.0)
    # ‖A‖_F
    return jnp.sqrt(clamped_norm_squared)
