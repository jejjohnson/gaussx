"""Structured Frobenius norm ‖A‖_F, dispatched on the operator type of A.

A is an operator with entries A_ij and ‖A‖_F = √(∑_ij |A_ij|²) its
Frobenius norm, √tr(Aᵀ A) for a real A. A Kronecker product is
A = A_1 ⊗ … ⊗ A_D with factors A_d, d = 1, …, D, and a block-diagonal
operator is blockdiag(A_1, …, A_D). The stochastic estimator averages
‖A z^(i)‖₂² over probes z^(i), i = 1, …, N_probe.
"""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
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
