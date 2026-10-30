"""Selected inverse: the band of Σ = Λ⁻¹ for a block-tridiagonal precision Λ.

Λ is a symmetric positive-definite block-tridiagonal precision with N_block
diagonal blocks of size d × d, indexed k = 1, …, N_block. Its block Cholesky
factor Λ = L Lᵀ is lower block-bidiagonal, with diagonal blocks L_k and
sub-diagonal blocks B_k in block position (k+1, k). Σ_k,k and Σ_k+1,k are the
diagonal and first sub-diagonal blocks of the covariance Σ = Λ⁻¹.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._block_tridiag import BlockTriDiag, LowerBlockTriDiag
from gaussx._primitives._cholesky import _cholesky_block_tridiag


def selected_inverse(operator: BlockTriDiag) -> BlockTriDiag:
    """The band of Σ = Λ⁻¹ for a symmetric positive-definite `BlockTriDiag`.

        Σ_k+1,k = −Σ_k+1,k+1 B_k L_k⁻¹
        Σ_k,k = L_k⁻ᵀ (L_k⁻¹ − B_kᵀ Σ_k+1,k)

    Takahashi's recursion in block form, which is the RTS smoother written in
    precision form. The identity Lᵀ Σ = L⁻¹ gives the recursion above,
    swept backwards from Σ_N,N = L_N⁻ᵀ L_N⁻¹ with N = N_block. Only the
    diagonal and first off-diagonal blocks are computed, at O(N_block d³)
    cost in a ``lax.scan``; with d = 1 this is Rue & Held's scalar
    recursion. `gaussx.diag_inv` takes the marginal variances from the
    diagonal blocks; the off-diagonal blocks are the lag-one covariances.

    Algorithm:
        1. Factor Λ = L Lᵀ block by block.
        2. Invert each diagonal block, L_k⁻¹, by a triangular solve.
        3. Start from the last block, Σ_N,N = L_N⁻ᵀ L_N⁻¹.
        4. Sweep backwards: Σ_k+1,k, then Σ_k,k, symmetrised.

    Args:
        operator: Symmetric positive-definite block-tridiagonal precision Λ.

    Returns:
        A `BlockTriDiag` holding Σ_k,k and Σ_k+1,k. It is the band of Λ⁻¹,
        not an operator equal to Λ⁻¹ (whose blocks are all dense).

    Raises:
        TypeError: If ``operator`` is not a `BlockTriDiag`.
        ValueError: If ``operator`` is not symmetric.

    Examples:
        An AR(1) precision with unit innovation variance and coefficient 0.5;
        the band matches the dense inverse:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> coefficient, num_blocks = 0.5, 6
        >>> diagonal = jnp.full((num_blocks, 1, 1), 1.0 + coefficient**2)
        >>> diagonal = diagonal.at[-1].set(1.0)
        >>> sub_diagonal = jnp.full((num_blocks - 1, 1, 1), -coefficient)
        >>> precision = gaussx.BlockTriDiag(diagonal, sub_diagonal)
        >>> band = gaussx.selected_inverse(precision)
        >>> covariance = jnp.linalg.inv(precision.as_matrix())
        >>> variances = band.diagonal[:, 0, 0]
        >>> bool(jnp.allclose(variances, jnp.diag(covariance), atol=1e-5))
        True
        >>> lag_one = band.sub_diagonal[:, 0, 0]
        >>> bool(jnp.allclose(lag_one, jnp.diag(covariance, -1), atol=1e-5))
        True

        Any other operator is rejected:

        >>> import lineax as lx
        >>> diagonal_operator = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> gaussx.selected_inverse(diagonal_operator)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: selected_inverse supports BlockTriDiag operators, ...
    """
    if not isinstance(operator, BlockTriDiag):
        msg = (
            "selected_inverse supports BlockTriDiag operators, got "
            f"{type(operator).__name__}."
        )
        raise TypeError(msg)
    if not operator.symmetric:
        msg = (
            "selected_inverse needs a symmetric positive-definite BlockTriDiag "
            "(it runs through the block Cholesky factor)."
        )
        raise ValueError(msg)

    # Λ = L Lᵀ, block by block
    factor = _cholesky_block_tridiag(operator)
    return _block_takahashi(factor)


def _block_takahashi(factor: LowerBlockTriDiag) -> BlockTriDiag:
    """The band of Σ = (L Lᵀ)⁻¹ from a lower block-bidiagonal factor L.

    Shared with the sparse Takahashi sweep (`gaussx._sparse._takahashi`).

    Algorithm:
        1. Invert each diagonal block, L_k⁻¹, by a triangular solve.
        2. Start from Σ_N,N = L_N⁻ᵀ L_N⁻¹.
        3. Sweep backwards: Σ_k+1,k = −Σ_k+1,k+1 B_k L_k⁻¹, then
           Σ_k,k = L_k⁻ᵀ (L_k⁻¹ − B_kᵀ Σ_k+1,k), symmetrised.

    Args:
        factor: Lower block-bidiagonal factor L, with diagonal blocks L_k and
            sub-diagonal blocks B_k.

    Returns:
        A `BlockTriDiag` holding Σ_k,k and Σ_k+1,k.
    """
    block_size = factor._block_size
    identity = jnp.eye(block_size, dtype=factor.diagonal.dtype)
    # L_k⁻¹ for every diagonal block
    factor_inverses = jax.vmap(
        lambda block: jax.scipy.linalg.solve_triangular(block, identity, lower=True)
    )(factor.diagonal)
    # Σ_N,N = L_N⁻ᵀ L_N⁻¹
    last = einsum(
        factor_inverses[-1], factor_inverses[-1], "mid row, mid col -> row col"
    )
    last_block = rearrange(last, "row col -> 1 row col")
    if factor._num_blocks == 1:
        return BlockTriDiag(last_block, factor.sub_diagonal)

    def step(
        sigma_next: Float[Array, "d d"],
        blocks: tuple[Float[Array, "d d"], Float[Array, "d d"]],
    ) -> tuple[Float[Array, "d d"], tuple[Float[Array, "d d"], Float[Array, "d d"]]]:
        factor_inverse_k, sub_block_k = blocks
        # Σ_k+1,k = −Σ_k+1,k+1 B_k L_k⁻¹
        sigma_sub = -einsum(
            sigma_next,
            sub_block_k,
            factor_inverse_k,
            "row a, a b, b col -> row col",
        )
        # B_kᵀ Σ_k+1,k
        coupling = einsum(sub_block_k, sigma_sub, "mid row, mid col -> row col")
        # L_k⁻¹ − B_kᵀ Σ_k+1,k
        rhs = factor_inverse_k - coupling
        # Σ_k,k = L_k⁻ᵀ (L_k⁻¹ − B_kᵀ Σ_k+1,k)
        sigma_kk = einsum(factor_inverse_k, rhs, "mid row, mid col -> row col")
        # Symmetric in exact arithmetic; remove the round-off asymmetry.
        sigma_kk = symmetrize(sigma_kk)
        return sigma_kk, (sigma_kk, sigma_sub)

    _, (diagonal, sub_diagonal) = jax.lax.scan(
        step, last, (factor_inverses[:-1], factor.sub_diagonal), reverse=True
    )
    return BlockTriDiag(jnp.concatenate([diagonal, last_block]), sub_diagonal)
