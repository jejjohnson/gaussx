"""Selected inverse: the entries of ``Q⁻¹`` on the sparsity pattern of ``Q``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag, LowerBlockTriDiag


def selected_inverse(operator: BlockTriDiag) -> BlockTriDiag:
    r"""The band of ``Σ = Q⁻¹`` for a symmetric positive-definite `BlockTriDiag`.

    Takahashi's recursion in block form — the RTS smoother written in
    precision form. With the block Cholesky factor ``Q = L Lᵀ`` (diagonal
    blocks ``L_k``, sub-diagonal blocks ``B_k`` at ``(k+1, k)``), the identity
    ``Lᵀ Σ = L⁻¹`` gives, sweeping backwards from
    ``Σ_NN = L_N⁻ᵀ L_N⁻¹``,

    $$
    \Sigma_{k+1,k} = -\Sigma_{k+1,k+1} B_k L_k^{-1},\qquad
    \Sigma_{kk} = L_k^{-\top}\big(L_k^{-1} - B_k^\top \Sigma_{k+1,k}\big).
    $$

    Only the diagonal and first off-diagonal blocks are computed, at
    ``O(N d³)`` cost in a ``lax.scan``; with ``d = 1`` this is Rue & Held's
    scalar recursion. `gaussx.diag_inv` takes the marginal variances from the
    diagonal blocks; the off-diagonal blocks are the lag-one covariances.

    Args:
        operator: A symmetric positive-definite block-tridiagonal precision.

    Returns:
        A `BlockTriDiag` holding ``Σ_kk`` and ``Σ_{k+1,k}``. It is the band of
        ``Q⁻¹``, not an operator equal to ``Q⁻¹`` (whose blocks are dense).

    Raises:
        TypeError: If ``operator`` is not a `BlockTriDiag`.
        ValueError: If ``operator`` is not symmetric.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx

        # An AR(1) precision with unit innovation variance and phi = 0.5
        phi, n = 0.5, 6
        diagonal = jnp.full((n, 1, 1), 1.0 + phi**2).at[-1].set(1.0)
        sub_diagonal = jnp.full((n - 1, 1, 1), -phi)
        Q = gaussx.BlockTriDiag(diagonal, sub_diagonal)
        band = gaussx.selected_inverse(Q)
        Sigma = jnp.linalg.inv(Q.as_matrix())
        assert jnp.allclose(band.diagonal[:, 0, 0], jnp.diag(Sigma), atol=1e-5)
        assert jnp.allclose(band.sub_diagonal[:, 0, 0], jnp.diag(Sigma, -1), atol=1e-5)
        ```
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

    from gaussx._primitives._cholesky import _cholesky_block_tridiag

    return _block_takahashi(_cholesky_block_tridiag(operator))


def _block_takahashi(factor: LowerBlockTriDiag) -> BlockTriDiag:
    """The band of ``Σ = (L Lᵀ)⁻¹`` from a lower block-bidiagonal factor ``L``."""
    d = factor._block_size
    eye = jnp.eye(d, dtype=factor.diagonal.dtype)
    factor_inv = jax.vmap(
        lambda block: jax.scipy.linalg.solve_triangular(block, eye, lower=True)
    )(factor.diagonal)
    last = einsum(factor_inv[-1], factor_inv[-1], "j i, j m -> i m")
    last_block = rearrange(last, "i j -> 1 i j")
    if factor._num_blocks == 1:
        return BlockTriDiag(last_block, factor.sub_diagonal)

    def step(
        sigma_next: Float[Array, "d d"],
        blocks: tuple[Float[Array, "d d"], Float[Array, "d d"]],
    ) -> tuple[Float[Array, "d d"], tuple[Float[Array, "d d"], Float[Array, "d d"]]]:
        inv_k, sub_k = blocks
        # Σ_{k+1,k} = -Σ_{k+1,k+1} B_k L_k⁻¹
        sigma_sub = -einsum(sigma_next, sub_k, inv_k, "i j, j l, l m -> i m")
        # Σ_kk = L_k⁻ᵀ (L_k⁻¹ - B_kᵀ Σ_{k+1,k})
        rhs = inv_k - einsum(sub_k, sigma_sub, "j i, j m -> i m")
        sigma_kk = einsum(inv_k, rhs, "j i, j m -> i m")
        # Symmetric in exact arithmetic; remove the round-off asymmetry.
        sigma_kk = 0.5 * (sigma_kk + rearrange(sigma_kk, "i j -> j i"))
        return sigma_kk, (sigma_kk, sigma_sub)

    _, (diagonal, sub_diagonal) = jax.lax.scan(
        step, last, (factor_inv[:-1], factor.sub_diagonal), reverse=True
    )
    return BlockTriDiag(jnp.concatenate([diagonal, last_block]), sub_diagonal)
