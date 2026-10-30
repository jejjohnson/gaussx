"""UDL factorisation Λ_x = U D Uᵀ of block-tridiagonal precision matrices.

U is unit upper block-bidiagonal and D block-diagonal: the banded analogue of
an L D Lᵀ decomposition, computed last block first. Unlike the Cholesky
factor from `gaussx.cholesky`, its blocks have a physical reading: for a
Gauss-Markov chain they are the transitions and model-error precisions of
the state-space model whose joint precision is Λ_x.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag


def _cho_solve(chol: Float[Array, "d d"], rhs: Array) -> Array:
    """Solve A v = b from the lower Cholesky factor L_A of A.

    Args:
        chol: Lower Cholesky factor L_A, with L_A L_Aᵀ = A.
        rhs: Right-hand side b, a vector or a matrix.

    Returns:
        The solution v = A⁻¹ b.
    """
    # v = (L_A L_Aᵀ)⁻¹ b
    return jsl.cho_solve((chol, True), rhs)


def _cho_inv(chol: Float[Array, "d d"]) -> Float[Array, "d d"]:
    """Inverse A⁻¹ from the lower Cholesky factor L_A of A.

    Args:
        chol: Lower Cholesky factor L_A.

    Returns:
        The dense inverse A⁻¹.
    """
    identity = jnp.eye(chol.shape[0], dtype=chol.dtype)
    # A⁻¹ = (L_A L_Aᵀ)⁻¹ I
    return _cho_solve(chol, identity)


class UDLDecomposition(eqx.Module):
    """Banded factorisation Λ_x = U D Uᵀ of a block-tridiagonal precision.

        D_T−1 = [Λ_x]_T−1,T−1
        U_tᵀ = D_t+1⁻¹ [Λ_x]_t+1,t
        D_t = [Λ_x]_t,t − [Λ_x]_t+1,tᵀ D_t+1⁻¹ [Λ_x]_t+1,t

    for blocks t = 0, …, T − 1 of size D_x. U is unit upper
    block-bidiagonal with super-diagonal blocks U_t, D = blockdiag(D_0, …,
    D_T−1), and [Λ_x]_t,t and [Λ_x]_t+1,t are the diagonal and sub-diagonal
    blocks of Λ_x; the recurrence runs backward through Schur complements.

    This is not the Cholesky factor of Λ_x. For a Gauss-Markov chain
    x_t+1 = F_t x_t + η_t+1 with η_t+1 ~ 𝒩(0, Σ_η,t+1) and
    x₀ ~ 𝒩(μ_x,0, Σ_x,0), the blocks are exactly U_tᵀ = −F_t, D_t = Σ_η,t⁻¹
    for t ≥ 1 and D_0 = Σ_x,0⁻¹, so one pass turns a precision-form
    posterior into a state-space model that can be sampled (see
    `udl_to_ssm_params`). Here F_t maps x_t to x_t+1, so it is the array
    ``A[t]``; `kalman_filter` indexes the same matrix by the state it
    arrives at.

    Every method costs O(T D_x³) (or O(T D_x²) for `solve`) and never forms
    the dense T D_x × T D_x matrix.

    Attributes:
        U_sub: Sub-diagonal blocks U_tᵀ = D_t+1⁻¹ [Λ_x]_t+1,t of Uᵀ.
        D_diag: Block-diagonal factors D_t.
        chol_D: Lower Cholesky factors L_D,t of each D_t, cached for
            `solve` and `logdet`.

    Examples:
        Factorise a small chain precision, then solve and take the
        log-determinant through the factors:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> diag = jnp.broadcast_to(2.0 * jnp.eye(2), (4, 2, 2))
        >>> sub = jnp.broadcast_to(-0.5 * jnp.eye(2), (3, 2, 2))
        >>> precision = gaussx.BlockTriDiag(diag, sub)
        >>> udl = gaussx.udl_decomposition(precision)
        >>> udl.num_blocks, udl.block_size
        (4, 2)
        >>> dense = precision.as_matrix()
        >>> rhs = jnp.arange(8.0)
        >>> bool(jnp.allclose(dense @ udl.solve(rhs), rhs, atol=1e-4))
        True
        >>> bool(jnp.allclose(udl.logdet(), jnp.linalg.slogdet(dense)[1]))
        True

        Reassembling the factors gives back Λ_x:

        >>> rebuilt = udl.as_block_tridiag().as_matrix()
        >>> bool(jnp.allclose(rebuilt, dense, atol=1e-5))
        True
    """

    U_sub: Float[Array, "Tm1 d d"]
    D_diag: Float[Array, "T d d"]
    chol_D: Float[Array, "T d d"]

    @property
    def num_blocks(self) -> int:
        """Number of blocks T."""
        return self.D_diag.shape[0]

    @property
    def block_size(self) -> int:
        """Block size D_x."""
        return self.D_diag.shape[1]

    def solve(self, rhs: Float[Array, " Td"]) -> Float[Array, " Td"]:
        """Solve Λ_x v = b through the factors in O(T D_x²).

            U a = b,   D c = a,   Uᵀ v = c

        Algorithm:
            1. Backward sweep: a_T−1 = b_T−1, a_t = b_t − U_t a_t+1.
            2. Block-diagonal solve c_t = D_t⁻¹ a_t with the cached
               Cholesky factors.
            3. Forward sweep: v_0 = c_0, v_t = c_t − U_t−1ᵀ v_t−1.

        Args:
            rhs: Right-hand side b, stacked over the T blocks.

        Returns:
            The solution v = Λ_x⁻¹ b, stacked like ``rhs``.
        """
        num_blocks, block_size = self.num_blocks, self.block_size
        rhs_blocks = rearrange(
            rhs, "(block state) -> block state", block=num_blocks, state=block_size
        )

        # --- backward sweep ---
        def _backward(upper_solution_next, inputs):
            rhs_t, upper_t_transpose = inputs
            # U_t a_t+1
            coupling = einsum(
                upper_t_transpose, upper_solution_next, "next state, next -> state"
            )
            # a_t = b_t − U_t a_t+1
            upper_solution_t = rhs_t - coupling
            return upper_solution_t, upper_solution_t

        _, upper_solution_rest = jax.lax.scan(
            _backward, rhs_blocks[-1], (rhs_blocks[:-1], self.U_sub), reverse=True
        )
        upper_solution = jnp.concatenate([upper_solution_rest, rhs_blocks[-1:]])

        # --- block-diagonal solve ---
        # c_t = D_t⁻¹ a_t
        diagonal_solution = jax.vmap(_cho_solve)(self.chol_D, upper_solution)

        # --- forward sweep ---
        def _forward(solution_prev, inputs):
            diagonal_solution_t, upper_prev_transpose = inputs
            # v_t = c_t − U_t−1ᵀ v_t−1
            solution_t = diagonal_solution_t - upper_prev_transpose @ solution_prev
            return solution_t, solution_t

        _, solution_rest = jax.lax.scan(
            _forward, diagonal_solution[0], (diagonal_solution[1:], self.U_sub)
        )
        solution = jnp.concatenate([diagonal_solution[:1], solution_rest])
        return rearrange(solution, "block state -> (block state)")

    def logdet(self) -> Float[Array, ""]:
        """Log-determinant of Λ_x from the cached factors.

            log det Λ_x = ∑_t log det D_t = 2 ∑_t ∑_d log [L_D,t]_dd

        Returns:
            The scalar log det Λ_x.
        """
        # log [L_D,t]_dd
        log_diag = jnp.log(jnp.diagonal(self.chol_D, axis1=-2, axis2=-1))
        # log det Λ_x = 2 ∑_t ∑_d log [L_D,t]_dd
        return 2.0 * jnp.sum(log_diag)

    def as_block_tridiag(self) -> BlockTriDiag:
        """Reassemble Λ_x = U D Uᵀ as a `BlockTriDiag`.

            [Λ_x]_t+1,t = D_t+1 U_tᵀ
            [Λ_x]_t,t = D_t + U_t D_t+1 U_tᵀ

        which inverts the recurrence of the factorisation.

        Returns:
            The block-tridiagonal precision Λ_x.
        """
        if self.U_sub.shape[0] == 0:
            # One block: Λ_x = D_0 (einx cannot contract the empty axis).
            return BlockTriDiag(self.D_diag, self.U_sub)
        # [Λ_x]_t+1,t = D_t+1 U_tᵀ
        sub_blocks = einsum(
            self.D_diag[1:],
            self.U_sub,
            "block row mid, block mid col -> block row col",
        )
        # U_t D_t+1 U_tᵀ
        future = einsum(
            self.U_sub,
            sub_blocks,
            "block mid row, block mid col -> block row col",
        )
        # [Λ_x]_t,t = D_t + U_t D_t+1 U_tᵀ, for t < T − 1
        diag_blocks = self.D_diag.at[:-1].add(future)
        return BlockTriDiag(diag_blocks, sub_blocks)


def udl_decomposition(precision: BlockTriDiag) -> UDLDecomposition:
    """Factorise a symmetric block-tridiagonal precision as Λ_x = U D Uᵀ.

        D_T−1 = [Λ_x]_T−1,T−1
        U_tᵀ = D_t+1⁻¹ [Λ_x]_t+1,t
        D_t = [Λ_x]_t,t − [Λ_x]_t+1,tᵀ D_t+1⁻¹ [Λ_x]_t+1,t

    Runs the backward Schur-complement recurrence as a `jax.lax.scan` from
    the last block to the first, at O(T D_x³) time and O(T D_x²) memory.
    Each D_t is Cholesky-factored on the fly, so the input must be positive
    definite.

    Algorithm:
        1. Factor the last diagonal block, D_T−1 = [Λ_x]_T−1,T−1.
        2. For t = T − 2, …, 0, solve for U_tᵀ against D_t+1, form the
           Schur complement D_t and factor it.

    Args:
        precision: Symmetric positive-definite precision Λ_x with T
            diagonal blocks of size D_x.

    Returns:
        The `UDLDecomposition` of ``precision``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> diag = jnp.broadcast_to(2.0 * jnp.eye(2), (4, 2, 2))
        >>> sub = jnp.broadcast_to(-0.5 * jnp.eye(2), (3, 2, 2))
        >>> udl = gaussx.udl_decomposition(gaussx.BlockTriDiag(diag, sub))
        >>> udl.U_sub.shape, udl.D_diag.shape
        ((3, 2, 2), (4, 2, 2))

        The last factor block is the last diagonal block of Λ_x:

        >>> bool(jnp.allclose(udl.D_diag[-1], diag[-1]))
        True
    """
    last_factor = precision.diagonal[-1]
    # L_D,T−1 L_D,T−1ᵀ = D_T−1
    last_chol = jnp.linalg.cholesky(last_factor)

    def _step(carry, inputs):
        _, chol_next = carry
        diag_block, sub_block = inputs
        # U_tᵀ = D_t+1⁻¹ [Λ_x]_t+1,t
        upper_transpose = _cho_solve(chol_next, sub_block)
        # [Λ_x]_t+1,tᵀ D_t+1⁻¹ [Λ_x]_t+1,t
        schur_correction = einsum(
            sub_block, upper_transpose, "next row, next col -> row col"
        )
        # D_t = [Λ_x]_t,t − [Λ_x]_t+1,tᵀ D_t+1⁻¹ [Λ_x]_t+1,t
        factor = diag_block - schur_correction
        # L_D,t L_D,tᵀ = D_t
        chol = jnp.linalg.cholesky(factor)
        return (factor, chol), (upper_transpose, factor, chol)

    _, (upper_transposes, factor_rest, chol_rest) = jax.lax.scan(
        _step,
        (last_factor, last_chol),
        (precision.diagonal[:-1], precision.sub_diagonal),
        reverse=True,
    )
    factors = jnp.concatenate(
        [factor_rest, rearrange(last_factor, "row col -> 1 row col")]
    )
    chols = jnp.concatenate([chol_rest, rearrange(last_chol, "row col -> 1 row col")])
    return UDLDecomposition(U_sub=upper_transposes, D_diag=factors, chol_D=chols)


def udl_to_ssm_params(
    udl: UDLDecomposition,
) -> tuple[
    Float[Array, "Tm1 d d"],
    Float[Array, "T d d"],
    Float[Array, "T d d"],
]:
    """Read the equivalent Gauss-Markov chain off a `UDLDecomposition`.

        F_t = −U_tᵀ,   Σ_η,t = D_t⁻¹ (t ≥ 1),   Σ_x,0 = D_0⁻¹

    for the chain x₀ ~ 𝒩(μ_x,0, Σ_x,0), x_t+1 = F_t x_t + η_t+1 with
    η_t+1 ~ 𝒩(0, Σ_η,t+1), whose precision is Λ_x = U D Uᵀ: the factors
    are the state-space model. The returned covariances stack the two:
    ``Q[0]`` is Σ_x,0 and ``Q[t]`` for t ≥ 1 is Σ_η,t, the model error
    entering x_t (``Q[1:]`` is the `gaussx.MarkovGaussian` layout).

    Args:
        udl: Factorisation of the chain's precision Λ_x.

    Returns:
        Tuple ``(A, Q, chol_Q)``: the transitions F_t, the stacked
        covariances [Σ_x,0; Σ_η,1:T−1] and their lower Cholesky factors.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> diag = jnp.broadcast_to(2.0 * jnp.eye(2), (4, 2, 2))
        >>> sub = jnp.broadcast_to(-0.5 * jnp.eye(2), (3, 2, 2))
        >>> udl = gaussx.udl_decomposition(gaussx.BlockTriDiag(diag, sub))
        >>> transitions, covariances, chols = gaussx.udl_to_ssm_params(udl)
        >>> transitions.shape, covariances.shape
        ((3, 2, 2), (4, 2, 2))

        The chain round-trips through `udl_from_ssm_params`:

        >>> rebuilt = gaussx.udl_from_ssm_params(
        ...     transitions, covariances[1:], covariances[0]
        ... )
        >>> bool(jnp.allclose(rebuilt.D_diag, udl.D_diag, atol=1e-5))
        True
    """
    # F_t = −U_tᵀ
    transitions = -udl.U_sub
    # [Σ_x,0; Σ_η,t] = D_t⁻¹
    covariances = jax.vmap(_cho_inv)(udl.chol_D)
    # L_Σ,t L_Σ,tᵀ = D_t⁻¹
    chols = jnp.linalg.cholesky(covariances)
    return transitions, covariances, chols


def udl_from_ssm_params(
    A: Float[Array, "Tm1 d d"],
    Q: Float[Array, "Tm1 d d"] | Float[Array, "T d d"],
    P0: Float[Array, "d d"] | None = None,
) -> UDLDecomposition:
    """Build the `UDLDecomposition` of a Gauss-Markov chain's precision.

        U_tᵀ = −F_t,   D_t = Σ_η,t⁻¹ (t ≥ 1),   D_0 = Σ_x,0⁻¹

    The inverse of `udl_to_ssm_params`: the factors are set directly, so
    the chain's block-tridiagonal precision Λ_x is available through
    `UDLDecomposition.as_block_tridiag` without ever factorising it.

    Args:
        A: Transitions F_t, with ``A[t]`` mapping x_t to x_t+1.
        Q: Model-error covariances, with ``Q[t]`` = Σ_η,t+1 driving x_t+1
            (the `gaussx.MarkovGaussian` layout). Without ``P0``, the older
            stacked layout with ``Q[0]`` = Σ_x,0 and one more block is
            accepted with a ``DeprecationWarning`` until gaussx 0.7.0.
        P0: Prior covariance Σ_x,0 of the initial state x₀.

    Raises:
        ValueError: If ``Q`` does not have one block per transition.

    Returns:
        The factorisation of the chain's precision Λ_x.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transitions = jnp.broadcast_to(0.9 * jnp.eye(2), (3, 2, 2))
        >>> process_noise = jnp.broadcast_to(0.1 * jnp.eye(2), (3, 2, 2))
        >>> udl = gaussx.udl_from_ssm_params(transitions, process_noise, jnp.eye(2))
        >>> udl.num_blocks, udl.block_size
        (4, 2)
        >>> bool(jnp.allclose(udl.U_sub, -transitions))
        True
    """
    if P0 is not None:
        # [Σ_x,0; Σ_η,1:T−1]
        Q = jnp.concatenate([rearrange(P0, "row col -> 1 row col"), Q])
    else:
        warn_deprecated(
            "udl_from_ssm_params(A, Q) with P_0 stacked as Q[0] is deprecated; "
            "pass the transition noise and P0 separately, "
            "udl_from_ssm_params(A, Q, P0). The stacked layout will be "
            "removed in gaussx 0.7.0."
        )
    if Q.shape[0] != A.shape[0] + 1:
        msg = (
            f"udl_from_ssm_params: expected {A.shape[0]} transition-noise blocks "
            f"to match A, got {Q.shape[0] - (P0 is not None)}."
        )
        raise ValueError(msg)
    # L_Σ,t L_Σ,tᵀ = [Σ_x,0; Σ_η,t]
    chol_covariances = jnp.linalg.cholesky(Q)
    # D_t = [Σ_x,0; Σ_η,t]⁻¹
    factors = jax.vmap(_cho_inv)(chol_covariances)
    # L_D,t L_D,tᵀ = D_t
    chols = jnp.linalg.cholesky(factors)
    # U_tᵀ = −F_t
    return UDLDecomposition(U_sub=-A, D_diag=factors, chol_D=chols)
