r"""UDL factorisation of block-tridiagonal precision matrices.

The factorisation $\Lambda = U \tilde{D} U^{\top}$ with unit upper
block-bidiagonal $U$ and block-diagonal $\tilde{D}$ is the banded
analogue of an $L D L^{\top}$ decomposition, computed *last block first*.
Unlike the Cholesky factor from `gaussx.cholesky`, its blocks have a
physical reading: for a Gauss-Markov chain they are the transition
matrices and process-noise precisions of the state-space model whose
joint precision is $\Lambda$.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag


def _cho_solve(chol: Float[Array, "d d"], rhs: Array) -> Array:
    """Solve ``(L L^T) x = rhs`` from a lower Cholesky factor ``L``."""
    return jsl.cho_solve((chol, True), rhs)


def _cho_inv(chol: Float[Array, "d d"]) -> Float[Array, "d d"]:
    """Inverse ``(L L^T)^{-1}`` from a lower Cholesky factor ``L``."""
    return _cho_solve(chol, jnp.eye(chol.shape[0], dtype=chol.dtype))


class UDLDecomposition(eqx.Module):
    r"""Banded factorisation $\Lambda = U \tilde{D} U^{\top}$ of a `BlockTriDiag`.

    $U$ is unit upper block-bidiagonal with super-diagonal blocks $U_k$
    and $\tilde{D} = \mathrm{blockdiag}(\tilde{D}_1, \dots, \tilde{D}_T)$.
    The factors are produced by the backward Schur-complement recurrence

    $$
    \tilde{D}_T = D_T, \qquad
    \tilde{D}_k = D_k - A_k^{\top} \tilde{D}_{k+1}^{-1} A_k, \qquad
    U_k^{\top} = \tilde{D}_{k+1}^{-1} A_k,
    $$

    where $D_k$ / $A_k$ are the diagonal / sub-diagonal blocks of
    $\Lambda$. This is **not** the Cholesky factor of $\Lambda$: for a
    Gauss-Markov chain $x_{k+1} = A^{\mathrm{ssm}}_k x_k + \varepsilon_k$
    the blocks are exactly $U_k^{\top} = -A^{\mathrm{ssm}}_k$ and
    $\tilde{D}_k = Q_k^{-1}$, so one pass converts a precision-form
    posterior into a sampleable state-space model
    (see `udl_to_ssm_params`).

    Every method costs $O(T d^3)$ (or $O(T d^2)$ for `solve`) and never
    forms the dense $(Td) \times (Td)$ matrix.

    Attributes:
        U_sub: Sub-diagonal blocks $U_k^{\top} = \tilde{D}_{k+1}^{-1} A_k$
            of $U^{\top}$, shape ``(T-1, d, d)``.
        D_diag: Block-diagonal factors $\tilde{D}_k$, shape ``(T, d, d)``.
        chol_D: Lower Cholesky factor of each $\tilde{D}_k$, shape
            ``(T, d, d)``, cached for `solve` and `logdet`.
    """

    U_sub: Float[Array, "Tm1 d d"]
    D_diag: Float[Array, "T d d"]
    chol_D: Float[Array, "T d d"]

    @property
    def num_blocks(self) -> int:
        """Number of blocks ``T``."""
        return self.D_diag.shape[0]

    @property
    def block_size(self) -> int:
        """Block dimension ``d``."""
        return self.D_diag.shape[1]

    def solve(self, rhs: Float[Array, " Td"]) -> Float[Array, " Td"]:
        r"""Solve $\Lambda x = b$ through the factors in $O(T d^2)$.

        Three banded sweeps: $U z = b$ (backward), $\tilde{D} w = z$
        (block-wise, via the cached Cholesky factors) and
        $U^{\top} x = w$ (forward).

        Args:
            rhs: Right-hand side, shape ``(T * d,)``.

        Returns:
            Solution, shape ``(T * d,)``.
        """
        T, d = self.num_blocks, self.block_size
        b = rearrange(rhs, "(T d) -> T d", T=T, d=d)

        # Backward sweep: z_T = b_T, z_k = b_k - U_k z_{k+1}.
        def _backward(z_next, inputs):
            b_k, U_sub_k = inputs
            z_k = b_k - einsum(U_sub_k, z_next, "j i, j -> i")
            return z_k, z_k

        _, z_rest = jax.lax.scan(_backward, b[-1], (b[:-1], self.U_sub), reverse=True)
        z = jnp.concatenate([z_rest, b[-1:]], axis=0)

        # Block-diagonal solve with the cached Cholesky factors.
        w = jax.vmap(_cho_solve)(self.chol_D, z)

        # Forward sweep: x_1 = w_1, x_k = w_k - U_{k-1}^T x_{k-1}.
        def _forward(x_prev, inputs):
            w_k, U_sub_k = inputs
            x_k = w_k - U_sub_k @ x_prev
            return x_k, x_k

        _, x_rest = jax.lax.scan(_forward, w[0], (w[1:], self.U_sub))
        x = jnp.concatenate([w[:1], x_rest], axis=0)
        return rearrange(x, "T d -> (T d)")

    def logdet(self) -> Float[Array, ""]:
        r"""$\log|\Lambda| = \sum_k \log|\tilde{D}_k|$, from the cached factors."""
        log_diag = jnp.log(jnp.diagonal(self.chol_D, axis1=-2, axis2=-1))
        return 2.0 * jnp.sum(log_diag)

    def as_block_tridiag(self) -> BlockTriDiag:
        r"""Reassemble $\Lambda = U \tilde{D} U^{\top}$ as a `BlockTriDiag`.

        Inverts the recurrence: $A_k = \tilde{D}_{k+1} U_k^{\top}$ and
        $D_k = \tilde{D}_k + A_k^{\top} \tilde{D}_{k+1}^{-1} A_k
        = \tilde{D}_k + U_k \tilde{D}_{k+1} U_k^{\top}$.
        """
        sub = einsum(self.D_diag[1:], self.U_sub, "T i j, T j k -> T i k")
        future = einsum(self.U_sub, sub, "T j i, T j k -> T i k")
        diag = self.D_diag.at[:-1].add(future)
        return BlockTriDiag(diag, sub)


def udl_decomposition(precision: BlockTriDiag) -> UDLDecomposition:
    r"""Factorise a symmetric block-tridiagonal precision as $U \tilde{D} U^{\top}$.

    Runs the backward Schur-complement recurrence as a `jax.lax.scan`
    from the last block to the first, at $O(T d^3)$ time and $O(T d^2)$
    memory. Each $\tilde{D}_k$ is Cholesky-factored on the fly, so the
    input must be positive definite.

    Args:
        precision: Symmetric positive-definite `BlockTriDiag` with ``T``
            diagonal blocks of size ``(d, d)``.

    Returns:
        The `UDLDecomposition` of ``precision``.

    Examples:
        >>> import jax.numpy as jnp, gaussx
        >>> diag = jnp.broadcast_to(2.0 * jnp.eye(2), (4, 2, 2))
        >>> sub = jnp.broadcast_to(-0.5 * jnp.eye(2), (3, 2, 2))
        >>> udl = gaussx.udl_decomposition(gaussx.BlockTriDiag(diag, sub))
        >>> udl.U_sub.shape, udl.D_diag.shape
        ((3, 2, 2), (4, 2, 2))
    """
    D_last = precision.diagonal[-1]
    chol_last = jnp.linalg.cholesky(D_last)

    def _step(carry, inputs):
        _, chol_next = carry
        D_k, A_k = inputs
        # U_k^T = D~_{k+1}^{-1} A_k  and  D~_k = D_k - A_k^T D~_{k+1}^{-1} A_k.
        U_sub_k = _cho_solve(chol_next, A_k)
        D_tilde_k = D_k - einsum(A_k, U_sub_k, "j i, j k -> i k")
        chol_k = jnp.linalg.cholesky(D_tilde_k)
        return (D_tilde_k, chol_k), (U_sub_k, D_tilde_k, chol_k)

    _, (U_sub, D_rest, chol_rest) = jax.lax.scan(
        _step,
        (D_last, chol_last),
        (precision.diagonal[:-1], precision.sub_diagonal),
        reverse=True,
    )
    D_diag = jnp.concatenate([D_rest, D_last[None]], axis=0)
    chol_D = jnp.concatenate([chol_rest, chol_last[None]], axis=0)
    return UDLDecomposition(U_sub=U_sub, D_diag=D_diag, chol_D=chol_D)


def udl_to_ssm_params(
    udl: UDLDecomposition,
) -> tuple[
    Float[Array, "Tm1 d d"],
    Float[Array, "T d d"],
    Float[Array, "T d d"],
]:
    r"""Read the equivalent Gauss-Markov chain off a `UDLDecomposition`.

    A chain $x_0 \sim \mathcal{N}(\mu_0, P_0)$,
    $x_{k+1} = A_k x_k + \varepsilon_k$, $\varepsilon_k \sim \mathcal{N}(0, Q_{k+1})$
    has precision $\Lambda = U \tilde{D} U^{\top}$ with

    $$
    A_k = -U_k^{\top}, \qquad Q_{k}^{-1} = \tilde{D}_{k}, \qquad P_0^{-1} = \tilde{D}_0,
    $$

    so the factors *are* the SSM. The returned ``Q`` follows the
    `ssm_to_naturals` / `naturals_to_ssm` layout: ``Q[0]`` is $P_0$ and
    ``Q[k]`` for $k \ge 1$ is the process noise entering state $k$.

    Args:
        udl: Factorisation of the chain's precision.

    Returns:
        Tuple ``(A, Q, chol_Q)``: transitions of shape ``(T-1, d, d)``,
        covariances of shape ``(T, d, d)`` and their lower Cholesky
        factors of shape ``(T, d, d)``.
    """
    A = -udl.U_sub
    Q = jax.vmap(_cho_inv)(udl.chol_D)
    chol_Q = jnp.linalg.cholesky(Q)
    return A, Q, chol_Q


def udl_from_ssm_params(
    A: Float[Array, "Tm1 d d"],
    Q: Float[Array, "T d d"],
) -> UDLDecomposition:
    r"""Build the `UDLDecomposition` of a Gauss-Markov chain's precision.

    Inverse of `udl_to_ssm_params`: sets $U_k^{\top} = -A_k$ and
    $\tilde{D}_k = Q_k^{-1}$ directly, so the chain's block-tridiagonal
    precision is available through `UDLDecomposition.as_block_tridiag`
    without ever factorising it.

    Args:
        A: Transition matrices, shape ``(T-1, d, d)``.
        Q: Covariances, shape ``(T, d, d)``; ``Q[0]`` is the initial
            covariance $P_0$ and ``Q[k]`` the process noise entering
            state $k$.

    Returns:
        The factorisation of the chain's precision.
    """
    chol_Q = jnp.linalg.cholesky(Q)
    D_diag = jax.vmap(_cho_inv)(chol_Q)
    chol_D = jnp.linalg.cholesky(D_diag)
    return UDLDecomposition(U_sub=-A, D_diag=D_diag, chol_D=chol_D)
