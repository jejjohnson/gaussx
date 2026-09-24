r"""Falkon: preconditioned conjugate gradients for Nyström kernel ridge regression.

Nyström KRR with ``N`` data points and ``M`` inducing points solves

$$(K_{nm}^{\top} K_{nm} + \lambda n K_{mm})\, \alpha = K_{nm}^{\top} y .$$

Falkon (Rudi et al. 2017; Meanti et al. 2020) preconditions it with the
Nyström approximation $K_{nm}^{\top} K_{nm} \approx (n/m) K_{mm}^2$, which
needs only two ``M x M`` Cholesky factors, so the rectangular ``K_{nm}`` is
touched only through matvecs.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float


class FalkonPreconditioner(eqx.Module):
    r"""The two upper-triangular Cholesky factors of the Falkon preconditioner.

    With $K_{mm} = T^{\top} T$ and $A^{\top} A = T T^{\top} / m + \lambda I$,
    the preconditioner $P = T^{-1} A^{-1}$ satisfies

    $$P P^{\top} = n \bigl( (n/m) K_{mm}^2 + \lambda n K_{mm} \bigr)^{-1},$$

    the inverse of the Nyström approximation to the KRR system matrix (up
    to the constant $n$, which CG does not see). Both factors are *upper*
    triangular: with lower factors $T T^{\top}$ would be a different matrix
    and the identity above would fail.

    In the preconditioned variable $\beta = P^{-1} \alpha$ the system matrix
    becomes

    $$P^{\top} (K_{nm}^{\top} K_{nm} + \lambda n K_{mm}) P
      = A^{-\top} \bigl[ T^{-\top} K_{nm}^{\top} K_{nm} T^{-1}
        + \lambda n I \bigr] A^{-1},$$

    in which $K_{mm}$ has cancelled: applying it needs only triangular
    solves with ``T`` and ``A`` and matvecs with $K_{nm}$.

    Attributes:
        T: Upper Cholesky factor of the (jittered) $K_{mm}$, shape ``(M, M)``.
        A: Upper Cholesky factor of $T T^{\top} / m + \lambda I$, shape
            ``(M, M)``.
    """

    T: Float[Array, "M M"]
    A: Float[Array, "M M"]

    def precondition(self, beta: Float[Array, " M *C"]) -> Float[Array, " M *C"]:
        r"""Map the preconditioned variable back: $\alpha = T^{-1} A^{-1} \beta$."""
        return _solve_upper(self.T, _solve_upper(self.A, beta))

    def precondition_transpose(
        self, vector: Float[Array, " M *C"]
    ) -> Float[Array, " M *C"]:
        r"""Apply $P^{\top} = A^{-\top} T^{-\top}$."""
        return _solve_upper(self.A, _solve_upper(self.T, vector, trans=1), trans=1)


def falkon_preconditioner(
    K_mm: Float[Array, "M M"] | lx.AbstractLinearOperator,
    regularization: float | Float[Array, ""],
    *,
    jitter: float | Float[Array, ""] | None = None,
) -> FalkonPreconditioner:
    r"""Build the Falkon preconditioner from the inducing-point kernel matrix.

    Costs two ``M x M`` Cholesky factorisations, $O(M^3)$, once; see
    `FalkonPreconditioner` for the factors and the identity they satisfy.

    Kernel matrices are often numerically singular, so ``jitter`` is added
    to the diagonal of $K_{mm}$ before factorising. The default is the
    pstrf-style ``M * eps * max(diag(K_mm))``: large enough to keep the
    Cholesky finite, small enough not to change the solution at working
    precision. An all-zero diagonal takes a scale of 1 instead, so the
    jitter stays positive.

    Args:
        K_mm: Kernel matrix of the inducing points, shape ``(M, M)``, as an
            array or a lineax operator (materialised).
        regularization: Ridge parameter $\lambda > 0$ of the KRR objective.
        jitter: Diagonal jitter added to ``K_mm``. ``None`` uses the default
            above.

    Returns:
        The preconditioner's factors.

    Raises:
        ValueError: If ``K_mm`` is not square.
    """
    if isinstance(K_mm, lx.AbstractLinearOperator):
        K_mm = K_mm.as_matrix()
    K_mm = jnp.asarray(K_mm)
    if K_mm.ndim != 2 or K_mm.shape[0] != K_mm.shape[1]:
        raise ValueError(f"K_mm must be a square matrix, got shape {K_mm.shape}.")
    m = K_mm.shape[0]
    # One dtype for both factors: a float64 ``regularization`` with a float32
    # ``K_mm`` would otherwise give a float32 T and a float64 A.
    operands = [K_mm, regularization] + ([] if jitter is None else [jitter])
    dtype = jnp.result_type(*operands, jnp.float32)
    K_mm = K_mm.astype(dtype)
    regularization = jnp.asarray(regularization, dtype=dtype)
    identity = jnp.eye(m, dtype=dtype)
    if jitter is None:
        scale = jnp.max(jnp.abs(jnp.diag(K_mm)))
        # A zero-scale kernel (e.g. a linear kernel at zero inputs) would
        # otherwise get zero jitter and a NaN Cholesky.
        scale = jnp.where(scale > 0, scale, 1.0)
        jitter = m * jnp.finfo(dtype).eps * scale
    jitter = jnp.asarray(jitter, dtype=dtype)

    T = jax.scipy.linalg.cholesky(K_mm + jitter * identity, lower=False)
    A = jax.scipy.linalg.cholesky(T @ T.T / m + regularization * identity, lower=False)
    return FalkonPreconditioner(T=T, A=A)


def _solve_upper(factor: Float[Array, "M M"], rhs: Array, trans: int = 0) -> Array:
    """Solve ``factor x = rhs`` (``trans=1``: ``factorᵀ x = rhs``), upper."""
    return jax.scipy.linalg.solve_triangular(factor, rhs, lower=False, trans=trans)
