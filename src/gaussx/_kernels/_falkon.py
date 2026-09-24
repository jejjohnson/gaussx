r"""Falkon: preconditioned conjugate gradients for Nyström kernel ridge regression.

Nyström KRR with ``N`` data points and ``M`` inducing points solves

$$(K_{nm}^{\top} K_{nm} + \lambda n K_{mm})\, \alpha = K_{nm}^{\top} y .$$

Falkon (Rudi et al. 2017; Meanti et al. 2020) preconditions it with the
Nyström approximation $K_{nm}^{\top} K_{nm} \approx (n/m) K_{mm}^2$, which
needs only two ``M x M`` Cholesky factors, so the rectangular ``K_{nm}`` is
touched only through matvecs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._implicit_cross_kernel import ImplicitCrossKernelOperator


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
    precision. It is floored at the dtype's smallest normal number, so a
    zero or tiny-scale ``K_mm`` still gets a positive jitter.

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
        # Floored at the smallest normal number: a zero or tiny-scale kernel
        # (e.g. a linear kernel at zero inputs) would otherwise get a jitter
        # that is, or underflows to, zero, and a NaN Cholesky.
        jitter = jnp.maximum(m * jnp.finfo(dtype).eps * scale, jnp.finfo(dtype).tiny)
    jitter = jnp.asarray(jitter, dtype=dtype)

    T = jax.scipy.linalg.cholesky(K_mm + jitter * identity, lower=False)
    A = jax.scipy.linalg.cholesky(T @ T.T / m + regularization * identity, lower=False)
    return FalkonPreconditioner(T=T, A=A)


def falkon_solve(
    K_nm: lx.AbstractLinearOperator,
    y: Float[Array, " N"],
    preconditioner: FalkonPreconditioner,
    regularization: float | Float[Array, ""],
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Float[Array, " M"]:
    r"""Nyström kernel ridge regression weights by Falkon's preconditioned CG.

    Solves $(K_{nm}^{\top} K_{nm} + \lambda n K_{mm})\, \alpha = K_{nm}^{\top} y$
    for the Nyström weights $\alpha$, with $K_{mm}$ the jittered matrix the
    preconditioner was built from. CG runs on $\beta = P^{-1} \alpha$
    against

    $$A^{-\top} \bigl[ T^{-\top} K_{nm}^{\top} K_{nm} T^{-1}
      + \lambda n I \bigr] A^{-1} \beta = A^{-\top} T^{-\top} K_{nm}^{\top} y,$$

    so each iteration costs four $M \times M$ triangular solves and one
    matvec each with $K_{nm}$ and $K_{nm}^{\top}$ -- pass an
    `ImplicitCrossKernelOperator` and the $N \times M$ matrix is never
    formed. The weights are recovered as $\alpha = T^{-1} A^{-1} \beta$.

    Following Falkon, ``max_iter`` is a budget rather than a failure: the
    preconditioned system is well conditioned enough that a few tens of
    iterations reach the statistical accuracy of the estimator, and the
    iterate at the budget is returned without raising.

    Args:
        K_nm: Cross-kernel operator between the ``N`` data points and the
            ``M`` inducing points, shape ``(N, M)``. Only ``mv`` and
            ``transpose().mv`` are used.
        y: Targets, shape ``(N,)``.
        preconditioner: `FalkonPreconditioner` built from the same inducing
            points and ``regularization``.
        regularization: Ridge parameter $\lambda$; must match the
            preconditioner's.
        max_iter: CG iteration budget.
        tol: Relative residual tolerance for stopping early.

    Returns:
        Nyström weights $\alpha$, shape ``(M,)``.

    Raises:
        ValueError: If the shapes of ``K_nm``, ``y`` and ``preconditioner``
            disagree, or ``max_iter`` is below one.
    """
    y = jnp.asarray(y)
    m = preconditioner.T.shape[0]
    if y.ndim != 1:
        raise ValueError(f"y must be a vector of shape (N,), got {y.shape}.")
    n = y.shape[0]
    if (K_nm.out_size(), K_nm.in_size()) != (n, m):
        raise ValueError(
            f"K_nm must have shape ({n}, {m}) to match y and the preconditioner, "
            f"got ({K_nm.out_size()}, {K_nm.in_size()})."
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}.")

    # One dtype for every operand -- y, both factors, the regularization and
    # the cross kernel -- so the CG operator's input and output structures
    # agree however the caller's dtypes are mixed.
    dtype = jnp.result_type(
        y,
        preconditioner.T,
        preconditioner.A,
        regularization,
        K_nm.in_structure().dtype,
        K_nm.out_structure().dtype,
        jnp.float32,
    )
    T = preconditioner.T.astype(dtype)
    A = preconditioner.A.astype(dtype)
    ridge = jnp.asarray(regularization, dtype=dtype) * n
    K_mn = K_nm.transpose()

    # Each cross-kernel product runs in the operator's declared input dtype:
    # an implicit operator's scan carries that dtype and rejects a wider
    # vector. Only the results are promoted.
    def forward(w: Float[Array, " M"]) -> Float[Array, " N"]:
        return K_nm.mv(w.astype(K_nm.in_structure().dtype)).astype(dtype)

    def adjoint(u: Float[Array, " N"]) -> Float[Array, " M"]:
        return K_mn.mv(u.astype(K_mn.in_structure().dtype)).astype(dtype)

    def gram(w: Float[Array, " M"]) -> Float[Array, " M"]:
        return adjoint(forward(w))

    def preconditioned_system(beta: Float[Array, " M"]) -> Float[Array, " M"]:
        v = _solve_upper(A, beta)
        w = _solve_upper(T, v)
        c = _solve_upper(T, gram(w), trans=1) + ridge * v
        return _solve_upper(A, c, trans=1)

    operator = lx.FunctionLinearOperator(
        preconditioned_system,
        jax.ShapeDtypeStruct((m,), dtype),
        lx.positive_semidefinite_tag,
    )
    projected = adjoint(y.astype(dtype))
    rhs = _solve_upper(A, _solve_upper(T, projected, trans=1), trans=1)
    solution = lx.linear_solve(
        operator,
        rhs,
        lx.CG(rtol=tol, atol=0.0, max_steps=max_iter),
        throw=False,
    )
    return _solve_upper(T, _solve_upper(A, solution.value))


def falkon_predict(
    kernel_fn: Callable,
    X_inducing: Float[Array, "M D"],
    alpha: Float[Array, " M"],
    X_test: Float[Array, "Nt D"],
    *,
    batch_size: int = 1024,
    params: Any | None = None,
) -> Float[Array, " Nt"]:
    r"""Predict with Nyström KRR weights: $f(x_*) = \sum_j \alpha_j\, k(x_*, z_j)$.

    Evaluates $K(X_*, Z)\, \alpha$ through an `ImplicitCrossKernelOperator`,
    so the ``(Nt, M)`` test kernel is streamed in ``batch_size`` rows and
    never held in memory at once.

    Args:
        kernel_fn: Kernel ``k(x, z) -> scalar``, or ``k(params, x, z)`` when
            ``params`` is given -- the same signatures as
            `ImplicitCrossKernelOperator`.
        X_inducing: Inducing points $Z$, shape ``(M, D)``.
        alpha: Nyström weights, e.g. from `falkon_solve`, shape ``(M,)``.
        X_test: Test points, shape ``(Nt, D)``.
        batch_size: Test rows evaluated per scan step.
        params: Optional kernel hyperparameters.

    Returns:
        Predictions at ``X_test``, shape ``(Nt,)``.

    Raises:
        ValueError: If ``alpha`` does not have one weight per inducing point.
    """
    alpha = jnp.asarray(alpha)
    if alpha.shape != (X_inducing.shape[0],):
        raise ValueError(
            f"alpha must have shape ({X_inducing.shape[0]},), one weight per "
            f"inducing point, got {alpha.shape}."
        )
    cross = ImplicitCrossKernelOperator(
        kernel_fn, X_test, X_inducing, batch_size, params=params
    )
    return cross.mv(alpha)


def _solve_upper(factor: Float[Array, "M M"], rhs: Array, trans: int = 0) -> Array:
    """Solve ``factor x = rhs`` (``trans=1``: ``factorᵀ x = rhs``), upper."""
    return jax.scipy.linalg.solve_triangular(factor, rhs, lower=False, trans=trans)
