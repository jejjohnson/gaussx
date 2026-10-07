"""Sparse-GP conditional via Schur complement (`sparse_conditional`)."""

from __future__ import annotations

import einx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange, reduce, repeat
from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky
from gaussx._strategies._base import AbstractSolverStrategy


def _as_psd_operator(
    K: Float[Array, "M M"] | lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Wrap a dense array as a PSD operator; pass operators through."""
    if isinstance(K, lx.AbstractLinearOperator):
        return K
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)


def sparse_conditional(
    K_zz: Float[Array, "M M"] | lx.AbstractLinearOperator,
    K_xz: Float[Array, "N M"],
    K_xx: Float[Array, "N N"] | Float[Array, " N"],
    q_mu: Float[Array, "M R"] | Float[Array, " M"],
    *,
    q_sqrt: Float[Array, "R M M"]
    | Float[Array, "M R"]
    | Float[Array, "M M"]
    | Float[Array, " M"]
    | None = None,
    white: bool = False,
) -> tuple[Float[Array, "N R"], Float[Array, ...]]:
    r"""Sparse-GP conditional ``q(f_x) = \int p(f_x | u) q(u) du``.

    Follows the `gaussx._gp` sparse-GP convention: inducing covariance
    ``K_zz`` first (array or operator), cross-covariance ``K_xz`` with
    shape ``(N, M)`` (data × inducing), prior ``K_xx`` at the evaluation
    points, variational ``q(u) = N(q_mu, q_sqrt q_sqrtᵀ)``.

    With ``L = chol(K_zz)`` and ``A = L^{-1} K_xz^\top`` (shape ``(M, N)``),

    $$
    \begin{aligned}
    \mu &= A^\top L^{-1} q_\mu
        \;(= K_{xz} K_{zz}^{-1} q_\mu), \qquad
        \mu = A^\top q_\mu \text{ if white}, \\
    \Sigma_r &= K_{xx} - A^\top A + W_r^\top W_r, \qquad
    W_r = S_r^\top P, \quad
    P = \begin{cases} A & \text{white} \\ L^{-\top} A & \text{otherwise}
        \end{cases},
    \end{aligned}
    $$

    where ``S_r`` is the ``r``-th ``q_sqrt`` factor (``diag(q_sqrt[:, r])``
    for the diagonal layout). Only the diagonal of ``Σ_r`` is formed when
    ``K_xx`` is a vector. ``K_zz`` is factorised with `gaussx.cholesky`
    and every solve with its factor is a structured triangular solve, so a
    `Kronecker` or `BlockDiag` ``K_zz`` is never densified.

    Args:
        K_zz: Prior covariance at the inducing points, ``(M, M)`` array or
            PSD operator.
        K_xz: Cross-covariance ``k(X, Z)``, shape ``(N, M)``.
        K_xx: Prior covariance at ``X``: full ``(N, N)`` or diagonal ``(N,)``.
        q_mu: Variational mean ``(M, R)``, or ``(M,)`` for a single output
            (the outputs then drop their trailing ``R`` axis).
        q_sqrt: Optional variational root. Full ``(R, M, M)`` (lower
            triangular) or diagonal ``(M, R)``; with a 1-D ``q_mu`` also
            ``(M, M)`` or ``(M,)``.
        white: If ``True``, ``q_mu`` and ``q_sqrt`` are in whitened space
            (prior ``N(0, I)``).

    Returns:
        ``(mean, var)``: ``mean`` ``(N, R)``; ``var`` ``(N, N, R)`` (full
        ``K_xx``) or ``(N, R)`` (diagonal); no trailing ``R`` for a 1-D
        ``q_mu``. Diagonal variances are clipped at 0 against round-off; a
        full covariance is returned unclipped.

    Raises:
        ValueError: If ``q_mu`` / ``q_sqrt`` do not match ``M`` and ``R``.
    """
    K_zz_op = _as_psd_operator(K_zz)
    M = K_zz_op.in_size()
    single_output = q_mu.ndim == 1
    f, q_sqrt = _check_shapes(M, q_mu, q_sqrt)
    R = f.shape[1]

    L = cholesky(K_zz_op)  # lower factor, structured where possible
    A = solve_columns(L, rearrange(K_xz, "n m -> m n"))  # (M, N)

    if white:
        mean = einsum(A, f, "m n, m r -> n r")
        P = A
    else:
        mean = einsum(A, solve_columns(L, f), "m n, m r -> n r")
        P = solve_columns(L.transpose(), A)  # L^{-T} A = K_zz^{-1} K_zx

    full_cov = K_xx.ndim == 2
    if full_cov:
        var_base = K_xx - einsum(A, A, "m i, m j -> i j")  # (N, N)
    else:
        var_base = K_xx - reduce(A**2, "m n -> n", "sum")  # (N,)

    if q_sqrt is None:
        if full_cov:
            var = repeat(var_base, "i j -> i j r", r=R)
        else:
            var = repeat(var_base, "n -> n r", r=R)
    else:
        if q_sqrt.ndim == 2:  # (M, R) diagonal roots
            W = einx.multiply("m r, m n -> r m n", q_sqrt, P)
        else:  # (R, M, M) lower-triangular roots: W_r = S_rᵀ P
            W = einsum(q_sqrt, P, "r m k, m n -> r k n")
        if full_cov:
            var = einx.add(
                "i j, i j r -> i j r", var_base, einsum(W, W, "r k i, r k j -> i j r")
            )
        else:
            var = einx.add(
                "n, n r -> n r", var_base, reduce(W**2, "r k n -> n r", "sum")
            )

    if not full_cov:
        # K_xx - diag(Q_xx) can round below zero (gh-363).
        var = jnp.maximum(var, 0.0)
    if single_output:
        return mean[:, 0], var[..., 0]
    return mean, var


def base_conditional(
    K_mm: Float[Array, "M M"],
    K_mn: Float[Array, "M N"],
    K_nn: Float[Array, "N N"] | Float[Array, " N"],
    f: Float[Array, "M R"] | Float[Array, " M"],
    *,
    q_sqrt: Float[Array, "R M M"]
    | Float[Array, "M R"]
    | Float[Array, "M M"]
    | Float[Array, " M"]
    | None = None,
    white: bool = False,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, "N R"], Float[Array, ...]]:
    r"""Deprecated: use `sparse_conditional` (note ``K_xz = K_mn.T``).

    ``base_conditional(K_mm, K_mn, K_nn, f, q_sqrt=..., white=...)`` equals
    ``sparse_conditional(K_mm, K_mn.T, K_nn, f, q_sqrt=..., white=...)``.
    It takes the cross-covariance as ``(M, N)`` (inducing × data), the
    transpose of every other sparse-GP helper; for ``M == N`` a transposed
    argument is not detectable, so the convention is changed under a new
    name rather than in place. ``solver`` was never used. Removed in
    gaussx 0.7.0.

    Args:
        K_mm: Prior covariance at inducing points, shape ``(M, M)``.
        K_mn: Cross-covariance, shape ``(M, N)``.
        K_nn: Test-point covariance, full ``(N, N)`` or diagonal ``(N,)``.
        f: Inducing function values, shape ``(M, R)`` or ``(M,)``.
        q_sqrt: Optional variational root, as in `sparse_conditional`.
        white: Whitened parameterisation, as in `sparse_conditional`.
        solver: Ignored.

    Returns:
        ``(mean, var)`` as in `sparse_conditional`.
    """
    del solver
    warn_deprecated(
        "base_conditional(K_mm, K_mn, ...) is deprecated and will be removed in "
        "gaussx 0.7.0; use sparse_conditional(K_zz, K_xz, K_xx, q_mu, "
        "...) with K_xz = K_mn.T (shape (N, M))."
    )
    return sparse_conditional(
        K_mm, rearrange(K_mn, "m n -> n m"), K_nn, f, q_sqrt=q_sqrt, white=white
    )


def _check_shapes(
    M: int,
    f: Float[Array, "M R"] | Float[Array, " M"],
    q_sqrt: Array | None,
) -> tuple[Float[Array, "M R"], Array | None]:
    """Validate ``f`` / ``q_sqrt`` and promote the single-output layouts."""
    if f.ndim == 1:
        f = rearrange(f, "m -> m 1")
        if q_sqrt is not None and q_sqrt.shape == (M,):
            q_sqrt = rearrange(q_sqrt, "m -> m 1")
        elif q_sqrt is not None and q_sqrt.shape == (M, M):
            q_sqrt = rearrange(q_sqrt, "m k -> 1 m k")
    if f.ndim != 2 or f.shape[0] != M:
        raise ValueError(f"f must have shape (M, R) or (M,) with M={M}, got {f.shape}.")
    R = f.shape[1]
    if q_sqrt is not None and q_sqrt.shape not in ((M, R), (R, M, M)):
        raise ValueError(
            f"q_sqrt must have shape (M, R)=({M}, {R}) or (R, M, M)=({R}, {M}, {M}) "
            f"to match f, got {q_sqrt.shape}."
        )
    return f, q_sqrt
