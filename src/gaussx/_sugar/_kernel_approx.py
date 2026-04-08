"""Kernel approximation sugar: Nystrom, RFF, centering, HSIC, MMD."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._primitives._cholesky import cholesky


def nystrom_operator(
    K_XZ: Float[Array, "N M"],
    K_ZZ_op: lx.AbstractLinearOperator,
) -> LowRankUpdate:
    r"""Nystrom low-rank kernel approximation.

    Approximates ``K_{XX} \approx K_{XZ} K_{ZZ}^{-1} K_{ZX}`` as a
    :class:`~gaussx.LowRankUpdate` with zero base::

        K_{XX} \approx U D U^T

    where ``U = K_{XZ} L_{ZZ}^{-T}`` and ``D = I`` (i.e. ``UU^T``),
    with ``L_{ZZ} = cholesky(K_{ZZ})``.

    Args:
        K_XZ: Cross-covariance between data and inducing points,
            shape ``(N, M)``.
        K_ZZ_op: Inducing-point covariance operator, shape ``(M, M)``.

    Returns:
        :class:`~gaussx.LowRankUpdate` operator of shape ``(N, N)``.
    """

    L = cholesky(K_ZZ_op)
    # U = K_XZ @ L^{-T} = solve(L^T, K_XZ^T)^T
    # Solve L @ A_col = K_XZ^T_col for each column
    from gaussx._sugar._linalg import solve_columns

    K_ZX = K_XZ.T  # (M, N)
    A = solve_columns(L, K_ZX)
    U = A.T  # (N, M)

    N = K_XZ.shape[0]
    M = K_XZ.shape[1]
    base = lx.DiagonalLinearOperator(jnp.zeros(N))
    D = jnp.ones(M)
    return LowRankUpdate(
        base=base,
        U=U,
        d=D,
        V=U,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def rff_operator(
    X: Float[Array, "N D"],
    omega: Float[Array, "D_rff D"],
    b: Float[Array, " D_rff"],
) -> LowRankUpdate:
    r"""Random Fourier Features kernel approximation.

    Approximates ``K_{XX} \approx \Phi \Phi^T`` where::

        \Phi_{i,j} = \sqrt{2/D_{rff}} \cos(X_i \cdot \omega_j + b_j)

    Returns a :class:`~gaussx.LowRankUpdate` that never materializes
    the ``N x N`` matrix.

    Args:
        X: Data points, shape ``(N, D)``.
        omega: Random frequencies, shape ``(D_rff, D)``.
            Sample from the spectral density of the kernel.
        b: Random phase offsets, shape ``(D_rff,)``.
            Sample uniformly from ``[0, 2*pi]``.

    Returns:
        :class:`~gaussx.LowRankUpdate` operator of shape ``(N, N)``.
    """
    D_rff = omega.shape[0]
    N = X.shape[0]
    Phi = jnp.sqrt(2.0 / D_rff) * jnp.cos(X @ omega.T + b[None, :])  # (N, D_rff)

    base = lx.DiagonalLinearOperator(jnp.zeros(N))
    D = jnp.ones(D_rff)
    return LowRankUpdate(
        base=base,
        U=Phi,
        d=D,
        V=Phi,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def centering_operator(n: int) -> LowRankUpdate:
    r"""Centering matrix ``H = I - (1/n) \mathbf{1}\mathbf{1}^T``.

    Returns as a :class:`~gaussx.LowRankUpdate` so the structure is
    preserved for downstream operations like ``H K H``.

    Args:
        n: Dimension of the centering matrix.

    Returns:
        :class:`~gaussx.LowRankUpdate` operator of shape ``(n, n)``.
    """
    base = lx.DiagonalLinearOperator(jnp.ones(n))
    ones = jnp.ones((n, 1))
    D = jnp.array([-1.0 / n])
    return LowRankUpdate(base=base, U=ones, d=D, V=ones, tags=frozenset())


def center_kernel(
    K: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    r"""Center a kernel matrix: ``H K H``.

    Computes the centered Gram matrix where ``H = I - (1/n) 11^T``.
    Used in kernel PCA, HSIC, and other centered kernel methods.

    Args:
        K: Kernel (Gram) matrix operator, shape ``(n, n)``.

    Returns:
        Centered kernel operator, shape ``(n, n)``.
    """
    K_mat = K.as_matrix()
    row_mean = jnp.mean(K_mat, axis=1, keepdims=True)
    col_mean = jnp.mean(K_mat, axis=0, keepdims=True)
    total_mean = jnp.mean(K_mat)
    K_centered = K_mat - row_mean - col_mean + total_mean
    tags = frozenset()
    if lx.is_symmetric(K):
        tags = frozenset({lx.symmetric_tag})
    return lx.MatrixLinearOperator(K_centered, tags)


def hsic(
    K_f: lx.AbstractLinearOperator,
    K_q: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    r"""Biased HSIC estimator.

    Computes the Hilbert-Schmidt Independence Criterion::

        HSIC = (1/n^2) \mathrm{tr}(K_f H K_q H)

    where ``H = I - (1/n) 11^T`` is the centering matrix.

    Args:
        K_f: First kernel matrix, shape ``(n, n)``.
        K_q: Second kernel matrix, shape ``(n, n)``.

    Returns:
        Scalar HSIC estimate.
    """
    from gaussx._sugar._linalg import trace_product

    K_f_centered = center_kernel(K_f)
    K_q_centered = center_kernel(K_q)
    n = K_f_centered.as_matrix().shape[0]
    return trace_product(K_f_centered, K_q_centered) / (n * n)


def mmd_squared(
    K_xx: Float[Array, "Nx Nx"],
    K_yy: Float[Array, "Ny Ny"],
    K_xy: Float[Array, "Nx Ny"],
) -> Float[Array, ""]:
    r"""Biased squared Maximum Mean Discrepancy.

    Computes::

        MMD^2 = mean(K_{xx}) + mean(K_{yy}) - 2 \cdot mean(K_{xy})

    Args:
        K_xx: Kernel matrix within first sample, shape ``(Nx, Nx)``.
        K_yy: Kernel matrix within second sample, shape ``(Ny, Ny)``.
        K_xy: Cross-kernel matrix between samples, shape ``(Nx, Ny)``.

    Returns:
        Scalar biased MMD^2 estimate.
    """
    return jnp.mean(K_xx) + jnp.mean(K_yy) - 2.0 * jnp.mean(K_xy)
