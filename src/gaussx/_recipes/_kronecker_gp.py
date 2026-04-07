"""Kronecker GP recipes — exact inference on grid-structured data."""

from __future__ import annotations

import functools as ft

import jax.numpy as jnp
import lineax as lx
from einops import rearrange
from jaxtyping import Array, Float

from gaussx._primitives._eig import eig


def kronecker_mll(
    K_factors: list[lx.AbstractLinearOperator],
    y: Float[Array, " N"],
    noise_var: float,
    grid_shape: tuple[int, ...],
) -> Float[Array, ""]:
    r"""Exact marginal log-likelihood for a Kronecker-structured GP.

    For a GP with covariance ``K = K_1 \otimes K_2 \otimes \ldots + sigma^2 I``,
    computes the log marginal likelihood via per-factor eigendecomposition::

        log p(y) = -0.5 * (y^T (K + sigma^2 I)^{-1} y
                           + log|K + sigma^2 I|
                           + N log(2 pi))

    The Kronecker eigendecomposition avoids forming the full ``N x N`` matrix:
    if ``K_i = Q_i Lambda_i Q_i^T``, the combined eigenvalues are the outer
    products of per-factor eigenvalues and the combined eigenvectors are the
    Kronecker product of per-factor eigenvectors.

    Complexity: ``O(sum n_i^3 + N)`` instead of ``O(N^3)`` where ``N = prod n_i``.

    Args:
        K_factors: List of per-dimension kernel operators. Each must be
            square and symmetric.
        y: Observations, shape ``(N,)`` where ``N = prod(grid_shape)``.
        noise_var: Observation noise variance ``sigma^2``.
        grid_shape: Shape of the grid, e.g. ``(n1, n2)`` for 2D.

    Returns:
        Scalar log marginal likelihood.
    """
    N = y.shape[0]

    # Per-factor eigendecomposition
    factor_eigs = [eig(K_i) for K_i in K_factors]
    all_vals = [vals for vals, _ in factor_eigs]
    all_vecs = [vecs for _, vecs in factor_eigs]

    # Combined eigenvalues: outer product of per-factor eigenvalues
    combined_vals = ft.reduce(jnp.kron, all_vals)  # (N,)
    noisy_vals = combined_vals + noise_var  # (N,)

    # Rotate data into eigenbasis: alpha = Q^T y
    # Q = Q_1 kron Q_2 kron ..., so Q^T y can be computed factor-by-factor
    alpha = _kron_rotate(all_vecs, y, grid_shape)

    # MLL in eigenbasis: all operations are O(N)
    data_fit = jnp.sum(alpha**2 / noisy_vals)
    log_det = jnp.sum(jnp.log(noisy_vals))
    const = N * jnp.log(2.0 * jnp.pi)

    return -0.5 * (data_fit + log_det + const)


def kronecker_posterior_predictive(
    K_factors: list[lx.AbstractLinearOperator],
    y: Float[Array, " N"],
    noise_var: float,
    grid_shape: tuple[int, ...],
    K_cross_factors: list[Float[Array, "Ni_test Ni_train"]],
    *,
    K_test_diag_factors: list[Float[Array, " Ni_test"]],
) -> tuple[Float[Array, " N_test"], Float[Array, " N_test"]]:
    r"""Posterior mean and variance for a Kronecker GP at test points.

    Uses the eigendecomposition trick: projects cross-covariances onto
    the eigenbasis and weights by inverse eigenvalues::

        mu_* = K_{*f} (K_{ff} + sigma^2 I)^{-1} y
        var_* = k_{**} - K_{*f} (K_{ff} + sigma^2 I)^{-1} K_{f*}

    Both computed via per-factor eigendecomposition in
    ``O(sum n_i^3 + N + N_test)`` time.

    Args:
        K_factors: List of per-dimension training kernel operators.
        y: Observations, shape ``(N,)`` where ``N = prod(grid_shape)``.
        noise_var: Observation noise variance ``sigma^2``.
        grid_shape: Grid shape, e.g. ``(n1, n2)``.
        K_cross_factors: Per-dimension cross-covariance matrices,
            each shape ``(n_i_test, n_i_train)``.
        K_test_diag_factors: Per-dimension prior diagonals at the test points,
            each shape ``(n_i_test,)``.

    Returns:
        Tuple ``(mean, variance)`` at test points.
    """
    if len(K_factors) != len(grid_shape):
        msg = "grid_shape must have one entry per Kronecker factor"
        raise ValueError(msg)
    if len(K_cross_factors) != len(K_factors):
        msg = "K_cross_factors must have one matrix per Kronecker factor"
        raise ValueError(msg)
    if len(K_test_diag_factors) != len(K_factors):
        msg = "K_test_diag_factors must have one vector per Kronecker factor"
        raise ValueError(msg)

    # Per-factor eigendecomposition of training kernels
    factor_eigs = [eig(K_i) for K_i in K_factors]
    all_vals = [vals for vals, _ in factor_eigs]
    all_vecs = [vecs for _, vecs in factor_eigs]

    # Combined training eigenvalues
    combined_vals = ft.reduce(jnp.kron, all_vals)
    inv_noisy_vals = 1.0 / (combined_vals + noise_var)

    # Rotate observations: alpha = Q^T y
    alpha = _kron_rotate(all_vecs, y, grid_shape)

    # Weights in eigenbasis: w = diag(1/(lambda + sigma^2)) Q^T y
    w = inv_noisy_vals * alpha

    # Per-factor cross-covariance projected onto eigenbasis: A_i = K_cross_i @ Q_i
    A_factors = [K_cross_factors[i] @ all_vecs[i] for i in range(len(K_factors))]

    # Posterior mean: mu_* = (A_1 kron A_2 kron ...) w
    # Mean: project back from eigenbasis
    mean = _kron_matvec(A_factors, w, grid_shape)

    # Variance: k_** - sum_j (A_j^2 / (lambda_j + sigma^2))
    # = k_** - (A_1 kron A_2 kron ...)^2 @ inv_noisy_vals element-wise
    A_sq_factors = [A**2 for A in A_factors]
    var_reduction = _kron_matvec(A_sq_factors, inv_noisy_vals, grid_shape)

    # Prior diagonal at test points: diag(K_**) = kron_i diag(K_i(test, test))
    K_test_prior = ft.reduce(jnp.kron, K_test_diag_factors)

    variance = jnp.clip(K_test_prior - var_reduction, 0.0)

    return mean, variance


def _kron_rotate(
    vecs_list: list[Float[Array, "Ni Ni"]],
    y: Float[Array, " N"],
    grid_shape: tuple[int, ...],
) -> Float[Array, " N"]:
    r"""Compute ``(Q_1 \otimes Q_2 \otimes ...)^T y`` factor-by-factor.

    Reshapes y to the grid, applies each ``Q_i^T`` along its axis,
    then flattens back.
    """
    x = rearrange(y, "(total) -> total", total=y.shape[0])
    x = x.reshape(grid_shape)

    for i, Q_i in enumerate(vecs_list):
        # Move axis i to last position, apply Qᵢᵀ, move back
        x = jnp.moveaxis(x, i, -1)
        x = x @ Q_i  # (..., n_i) @ (n_i, n_i) -> (..., n_i)
        x = jnp.moveaxis(x, -1, i)

    return rearrange(x, "... -> (...)")


def _kron_matvec(
    A_factors: list[Float[Array, "Mi Ni"]],
    v: Float[Array, " N"],
    grid_shape: tuple[int, ...],
) -> Float[Array, " M"]:
    r"""Compute ``(A_1 \otimes A_2 \otimes ...) v`` factor-by-factor.

    Similar to the Roth column lemma approach.
    """
    x = v.reshape(grid_shape)

    out_shape = []
    for i, A_i in enumerate(A_factors):
        x = jnp.moveaxis(x, i, -1)
        x = x @ A_i.T  # (..., n_i) @ (n_i, m_i) -> (..., m_i)
        out_shape.append(A_i.shape[0])
        x = jnp.moveaxis(x, -1, i)

    return rearrange(x, "... -> (...)")
