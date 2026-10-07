"""LOVE — Lanczos Variance Estimates for fast GP predictive variance."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._primitives._root import _safe_psd, _top_real_eigenpairs


class LOVECache(eqx.Module):
    r"""Cached rank-``k`` Lanczos approximation of ``K^{-1}``.

    ``k`` steps of symmetric Lanczos from a start vector ``v_0`` build an
    orthonormal basis of the Krylov space
    ``\mathcal{K}_k(K, v_0) = \mathrm{span}\{v_0, K v_0, \dots, K^{k-1} v_0\}``
    and the tridiagonal ``T_k = Q_k^\top K Q_k``. With the eigendecomposition
    ``T_k = S \Theta S^\top``, the cache stores the Ritz vectors
    ``Q = Q_k S`` and the inverse Ritz values ``1 / \theta_i``, so that

    $$
    K^{-1} \approx Q_k T_k^{-1} Q_k^\top = Q \Theta^{-1} Q^\top .
    $$

    Ritz pairs approximate eigenpairs of ``K`` only once they have converged
    (first at the extremes of the spectrum); for ``k < N`` they are not the
    eigenpairs of ``K``. ``K^{-1} - Q \Theta^{-1} Q^\top`` is positive
    semi-definite, so the approximation never over-estimates a quadratic form
    ``k_*^\top K^{-1} k_*`` (see `love_variance`).

    Attributes:
        Q: Ritz vectors (an orthonormal basis of the Krylov space),
            shape ``(N, k)``.
        inv_eigvals: Inverse Ritz values ``1 / \theta_i``, shape ``(k,)``.
    """

    Q: Float[Array, "N k"]
    inv_eigvals: Float[Array, " k"]


def love_cache(
    K_op: lx.AbstractLinearOperator,
    lanczos_order: int = 50,
    key: jax.Array | None = None,
    *,
    initial_vector: Float[Array, " N"] | None = None,
) -> LOVECache:
    r"""Precompute a Lanczos approximation of ``K^{-1}`` for fast variance.

    Runs ``k = min(lanczos_order, N)`` steps of symmetric Lanczos (with full
    reorthogonalisation) on ``K`` from the start vector ``v_0`` and returns
    the Galerkin approximation

    $$
    K^{-1} \approx Q_k T_k^{-1} Q_k^\top
    = K^{-1/2} P_k K^{-1/2},
    $$

    where ``P_k`` is the orthogonal projector onto
    ``K^{1/2} \mathcal{K}_k(K, v_0)``. Because ``P_k \preceq I`` the
    approximation is a lower bound on ``K^{-1}`` in the Loewner order: the
    quadratic form returned by `love_variance` is biased low and the GP
    predictive variance ``k_{**} - k_*^\top K^{-1} k_*`` built from it is
    biased **high**, until the Krylov space captures every direction in
    which ``k_*`` has weight. At ``k = N`` the result is exact.

    Choosing ``lanczos_order``: the required ``k`` is set by the number of
    eigenvalues of ``K`` well above the noise level (the kernel's effective
    rank at the noise variance), not by ``N``. A smooth kernel with small
    noise can need noticeably more steps than its numerical rank suggests.
    Check convergence with `love_residual` on representative test points.

    Pseudocode:

    ```text
    v0 = initial_vector or normal(key, (N,))
    Q_k, T_k = lanczos(K, v0, k)            # k matvecs with K
    theta, S = eigh(T_k)
    Q = Q_k @ S;  inv_eigvals = 1 / theta
    ```

    Once cached, each test point costs ``O(Nk)`` instead of an ``O(N^2)``
    solve (Pleiss et al., 2018).

    Args:
        K_op: Training kernel operator ``K``, shape ``(N, N)``. Must be
            symmetric positive definite.
        lanczos_order: Number of Lanczos steps ``k`` (rank of the
            approximation), clamped to ``N``. Default ``50``.
        key: PRNG key for a random start vector. If ``None`` (and no
            ``initial_vector`` is given), uses ``jax.random.PRNGKey(0)``.
        initial_vector: Deterministic start vector ``v_0``, shape ``(N,)``,
            e.g. the training targets ``y`` (the right-hand side of the mean
            solve, as in LOVE). Takes precedence over ``key``.

    Returns:
        A `LOVECache` object.
    """
    n = K_op.in_size()
    rank = min(lanczos_order, n)
    dtype = K_op.in_structure().dtype
    if initial_vector is None:
        if key is None:
            key = jr.PRNGKey(0)
        v0 = jr.normal(key, (n,), dtype=dtype)
    else:
        v0 = jnp.asarray(initial_vector, dtype=dtype)

    tridiag = matfree.decomp.tridiag_sym(rank, reortho="full")
    # matfree returns Ritz values (k,) and Ritz vectors (k, N).
    vals, vecs = matfree.eig.eigh_partial(tridiag)(K_op.mv, v0)
    vals, Q = _top_real_eigenpairs(vals, rearrange(vecs, "k n -> n k"), rank)
    return LOVECache(Q=Q, inv_eigvals=1.0 / _safe_psd(vals))


def love_variance(
    cache: LOVECache,
    K_star_row: Float[Array, " N"],
) -> Float[Array, ""]:
    r"""Approximate ``k_*^\top K^{-1} k_*`` from a LOVE cache.

    Computes, in ``O(Nk)``,

    $$
    k_*^\top Q \Theta^{-1} Q^\top k_*
    = \sum_i (q_i^\top k_*)^2 / \theta_i
    \;\le\; k_*^\top K^{-1} k_* .
    $$

    The inequality is the one-signed bias of the Lanczos approximation
    (see `love_cache`): the GP predictive variance

    $$
    \sigma_*^2 = k_{**} - \mathrm{love\_variance}(\mathrm{cache}, k_*)
    $$

    is an **upper bound** on the exact predictive variance, tight once the
    cache has converged for this ``k_*`` (check with `love_residual`).

    Args:
        cache: A `LOVECache` from `love_cache`.
        K_star_row: Cross-covariance vector ``k(X_{train}, x_*)``,
            shape ``(N,)``.

    Returns:
        Scalar approximation of ``k_*^T K^{-1} k_*``.
    """
    z = einsum(cache.Q, K_star_row, "n k, n -> k")
    return jnp.sum(z**2 * cache.inv_eigvals)


def love_residual(
    cache: LOVECache,
    K_op: lx.AbstractLinearOperator,
    K_star_row: Float[Array, " N"],
) -> Float[Array, ""]:
    r"""Relative solve residual of a LOVE cache for one cross-covariance.

    Convergence diagnostic for `love_variance`: with
    ``a = Q \Theta^{-1} Q^\top k_*`` (the cache's approximation of
    ``K^{-1} k_*``), returns

    $$
    r(k_*) = \frac{\lVert K a - k_* \rVert_2}{\lVert k_* \rVert_2}.
    $$

    ``r = 0`` exactly when the cache solves ``K a = k_*``, in which case
    `love_variance` is exact for this ``k_*``. A value well above the
    working precision (e.g. ``> 1e-3``) means ``lanczos_order`` is too
    small for this test point and the predictive variance is
    over-estimated. Costs one matvec with ``K``.

    Args:
        cache: A `LOVECache` from `love_cache`.
        K_op: The operator the cache was built from, shape ``(N, N)``.
        K_star_row: Cross-covariance vector ``k(X_{train}, x_*)``,
            shape ``(N,)``.

    Returns:
        Scalar relative residual ``r(k_*)``.
    """
    z = einsum(cache.Q, K_star_row, "n k, n -> k") * cache.inv_eigvals
    a = einsum(cache.Q, z, "n k, k -> n")
    return jnp.linalg.norm(K_op.mv(a) - K_star_row) / jnp.linalg.norm(K_star_row)
