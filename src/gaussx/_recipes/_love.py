"""LOVE — Lanczos Variance Estimates for fast GP predictive variance."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._eig import eig


class LOVECache(eqx.Module):
    r"""Cached Lanczos factorization for fast predictive variance.

    Stores the eigenvector basis ``Q`` and inverse eigenvalues such that
    ``K^{-1} \approx Q \Lambda^{-1} Q^T``.

    Attributes:
        Q: Lanczos eigenvector basis, shape ``(N, k)``.
        inv_eigvals: Inverse eigenvalues ``1 / lambda_i``, shape ``(k,)``.
    """

    Q: Float[Array, "N k"]
    inv_eigvals: Float[Array, " k"]


def love_cache(
    K_op: lx.AbstractLinearOperator,
    lanczos_order: int = 50,
    key: jnp.ndarray | None = None,
) -> LOVECache:
    r"""Precompute Lanczos factorization of ``K^{-1}`` for fast variance.

    Builds a rank-``k`` approximation ``K^{-1} \approx Q \Lambda^{-1} Q^T``
    using the symmetric Lanczos algorithm via partial eigendecomposition.

    This amortizes the cost of predictive variance: once cached, each
    test point needs only ``O(Nk)`` instead of ``O(N^2)`` for a solve.

    Args:
        K_op: Training kernel operator, shape ``(N, N)``. Must be
            symmetric positive definite.
        lanczos_order: Number of Lanczos iterations (rank of approximation).
            Default ``50``.
        key: PRNG key for the initial random vector. If ``None``, uses
            ``jax.random.PRNGKey(0)``.

    Returns:
        A :class:`LOVECache` object.
    """
    import jax.random as jr

    if key is None:
        key = jr.PRNGKey(0)

    n = K_op.in_size()
    lanczos_order = min(lanczos_order, n)

    # Partial eigendecomposition via matfree Lanczos
    eigvals, eigvecs = eig(K_op, rank=lanczos_order, key=key)

    # Guard against complex values from numerical round-off in
    # Lanczos / eigendecomposition of ill-conditioned matrices.
    eigvals = jnp.real(eigvals)
    eigvecs = jnp.real(eigvecs)

    # Clamp small / negative eigenvalues to a safe floor so that
    # 1/lambda stays finite and positive.
    floor = jnp.finfo(eigvals.dtype).tiny
    eigvals = jnp.maximum(eigvals, floor)

    # Invert eigenvalues for K^{-1} approximation
    inv_eigvals = 1.0 / eigvals

    return LOVECache(Q=eigvecs, inv_eigvals=inv_eigvals)


def love_variance(
    cache: LOVECache,
    K_star_row: Float[Array, " N"],
) -> Float[Array, ""]:
    r"""Fast predictive variance using a LOVE cache.

    Computes ``k_*^T K^{-1} k_*`` in ``O(Nk)`` via the cached Lanczos
    factorization::

        k_*^T K^{-1} k_* \approx k_*^T Q \Lambda^{-1} Q^T k_*
                              = \sum_i (q_i^T k_*)^2 / \lambda_i

    The predictive variance for a GP is then::

        var_* = k(x_*, x_*) - love\_variance(cache, k_*)

    Args:
        cache: A :class:`LOVECache` from :func:`love_cache`.
        K_star_row: Cross-covariance vector ``k(X_{train}, x_*)``,
            shape ``(N,)``.

    Returns:
        Scalar ``k_*^T K^{-1} k_*``.
    """
    # Project onto eigenvector basis: z = Q^T k_*  -> (k,)
    z = cache.Q.T @ K_star_row
    # Weighted sum: sum_i z_i^2 / lambda_i
    return jnp.sum(z**2 * cache.inv_eigvals)
