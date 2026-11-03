"""LOVE: Lanczos variance estimates for fast GP predictive variances.

Σ_y is the N × N symmetric positive-definite training covariance (K_XX +
σ_ε² I for a GP) and k_X* ∈ ℝᴺ the cross-covariance column between the
training inputs and one test input x_*. k steps of the Lanczos recurrence
from v₁,

    α_j = v_jᵀ Σ_y v_j,   β_j+1 v_j+1 = Σ_y v_j − α_j v_j − β_j v_j−1,

build an orthonormal basis V_k = [v₁, …, v_k] of the Krylov space
span{v₁, Σ_y v₁, …, Σ_y^{k−1} v₁} and the tridiagonal
T_k = V_kᵀ Σ_y V_k = S diag(θ) Sᵀ, with Ritz values θ_j and Ritz vectors
Q = V_k S.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._primitives._root import (
    _safe_psd,
    _top_real_eigenpairs,
    root_inv_decomposition,
)


_EXACT_EIG = (lx.DiagonalLinearOperator, BlockDiag, Kronecker, KroneckerSum)


class LOVECache(eqx.Module):
    """Cached rank-k Lanczos approximation Q diag(θ)⁻¹ Qᵀ of Σ_y⁻¹.

        Σ_y⁻¹ ≈ V_k T_k⁻¹ V_kᵀ = Q diag(θ)⁻¹ Qᵀ

    Ritz pairs (θ_j, q_j) approximate eigenpairs of Σ_y only once they have
    converged (first at the extremes of the spectrum); for k < N they are
    not the eigenpairs of Σ_y. Σ_y⁻¹ − Q diag(θ)⁻¹ Qᵀ is positive
    semi-definite, so the approximation never over-estimates a quadratic
    form ``k_X*ᵀ Σ_y⁻¹ k_X*`` (see `love_variance`).

    Attributes:
        Q: Ritz vectors Q = V_k S, an orthonormal basis of the Krylov
            space, one per column.
        inv_eigvals: Inverse Ritz values 1 / θ_j.
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
    """Precompute a Lanczos approximation of Σ_y⁻¹ for fast GP variances.

        Σ_y⁻¹ ≈ V_k T_k⁻¹ V_kᵀ = Σ_y^−½ P_k Σ_y^−½

    with k = min(``lanczos_order``, N) steps of symmetric Lanczos (with full
    reorthogonalisation) from v₁ and P_k the orthogonal projector onto
    Σ_y^½ span{v₁, …, Σ_y^{k−1} v₁}. Because P_k ⪯ I the approximation is a
    lower bound on Σ_y⁻¹ in the Loewner order: the quadratic form returned
    by `love_variance` is biased low and the GP predictive variance
    ``k_** − k_X*ᵀ Σ_y⁻¹ k_X*`` built from it is biased **high**, until the
    Krylov space captures every direction in which ``k_X*`` has weight. At
    k = N the result is exact.

    Choosing ``lanczos_order``: the required k is set by the number of
    eigenvalues of Σ_y well above the noise level (the kernel's effective
    rank at the noise variance), not by N. A smooth kernel with small
    noise can need noticeably more steps than its numerical rank suggests.
    Check convergence with `love_residual` / `love_variance_error_bound` on
    representative test points. Once cached, each test point costs O(N k)
    instead of an O(N²) solve.

    A `lineax.DiagonalLinearOperator`, `BlockDiag`, `Kronecker` or
    `KroneckerSum` Σ_y is not run through Lanczos: its exact structured
    eigendecomposition is truncated to the top k eigenpairs (``key`` and
    ``initial_vector`` are unused), whose eigenvalues γ_j take the place
    of the Ritz values θ_j. That is also a Galerkin approximation, with
    the same one-signed bias.

    Algorithm:
        1. For a structured Σ_y, take the inverse root
           R⁻ = Q_k diag(γ_k^−½) of its top k exact eigenpairs and split
           it into Q_k and 1 / γ_j.
        2. Otherwise draw v₁ from ``key``, or take ``initial_vector``
           (ones if it is zero).
        3. Run k Lanczos steps on Σ_y from v₁ and diagonalise
           T_k = S diag(θ) Sᵀ.
        4. Keep the k largest real Ritz pairs: Q = V_k S and 1 / θ_j.

    Args:
        K_op: Training covariance Σ_y, symmetric positive definite.
        lanczos_order: Number of Lanczos steps k (rank of the
            approximation), clamped to N. Default ``50``.
        key: PRNG key for a random start vector. If ``None`` (and no
            ``initial_vector`` is given), uses ``jax.random.PRNGKey(0)``.
        initial_vector: Deterministic start vector v₁, e.g. the training
            targets y (the right-hand side of the mean solve, as in LOVE).
            Takes precedence over ``key``. A zero vector is replaced by the
            all-ones vector.

    Returns:
        A `LOVECache` holding Q and 1 / θ_j.

    Examples:
        At k = N the cache reproduces Σ_y⁻¹:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> covariance = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> cache = gaussx.love_cache(covariance, 3, initial_vector=jnp.ones(3))
        >>> cache.Q.shape, cache.inv_eigvals.shape
        ((3, 3), (3,))
        >>> approx = cache.Q @ jnp.diag(cache.inv_eigvals) @ cache.Q.T
        >>> bool(jnp.allclose(approx, jnp.linalg.inv(matrix), atol=1e-5))
        True

        A diagonal Σ_y keeps its top k exact eigenpairs:

        >>> diagonal = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 4.0]))
        >>> top = gaussx.love_cache(diagonal, 2)
        >>> [round(float(value), 4) for value in top.inv_eigvals]
        [0.25, 0.5]

    References:
        Pleiss, Gardner, Weinberger and Wilson (2018), Constant-time
        predictive distributions for Gaussian processes, ICML.
    """
    size = K_op.in_size()
    rank = min(lanczos_order, size)
    if isinstance(K_op, _EXACT_EIG):
        # Exact (structured) eigendecomposition, top-``rank`` pairs: no
        # Krylov breakdown on repeated eigenvalues (e.g. a diagonal Σ_y).
        decomposition = root_inv_decomposition(
            K_op, rank=rank, method="lanczos", key=key
        )
        # R⁻ = Q_k diag(γ_k^−½)
        inverse_root = decomposition.root
        # [R⁻]²
        inverse_root_squared = inverse_root**2
        # 1 / γ_j = ‖R⁻ e_j‖₂², since the columns of Q_k are orthonormal
        inv_eigvals = reduce(inverse_root_squared, "sample ritz -> ritz", "sum")
        # max(1 / γ_j, tiny)
        floored_inv_eigvals = _safe_psd(inv_eigvals)
        # γ_j^−½
        scale = jnp.sqrt(floored_inv_eigvals)
        # Q_k = R⁻ diag(γ_k^½)
        eigenvectors = einx.divide(
            "sample ritz, ritz -> sample ritz", inverse_root, scale
        )
        return LOVECache(Q=eigenvectors, inv_eigvals=inv_eigvals)

    dtype = K_op.in_structure().dtype
    if initial_vector is None:
        if key is None:
            key = jr.PRNGKey(0)
        # v₁ ~ 𝒩(0, I), normalised inside the Lanczos run
        start_vector = jr.normal(key, (size,), dtype=dtype)
    else:
        start_vector = jnp.asarray(initial_vector, dtype=dtype)
        # A zero start vector would make the first Lanczos normalisation 0/0.
        # ‖v₁‖₂
        start_norm = jnp.linalg.norm(start_vector)
        is_nonzero = start_norm > 0
        start_vector = jnp.where(is_nonzero, start_vector, jnp.ones_like(start_vector))

    # --- Lanczos ---
    tridiag = matfree.decomp.tridiag_sym(rank, reortho="full")
    eigh_partial = matfree.eig.eigh_partial(tridiag)
    # T_k = S diag(θ) Sᵀ; matfree returns θ and the Ritz vectors as rows
    ritz_values, ritz_vector_rows = eigh_partial(K_op.mv, start_vector)
    # Q = V_k S, one Ritz vector per column
    ritz_vectors = rearrange(ritz_vector_rows, "ritz sample -> sample ritz")
    ritz_values, ritz_vectors = _top_real_eigenpairs(ritz_values, ritz_vectors, rank)
    # max(θ_j, tiny)
    floored_ritz_values = _safe_psd(ritz_values)
    # 1 / θ_j
    inv_eigvals = 1.0 / floored_ritz_values
    return LOVECache(Q=ritz_vectors, inv_eigvals=inv_eigvals)


def love_variance(
    cache: LOVECache,
    K_star_row: Float[Array, " N"],
) -> Float[Array, ""]:
    """Approximate the variance reduction ``k_X*ᵀ Σ_y⁻¹ k_X*`` from a LOVE cache.

        k_X*ᵀ Q diag(θ)⁻¹ Qᵀ k_X* = ∑_j (q_jᵀ k_X*)² / θ_j ≤ k_X*ᵀ Σ_y⁻¹ k_X*

    It costs O(N k). The inequality is the one-signed bias of the Lanczos
    approximation (see `love_cache`): the GP predictive variance

        σ²_* = k_** − love_variance(cache, k_X*)

    is an **upper bound** on the exact predictive variance, tight once the
    cache has converged for this ``k_X*`` (check with `love_residual`).

    Args:
        cache: A `LOVECache` from `love_cache`.
        K_star_row: Cross-covariance ``k_X*`` between the training inputs
            and the test input ``x_*``.

    Returns:
        The approximation of ``k_X*ᵀ Σ_y⁻¹ k_X*``.

    Examples:
        Exact at k = N, and a lower bound below it:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> covariance = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> cross = jnp.array([1.0, 0.0, 0.0])
        >>> exact = cross @ jnp.linalg.solve(matrix, cross)
        >>> full = gaussx.love_cache(covariance, 3, initial_vector=jnp.ones(3))
        >>> bool(jnp.allclose(gaussx.love_variance(full, cross), exact))
        True
        >>> short = gaussx.love_cache(covariance, 1, initial_vector=jnp.ones(3))
        >>> bool(gaussx.love_variance(short, cross) <= exact)
        True
    """
    # Qᵀ k_X*
    projection = einsum(cache.Q, K_star_row, "sample ritz, sample -> ritz")
    # (q_jᵀ k_X*)²
    projection_squared = projection**2
    # (q_jᵀ k_X*)² / θ_j
    weighted = projection_squared * cache.inv_eigvals
    # ∑_j (q_jᵀ k_X*)² / θ_j
    return jnp.sum(weighted)


def love_residual(
    cache: LOVECache,
    K_op: lx.AbstractLinearOperator,
    K_star_row: Float[Array, " N"],
) -> Float[Array, ""]:
    """Relative solve residual of a LOVE cache for one cross-covariance.

        r_rel(k_X*) = ‖Σ_y a − k_X*‖₂ / ‖k_X*‖₂,   a = Q diag(θ)⁻¹ Qᵀ k_X*

    A convergence diagnostic for `love_variance`, with a the cache's
    approximation of ``Σ_y⁻¹ k_X*``. r_rel = 0 exactly when the cache solves
    ``Σ_y a = k_X*``, in which case `love_variance` is exact for this
    ``k_X*`` (``k_X* = 0`` returns 0). A value well above the working precision means
    ``lanczos_order`` is too small for this test input. A *small* value is
    not sufficient on its own for an ill-conditioned Σ_y: the variance
    error is rᵀ Σ_y⁻¹ r with ``r = Σ_y a − k_X*``, which can be large for a
    small ‖r‖₂ when Σ_y has tiny eigenvalues; `love_variance_error_bound`
    turns the residual into a bound in variance units. Costs one matvec
    with Σ_y.

    Args:
        cache: A `LOVECache` from `love_cache`.
        K_op: The covariance Σ_y the cache was built from.
        K_star_row: Cross-covariance ``k_X*`` between the training inputs
            and the test input ``x_*``.

    Returns:
        The relative residual ``r_rel(k_X*)``.

    Examples:
        Converged at k = N, not after one step:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> covariance = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> cross = jnp.array([1.0, 0.0, 0.0])
        >>> full = gaussx.love_cache(covariance, 3, initial_vector=jnp.ones(3))
        >>> bool(gaussx.love_residual(full, covariance, cross) < 1e-4)
        True
        >>> short = gaussx.love_cache(covariance, 1, initial_vector=jnp.ones(3))
        >>> bool(gaussx.love_residual(short, covariance, cross) > 0.1)
        True
    """
    # r = Σ_y a − k_X*
    residual = _residual(cache, K_op, K_star_row)
    # ‖r‖₂
    residual_norm = jnp.linalg.norm(residual)
    # ‖k_X*‖₂
    cross_norm = jnp.linalg.norm(K_star_row)
    # k_X* = 0 is solved exactly (a = 0, r = 0): report 0, not 0/0.
    is_nonzero = cross_norm > 0
    safe_cross_norm = jnp.where(is_nonzero, cross_norm, 1.0)
    # r_rel = ‖r‖₂ / ‖k_X*‖₂
    relative_residual = residual_norm / safe_cross_norm
    return jnp.where(is_nonzero, relative_residual, 0.0)


def love_variance_error_bound(
    cache: LOVECache,
    K_op: lx.AbstractLinearOperator,
    K_star_row: Float[Array, " N"],
    lambda_min: float | Float[Array, ""],
) -> Float[Array, ""]:
    """Upper bound on the variance over-estimate of a LOVE cache.

        0 ≤ k_X*ᵀ Σ_y⁻¹ k_X* − love_variance(cache, k_X*) = rᵀ Σ_y⁻¹ r
          ≤ ‖r‖₂² / γ_min(Σ_y)

    The cache is a Galerkin approximation, so with ``a = Q diag(θ)⁻¹ Qᵀ k_X*``
    and ``r = Σ_y a − k_X*`` (Qᵀ r = 0) the error of the quadratic form is
    exactly rᵀ Σ_y⁻¹ r. Unlike the relative residual of `love_residual`,
    this is in the units of the variance and accounts for the conditioning
    of Σ_y. For a GP with Σ_y = K_XX + σ_ε² I the noise variance σ_ε² is a
    valid ``lambda_min``.

    Args:
        cache: A `LOVECache` from `love_cache`.
        K_op: The covariance Σ_y the cache was built from.
        K_star_row: Cross-covariance ``k_X*`` between the training inputs
            and the test input ``x_*``.
        lambda_min: A lower bound on the smallest eigenvalue γ_min(Σ_y).

    Returns:
        The upper bound on the absolute predictive-variance error.

    Examples:
        The bound covers the error of a one-step cache:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> covariance = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> cross = jnp.array([1.0, 0.0, 0.0])
        >>> short = gaussx.love_cache(covariance, 1, initial_vector=jnp.ones(3))
        >>> error = cross @ jnp.linalg.solve(matrix, cross) - gaussx.love_variance(
        ...     short, cross
        ... )
        >>> smallest = jnp.linalg.eigvalsh(matrix)[0]
        >>> bound = gaussx.love_variance_error_bound(
        ...     short, covariance, cross, smallest
        ... )
        >>> bool(error <= bound)
        True
    """
    # r = Σ_y a − k_X*
    residual = _residual(cache, K_op, K_star_row)
    # r²
    residual_squared = residual**2
    # ‖r‖₂²
    residual_norm_squared = jnp.sum(residual_squared)
    # ‖r‖₂² / γ_min
    return residual_norm_squared / lambda_min


def _residual(
    cache: LOVECache,
    K_op: lx.AbstractLinearOperator,
    K_star_row: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Galerkin residual of a LOVE cache for one cross-covariance.

        r = Σ_y a − k_X*,   a = Q diag(θ)⁻¹ Qᵀ k_X*

    Args:
        cache: A `LOVECache`, holding Q and 1 / θ_j.
        K_op: The covariance Σ_y the cache was built from.
        K_star_row: Cross-covariance ``k_X*``.

    Returns:
        The residual r.
    """
    # Qᵀ k_X*
    projection = einsum(cache.Q, K_star_row, "sample ritz, sample -> ritz")
    # diag(θ)⁻¹ Qᵀ k_X*
    weighted = projection * cache.inv_eigvals
    # a = Q diag(θ)⁻¹ Qᵀ k_X*
    solution = einsum(cache.Q, weighted, "sample ritz, ritz -> sample")
    # Σ_y a
    applied = K_op.mv(solution)
    # r = Σ_y a − k_X*
    return applied - K_star_row
