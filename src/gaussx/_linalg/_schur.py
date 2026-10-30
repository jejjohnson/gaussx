"""Schur complement K_XX − K_XZ K_ZZ⁻¹ K_ZX and the GP predictive variance.

X stacks the N inputs and Z the M inducing inputs, so K_XX is N × N, K_XZ is
N × M (with K_ZX = K_XZᵀ) and K_ZZ is M × M. A_X = K_XZ K_ZZ⁻¹ is the N × M
projection onto the inducing inputs and Σ_Z the M × M covariance of the
variational distribution over the inducing values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, reduce
from gaussx._operators._low_rank_update import LowRankUpdate


def schur_complement(
    K_XX: lx.AbstractLinearOperator,
    K_XZ: Float[Array, "N M"],
    K_ZZ: lx.AbstractLinearOperator,
) -> LowRankUpdate:
    """Schur complement K_XX − K_XZ K_ZZ⁻¹ K_ZX, kept as a low-rank update.

        K_XX − K_XZ K_ZZ⁻¹ K_ZX = K_XX + K_XZ (−I_M) (K_XZ K_ZZ⁻¹)ᵀ

    The covariance of a GP conditioned on its inducing values. It comes back
    as a `gaussx.LowRankUpdate` on K_XX, so later solves and
    log-determinants exploit the rank-M structure.

    Algorithm:
        1. Solve row by row for A_X = K_XZ K_ZZ⁻¹ (K_ZZ is symmetric).
        2. Return K_XX + U diag(−1) Vᵀ with U = K_XZ and V = A_X.

    Args:
        K_XX: Kernel matrix K_XX of the inputs, N × N.
        K_XZ: Cross-kernel matrix K_XZ between the inputs and the inducing
            inputs, N × M.
        K_ZZ: Kernel matrix K_ZZ of the inducing inputs, M × M.

    Returns:
        A `gaussx.LowRankUpdate` equal to K_XX − K_XZ K_ZZ⁻¹ K_ZX.

    Examples:
        The low-rank form matches the dense Schur complement:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> K_XX = lx.MatrixLinearOperator(
        ...     jnp.array([[2.0, 0.5], [0.5, 2.0]]), lx.positive_semidefinite_tag
        ... )
        >>> K_XZ = jnp.array([[1.0], [0.5]])
        >>> K_ZZ = lx.MatrixLinearOperator(
        ...     jnp.array([[2.0]]), lx.positive_semidefinite_tag
        ... )
        >>> schur = gaussx.schur_complement(K_XX, K_XZ, K_ZZ)
        >>> dense = K_XX.as_matrix() - jnp.outer(K_XZ, K_XZ) / 2.0
        >>> bool(jnp.allclose(schur.as_matrix(), dense))
        True
    """
    num_inducing = K_XZ.shape[1]

    # lazy import, cycle: _linalg._linalg -> _linalg._schur
    from gaussx._linalg._linalg import solve_rows

    # A_X = K_XZ K_ZZ⁻¹, one solve per row of K_XZ
    projection = solve_rows(K_ZZ, K_XZ)
    # diag(−1) of size M
    negative_ones = -jnp.ones(num_inducing, dtype=projection.dtype)
    # K_XX + K_XZ (−I_M) A_Xᵀ = K_XX − K_XZ K_ZZ⁻¹ K_ZX
    return LowRankUpdate(K_XX, K_XZ, negative_ones, projection)


def conditional_variance(
    K_XX_diag: Float[Array, " N"],
    K_XZ: Float[Array, "N M"] | lx.AbstractLinearOperator | None = None,
    A_X: Float[Array, "N M"] | None = None,
    S_u: lx.AbstractLinearOperator | None = None,
) -> Float[Array, " N"]:
    """Predictive variance: the Schur-complement diagonal plus a variational term.

        σ_n² = max([K_XX]_nn − [A_X K_ZX]_nn, 0) + [A_X Σ_Z A_Xᵀ]_nn

    Without ``S_u`` this is the exact GP predictive variance, the clamped
    diagonal of the Schur complement. With ``S_u`` it is the sparse
    variational GP predictive variance, which adds the variational
    covariance term.

    Until gaussx 0.7.0 the legacy three-positional form
    ``conditional_variance(base_diag, A_X, S_u)`` is still accepted. In it
    ``base_diag`` is the Schur diagonal already, ``A_X`` the projection and
    ``S_u`` the variational covariance; it is detected when the third
    positional argument is a `lineax.AbstractLinearOperator` (the old
    ``S_u`` slot). Such calls warn and compute
    max(base_diag, 0) + [A_X Σ_Z A_Xᵀ]_nn, with no Schur subtraction.

    Args:
        K_XX_diag: Prior variances [K_XX]_nn, the diagonal of K_XX.
        K_XZ: Cross-kernel matrix K_XZ between the inputs and the inducing
            inputs.
        A_X: Projection A_X = K_XZ K_ZZ⁻¹.
        S_u: Optional variational covariance Σ_Z of the inducing values.
            When given, adds [A_X Σ_Z A_Xᵀ]_nn to the base variance.

    Returns:
        The predictive variances σ_n².

    Raises:
        TypeError: If ``K_XZ`` or ``A_X`` is missing outside the legacy
            form, or ``K_XZ`` is not a JAX array.

    Examples:
        The exact predictive variance of two inputs with one inducing input:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> K_XX_diag = jnp.array([2.0, 2.0])
        >>> K_XZ = jnp.array([[1.0], [0.5]])
        >>> A_X = K_XZ / 2.0
        >>> variance = gaussx.conditional_variance(K_XX_diag, K_XZ, A_X)
        >>> bool(jnp.allclose(variance, jnp.array([1.5, 1.875])))
        True

        A variational covariance Σ_Z adds [A_X Σ_Z A_Xᵀ]_nn:

        >>> S_u = lx.MatrixLinearOperator(
        ...     jnp.array([[0.4]]), lx.positive_semidefinite_tag
        ... )
        >>> corrected = gaussx.conditional_variance(K_XX_diag, K_XZ, A_X, S_u)
        >>> bool(jnp.allclose(corrected - variance, 0.4 * A_X[:, 0] ** 2))
        True
    """
    # Backwards compatibility: the legacy 3-positional signature was
    # ``conditional_variance(base_diag, A_X, S_u)``, as the pre-#152 docs
    # called it, with the projection second and the variational covariance
    # operator third. The third positional being an AbstractLinearOperator
    # (the old ``S_u``) while ``S_u`` is left at ``None`` detects it.
    if (
        S_u is None
        and isinstance(A_X, lx.AbstractLinearOperator)
        and isinstance(K_XZ, jax.Array)
    ):
        warn_deprecated(
            "conditional_variance(base_diag, A_X, S_u) is deprecated and will be "
            "removed in gaussx 0.7.0; "
            "use conditional_variance(K_XX_diag, K_XZ, A_X, S_u=S_u). "
            "The legacy form treats the first argument as the "
            "precomputed Schur diagonal and skips the K_XZ-based "
            "subtraction."
        )
        # the second positional was the projection, the third Σ_Z
        legacy_projection = K_XZ
        legacy_variational_covariance: lx.AbstractLinearOperator = A_X
        variational_covariance = legacy_variational_covariance.as_matrix()
        # A_X Σ_Z
        projected = einsum(
            legacy_projection,
            variational_covariance,
            "sample mid, mid inducing -> sample inducing",
        )
        # [A_X Σ_Z A_Xᵀ]_nn
        variational_term = reduce(
            projected * legacy_projection, "sample inducing -> sample", "sum"
        )
        # max(base_diag, 0) + [A_X Σ_Z A_Xᵀ]_nn
        return jnp.clip(K_XX_diag, 0.0) + variational_term

    if K_XZ is None or A_X is None:
        raise TypeError(
            "conditional_variance requires K_XZ and A_X (or the legacy "
            f"(base_diag, A_X, S_u) call). Got K_XZ={K_XZ!r}, "
            f"A_X={A_X!r}, S_u={S_u!r}."
        )
    if not isinstance(K_XZ, jax.Array):
        raise TypeError(
            "K_XZ must be a jax array; the legacy 3-positional form "
            "passes an AbstractLinearOperator as the third positional "
            "argument."
        )

    # [A_X K_ZX]_nn
    explained = reduce(A_X * K_XZ, "sample inducing -> sample", "sum")
    # max([K_XX]_nn − [A_X K_ZX]_nn, 0)
    schur_diagonal = jnp.clip(K_XX_diag - explained, 0.0)
    if S_u is None:
        return schur_diagonal
    variational_covariance = S_u.as_matrix()
    # A_X Σ_Z
    projected = einsum(
        A_X, variational_covariance, "sample mid, mid inducing -> sample inducing"
    )
    # [A_X Σ_Z A_Xᵀ]_nn
    variational_term = reduce(projected * A_X, "sample inducing -> sample", "sum")
    # σ_n² = max([K_XX]_nn − [A_X K_ZX]_nn, 0) + [A_X Σ_Z A_Xᵀ]_nn
    return schur_diagonal + variational_term
