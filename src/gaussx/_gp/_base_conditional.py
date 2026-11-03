"""Sparse-GP conditional q(f_X) = ∫ p(f_X ∣ f_ℐ) q(f_ℐ) df_ℐ (`sparse_conditional`).

X stacks the N evaluation inputs and X_ℐ the N_ℐ inducing inputs of the
inducing set ℐ, so K_XX is the N × N prior kernel matrix, K_Xℐ the N × N_ℐ
cross-covariance (K_ℐX = K_Xℐᵀ) and K_ℐℐ = L_ℐℐ L_ℐℐᵀ the inducing kernel
matrix with lower Cholesky factor L_ℐℐ. Each of the R outputs, indexed by r,
has the variational distribution q(f_ℐ,r) = 𝒩(μ_ℐ,r, L_ℐ,r L_ℐ,rᵀ) over its
inducing values, and Q_XX = K_Xℐ K_ℐℐ⁻¹ K_ℐX is the Nyström matrix.
"""

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
    """Wrap a dense kernel matrix as a PSD operator; pass operators through.

    Args:
        K: Kernel matrix, as a dense array or a lineax operator.

    Returns:
        ``K`` itself when it is an operator, otherwise a PSD-tagged
        `lineax.MatrixLinearOperator`.
    """
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
    """Sparse-GP conditional q(f_X) = ∫ p(f_X ∣ f_ℐ) q(f_ℐ) df_ℐ, per output r.

        μ_X,r = K_Xℐ K_ℐℐ⁻¹ μ_ℐ,r
        Σ_X,r = K_XX − Q_XX + Pᵀ L_ℐ,r L_ℐ,rᵀ P,   P = K_ℐℐ⁻¹ K_ℐX

    In the whitened parameterisation (``white=True``) μ_ℐ,r and L_ℐ,r
    describe the whitened inducing values z_ℐ = L_ℐℐ⁻¹ f_ℐ instead, whose
    prior is 𝒩(0, I), and P = L_ℐℐ⁻¹ K_ℐX.

    Follows the `gaussx._gp` sparse-GP convention: inducing kernel matrix
    ``K_zz`` first (array or operator), cross-covariance ``K_xz`` with one
    row per evaluation input, prior ``K_xx`` at the evaluation inputs and
    the variational parameters ``q_mu`` and ``q_sqrt``. K_ℐℐ is factorised
    with `gaussx.cholesky` and every solve with its factor is a structured
    triangular solve, so a `Kronecker` or `BlockDiag` ``K_zz`` is never
    densified. Only the diagonal of Σ_X,r is formed when ``K_xx`` is a
    vector.

    Algorithm:
        1. Factorise K_ℐℐ = L_ℐℐ L_ℐℐᵀ and whiten the cross-covariance,
           V_X = L_ℐℐ⁻¹ K_ℐX, so that Q_XX = V_Xᵀ V_X.
        2. Form the mean μ_X,r = V_Xᵀ μ_ℐ,r (whitened) or
           V_Xᵀ L_ℐℐ⁻¹ μ_ℐ,r, and the projection P = V_X (whitened) or
           L_ℐℐ⁻ᵀ V_X.
        3. Form the Nyström residual K_XX − V_Xᵀ V_X, or its diagonal.
        4. Add the variational term W_rᵀ W_r with W_r = L_ℐ,rᵀ P, and clip
           diagonal variances at zero.

    Args:
        K_zz: Inducing kernel matrix K_ℐℐ, as a dense array or a PSD
            operator.
        K_xz: Cross-covariance K_Xℐ between the evaluation inputs and the
            inducing inputs.
        K_xx: Prior kernel matrix K_XX at the evaluation inputs, full or
            only its diagonal.
        q_mu: Variational means μ_ℐ,r, one column per output, or a single
            vector μ_ℐ (the outputs then drop their trailing output axis).
        q_sqrt: Optional variational roots L_ℐ,r: lower-triangular and
            stacked over outputs first, or diagonal with one column per
            output; with a 1-D ``q_mu`` also a single square root or a
            single diagonal. ``None`` drops the variational term.
        white: If ``True``, ``q_mu`` and ``q_sqrt`` describe the whitened
            values z_ℐ = L_ℐℐ⁻¹ f_ℐ, with prior 𝒩(0, I).

    Returns:
        Tuple ``(mean, var)`` holding μ_X,r, one column per output, and
        Σ_X,r: the full covariances stacked on a trailing output axis (full
        ``K_xx``) or the variances σ²_n,r (diagonal ``K_xx``). A 1-D
        ``q_mu`` drops the output axis. Diagonal variances are clipped at 0
        against round-off; a full covariance is returned unclipped.

    Raises:
        ValueError: If ``q_mu`` or ``q_sqrt`` does not match N_ℐ and R.

    Examples:
        Two inducing inputs with K_ℐℐ = 2 I and no variational term:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> K_zz = 2.0 * jnp.eye(2)
        >>> K_xz = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> K_xx_diag = jnp.full(3, 2.0)
        >>> q_mu = jnp.array([2.0, -2.0])
        >>> mean, var = gaussx.sparse_conditional(K_zz, K_xz, K_xx_diag, q_mu)
        >>> bool(jnp.allclose(mean, jnp.array([1.0, -1.0, 0.0]), atol=1e-6))
        True
        >>> bool(jnp.allclose(var, jnp.array([1.5, 1.5, 1.0])))
        True

        Whitened unit roots recover the prior covariance for every output:

        >>> q_sqrt = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
        >>> mean, cov = gaussx.sparse_conditional(
        ...     K_zz, K_xz, 2.0 * jnp.eye(3), jnp.ones((2, 4)), q_sqrt=q_sqrt,
        ...     white=True,
        ... )
        >>> mean.shape, cov.shape
        ((3, 4), (3, 3, 4))
        >>> bool(jnp.allclose(cov[..., 0], 2.0 * jnp.eye(3), atol=1e-5))
        True

        A mean of the wrong length is rejected:

        >>> gaussx.sparse_conditional(
        ...     K_zz, K_xz, K_xx_diag, jnp.ones(3)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: f must have shape (M, R) or (M,) with M=2, got (3, 1).
    """
    inducing_operator = _as_psd_operator(K_zz)
    inducing_size = inducing_operator.in_size()
    single_output = q_mu.ndim == 1
    inducing_mean, q_sqrt = _check_shapes(inducing_size, q_mu, q_sqrt)
    num_outputs = inducing_mean.shape[1]

    # --- whiten ---
    # K_ℐℐ = L_ℐℐ L_ℐℐᵀ, structured where possible
    inducing_factor = cholesky(inducing_operator)
    # K_ℐX = K_Xℐᵀ
    cross_covariance_t = rearrange(K_xz, "sample inducing -> inducing sample")
    # V_X = L_ℐℐ⁻¹ K_ℐX
    whitened_cross = solve_columns(inducing_factor, cross_covariance_t)

    # --- mean ---
    if white:
        # μ_X,r = V_Xᵀ μ_ℐ,r
        mean = einsum(
            whitened_cross,
            inducing_mean,
            "inducing sample, inducing output -> sample output",
        )
        # P = V_X
        projection = whitened_cross
    else:
        # L_ℐℐ⁻¹ μ_ℐ,r
        whitened_mean = solve_columns(inducing_factor, inducing_mean)
        # μ_X,r = V_Xᵀ L_ℐℐ⁻¹ μ_ℐ,r = K_Xℐ K_ℐℐ⁻¹ μ_ℐ,r
        mean = einsum(
            whitened_cross,
            whitened_mean,
            "inducing sample, inducing output -> sample output",
        )
        # P = L_ℐℐ⁻ᵀ V_X = K_ℐℐ⁻¹ K_ℐX
        projection = solve_columns(inducing_factor.transpose(), whitened_cross)

    # --- Nyström residual ---
    full_cov = K_xx.ndim == 2
    if full_cov:
        # Q_XX = V_Xᵀ V_X
        nystrom = einsum(
            whitened_cross, whitened_cross, "inducing row, inducing col -> row col"
        )
        # K_XX − Q_XX
        nystrom_residual = K_xx - nystrom
    else:
        # [V_X]²
        whitened_cross_squared = whitened_cross**2
        # diag(Q_XX) = diag(V_Xᵀ V_X)
        nystrom_diagonal = reduce(
            whitened_cross_squared, "inducing sample -> sample", "sum"
        )
        # diag(K_XX) − diag(Q_XX)
        nystrom_residual = K_xx - nystrom_diagonal

    # --- variational term ---
    if q_sqrt is None:
        # Σ_X,r = K_XX − Q_XX, the same for every output
        if full_cov:
            variance = repeat(
                nystrom_residual, "row col -> row col output", output=num_outputs
            )
        else:
            variance = repeat(
                nystrom_residual, "sample -> sample output", output=num_outputs
            )
    else:
        if q_sqrt.ndim == 2:
            # W_r = diag(L_ℐ,r) P, one diagonal root per output column
            root_projection = einx.multiply(
                "inducing output, inducing sample -> output inducing sample",
                q_sqrt,
                projection,
            )
        else:
            # W_r = L_ℐ,rᵀ P, one lower-triangular root per output
            root_projection = einsum(
                q_sqrt,
                projection,
                "output inducing root, inducing sample -> output root sample",
            )
        if full_cov:
            # W_rᵀ W_r = Pᵀ L_ℐ,r L_ℐ,rᵀ P
            variational_term = einsum(
                root_projection,
                root_projection,
                "output root row, output root col -> row col output",
            )
            # Σ_X,r = K_XX − Q_XX + W_rᵀ W_r
            variance = einx.add(
                "row col, row col output -> row col output",
                nystrom_residual,
                variational_term,
            )
        else:
            # W_r²
            root_projection_squared = root_projection**2
            # diag(W_rᵀ W_r)
            variational_term = reduce(
                root_projection_squared, "output root sample -> sample output", "sum"
            )
            # σ²_n,r = [K_XX − Q_XX]_nn + [W_rᵀ W_r]_nn
            variance = einx.add(
                "sample, sample output -> sample output",
                nystrom_residual,
                variational_term,
            )

    if not full_cov:
        # K_xx - diag(Q_xx) can round below zero (gh-363).
        # max(σ²_n,r, 0)
        variance = jnp.maximum(variance, 0.0)
    if single_output:
        return mean[:, 0], variance[..., 0]
    return mean, variance


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
    """Deprecated: use `sparse_conditional` (note ``K_xz = K_mn.T``).

        base_conditional(K_ℐℐ, K_ℐX, …) = sparse_conditional(K_ℐℐ, K_ℐXᵀ, …)

    ``base_conditional(K_mm, K_mn, K_nn, f, q_sqrt=..., white=...)`` equals
    ``sparse_conditional(K_mm, K_mn.T, K_nn, f, q_sqrt=..., white=...)``.
    It takes the cross-covariance as K_ℐX (inducing × data), the transpose
    of every other sparse-GP helper; for N = N_ℐ a transposed argument is
    not detectable, so the convention is changed under a new name rather
    than in place. ``solver`` was never used. Removed in gaussx 0.7.0.

    Args:
        K_mm: Inducing kernel matrix K_ℐℐ.
        K_mn: Cross-covariance K_ℐX between the inducing inputs and the
            evaluation inputs, the transpose of `sparse_conditional`'s
            ``K_xz``.
        K_nn: Prior kernel matrix K_XX at the evaluation inputs, full or
            only its diagonal.
        f: Variational means μ_ℐ,r, as ``q_mu`` in `sparse_conditional`.
        q_sqrt: Optional variational roots L_ℐ,r, as in
            `sparse_conditional`.
        white: Whitened parameterisation, as in `sparse_conditional`.
        solver: Ignored.

    Returns:
        ``(mean, var)`` as in `sparse_conditional`.

    Examples:
        It warns and matches `sparse_conditional` on the transposed
        cross-covariance:

        >>> import warnings
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> K_mm = 2.0 * jnp.eye(2)
        >>> K_mn = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        >>> f = jnp.array([2.0, -2.0])
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     mean, _ = gaussx.base_conditional(K_mm, K_mn, jnp.full(3, 2.0), f)
        >>> caught[0].category.__name__
        'GaussxDeprecationWarning'
        >>> bool(jnp.allclose(mean, jnp.array([1.0, -1.0, 0.0]), atol=1e-6))
        True
    """
    del solver
    warn_deprecated(
        "base_conditional(K_mm, K_mn, ...) is deprecated and will be removed in "
        "gaussx 0.7.0; use sparse_conditional(K_zz, K_xz, K_xx, q_mu, "
        "...) with K_xz = K_mn.T (shape (N, M))."
    )
    # K_Xℐ = K_ℐXᵀ
    cross_covariance = rearrange(K_mn, "inducing sample -> sample inducing")
    return sparse_conditional(
        K_mm, cross_covariance, K_nn, f, q_sqrt=q_sqrt, white=white
    )


def _check_shapes(
    inducing_size: int,
    f: Float[Array, "M R"] | Float[Array, " M"],
    q_sqrt: Array | None,
) -> tuple[Float[Array, "M R"], Array | None]:
    """Validate μ_ℐ,r and L_ℐ,r and promote the single-output layouts.

    Args:
        inducing_size: Number of inducing inputs N_ℐ.
        f: Variational means μ_ℐ,r, or a single vector μ_ℐ.
        q_sqrt: Optional variational roots L_ℐ,r, in any layout accepted by
            `sparse_conditional`.

    Returns:
        Tuple ``(f, q_sqrt)`` with the means as one column per output and
        the roots stacked over outputs (full) or one column per output
        (diagonal).

    Raises:
        ValueError: If ``f`` or ``q_sqrt`` does not match N_ℐ and R.
    """
    if f.ndim == 1:
        f = rearrange(f, "inducing -> inducing 1")
        if q_sqrt is not None and q_sqrt.shape == (inducing_size,):
            q_sqrt = rearrange(q_sqrt, "inducing -> inducing 1")
        elif q_sqrt is not None and q_sqrt.shape == (inducing_size, inducing_size):
            q_sqrt = rearrange(q_sqrt, "inducing root -> 1 inducing root")
    if f.ndim != 2 or f.shape[0] != inducing_size:
        raise ValueError(
            f"f must have shape (M, R) or (M,) with M={inducing_size}, got {f.shape}."
        )
    num_outputs = f.shape[1]
    diagonal_shape = (inducing_size, num_outputs)
    full_shape = (num_outputs, inducing_size, inducing_size)
    if q_sqrt is not None and q_sqrt.shape not in (diagonal_shape, full_shape):
        raise ValueError(
            f"q_sqrt must have shape (M, R)=({inducing_size}, {num_outputs}) or "
            f"(R, M, M)=({num_outputs}, {inducing_size}, {inducing_size}) "
            f"to match f, got {q_sqrt.shape}."
        )
    return f, q_sqrt
