"""Whitened SVGP prediction: the marginals of q(f_X) from whitened q(z_ℐ).

X stacks the N evaluation inputs and X_ℐ the N_ℐ inducing inputs of the
inducing set ℐ, with K_ℐℐ = L_ℐℐ L_ℐℐᵀ and cross-covariance K_Xℐ. The
inducing values are whitened, f_ℐ = L_ℐℐ z_ℐ, and the single-output
variational distribution q(z_ℐ) = 𝒩(μ_ℐ, L_ℐ L_ℐᵀ) is over z_ℐ, as in the
``white=True`` form of `gaussx.sparse_conditional`.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange, reduce
from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky


def whitened_svgp_predict(
    K_zz_op: lx.AbstractLinearOperator,
    K_xz: Float[Array, "N M"],
    u_mean: Float[Array, " M"],
    u_chol: Float[Array, "M M"],
    K_xx_diag: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """Whitened SVGP prediction: per-input means μ_n and variances σ²_n.

        μ_n = [V_Xᵀ μ_ℐ]_n,   V_X = L_ℐℐ⁻¹ K_ℐX
        σ²_n = [K_XX]_nn − [V_Xᵀ V_X]_nn + [V_Xᵀ L_ℐ L_ℐᵀ V_X]_nn

    These are the marginals q(f_n) = 𝒩(μ_n, σ²_n) of the single-output
    sparse-GP conditional with whitened inducing values z_ℐ = L_ℐℐ⁻¹ f_ℐ,
    the diagonal case of `gaussx.sparse_conditional` with ``white=True``.
    K_ℐℐ is factorised with `gaussx.cholesky`, so a structured operator
    keeps its structure through the triangular solves.

    Algorithm:
        1. Factorise K_ℐℐ = L_ℐℐ L_ℐℐᵀ and whiten the cross-covariance,
           V_X = L_ℐℐ⁻¹ K_ℐX.
        2. Form the means μ_n = [V_Xᵀ μ_ℐ]_n.
        3. Subtract the Nyström diagonal diag(Q_XX) = ∑ V_X² over the
           inducing axis from the prior variances [K_XX]_nn.
        4. Add the variational term diag(Wᵀ W) with W = L_ℐᵀ V_X, and
           clip the variances at zero.

    Args:
        K_zz_op: Inducing kernel matrix K_ℐℐ, as a PSD operator.
        K_xz: Cross-covariance K_Xℐ between the evaluation inputs and the
            inducing inputs.
        u_mean: Whitened variational mean μ_ℐ of q(z_ℐ).
        u_chol: Lower-triangular variational root L_ℐ of q(z_ℐ), whose
            covariance is L_ℐ L_ℐᵀ.
        K_xx_diag: Prior variances [K_XX]_nn at the evaluation inputs.

    Returns:
        Tuple ``(f_loc, f_var)`` holding the predictive means μ_n and the
        predictive variances σ²_n, clipped at zero against round-off.

    Examples:
        A unit whitened root recovers the prior variances:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> K_zz = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> K_xz = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> K_xx_diag = jnp.full(3, 2.0)
        >>> u_mean = jnp.array([1.0, -1.0])
        >>> f_loc, f_var = gaussx.whitened_svgp_predict(
        ...     K_zz, K_xz, u_mean, jnp.eye(2), K_xx_diag
        ... )
        >>> f_loc.shape, f_var.shape
        ((3,), (3,))
        >>> bool(jnp.allclose(f_var, K_xx_diag))
        True

        It agrees with `gaussx.sparse_conditional` in the whitened form:

        >>> u_chol = jnp.array([[0.5, 0.0], [0.2, 0.3]])
        >>> f_loc, f_var = gaussx.whitened_svgp_predict(
        ...     K_zz, K_xz, u_mean, u_chol, K_xx_diag
        ... )
        >>> mean, var = gaussx.sparse_conditional(
        ...     2.0 * jnp.eye(2), K_xz, K_xx_diag, u_mean, q_sqrt=u_chol,
        ...     white=True,
        ... )
        >>> bool(jnp.allclose(f_loc, mean)), bool(jnp.allclose(f_var, var))
        (True, True)
    """
    # einx transposes are safe here: N = 0 and N_ℐ = 0 already fail in this
    # function.

    # --- whiten ---
    # K_ℐℐ = L_ℐℐ L_ℐℐᵀ, structured where possible
    inducing_factor = cholesky(K_zz_op)
    # K_ℐX = K_Xℐᵀ
    cross_covariance_t = rearrange(K_xz, "sample inducing -> inducing sample")
    # V_X = L_ℐℐ⁻¹ K_ℐX
    whitened_cross = solve_columns(inducing_factor, cross_covariance_t)

    # --- mean ---
    # V_Xᵀ
    whitened_cross_t = rearrange(whitened_cross, "inducing sample -> sample inducing")
    # μ_n = [V_Xᵀ μ_ℐ]_n
    f_loc = whitened_cross_t @ u_mean

    # --- variance ---
    # V_X²
    whitened_cross_squared = whitened_cross**2
    # diag(Q_XX) = diag(V_Xᵀ V_X)
    nystrom_diagonal = reduce(
        whitened_cross_squared, "inducing sample -> sample", "sum"
    )
    # L_ℐᵀ
    u_chol_t = rearrange(u_chol, "inducing root -> root inducing")
    # W = L_ℐᵀ V_X
    # jnp, not einx: `@` keeps the original matmul lowering bit for bit
    root_projection = u_chol_t @ whitened_cross
    # W²
    root_projection_squared = root_projection**2
    # diag(Wᵀ W) = diag(V_Xᵀ L_ℐ L_ℐᵀ V_X)
    variational_term = reduce(root_projection_squared, "root sample -> sample", "sum")
    # [K_XX]_nn − diag(Q_XX)
    nystrom_residual = K_xx_diag - nystrom_diagonal
    # σ²_n = [K_XX]_nn − diag(Q_XX) + diag(Wᵀ W)
    variance = nystrom_residual + variational_term
    # max(σ²_n, 0)
    f_var = jnp.clip(variance, 0.0)

    return f_loc, f_var
