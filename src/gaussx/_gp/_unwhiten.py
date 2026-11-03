"""Unwhitening: map whitened variational parameters back to the inducing values.

The inducing values are whitened as f_ℐ = L_ℐℐ z_ℐ, with K_ℐℐ = L_ℐℐ L_ℐℐᵀ
the N_ℐ × N_ℐ inducing kernel matrix. A variational distribution
q(z_ℐ) = 𝒩(μ_z, Σ_z) over the whitened values is then
q(f_ℐ) = 𝒩(μ_ℐ, Σ_ℐ) with μ_ℐ = L_ℐℐ μ_z and Σ_ℐ = L_ℐℐ Σ_z L_ℐℐᵀ.
"""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import cov_transform


def unwhiten(
    m_tilde: Float[Array, " M"],
    L: lx.AbstractLinearOperator,
) -> Float[Array, " M"]:
    """Unwhiten a variational mean: μ_ℐ = L_ℐℐ μ_z.

        μ_ℐ = L_ℐℐ μ_z,   K_ℐℐ = L_ℐℐ L_ℐℐᵀ

    Args:
        m_tilde: Whitened variational mean μ_z of q(z_ℐ).
        L: Lower Cholesky factor L_ℐℐ of K_ℐℐ, as an operator.

    Returns:
        The mean μ_ℐ of q(f_ℐ).

    Examples:
        A diagonal factor scales each whitened entry:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> factor = lx.DiagonalLinearOperator(jnp.array([2.0, 0.5]))
        >>> mean = gaussx.unwhiten(jnp.array([1.0, 4.0]), factor)
        >>> bool(jnp.allclose(mean, jnp.array([2.0, 2.0])))
        True

        A dense lower factor mixes the entries:

        >>> factor = lx.MatrixLinearOperator(jnp.array([[1.0, 0.0], [0.5, 2.0]]))
        >>> gaussx.unwhiten(jnp.array([1.0, 1.0]), factor).shape
        (2,)
    """
    # μ_ℐ = L_ℐℐ μ_z
    return L.mv(m_tilde)


def unwhiten_covariance(
    L: lx.AbstractLinearOperator,
    S_tilde: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator:
    """Unwhiten a variational covariance: Σ_ℐ = L_ℐℐ Σ_z L_ℐℐᵀ.

        Σ_ℐ = L_ℐℐ Σ_z L_ℐℐᵀ,   K_ℐℐ = L_ℐℐ L_ℐℐᵀ

    L_ℐℐ is materialised and the product delegated to `cov_transform`.

    Args:
        L: Lower Cholesky factor L_ℐℐ of K_ℐℐ, as an operator.
        S_tilde: Whitened variational covariance Σ_z of q(z_ℐ).

    Returns:
        The covariance Σ_ℐ of q(f_ℐ), as an operator.

    Examples:
        A unit whitened covariance unwhitens to the prior, Σ_ℐ = K_ℐℐ:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> lower = jnp.array([[1.0, 0.0], [0.5, 2.0]])
        >>> factor = lx.MatrixLinearOperator(lower)
        >>> unit = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> covariance = gaussx.unwhiten_covariance(factor, unit)
        >>> bool(jnp.allclose(covariance.as_matrix(), lower @ lower.T))
        True

        ``whiten_covariance`` is the old name of the same function:

        >>> gaussx.whiten_covariance is gaussx.unwhiten_covariance
        True
    """
    factor_matrix = L.as_matrix()
    # Σ_ℐ = L_ℐℐ Σ_z L_ℐℐᵀ
    return cov_transform(factor_matrix, S_tilde)


# Backward-compatible alias (the old name was misleading — this unwhitens).
whiten_covariance = unwhiten_covariance
