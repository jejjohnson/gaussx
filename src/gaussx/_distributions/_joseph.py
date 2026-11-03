"""Joseph-form covariance update of a Kalman measurement step.

The notation is that of `gaussx._ssm._kalman`: prior covariance Σ_x,
observation operator H, observation-error covariance Σ_ε, gain G and
posterior covariance Σ_x|y.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize


def joseph_update(
    P_pred: Float[Array, "N N"],
    K: Float[Array, "N M"],
    H: Float[Array, "M N"],
    R: Float[Array, "M M"],
) -> Float[Array, "N N"]:
    """Joseph-form posterior covariance of a Kalman update, stable for any gain.

        Σ_x|y = (I − G H) Σ_x (I − G H)ᵀ + G Σ_ε Gᵀ

    For the optimal gain G = Σ_x Hᵀ Σ_y⁻¹ this equals Σ_x − G Σ_y Gᵀ, but
    the Joseph form stays symmetric positive semi-definite when G is
    approximate or the problem is poorly conditioned, where the short forms
    Σ_x − G Σ_y Gᵀ and (I − G H) Σ_x can lose definiteness.

    Algorithm:
        1. Form I − G H.
        2. Sandwich the prior covariance, (I − G H) Σ_x (I − G H)ᵀ.
        3. Add the noise term G Σ_ε Gᵀ and symmetrise.

    Args:
        P_pred: Prior (predicted) covariance Σ_x of the state.
        K: Kalman gain G.
        H: Observation operator H.
        R: Observation-error covariance Σ_ε.

    Returns:
        The posterior covariance Σ_x|y, symmetrised.

    Examples:
        With the optimal gain it matches Σ_x − G Σ_y Gᵀ:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior_covariance = jnp.eye(2)
        >>> obs_operator = jnp.array([[1.0, 0.0]])
        >>> obs_noise_covariance = jnp.array([[1.0]])
        >>> gain = gaussx.kalman_gain(
        ...     lx.MatrixLinearOperator(prior_covariance),
        ...     lx.MatrixLinearOperator(obs_operator),
        ...     lx.MatrixLinearOperator(obs_noise_covariance),
        ... )
        >>> posterior = gaussx.joseph_update(
        ...     prior_covariance, gain, obs_operator, obs_noise_covariance
        ... )
        >>> bool(jnp.allclose(posterior, jnp.diag(jnp.array([0.5, 1.0]))))
        True

        A suboptimal gain still gives a symmetric covariance:

        >>> suboptimal = jnp.array([[0.3], [0.1]])
        >>> posterior = gaussx.joseph_update(
        ...     prior_covariance, suboptimal, obs_operator, obs_noise_covariance
        ... )
        >>> bool(jnp.allclose(posterior, posterior.T))
        True
    """
    state_size = P_pred.shape[0]
    # jnp, not einx: einx rejects a zero-length axis, and D_x = 0 or D_y = 0
    # is accepted here.
    identity = jnp.eye(state_size, dtype=P_pred.dtype)
    # G H
    gain_obs_operator = K @ H
    # I − G H
    joseph_factor = identity - gain_obs_operator
    # (I − G H) Σ_x
    factor_covariance = joseph_factor @ P_pred
    # (I − G H) Σ_x (I − G H)ᵀ
    propagated = factor_covariance @ joseph_factor.T
    # G Σ_ε
    gain_noise = K @ R
    # G Σ_ε Gᵀ
    noise_term = gain_noise @ K.T
    # Σ_x|y = (I − G H) Σ_x (I − G H)ᵀ + G Σ_ε Gᵀ
    posterior_covariance = propagated + noise_term
    # ½ (Σ_x|y + Σ_x|yᵀ)
    return symmetrize(posterior_covariance)
