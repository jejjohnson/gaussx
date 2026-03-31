"""Conditional Gaussian distribution from partial observations."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._primitives._solve import solve


def conditional(
    loc: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
    obs_idx: Int[Array, " M"],
    obs_values: Float[Array, " M"],
) -> tuple[Float[Array, " R"], lx.AbstractLinearOperator]:
    r"""Compute ``p(x_A | x_B = b)`` from a joint Gaussian ``p(x_A, x_B)``.

    Given a joint distribution :math:`\mathcal{N}(\mu, \Sigma)` and
    observed indices *B* with values *b*, returns the conditional
    distribution over the remaining indices *A*:

    .. math::

        \mu_{A|B} &= \mu_A + \Sigma_{AB} \Sigma_{BB}^{-1} (b - \mu_B) \\
        \Sigma_{A|B} &= \Sigma_{AA} - \Sigma_{AB} \Sigma_{BB}^{-1} \Sigma_{BA}

    Args:
        loc: Mean vector of the joint distribution, shape ``(N,)``.
        cov: Covariance operator of the joint distribution, shape ``(N, N)``.
        obs_idx: Indices of the observed variables, shape ``(M,)``.
        obs_values: Observed values, shape ``(M,)``.

    Returns:
        Tuple ``(cond_mean, cond_cov)`` — mean and covariance of the
        conditional distribution over unobserved variables.
    """
    N = loc.shape[0]
    obs_idx = jnp.asarray(obs_idx, dtype=jnp.int32)
    obs_values = jnp.asarray(obs_values, dtype=loc.dtype)

    if obs_idx.ndim != 1:
        raise ValueError("obs_idx must be a 1D array.")
    if obs_values.shape != obs_idx.shape:
        raise ValueError("obs_values must have the same shape as obs_idx.")
    if bool(jnp.any((obs_idx < 0) | (obs_idx >= N))):
        raise ValueError(f"obs_idx must be within bounds [0, {N}).")
    if bool(jnp.any(jnp.diff(jnp.sort(obs_idx)) == 0)):
        raise ValueError("obs_idx must not contain duplicates.")

    # Build mask for unobserved indices
    mask = jnp.ones(N, dtype=bool).at[obs_idx].set(False)
    free_idx = jnp.where(mask, size=N - obs_idx.shape[0])[0]

    # Materialize covariance (required for index-based slicing)
    Sigma = cov.as_matrix()

    # Extract sub-blocks
    Sigma_AA = Sigma[jnp.ix_(free_idx, free_idx)]
    Sigma_AB = Sigma[jnp.ix_(free_idx, obs_idx)]
    Sigma_BB = Sigma[jnp.ix_(obs_idx, obs_idx)]

    mu_A = loc[free_idx]
    mu_B = loc[obs_idx]

    # Sigma_BB^{-1} (b - mu_B)
    residual = obs_values - mu_B
    Sigma_BB_op = lx.MatrixLinearOperator(Sigma_BB, lx.positive_semidefinite_tag)
    alpha = solve(Sigma_BB_op, residual)

    # Conditional mean: mu_A + Sigma_AB @ alpha
    cond_mean = mu_A + Sigma_AB @ alpha

    # Sigma_BB^{-1} Sigma_BA
    Sigma_BA = Sigma_AB.T
    # Solve Sigma_BB @ X = Sigma_BA for X, column by column
    X = jnp.linalg.solve(Sigma_BB, Sigma_BA)

    # Conditional covariance: Sigma_AA - Sigma_AB @ X
    cond_cov_mat = Sigma_AA - Sigma_AB @ X

    # Symmetrize for numerical stability
    cond_cov_mat = 0.5 * (cond_cov_mat + cond_cov_mat.T)
    cond_cov = lx.MatrixLinearOperator(cond_cov_mat, lx.positive_semidefinite_tag)

    return cond_mean, cond_cov
