"""Sparse inverse Kalman filter (SpInGP) recipes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


def _build_likelihood_precision(
    emission_model: jnp.ndarray,
    obs_noise: lx.AbstractLinearOperator,
    N: int,
    d: int,
) -> BlockTriDiag:
    """Build block-tridiagonal likelihood precision from emission model.

    For scalar or vector observations at each time step, the likelihood
    precision contribution is block-diagonal (zero sub-diagonals)::

        Lambda_lik[k] = H_k^T R_k^{-1} H_k

    The observation noise inverse ``R^{-1}`` is computed once via
    structural dispatch and reused for all time steps.

    Args:
        emission_model: Emission matrix. Shape ``(N, d_obs, d)`` for
            per-timestep matrices or ``(d_obs, d)`` for shared.
        obs_noise: Observation noise covariance operator.
        N: Number of time steps.
        d: State dimension.

    Returns:
        Block-tridiagonal likelihood precision (block-diagonal).
    """
    # Compute R^{-1} once via structural dispatch (obs_noise is typically small)
    R_inv = inv(obs_noise).as_matrix()

    if emission_model.ndim == 2:
        # Shared emission model: H^T R^{-1} H for all time steps
        block = emission_model.T @ R_inv @ emission_model
        diag_blocks = jnp.tile(block[None, :, :], (N, 1, 1))
    else:
        # Per-timestep emission: H_k^T R^{-1} H_k
        diag_blocks = jax.vmap(lambda H_k: H_k.T @ R_inv @ H_k)(emission_model)

    sub_diag_blocks = jnp.zeros((N - 1, d, d), dtype=diag_blocks.dtype)
    return BlockTriDiag(diag_blocks, sub_diag_blocks)


def _build_data_vector(
    emission_model: jnp.ndarray,
    obs_noise: lx.AbstractLinearOperator,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Build the data contribution to the posterior mean equation.

    Computes ``H^T R^{-1} y`` for each time step.  The observation
    noise inverse is computed once and reused for all time steps.

    Args:
        emission_model: Emission matrix, shape ``(N, d_obs, d)`` or
            ``(d_obs, d)``.
        obs_noise: Observation noise covariance operator.
        observations: Observations, shape ``(N, d_obs)``.

    Returns:
        Data vector, shape ``(N * d,)``.
    """
    # Precompute R^{-1} once (obs_noise is typically small)
    R_inv = inv(obs_noise).as_matrix()

    if emission_model.ndim == 2:
        # Shared: H^T R^{-1} y_k for each k
        HtRinv = emission_model.T @ R_inv
        data_vec = jax.vmap(lambda y_k: HtRinv @ y_k)(observations)
    else:
        data_vec = jax.vmap(lambda H_k, y_k: H_k.T @ R_inv @ y_k)(
            emission_model, observations
        )

    return data_vec.reshape(-1)


def spingp_log_likelihood(
    prior_precision: BlockTriDiag,
    emission_model: jnp.ndarray,
    obs_noise: lx.AbstractLinearOperator,
    observations: jnp.ndarray,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> jnp.ndarray:
    r"""Log marginal likelihood via sparse inverse GP formulation.

    Computes the log marginal likelihood using the precision-form
    Kalman filter (SpInGP)::

        1. Likelihood precision sites: :math:`\Lambda_{lik} = H^T R^{-1} H`
        2. Posterior precision: :math:`\Lambda_{post} = \Lambda_{prior} + \Lambda_{lik}`
        3. log p(y) via banded Cholesky logdet and quadratic form

    The full expression is::

        log p(y) = -0.5 * (N_{obs} * log(2\pi) + log|R|_{total}
                   + y^T R^{-1} y - \eta^T \Lambda_{post}^{-1} \eta
                   + log|\Lambda_{post}| - log|\Lambda_{prior}|)

    where :math:`\eta = H^T R^{-1} y`.

    All operations exploit banded structure for O(Nd³) cost.

    The ``solver`` parameter controls the algorithm used for the
    large-scale posterior precision operations (solve, logdet).
    Observation noise operations always use structural dispatch
    since ``obs_noise`` is typically a small dense matrix.

    Args:
        prior_precision: Prior precision as ``BlockTriDiag``,
            shape ``(N, d, d)`` diagonal and ``(N-1, d, d)`` sub-diagonal.
        emission_model: Emission matrix H. Shape ``(d_obs, d)`` for
            shared or ``(N, d_obs, d)`` per time step.
        obs_noise: Observation noise covariance R operator.
        observations: Observations y, shape ``(N, d_obs)``.
        solver: Optional solver strategy for posterior precision
            operations. When ``None``, uses structural dispatch.
            Observation noise operations always use structural dispatch.

    Returns:
        Scalar log marginal likelihood.
    """
    N = prior_precision._num_blocks
    d = prior_precision._block_size
    N_obs = observations.size
    log_2pi = jnp.log(2.0 * jnp.pi)

    # Build likelihood precision and posterior precision
    lik_prec = _build_likelihood_precision(emission_model, obs_noise, N, d)
    post_prec = prior_precision.add(lik_prec)

    # Data vector: eta = H^T R^{-1} y
    eta = _build_data_vector(emission_model, obs_noise, observations)

    # Quadratic term: eta^T Lambda_post^{-1} eta
    post_solve = dispatch_solve(post_prec, eta, solver)
    quad_term = jnp.dot(eta, post_solve)

    # Observation quadratic: y^T R^{-1} y (obs_noise is small, use inv)
    R_inv = inv(obs_noise).as_matrix()
    obs_quad = jnp.sum(jax.vmap(lambda y_k: y_k @ R_inv @ y_k)(observations))

    # Log determinants (posterior precision: may be large, use solver)
    ld_post = dispatch_logdet(post_prec, solver)
    ld_prior = dispatch_logdet(prior_precision, solver)

    # Total observation noise logdet: N * log|R| (small, structural dispatch)
    ld_R = logdet(obs_noise)
    ld_R_total = N * ld_R

    return -0.5 * (
        N_obs * log_2pi + ld_R_total + obs_quad - quad_term + ld_post - ld_prior
    )


def spingp_posterior(
    prior_precision: BlockTriDiag,
    emission_model: jnp.ndarray,
    obs_noise: lx.AbstractLinearOperator,
    observations: jnp.ndarray,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[jnp.ndarray, BlockTriDiag]:
    r"""Posterior mean and precision via SpInGP.

    Computes the posterior by adding likelihood precision sites to the
    prior precision and solving for the posterior mean::

        \Lambda_{post} = \Lambda_{prior} + H^T R^{-1} H
        \mu_{post} = \Lambda_{post}^{-1} H^T R^{-1} y

    Args:
        prior_precision: Prior precision as ``BlockTriDiag``.
        emission_model: Emission matrix H. Shape ``(d_obs, d)`` for
            shared or ``(N, d_obs, d)`` per time step.
        obs_noise: Observation noise covariance R operator.
        observations: Observations y, shape ``(N, d_obs)``.
        solver: Optional solver strategy for posterior precision
            operations. When ``None``, uses structural dispatch.

    Returns:
        Tuple ``(posterior_mean, posterior_precision)`` where
        ``posterior_mean`` has shape ``(N * d,)`` and
        ``posterior_precision`` is ``BlockTriDiag``.
    """
    N = prior_precision._num_blocks
    d = prior_precision._block_size

    # Build likelihood precision and posterior precision
    lik_prec = _build_likelihood_precision(emission_model, obs_noise, N, d)
    post_prec = prior_precision.add(lik_prec)

    # Data vector: eta = H^T R^{-1} y
    eta = _build_data_vector(emission_model, obs_noise, observations)

    # Posterior mean: Lambda_post^{-1} eta
    post_mean = dispatch_solve(post_prec, eta, solver)

    return post_mean, post_prec
