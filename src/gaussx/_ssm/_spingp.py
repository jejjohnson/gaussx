"""Sparse inverse GP (SpInGP): the precision-form Kalman recipes.

The prior over the stacked states x_1:T is a GMRF whose precision Λ_x is
block-tridiagonal, as for an SDE kernel discretised to the transitions
F_t = exp(F_c Δt_t) and model-noise covariances Σ_η,t (drift matrix F_c).
Observations y_t = H_t x_t + ε_t, ε_t ~ 𝒩(0, Σ_ε), add the block-diagonal
likelihood precision H_tᵀ Σ_ε⁻¹ H_t, so the posterior precision Λ_x|y
stays block-tridiagonal and every solve and log-determinant is banded.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, repeat
from gaussx._linalg._linalg import solve_columns
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


class _WhitenedObservations(NamedTuple):
    """Observation model whitened by one Cholesky factor L_ε L_εᵀ = Σ_ε.

    Attributes:
        whitened_operator: H_w,t = L_ε⁻¹ H_t, one for all steps or a stack.
        whitened_obs: y_w,t = L_ε⁻¹ y_t, one row per step.
        obs_noise_logdet: log det Σ_ε.
    """

    whitened_operator: Array
    whitened_obs: Float[Array, "N d_obs"]
    obs_noise_logdet: Float[Array, ""]


def _whiten(
    emission_model: Array,
    obs_noise: lx.AbstractLinearOperator,
    observations: Float[Array, "N d_obs"],
) -> _WhitenedObservations:
    """Factor Σ_ε once and whiten H_t and y_t by triangular solves.

        H_w,t = L_ε⁻¹ H_t,    y_w,t = L_ε⁻¹ y_t,    L_ε L_εᵀ = Σ_ε

    Then H_tᵀ Σ_ε⁻¹ H_t = H_w,tᵀ H_w,t, H_tᵀ Σ_ε⁻¹ y_t = H_w,tᵀ y_w,t and
    y_tᵀ Σ_ε⁻¹ y_t = ‖y_w,t‖₂², so no Σ_ε⁻¹ is materialised and Σ_ε is
    factorised exactly once (gh-403). `cholesky` dispatches structurally,
    so a diagonal Σ_ε stays O(D_y).

    Args:
        emission_model: Observation operator H_t, one for all steps or a
            stack over time.
        obs_noise: Observation-noise covariance Σ_ε.
        observations: Observations y_1:T, one row per step.

    Returns:
        The whitened H_w,t, y_w,t and log det Σ_ε.
    """
    # L_ε, with L_ε L_εᵀ = Σ_ε
    noise_cholesky = cholesky(obs_noise)
    # diag(L_ε)
    cholesky_diagonal = diag(noise_cholesky)
    # log det Σ_ε = 2 ∑ log diag(L_ε)
    obs_noise_logdet = 2.0 * jnp.sum(jnp.log(cholesky_diagonal))
    # y_1:Tᵀ, one column per step
    obs_columns = rearrange(observations, "N M -> M N")
    # L_ε⁻¹ y_t, column by column
    whitened_columns = solve_columns(noise_cholesky, obs_columns)
    # y_w,t = L_ε⁻¹ y_t, one row per step
    whitened_obs = rearrange(whitened_columns, "M N -> N M")
    if emission_model.ndim == 2:
        # H_w = L_ε⁻¹ H
        whitened_operator = solve_columns(noise_cholesky, emission_model)
    else:
        num_steps, _, state_size = emission_model.shape
        # [H_1, …, H_T] side by side, so one solve whitens every step
        stacked = rearrange(emission_model, "N M d -> M (N d)")
        # L_ε⁻¹ [H_1, …, H_T]
        whitened_stacked = solve_columns(noise_cholesky, stacked)
        # H_w,t = L_ε⁻¹ H_t
        whitened_operator = rearrange(
            whitened_stacked, "M (N d) -> N M d", N=num_steps, d=state_size
        )
    return _WhitenedObservations(
        whitened_operator=whitened_operator,
        whitened_obs=whitened_obs,
        obs_noise_logdet=obs_noise_logdet,
    )


def _build_likelihood_precision(
    whitened: _WhitenedObservations, num_steps: int, state_size: int
) -> BlockTriDiag:
    """Block-diagonal likelihood precision, blocks H_tᵀ Σ_ε⁻¹ H_t.

        Λ_lik,t = H_tᵀ Σ_ε⁻¹ H_t = H_w,tᵀ H_w,t

    Args:
        whitened: The observation model whitened by `_whiten`.
        num_steps: Number of time steps T.
        state_size: State length D_x.

    Returns:
        The likelihood precision as a `BlockTriDiag` with zero
        off-diagonal blocks.
    """
    whitened_operator = whitened.whitened_operator
    if whitened_operator.ndim == 2:
        # Λ_lik = H_wᵀ H_w
        block = einsum(
            whitened_operator, whitened_operator, "obs row, obs col -> row col"
        )
        # the same block at every step
        diag_blocks = repeat(block, "row col -> N row col", N=num_steps)
    else:
        # Λ_lik,t = H_w,tᵀ H_w,t
        diag_blocks = einsum(
            whitened_operator,
            whitened_operator,
            "N obs row, N obs col -> N row col",
        )

    # no coupling between steps
    sub_diag_blocks = jnp.zeros(
        (num_steps - 1, state_size, state_size), dtype=diag_blocks.dtype
    )
    return BlockTriDiag(diag_blocks, sub_diag_blocks)


def _build_data_vector(whitened: _WhitenedObservations) -> Float[Array, " Nd"]:
    """Data term H_tᵀ Σ_ε⁻¹ y_t = H_w,tᵀ y_w,t, stacked over t = 1, …, T.

    Args:
        whitened: The observation model whitened by `_whiten`.

    Returns:
        Hᵀ Σ_ε⁻¹ y, flattened to length T D_x.
    """
    whitened_operator = whitened.whitened_operator
    whitened_obs = whitened.whitened_obs
    if whitened_operator.ndim == 2:
        # H_wᵀ y_w,t
        data_vector = einsum(
            whitened_operator, whitened_obs, "obs state, N obs -> N state"
        )
    else:
        # H_w,tᵀ y_w,t
        data_vector = einsum(
            whitened_operator, whitened_obs, "N obs state, N obs -> N state"
        )
    return rearrange(data_vector, "N state -> (N state)")


def spingp_log_likelihood(
    prior_precision: BlockTriDiag,
    emission_model: Array,
    obs_noise: lx.AbstractLinearOperator,
    observations: Float[Array, "N d_obs"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Log marginal likelihood log p(y_1:T) of a zero-mean SpInGP prior.

        log p(y) = −½ (T D_y log 2π + T log det Σ_ε + ∑_t y_tᵀ Σ_ε⁻¹ y_t
                       − (Hᵀ Σ_ε⁻¹ y)ᵀ Λ_x|y⁻¹ (Hᵀ Σ_ε⁻¹ y)
                       + log det Λ_x|y − log det Λ_x)
        Λ_x|y = Λ_x + blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)

    with T time steps and D_y observations per step. Every operation
    exploits the banded structure, for O(T D_x³) cost.

    Algorithm:
        1. Whiten H_t and y_t by one Cholesky factor of Σ_ε.
        2. Add the likelihood precision: Λ_x|y = Λ_x + H_tᵀ Σ_ε⁻¹ H_t.
        3. Form the data term Hᵀ Σ_ε⁻¹ y and solve Λ_x|y against it.
        4. Take the banded log-determinants of Λ_x|y and Λ_x and assemble
           log p(y).

    Args:
        prior_precision: Prior precision Λ_x of x_1:T, block-tridiagonal
            with T diagonal and T − 1 sub-diagonal D_x × D_x blocks.
        emission_model: Observation operator H_t, one for all steps or a
            stack over time.
        obs_noise: Observation-noise covariance Σ_ε, as an operator.
        observations: Observations y_1:T, one row per step.
        solver: Solver strategy for the solve and log-determinant of
            Λ_x|y and Λ_x; ``None`` uses structural dispatch. Σ_ε always
            uses structural dispatch, since it is typically small.

    Returns:
        The log marginal likelihood log p(y_1:T).

    Examples:
        A random-walk prior over three steps, observed directly:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior = gaussx.BlockTriDiag(
        ...     jnp.array([[[2.0]], [[2.0]], [[1.0]]]),
        ...     jnp.array([[[-1.0]], [[-1.0]]]),
        ... )
        >>> obs_noise = lx.MatrixLinearOperator(0.5 * jnp.eye(1))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> log_likelihood = gaussx.spingp_log_likelihood(
        ...     prior, jnp.eye(1), obs_noise, observations
        ... )
        >>> log_likelihood.shape
        ()

        It equals the dense Gaussian density with Σ_y = Λ_x⁻¹ + Σ_ε:

        >>> obs_covariance = jnp.linalg.inv(prior.as_matrix()) + 0.5 * jnp.eye(3)
        >>> obs_covariance_op = lx.MatrixLinearOperator(
        ...     obs_covariance, lx.positive_semidefinite_tag
        ... )
        >>> dense = gaussx.gaussian_log_prob(
        ...     jnp.zeros(3), obs_covariance_op, observations[:, 0]
        ... )
        >>> bool(jnp.allclose(log_likelihood, dense, atol=1e-5))
        True
    """
    num_steps = prior_precision._num_blocks
    state_size = prior_precision._block_size
    # T D_y
    total_obs = observations.size
    # log 2π
    log_2pi = jnp.log(2.0 * jnp.pi)

    # --- posterior precision ---
    whitened = _whiten(emission_model, obs_noise, observations)
    # blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)
    likelihood_precision = _build_likelihood_precision(whitened, num_steps, state_size)
    # Λ_x|y = Λ_x + blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)
    posterior_precision = prior_precision.add(likelihood_precision)

    # --- quadratic terms ---
    # Hᵀ Σ_ε⁻¹ y
    data_vector = _build_data_vector(whitened)
    # Λ_x|y⁻¹ Hᵀ Σ_ε⁻¹ y
    posterior_solve = dispatch_solve(posterior_precision, data_vector, solver)
    # (Hᵀ Σ_ε⁻¹ y)ᵀ Λ_x|y⁻¹ Hᵀ Σ_ε⁻¹ y
    data_quadratic = jnp.dot(data_vector, posterior_solve)
    # ∑_t y_tᵀ Σ_ε⁻¹ y_t = ‖y_w‖₂²
    obs_quadratic = jnp.sum(whitened.whitened_obs**2)

    # --- log-determinants ---
    # log det Λ_x|y, large, so through the solver
    posterior_logdet = dispatch_logdet(posterior_precision, solver)
    # log det Λ_x
    prior_logdet = dispatch_logdet(prior_precision, solver)
    # T log det Σ_ε, from the same factor
    total_obs_noise_logdet = num_steps * whitened.obs_noise_logdet

    # log p(y) = −½ (T D_y log 2π + T log det Σ_ε + yᵀ Σ_ε⁻¹ y
    #     − (Hᵀ Σ_ε⁻¹ y)ᵀ Λ_x|y⁻¹ Hᵀ Σ_ε⁻¹ y + log det Λ_x|y − log det Λ_x)
    return -0.5 * (
        total_obs * log_2pi
        + total_obs_noise_logdet
        + obs_quadratic
        - data_quadratic
        + posterior_logdet
        - prior_logdet
    )


def spingp_posterior(
    prior_precision: BlockTriDiag,
    emission_model: Array,
    obs_noise: lx.AbstractLinearOperator,
    observations: Float[Array, "N d_obs"],
    *,
    prior_mean: Float[Array, " Nd"] | None = None,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    """Posterior mean μ_x|y and precision Λ_x|y of x_1:T under SpInGP.

        Λ_x|y = Λ_x + blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)
        μ_x|y = Λ_x|y⁻¹ (Hᵀ Σ_ε⁻¹ y + Λ_x μ_x)

    With ``prior_mean=None`` the prior is zero-mean and the Λ_x μ_x term
    vanishes.

    Algorithm:
        1. Whiten H_t and y_t by one Cholesky factor of Σ_ε.
        2. Add the likelihood precision: Λ_x|y = Λ_x + H_tᵀ Σ_ε⁻¹ H_t.
        3. Form Hᵀ Σ_ε⁻¹ y, plus Λ_x μ_x when a prior mean is given.
        4. Solve Λ_x|y μ_x|y against it.

    Args:
        prior_precision: Prior precision Λ_x of x_1:T, block-tridiagonal.
        emission_model: Observation operator H_t, one for all steps or a
            stack over time.
        obs_noise: Observation-noise covariance Σ_ε, as an operator.
        observations: Observations y_1:T, one row per step.
        prior_mean: Prior mean μ_x of x_1:T, stacked to length T D_x (for
            example the ``mean`` half of
            `gaussx.MarkovGaussian.to_precision_form`); ``None`` means zero.
        solver: Solver strategy for the solve against Λ_x|y; ``None`` uses
            structural dispatch.

    Returns:
        Tuple ``(posterior_mean, posterior_precision)`` holding μ_x|y,
        stacked to length T D_x, and Λ_x|y as a `BlockTriDiag`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior = gaussx.BlockTriDiag(
        ...     jnp.array([[[2.0]], [[2.0]], [[1.0]]]),
        ...     jnp.array([[[-1.0]], [[-1.0]]]),
        ... )
        >>> obs_noise = lx.MatrixLinearOperator(0.5 * jnp.eye(1))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> mean, precision = gaussx.spingp_posterior(
        ...     prior, jnp.eye(1), obs_noise, observations
        ... )
        >>> mean.shape, isinstance(precision, gaussx.BlockTriDiag)
        ((3,), True)

        The mean matches the dense Gaussian conditional:

        >>> dense_precision = prior.as_matrix() + 2.0 * jnp.eye(3)
        >>> dense = jnp.linalg.solve(dense_precision, 2.0 * observations[:, 0])
        >>> bool(jnp.allclose(mean, dense, atol=1e-5))
        True
    """
    num_steps = prior_precision._num_blocks
    state_size = prior_precision._block_size

    # --- posterior precision ---
    whitened = _whiten(emission_model, obs_noise, observations)
    # blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)
    likelihood_precision = _build_likelihood_precision(whitened, num_steps, state_size)
    # Λ_x|y = Λ_x + blockdiag_t(H_tᵀ Σ_ε⁻¹ H_t)
    posterior_precision = prior_precision.add(likelihood_precision)

    # --- posterior mean ---
    # Hᵀ Σ_ε⁻¹ y
    data_vector = _build_data_vector(whitened)
    if prior_mean is not None:
        # Hᵀ Σ_ε⁻¹ y + Λ_x μ_x
        data_vector = data_vector + prior_precision.mv(prior_mean)

    # μ_x|y = Λ_x|y⁻¹ (Hᵀ Σ_ε⁻¹ y + Λ_x μ_x)
    posterior_mean = dispatch_solve(posterior_precision, data_vector, solver)

    return posterior_mean, posterior_precision
