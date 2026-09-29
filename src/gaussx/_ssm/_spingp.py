"""Sparse inverse Kalman filter (SpInGP) recipes."""

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
    """Observation model whitened by one Cholesky factor ``R = L Lᵀ``.

    Attributes:
        W: ``L⁻¹ H``, shape ``(d_obs, d)`` or ``(N, d_obs, d)``.
        z: ``L⁻¹ y_k`` per step, shape ``(N, d_obs)``.
        logdet_R: ``log|R|``.
    """

    W: Array
    z: Float[Array, "N d_obs"]
    logdet_R: Float[Array, ""]


def _whiten(
    emission_model: Array,
    obs_noise: lx.AbstractLinearOperator,
    observations: Float[Array, "N d_obs"],
) -> _WhitenedObservations:
    """Factor ``R`` once and whiten ``H`` and ``y`` by triangular solves.

    Then ``Hᵀ R⁻¹ H = Wᵀ W``, ``Hᵀ R⁻¹ y = Wᵀ z`` and ``yᵀ R⁻¹ y = ‖z‖²``,
    so no ``R⁻¹`` is materialised and ``R`` is factorised exactly once
    (gh-403). `cholesky` dispatches structurally, so a diagonal ``R``
    stays O(d_obs).
    """
    L = cholesky(obs_noise)
    logdet_R = 2.0 * jnp.sum(jnp.log(diag(L)))
    z = rearrange(solve_columns(L, rearrange(observations, "N M -> M N")), "M N -> N M")
    if emission_model.ndim == 2:
        W = solve_columns(L, emission_model)
    else:
        N, _, d = emission_model.shape
        stacked = rearrange(emission_model, "N M d -> M (N d)")
        W = rearrange(solve_columns(L, stacked), "M (N d) -> N M d", N=N, d=d)
    return _WhitenedObservations(W=W, z=z, logdet_R=logdet_R)


def _build_likelihood_precision(
    whitened: _WhitenedObservations, N: int, d: int
) -> BlockTriDiag:
    """Block-diagonal likelihood precision ``Λ_lik[k] = Hₖᵀ R⁻¹ Hₖ = Wₖᵀ Wₖ``.

    Args:
        whitened: The observation model whitened by `_whiten`.
        N: Number of time steps.
        d: State dimension.

    Returns:
        Block-tridiagonal likelihood precision (block-diagonal).
    """
    W = whitened.W
    if W.ndim == 2:
        block = einsum(W, W, "M d1, M d2 -> d1 d2")
        diag_blocks = repeat(block, "d1 d2 -> N d1 d2", N=N)
    else:
        diag_blocks = einsum(W, W, "N M d1, N M d2 -> N d1 d2")

    sub_diag_blocks = jnp.zeros((N - 1, d, d), dtype=diag_blocks.dtype)
    return BlockTriDiag(diag_blocks, sub_diag_blocks)


def _build_data_vector(whitened: _WhitenedObservations) -> Float[Array, " Nd"]:
    """Data contribution ``Hₖᵀ R⁻¹ yₖ = Wₖᵀ zₖ``, flattened to ``(N * d,)``."""
    W, z = whitened.W, whitened.z
    if W.ndim == 2:
        data_vec = einsum(W, z, "M d, N M -> N d")
    else:
        data_vec = einsum(W, z, "N M d, N M -> N d")
    return rearrange(data_vec, "N d -> (N d)")


def spingp_log_likelihood(
    prior_precision: BlockTriDiag,
    emission_model: Array,
    obs_noise: lx.AbstractLinearOperator,
    observations: Float[Array, "N d_obs"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    r"""Log marginal likelihood via sparse inverse GP formulation.

    Computes the log marginal likelihood using the precision-form
    Kalman filter (SpInGP):

        1. Likelihood precision sites: $\Lambda_{lik} = H^T R^{-1} H$
        2. Posterior precision: $\Lambda_{post} = \Lambda_{prior} + \Lambda_{lik}$
        3. log p(y) via banded Cholesky logdet and quadratic form

    The full expression is:

        log p(y) = -0.5 * (N_{obs} * log(2\pi) + log|R|_{total}
                   + y^T R^{-1} y - \eta^T \Lambda_{post}^{-1} \eta
                   + log|\Lambda_{post}| - log|\Lambda_{prior}|)

    where $\eta = H^T R^{-1} y$.

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
    whitened = _whiten(emission_model, obs_noise, observations)
    lik_prec = _build_likelihood_precision(whitened, N, d)
    post_prec = prior_precision.add(lik_prec)

    # Data vector: eta = H^T R^{-1} y
    eta = _build_data_vector(whitened)

    # Quadratic term: eta^T Lambda_post^{-1} eta
    post_solve = dispatch_solve(post_prec, eta, solver)
    quad_term = jnp.dot(eta, post_solve)

    # Observation quadratic: y^T R^{-1} y = ||z||^2
    obs_quad = jnp.sum(whitened.z**2)

    # Log determinants (posterior precision: may be large, use solver)
    ld_post = dispatch_logdet(post_prec, solver)
    ld_prior = dispatch_logdet(prior_precision, solver)

    # Total observation noise logdet: N * log|R|, from the same factor
    ld_R_total = N * whitened.logdet_R

    return -0.5 * (
        N_obs * log_2pi + ld_R_total + obs_quad - quad_term + ld_post - ld_prior
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
    r"""Posterior mean and precision via SpInGP.

    Computes the posterior by adding likelihood precision sites to the
    prior precision and solving for the posterior mean:

        \Lambda_{post} = \Lambda_{prior} + H^T R^{-1} H
        \mu_{post} = \Lambda_{post}^{-1} (H^T R^{-1} y + \Lambda_{prior} \mu_{prior})

    With ``prior_mean=None`` the prior is taken to be zero-mean and the
    second term vanishes.

    Args:
        prior_precision: Prior precision as ``BlockTriDiag``.
        emission_model: Emission matrix H. Shape ``(d_obs, d)`` for
            shared or ``(N, d_obs, d)`` per time step.
        obs_noise: Observation noise covariance R operator.
        observations: Observations y, shape ``(N, d_obs)``.
        prior_mean: Optional prior mean ``mu_prior``, shape ``(N * d,)``
            — e.g. the ``mean`` half of
            `gaussx.MarkovGaussian.to_precision_form`. Defaults to
            zero.
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
    whitened = _whiten(emission_model, obs_noise, observations)
    lik_prec = _build_likelihood_precision(whitened, N, d)
    post_prec = prior_precision.add(lik_prec)

    # Data vector: eta = H^T R^{-1} y (+ Lambda_prior mu_prior)
    eta = _build_data_vector(whitened)
    if prior_mean is not None:
        eta = eta + prior_precision.mv(prior_mean)

    # Posterior mean: Lambda_post^{-1} eta
    post_mean = dispatch_solve(post_prec, eta, solver)

    return post_mean, post_prec
