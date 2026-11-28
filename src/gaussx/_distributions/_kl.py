"""Closed-form KL divergence between two Gaussians with operator covariances.

p = 𝒩(μ_p, Σ_p) and q = 𝒩(μ_q, Σ_q) share the event size D_x; L_p and L_q
are the lower Cholesky factors, L_p L_pᵀ = Σ_p and L_q L_qᵀ = Σ_q. A solver
strategy, when given, takes every solve and log-determinant with Σ_q, and a
log-determinant strategy the one of Σ_p (gh-884).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_columns, trace_product
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import _cholesky_logdet, _dense_psd_matrix, logdet
from gaussx._primitives._solve import solve
from gaussx._primitives._trace import trace_and_diag
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


def gaussian_kl(
    p_loc: Float[Array, " N"],
    p_cov: lx.AbstractLinearOperator,
    q_loc: Float[Array, " N"],
    q_cov: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
    p_logdet: AbstractLogdetStrategy | None = None,
    num_probes: int = 20,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """KL(first ‖ second): KL[p ‖ q] between two multivariate normals.

        KL[p ‖ q] = ½ (tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D_x
                       + log det Σ_q − log det Σ_p)

    with p = 𝒩(μ_p, Σ_p) and q = 𝒩(μ_q, Σ_q). This is the canonical KL for
    lineax-operator covariances; the specialised variants compute the same
    quantity in other parameterisations:

    - `kl_standard_normal`: the case q = 𝒩(0, I), with no solve.
    - `gauss_kl`: the Cholesky-parameterised form for GP and SVGP models,
      with multi-output and diagonal ``q_sqrt``.
    - `kl_divergence`: the Bregman-divergence form on natural parameters
      of the exponential family.

    With a ``solver``, every solve and log-determinant with Σ_q goes through
    it, and log det Σ_p through ``p_logdet`` (structural dispatch when
    ``None``): the two covariances have different conditioning, so one
    strategy is never applied to both. The trace is exact, D_x solves of
    the columns Σ_p e_j, when ``key`` is ``None``; with a ``key`` it is
    the Hutchinson estimate (1 / N_probe) ∑_i z^(i)ᵀ Σ_q⁻¹ Σ_p z^(i), which
    costs ``num_probes`` solves and suits a large D_x. A stochastic trace
    or log-determinant can make the estimate slightly negative; it is not
    clamped, since clamping would bias an ELBO.

    Algorithm:
        1. When both covariances are dense and PSD, factor Σ_p = L_p L_pᵀ
           and Σ_q = L_q L_qᵀ; then tr(Σ_q⁻¹ Σ_p) = ‖L_q⁻¹ L_p‖_F², the
           quadratic term is ‖L_q⁻¹ (μ_q − μ_p)‖² and both log-determinants
           come from the factors.
        2. Otherwise take tr(Σ_q⁻¹ Σ_p) through the lazy inverse of Σ_q, and
           the solve and log-determinants through structural dispatch.
        3. With a ``solver``, take the quadratic term from one solve with
           Σ_q, the trace from D_x solves (or ``num_probes`` with a
           ``key``), log det Σ_q from the solver and log det Σ_p from
           ``p_logdet``.
        4. Combine the terms.

    Args:
        p_loc: Mean μ_p of the first Gaussian p.
        p_cov: Covariance Σ_p of p.
        q_loc: Mean μ_q of the second Gaussian q.
        q_cov: Covariance Σ_q of q.
        solver: Solver strategy for every solve and log-determinant with
            Σ_q. ``None`` (the default) keeps structural dispatch for both
            covariances, and ignores ``p_logdet``, ``num_probes`` and ``key``.
        p_logdet: Log-determinant strategy for Σ_p, used with a ``solver``;
            ``None`` uses structural dispatch.
        num_probes: Number of Hutchinson probes N_probe for the trace, used
            only with a ``solver`` and a ``key``.
        key: PRNG key. With a ``solver``, ``None`` gives the exact trace and
            the strategies' own seeds; a key gives the Hutchinson trace and
            is split between it and the two log-determinants.

    Returns:
        The scalar KL divergence KL[p ‖ q].

    Examples:
        A shifted, wider second Gaussian:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> p_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> q_cov = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> q_loc = jnp.array([1.0, 0.0])
        >>> kl = gaussx.gaussian_kl(jnp.zeros(2), p_cov, q_loc, q_cov)
        >>> round(float(kl), 4)
        0.4431

        The divergence of a Gaussian from itself is zero:

        >>> self_kl = gaussx.gaussian_kl(q_loc, q_cov, q_loc, q_cov)
        >>> bool(jnp.allclose(self_kl, 0.0, atol=1e-6))
        True

        Diagonal operators take the structured path and agree:

        >>> p_diagonal = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> q_diagonal = lx.DiagonalLinearOperator(2.0 * jnp.ones(2))
        >>> structured = gaussx.gaussian_kl(
        ...     jnp.zeros(2), p_diagonal, q_loc, q_diagonal
        ... )
        >>> bool(jnp.allclose(structured, kl))
        True

        Any solver strategy for Σ_q gives the same value:

        >>> with_solver = gaussx.gaussian_kl(
        ...     jnp.zeros(2), p_cov, q_loc, q_cov, solver=gaussx.CGSolver()
        ... )
        >>> bool(jnp.allclose(with_solver, kl, atol=1e-4))
        True
    """
    if solver is not None:
        return _gaussian_kl_with_solver(
            p_loc,
            p_cov,
            q_loc,
            q_cov,
            solver=solver,
            p_logdet=p_logdet,
            num_probes=num_probes,
            key=key,
        )
    event_size = p_loc.shape[-1]
    # μ_q − μ_p
    mean_difference = q_loc - p_loc

    p_matrix, q_matrix = _dense_psd_matrix(p_cov), _dense_psd_matrix(q_cov)
    if p_matrix is not None and q_matrix is not None:
        # Two Choleskys for the whole KL (gh-329).
        # L_p L_pᵀ = Σ_p
        p_factor = jnp.linalg.cholesky(p_matrix)
        # L_q L_qᵀ = Σ_q
        q_factor = jnp.linalg.cholesky(q_matrix)
        # L_q⁻¹ L_p
        whitened_factor = jax.scipy.linalg.solve_triangular(
            q_factor, p_factor, lower=True
        )
        # L_q⁻¹ (μ_q − μ_p)
        whitened_difference = jax.scipy.linalg.solve_triangular(
            q_factor, mean_difference, lower=True
        )
        # log det Σ_q − log det Σ_p
        logdet_difference = _cholesky_logdet(q_factor) - _cholesky_logdet(p_factor)
        # tr(Σ_q⁻¹ Σ_p) = ‖L_q⁻¹ L_p‖_F², a full reduction with no axis
        trace_term = jnp.sum(whitened_factor**2)
        # (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) = ‖L_q⁻¹ (μ_q − μ_p)‖²
        mahalanobis = whitened_difference @ whitened_difference
        # KL[p ‖ q] = ½ (tr + quadratic − D_x + log det Σ_q − log det Σ_p)
        return 0.5 * (trace_term + mahalanobis - event_size + logdet_difference)

    # Σ_q⁻¹, lazy
    q_precision = inv(q_cov)
    # tr(Σ_q⁻¹ Σ_p)
    trace_term = trace_product(q_precision, p_cov)

    # Σ_q⁻¹ (μ_q − μ_p)
    solved_difference = solve(q_cov, mean_difference)
    # (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p), a full reduction with no axis
    mahalanobis = jnp.sum(mean_difference * solved_difference)

    # log det Σ_q
    q_logdet = logdet(q_cov)
    # log det Σ_p
    p_logdet_value = logdet(p_cov)

    # KL[p ‖ q] = ½ (tr + quadratic − D_x + log det Σ_q − log det Σ_p)
    return 0.5 * (trace_term + mahalanobis - event_size + q_logdet - p_logdet_value)


def _gaussian_kl_with_solver(
    p_loc: Float[Array, " N"],
    p_cov: lx.AbstractLinearOperator,
    q_loc: Float[Array, " N"],
    q_cov: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy,
    p_logdet: AbstractLogdetStrategy | None,
    num_probes: int,
    key: jax.Array | None,
) -> Float[Array, ""]:
    """KL[p ‖ q] with every Σ_q solve and log-determinant through ``solver``.

        KL[p ‖ q] = ½ (tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D_x
                       + log det Σ_q − log det Σ_p)

    Args:
        p_loc: Mean μ_p.
        p_cov: Covariance Σ_p.
        q_loc: Mean μ_q.
        q_cov: Covariance Σ_q.
        solver: Solver strategy for Σ_q.
        p_logdet: Log-determinant strategy for Σ_p, or ``None``.
        num_probes: Hutchinson probes for the trace, with a ``key``.
        key: PRNG key, or ``None`` for the exact trace.

    Returns:
        The scalar KL[p ‖ q].
    """
    event_size = p_loc.shape[-1]
    if key is None:
        trace_key = q_key = p_key = None
    else:
        trace_key, q_key, p_key = jr.split(key, 3)
    # μ_q − μ_p
    mean_difference = q_loc - p_loc
    # Σ_q⁻¹ (μ_q − μ_p)
    solved_difference = dispatch_solve(q_cov, mean_difference, solver)
    # (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p), a full reduction with no axis
    mahalanobis = jnp.sum(mean_difference * solved_difference)
    # tr(Σ_q⁻¹ Σ_p)
    trace_term = _solved_trace(p_cov, q_cov, solver, num_probes, trace_key)
    # log det Σ_q
    q_logdet = dispatch_logdet(q_cov, solver, key=q_key)
    # log det Σ_p
    p_logdet_value = dispatch_logdet(p_cov, p_logdet, key=p_key)
    # KL[p ‖ q] = ½ (tr + quadratic − D_x + log det Σ_q − log det Σ_p)
    return 0.5 * (trace_term + mahalanobis - event_size + q_logdet - p_logdet_value)


def _solved_trace(
    p_cov: lx.AbstractLinearOperator,
    q_cov: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy,
    num_probes: int,
    key: jax.Array | None,
) -> Float[Array, ""]:
    """tr(Σ_q⁻¹ Σ_p), exactly or by Hutchinson, with ``solver`` for Σ_q.

        tr(Σ_q⁻¹ Σ_p) = ∑_j [Σ_q⁻¹ Σ_p e_j]_j
                      ≈ (1 / N_probe) ∑_i z^(i)ᵀ Σ_q⁻¹ Σ_p z^(i)

    Args:
        p_cov: Covariance Σ_p.
        q_cov: Covariance Σ_q.
        solver: Solver strategy for Σ_q.
        num_probes: Number of probes N_probe, with a ``key``.
        key: PRNG key for the probes; ``None`` gives the exact trace.

    Returns:
        The scalar tr(Σ_q⁻¹ Σ_p).
    """
    if key is None:
        structure = p_cov.in_structure()
        identity = jnp.eye(structure.shape[0], dtype=structure.dtype)
        # Σ_p e_j, one matvec per column
        p_columns = jax.vmap(p_cov.mv, in_axes=1, out_axes=1)(identity)
        # Σ_q⁻¹ Σ_p e_j, one solve per column
        solved = solve_columns(q_cov, p_columns, solver=solver)
        # ∑_j [Σ_q⁻¹ Σ_p]_jj
        return jnp.trace(solved)

    def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        # Σ_q⁻¹ Σ_p v
        return dispatch_solve(q_cov, p_cov.mv(vector), solver)

    product = lx.FunctionLinearOperator(matvec, p_cov.in_structure())
    # (1 / N_probe) ∑_i z^(i)ᵀ Σ_q⁻¹ Σ_p z^(i)
    estimate, _ = trace_and_diag(product, num_probes=num_probes, key=key)
    return estimate
