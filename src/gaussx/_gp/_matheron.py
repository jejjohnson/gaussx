"""Matheron's-rule pathwise conditioning of joint Gaussian draws.

A indexes the target entries and B the conditioning entries of a joint
Gaussian, as in `gaussx.sample_joint_conditional`: x_A and x_B are the two
blocks of one draw, Σ_AB and Σ_BB their cross- and conditioning covariances,
and β the observed value of x_B.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


def matheron_update(
    prior_sample_target: Float[Array, "S N_star"],
    prior_sample_conditioning: Float[Array, "S M"],
    observed_value: Float[Array, " M"],
    cross_covariance: lx.AbstractLinearOperator,
    conditioning_covariance: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, "S N_star"]:
    """Posterior draws of x_A ∣ x_B = β from joint prior draws, by Matheron's rule.

        x_A|β = x_A + Σ_AB Σ_BB⁻¹ (β − x_B),   (x_A, x_B) ~ p(x_A, x_B)

    As in `gaussx.sample_joint_conditional`, A indexes the N_A target
    entries and B the N_B conditioning entries of a joint Gaussian, with
    cross-covariance Σ_AB and conditioning covariance Σ_BB. Each row of the
    inputs is one joint prior draw (x_A, x_B); the update turns it into a
    draw of x_A given the observed value β of x_B.

    Both covariance arguments stay lineax operators, so the conditioning
    solve uses gaussx's structural dispatch and the target correction is a
    rectangular matvec.

    Algorithm:
        1. Promote the draws and β to one dtype and check their shapes
           against Σ_AB and Σ_BB.
        2. Form the residuals β − x_B, one per draw.
        3. Solve Σ_BB⁻¹ (β − x_B) per draw, with ``solver`` or structural
           dispatch.
        4. Map each solution through Σ_AB and add it to x_A.

    Args:
        prior_sample_target: Joint prior draws of the target block x_A, one
            per row.
        prior_sample_conditioning: The matching draws of the conditioning
            block x_B, one per row.
        observed_value: Observed value β of x_B.
        cross_covariance: Cross-covariance Σ_AB between the target and the
            conditioning entries, N_A × N_B.
        conditioning_covariance: Conditioning covariance Σ_BB, N_B × N_B.
        solver: Optional solver strategy for the conditioning solve
            (e.g. `gaussx.CGSolver`, `gaussx.BBMMSolver`).
            When ``None``, routes through structural dispatch on
            ``conditioning_covariance``.

    Returns:
        The posterior draws x_A|β, one per row.

    Raises:
        ValueError: If the draws are not 2-D, β is not 1-D, the numbers of
            draws differ, or the sizes of β, Σ_AB and Σ_BB do not match the
            draws.

    Examples:
        Three draws of two target entries, conditioned on one entry with
        Σ_BB = 1, so the gain is Σ_AB:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> cross = jnp.array([[1.0], [0.5]])
        >>> cross_op = lx.MatrixLinearOperator(cross)
        >>> conditioning = lx.MatrixLinearOperator(
        ...     jnp.eye(1), lx.positive_semidefinite_tag
        ... )
        >>> target = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        >>> draws_b = jnp.array([[0.0], [1.0], [-1.0]])
        >>> beta = jnp.array([0.5])
        >>> posterior = gaussx.matheron_update(
        ...     target, draws_b, beta, cross_op, conditioning
        ... )
        >>> expected = target + (0.5 - draws_b) * cross[:, 0]
        >>> bool(jnp.allclose(posterior, expected))
        True

        An iterative solver gives the same draws:

        >>> iterative = gaussx.matheron_update(
        ...     target, draws_b, beta, cross_op, conditioning,
        ...     solver=gaussx.CGSolver(),
        ... )
        >>> bool(jnp.allclose(iterative, posterior, atol=1e-5))
        True

        No draws in, no draws out:

        >>> empty = gaussx.matheron_update(
        ...     jnp.zeros((0, 2)), jnp.zeros((0, 1)), beta, cross_op, conditioning
        ... )
        >>> empty.shape
        (0, 2)
    """
    prior_sample_target = jnp.asarray(prior_sample_target)
    prior_sample_conditioning = jnp.asarray(prior_sample_conditioning)
    observed_value = jnp.asarray(observed_value)
    dtype = jnp.result_type(
        prior_sample_target,
        prior_sample_conditioning,
        observed_value,
    )
    # Promote sample arrays and observations together so the residual,
    # structured solve, and matvec all use the same precision.
    prior_sample_target = prior_sample_target.astype(dtype)
    prior_sample_conditioning = prior_sample_conditioning.astype(dtype)
    observed_value = observed_value.astype(dtype)

    if prior_sample_target.ndim != 2:
        raise ValueError("prior_sample_target must have shape (S, N_star).")
    if prior_sample_conditioning.ndim != 2:
        raise ValueError("prior_sample_conditioning must have shape (S, M).")
    if observed_value.ndim != 1:
        raise ValueError("observed_value must have shape (M,).")
    if prior_sample_target.shape[0] != prior_sample_conditioning.shape[0]:
        raise ValueError("prior samples must have the same sample dimension S.")

    num_conditioning = prior_sample_conditioning.shape[1]
    num_target = prior_sample_target.shape[1]
    if observed_value.shape[0] != num_conditioning:
        raise ValueError("observed_value must match the conditioning dimension M.")
    if conditioning_covariance.in_size() != num_conditioning:
        raise ValueError(
            "conditioning_covariance input size must match the conditioning "
            "dimension M."
        )
    if conditioning_covariance.out_size() != num_conditioning:
        raise ValueError(
            "conditioning_covariance output size must match the conditioning "
            "dimension M."
        )
    if cross_covariance.in_size() != num_conditioning:
        raise ValueError(
            "cross_covariance input size must match the conditioning dimension M."
        )
    if cross_covariance.out_size() != num_target:
        raise ValueError(
            "cross_covariance output size must match the target dimension N_star."
        )

    # jnp, not einx: the number of draws may be 0, which einx rejects.
    # β − x_B, one row per draw
    residuals = observed_value - prior_sample_conditioning
    if solver is not None:
        solve_one = lambda r: dispatch_solve(conditioning_covariance, r, solver)
    else:
        solve_one = lambda r: solve(conditioning_covariance, r)
    # Σ_BB⁻¹ (β − x_B), one row at a time
    solves = jax.vmap(solve_one)(residuals)
    # Σ_AB Σ_BB⁻¹ (β − x_B)
    corrections = jax.vmap(cross_covariance.mv)(solves)
    # x_A|β = x_A + Σ_AB Σ_BB⁻¹ (β − x_B)
    return prior_sample_target + corrections
