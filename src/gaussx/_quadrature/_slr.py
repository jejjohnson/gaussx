"""Statistical linear regression: moment-matched linear-Gaussian surrogate."""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


class SLRResult(eqx.Module):
    """Linear-Gaussian surrogate produced by statistical linear regression.

        p(y ∣ f) ≈ 𝒩(y ∣ A f + b, Ω)

    Attributes:
        A: Linearisation gain A.
        b: Linearisation offset b.
        omega: Residual noise covariance Ω.
        mu: Expected conditional mean μ_y = 𝔼_q[𝔼[y ∣ f]].

    Examples:
        >>> import jax.numpy as jnp
        >>> surrogate = SLRResult(
        ...     A=jnp.eye(2), b=jnp.zeros(2), omega=jnp.eye(2), mu=jnp.zeros(2)
        ... )
        >>> surrogate.A.shape
        (2, 2)
    """

    A: Float[Array, "M D"]
    b: Float[Array, " M"]
    omega: Float[Array, "M M"]
    mu: Float[Array, " M"]


def statistical_linear_regression(
    conditional_mean_fn: Callable[[Float[Array, " D"]], Float[Array, " M"]],
    conditional_var_fn: Callable[[Float[Array, " D"]], Float[Array, "M M"]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> SLRResult:
    """Moment-match a nonlinear conditional to a linear-Gaussian surrogate.

        A = Σ_fyᵀ Σ_f⁻¹,    b = μ_y − A μ_f,    Ω = Σ_y − A Σ_fy

    Given a conditional p(y ∣ f) described by its first two moments and a
    Gaussian state q(f) = 𝒩(μ_f, Σ_f), returns the model
    p(y ∣ f) ≈ 𝒩(y ∣ A f + b, Ω) whose moments match those of the true
    conditional under q, with

        μ_y = 𝔼_q[𝔼[y ∣ f]]
        Σ_fy = Cov_q[f, 𝔼[y ∣ f]]
        Σ_y = 𝕍_q[𝔼[y ∣ f]] + 𝔼_q[𝕍[y ∣ f]]

    This is the shared primitive behind posterior linearisation, the
    iterated Kalman smoother (IPLF / IPLS), and the conjugate Gaussian site
    that an EP cavity meets. Feeding ``(A, b, omega)`` into a standard
    linear-Gaussian Kalman update reproduces posterior-linearisation
    inference exactly.

    All expectations are evaluated with ``integrator``, so the accuracy /
    cost trade-off is chosen by the caller. Only the conditional moments
    are needed — no autodiff runs through the quadrature.

    Algorithm:
        1. Take the rule's points and weights under q(f).
        2. Assemble μ_y, 𝕍_q[𝔼[y ∣ f]] and Σ_fy from the conditional means.
        3. Average the conditional covariances over the same points.
        4. Solve for A, then form b and Ω.

    Args:
        conditional_mean_fn: Maps a latent f to the conditional mean
            𝔼[y ∣ f].
        conditional_var_fn: Maps a latent f to the conditional covariance
            𝕍[y ∣ f]. Either the full covariance or, for conditionally
            independent outputs, a vector of variances.
        state: Gaussian state q(f). Its covariance must be symmetric.
        integrator: Point-based rule used for the expectations, e.g.
            `gaussx.FifthOrderCubatureIntegrator`.

    Returns:
        `SLRResult` with the gain, offset, residual covariance, and
        expected conditional mean.

    Raises:
        NotImplementedError: If ``integrator`` is not point-based.
        ValueError: If ``conditional_var_fn`` does not return an ``(M,)``
            or ``(M, M)`` array matching the ``M`` outputs of
            ``conditional_mean_fn``.

    Examples:
        A linear conditional with constant noise is recovered exactly:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> state = gaussx.GaussianState(jnp.ones(2), covariance)
        >>> surrogate = statistical_linear_regression(
        ...     lambda f: 2.0 * f + 1.0,
        ...     lambda f: 0.5 * jnp.ones(2),
        ...     state,
        ...     gaussx.CubatureIntegrator(),
        ... )
        >>> bool(jnp.allclose(surrogate.A, 2.0 * jnp.eye(2)))
        True
        >>> bool(jnp.allclose(surrogate.omega, 0.5 * jnp.eye(2)))
        True
    """
    points, mean_weights, covariance_weights = integrator.points_and_weights(state)

    # --- moments of the conditional mean under q(f) ---
    # 𝔼[y ∣ fᵢ]
    conditional_means = jax.vmap(conditional_mean_fn)(points)
    moments = assemble_propagation_result(
        points, conditional_means, state.mean, mean_weights, covariance_weights
    )
    # μ_y = 𝔼_q[𝔼[y ∣ f]]
    output_mean = moments.state.mean
    # 𝕍_q[𝔼[y ∣ f]]
    variance_of_mean = moments.state.cov.as_matrix()
    # Σ_fy = Cov_q[f, 𝔼[y ∣ f]]
    cross_covariance = moments.cross_cov
    assert cross_covariance is not None, "Integrator must return cross_cov"

    # --- 𝔼_q[𝕍[y ∣ f]], reusing the same points ---
    # The trailing shape is checked against D_y rather than just the rank: a
    # (1,) or (1, 1) return would otherwise broadcast across the whole of Σ_y
    # and silently corrupt every entry of Ω.
    output_size = conditional_means.shape[-1]
    # 𝕍[y ∣ fᵢ]
    conditional_variances = jax.vmap(conditional_var_fn)(points)
    if conditional_variances.shape[1:] == (output_size,):
        # w_m,i diag 𝕍[y ∣ fᵢ]
        weighted_variances = einx.multiply(
            "point, point out -> point out", mean_weights, conditional_variances
        )
        # ∑ᵢ w_m,i diag 𝕍[y ∣ fᵢ]
        diagonal_of_mean = reduce(weighted_variances, "point out -> out", "sum")
        # 𝔼_q[𝕍[y ∣ f]], diagonal
        mean_of_variance = jnp.diag(diagonal_of_mean)
    elif conditional_variances.shape[1:] == (output_size, output_size):
        # w_m,i 𝕍[y ∣ fᵢ]
        weighted_variances = einx.multiply(
            "point, point row col -> point row col",
            mean_weights,
            conditional_variances,
        )
        # 𝔼_q[𝕍[y ∣ f]] = ∑ᵢ w_m,i 𝕍[y ∣ fᵢ]
        mean_of_variance = reduce(weighted_variances, "point row col -> row col", "sum")
    else:
        msg = (
            f"conditional_var_fn must return an ({output_size},) or "
            f"({output_size}, {output_size}) array per point to match the "
            f"{output_size} outputs of conditional_mean_fn, got trailing "
            f"shape {conditional_variances.shape[1:]}."
        )
        raise ValueError(msg)

    # Σ_y = 𝕍_q[𝔼[y ∣ f]] + 𝔼_q[𝕍[y ∣ f]]
    output_covariance = variance_of_mean + mean_of_variance

    # --- surrogate ---
    # Σ_fyᵀ
    cross_covariance_transpose = rearrange(cross_covariance, "state out -> out state")
    # A = Σ_fyᵀ Σ_f⁻¹; solve_rows solves Σ_f x = row, and Σ_f is symmetric, so
    # stacking the rows of Σ_fyᵀ gives A directly
    gain = solve_rows(state.cov, cross_covariance_transpose)

    # A μ_f
    gain_mean = gain @ state.mean
    # b = μ_y − A μ_f
    offset = output_mean - gain_mean
    # A Σ_fy
    explained_covariance = einsum(
        gain, cross_covariance, "out state, state col -> out col"
    )
    # Ω = Σ_y − A Σ_fy, symmetrised against round-off
    residual_covariance = symmetrize(output_covariance - explained_covariance)

    return SLRResult(A=gain, b=offset, omega=residual_covariance, mu=output_mean)
