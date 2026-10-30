"""Gaussian-path linearisation of a nonlinear SDE and the drift path-KL.

Variational treatments of a nonlinear Itô SDE

    dx(t) = f(x(t)) dt + L_c db(t),

with drift f and diffusion covariance B_c = L_c Q_c L_cᵀ, approximate its
posterior by a Gaussian process q with marginals q(x_t) = 𝒩(μ_x,t, Σ_x,t)
and a linear drift f_lin,t(x) = F_c,t x + c_t (Archambeau, Cornford, Opper
& Shawe-Taylor, 2007). Two quantities recur in every such method: the
linear drift that is optimal along the path, and the path-KL between the
linear and the nonlinear SDE. Both are expectations under the Gaussian
marginals, so both are evaluated with a gaussx integrator.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange, repeat
from gaussx._linalg._linalg import solve_rows
from gaussx._quadrature._expectations import mean_expectation
from gaussx._quadrature._integrator import AbstractIntegrator, moment_transform
from gaussx._quadrature._taylor import TaylorIntegrator
from gaussx._quadrature._types import GaussianState
from gaussx._quadrature._unscented import UnscentedIntegrator


class LinearizedSDE(eqx.Module):
    """Linear surrogate dx(t) = (F_c,t x(t) + c_t) dt + L_c db(t) of an SDE.

    One linear drift per path point t = 1, …, T, on a state of length D_x,
    with the diffusion covariance B_c,t = L_c Q_c L_cᵀ carried alongside.

    Attributes:
        A: Linearised drift matrices F_c,t.
        b: Linearised drift offsets c_t.
        Q: Diffusion covariances B_c,t, unchanged by the linearisation.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> covs = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> means = jnp.zeros((3, 2))
        >>> linear = gaussx.linearize_sde(lambda x: -x, jnp.eye(2), means, covs)
        >>> linear.A.shape, linear.b.shape, linear.Q.shape
        ((3, 2, 2), (3, 2), (3, 2, 2))
    """

    A: Float[Array, "T d d"]
    b: Float[Array, "T d"]
    Q: Float[Array, "T d d"]


def _default_integrator(integrator: AbstractIntegrator | None) -> AbstractIntegrator:
    """The given rule, or ``UnscentedIntegrator(alpha=1.0)`` when ``None``.

    Args:
        integrator: Gaussian integration rule, or ``None``.

    Returns:
        The rule to use.
    """
    if integrator is not None:
        return integrator
    # alpha=1.0 for the reason given in `moment_transform`: the 1e-3 default
    # recovers the moments by cancellation and is unsafe in float32.
    return UnscentedIntegrator(alpha=1.0)


def _check_path(
    path_means: Float[Array, "T d"], path_covs: Float[Array, "T d d"]
) -> tuple[int, int]:
    """Check the path marginals and return their sizes.

    Args:
        path_means: Marginal means μ_x,t.
        path_covs: Marginal covariances Σ_x,t.

    Returns:
        Tuple ``(T, d)``: the number of path points and D_x.

    Raises:
        ValueError: If the shapes are not ``(T, d)`` and ``(T, d, d)``.
    """
    if path_means.ndim != 2:
        raise ValueError(f"path_means must have shape (T, d), got {path_means.shape}.")
    T, d = path_means.shape
    if path_covs.shape != (T, d, d):
        raise ValueError(
            f"path_covs must have shape (T, d, d) = {(T, d, d)}, got {path_covs.shape}."
        )
    return T, d


def linearize_sde(
    drift_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    diffusion: Float[Array, "d d"] | Float[Array, "T d d"],
    path_means: Float[Array, "T d"],
    path_covs: Float[Array, "T d d"],
    integrator: AbstractIntegrator | None = None,
) -> LinearizedSDE:
    """Optimal linear drift of a nonlinear SDE along a Gaussian path.

        F_c,t = 𝔼_q[∂f/∂x] = Σ_xf,tᵀ Σ_x,t⁻¹
        c_t = μ_f,t − F_c,t μ_x,t

    For each marginal q(x_t) = 𝒩(μ_x,t, Σ_x,t) of the path, the linear
    drift f_lin,t(x) = F_c,t x + c_t minimises
    𝔼_q[(f(x) − f_lin,t(x))ᵀ B_c⁻¹ (f(x) − f_lin,t(x))] (Archambeau et al.,
    2007), with μ_f,t = 𝔼_q[f(x_t)] and the cross-covariance
    Σ_xf,t = Cov_q(x_t, f(x_t)).

    The second form of F_c,t is Stein's lemma for a Gaussian q, so only f
    is evaluated, never its Jacobian. Both moments come from one pass of
    ``integrator`` over the marginal (`gaussx.moment_transform`), which
    makes (F_c,t, c_t) the statistical linear regression of f under
    q(x_t), the same surrogate `gaussx.statistical_linear_regression`
    builds for an observation model. The minimiser does not depend on B_c,
    which is only carried through to the result.

    A linear drift f(x) = M x + c is reproduced exactly (F_c,t = M,
    c_t = c) by any rule that integrates quadratics exactly.

    Algorithm:
        1. For each path point (vmapped), push 𝒩(μ_x,t, Σ_x,t) through f
           for μ_f,t and Σ_xf,t.
        2. Solve F_c,t = Σ_xf,tᵀ Σ_x,t⁻¹ row by row against Σ_x,t.
        3. Set c_t = μ_f,t − F_c,t μ_x,t.
        4. Broadcast B_c to one block per path point.

    Args:
        drift_fn: Drift f, mapping a state of length D_x to the same.
        diffusion: Diffusion covariance B_c, one for the whole path or one
            per path point.
        path_means: Marginal means μ_x,t.
        path_covs: Marginal covariances Σ_x,t; must be positive definite.
        integrator: Gaussian integration rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``, as in
            `gaussx.moment_transform`.

    Returns:
        The `LinearizedSDE` holding F_c,t, c_t and B_c,t.

    Raises:
        ValueError: If the path or diffusion shapes are inconsistent.

    References:
        Archambeau, C., Cornford, D., Opper, M. & Shawe-Taylor, J. (2007).
        Gaussian process approximations of stochastic differential
        equations. *JMLR Workshop and Conference Proceedings* 1, 1-16.

    Examples:
        The double-well drift f(x) = x − x³ around zero:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> lin = gaussx.linearize_sde(
        ...     lambda x: x - x**3,
        ...     jnp.eye(1),
        ...     jnp.zeros((5, 1)),
        ...     jnp.full((5, 1, 1), 0.1),
        ...     gaussx.GaussHermiteIntegrator(order=5),
        ... )
        >>> lin.A.shape
        (5, 1, 1)
        >>> round(float(lin.A[0, 0, 0]), 6)  # 𝔼[1 − 3x²] = 1 − 3 × 0.1
        0.7

        A linear drift comes back unchanged:

        >>> drift = lambda x: -2.0 * x + 1.0
        >>> lin = gaussx.linearize_sde(
        ...     drift, jnp.eye(1), jnp.zeros((2, 1)), jnp.full((2, 1, 1), 0.3)
        ... )
        >>> bool(jnp.allclose(lin.A, -2.0)), bool(jnp.allclose(lin.b, 1.0))
        (True, True)
    """
    T, d = _check_path(path_means, path_covs)
    integrator = _default_integrator(integrator)

    def _linearize_one(
        mean: Float[Array, " d"], covariance: Float[Array, "d d"]
    ) -> tuple[Float[Array, "d d"], Float[Array, " d"]]:
        # μ_f,t = 𝔼_q[f(x_t)] and Σ_xf,t = Cov_q(x_t, f(x_t))
        drift_mean, _, cross_covariance = moment_transform(
            drift_fn, mean, covariance, integrator=integrator
        )
        covariance_op = lx.MatrixLinearOperator(
            covariance, lx.positive_semidefinite_tag
        )
        # Σ_xf,tᵀ
        cross_covariance_transpose = rearrange(
            cross_covariance, "state drift -> drift state"
        )
        # F_c,t = Σ_xf,tᵀ Σ_x,t⁻¹: Σ_x,t is symmetric, so solving
        # Σ_x,t a = row for each row of Σ_xf,tᵀ gives the rows of F_c,t
        # (as in statistical_linear_regression).
        drift_matrix = solve_rows(covariance_op, cross_covariance_transpose)
        # F_c,t μ_x,t
        drift_matrix_mean = drift_matrix @ mean
        # c_t = μ_f,t − F_c,t μ_x,t
        drift_offset = drift_mean - drift_matrix_mean
        return drift_matrix, drift_offset

    drift_matrices, drift_offsets = jax.vmap(_linearize_one)(path_means, path_covs)
    # B_c,t = B_c for every t
    diffusion_covariances = _broadcast_diffusion(jnp.asarray(diffusion), T, d)
    return LinearizedSDE(A=drift_matrices, b=drift_offsets, Q=diffusion_covariances)


def _broadcast_diffusion(
    diffusion: Float[Array, ...], T: int, d: int
) -> Float[Array, "T d d"]:
    """One diffusion covariance B_c,t per path point.

    Args:
        diffusion: Diffusion covariance B_c, shared or one per path point.
        T: Number of path points.
        d: State length D_x.

    Returns:
        The stack of B_c,t.

    Raises:
        ValueError: If ``diffusion`` is neither ``(d, d)`` nor ``(T, d, d)``.
    """
    if diffusion.shape == (d, d):
        # B_c,t = B_c
        return repeat(diffusion, "row col -> step row col", step=T)
    if diffusion.shape == (T, d, d):
        return diffusion
    raise ValueError(
        f"diffusion must have shape (d, d) = {(d, d)} or (T, d, d) = "
        f"{(T, d, d)}, got {diffusion.shape}."
    )


def sde_kl_divergence(
    drift_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    linear_drift: LinearizedSDE,
    path_means: Float[Array, "T d"],
    path_covs: Float[Array, "T d d"],
    dt: float | Float[Array, ""] | Float[Array, " T"],
    integrator: AbstractIntegrator | None = None,
) -> Float[Array, ""]:
    """Drift path-KL between a linear and a nonlinear SDE along a Gaussian path.

        KL[q ‖ p] = ½ ∫ 𝔼_q(x(t))[r(t)ᵀ B_c⁻¹ r(t)] dt,
        r(t) = f(x(t)) − f_lin(x(t))

    Two SDEs with the same diffusion covariance B_c and drifts f_lin and f
    have, by Girsanov's theorem, this path-space divergence over the path's
    time window, where q is the law of the linear SDE (Gaussian marginals
    𝒩(μ_x,t, Σ_x,t)) and p the law of the nonlinear one (Archambeau et al.,
    2007). The KL between the two initial-state distributions is not
    included; add it separately when q(x₀) ≠ p(x₀).

    **Discretisation.** This is the drift-only form with a left Riemann
    sum over the T path points,

        KL ≈ ½ ∑_t Δt_t 𝔼_q(x_t)[r_tᵀ B_c,t⁻¹ r_t],
        r_t = f(x_t) − F_c,t x_t − c_t,

    which needs only the marginals. The transition-form alternative, a KL
    between Euler-Maruyama transition densities, needs the pairwise
    marginals q(x_t, x_t+1) as well and agrees with this one to O(Δt); it
    is not what this function computes.

    Each expectation is the integrator's weighted sum over its points of
    the whitened residual's squared norm, so for a rule with non-negative
    mean weights the result is non-negative by construction.
    `gaussx.TaylorIntegrator` is rejected at every order: order 1 sees the
    residual only at the mean, where the drift of `linearize_sde` makes it
    vanish (a zero KL for every drift), and order 2 adds a Hessian term
    that can make the expectation of a square negative (drift x² at
    μ = 0, σ² = 1 gives 𝔼[(x² − 1)²] ≈ −1). For the optimal drift and a
    linear f the residual vanishes everywhere and the KL is zero.

    Algorithm:
        1. For each path point (vmapped), factor B_c,t = L_B,t L_B,tᵀ.
        2. Integrate e_t = 𝔼_q(x_t)[‖L_B,t⁻¹ r_t‖²] with ``integrator``.
        3. Return ½ ∑_t Δt_t e_t.

    Args:
        drift_fn: Nonlinear drift f, mapping a state of length D_x to the
            same.
        linear_drift: Linear drift and diffusion, e.g. from
            `linearize_sde`. ``linear_drift.Q`` supplies B_c,t, which must
            be positive definite.
        path_means: Marginal means μ_x,t.
        path_covs: Marginal covariances Σ_x,t.
        dt: Step sizes Δt_t, a non-negative scalar or one per path point.
        integrator: Gaussian integration rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``.

    Returns:
        The scalar path-KL.

    Raises:
        ValueError: If the path and ``linear_drift`` shapes disagree, or
            ``integrator`` is a `gaussx.TaylorIntegrator`.
        EquinoxRuntimeError: If any step size is negative (also under
            ``jit``).

    References:
        Archambeau, C., Cornford, D., Opper, M. & Shawe-Taylor, J. (2007).
        Gaussian process approximations of stochastic differential
        equations. *JMLR Workshop and Conference Proceedings* 1, 1-16.

    Examples:
        The optimal drift of a linear SDE has zero path-KL:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> drift = lambda x: -2.0 * x + 1.0
        >>> m, S = jnp.zeros((4, 1)), jnp.full((4, 1, 1), 0.3)
        >>> lin = gaussx.linearize_sde(drift, jnp.eye(1), m, S)
        >>> abs(float(gaussx.sde_kl_divergence(drift, lin, m, S, dt=0.1))) < 1e-10
        True

        A nonlinear drift leaves a positive residual, which a five-point
        Gauss-Hermite rule sees (the three unscented points of a cubic drift
        lie on its linear fit):

        >>> double_well = lambda x: x - x**3
        >>> rule = gaussx.GaussHermiteIntegrator(order=5)
        >>> lin = gaussx.linearize_sde(double_well, jnp.eye(1), m, S, rule)
        >>> kl = gaussx.sde_kl_divergence(double_well, lin, m, S, 0.1, rule)
        >>> float(kl) > 1e-4
        True
    """
    T, d = _check_path(path_means, path_covs)
    if linear_drift.A.shape != (T, d, d) or linear_drift.b.shape != (T, d):
        raise ValueError(
            f"linear_drift must have A of shape {(T, d, d)} and b of shape "
            f"{(T, d)} to match the path, got {linear_drift.A.shape} and "
            f"{linear_drift.b.shape}."
        )
    diffusion_covariances = _broadcast_diffusion(linear_drift.Q, T, d)
    integrator = _default_integrator(integrator)
    if isinstance(integrator, TaylorIntegrator):
        raise ValueError(
            "sde_kl_divergence needs a rule that keeps the expectation of a "
            "squared residual non-negative: TaylorIntegrator returns zero for "
            "every drift at order 1 and can go negative at order 2. Use a "
            "point-based rule with non-negative weights (the default "
            "UnscentedIntegrator(alpha=1.0), cubature or Gauss-Hermite)."
        )

    def _expected_whitened_residual(
        mean, covariance, drift_matrix, drift_offset, diffusion_covariance
    ):
        # L_B,t L_B,tᵀ = B_c,t
        diffusion_chol = jnp.linalg.cholesky(diffusion_covariance)

        def whitened_sq(x: Float[Array, " d"]) -> Float[Array, " 1"]:
            # f(x) is traced before the linear drift: reordering changes the
            # order in which the gradient's cotangents for x are summed.
            drift_value = drift_fn(x)
            # F_c,t x
            drift_matrix_state = drift_matrix @ x
            # f_lin,t(x) = F_c,t x + c_t
            linear_drift_value = drift_matrix_state + drift_offset
            # r_t = f(x) − f_lin,t(x)
            residual = drift_value - linear_drift_value
            # w_t = L_B,t⁻¹ r_t
            whitened = jax.scipy.linalg.solve_triangular(
                diffusion_chol, residual, lower=True
            )
            # w_t ⊙ w_t
            whitened_squares = whitened**2
            # ‖w_t‖² = r_tᵀ B_c,t⁻¹ r_t
            squared_norm = jnp.sum(whitened_squares)
            return jnp.atleast_1d(squared_norm)

        state = GaussianState(
            mean=mean,
            cov=lx.MatrixLinearOperator(covariance, lx.positive_semidefinite_tag),
        )
        # e_t = 𝔼_q(x_t)[‖L_B,t⁻¹ r_t‖²]
        return mean_expectation(whitened_sq, state, integrator)[0]

    expected_residuals = jax.vmap(_expected_whitened_residual)(
        path_means,
        path_covs,
        linear_drift.A,
        linear_drift.b,
        diffusion_covariances,
    )
    # Δt_t, one per path point
    step_sizes = jnp.broadcast_to(jnp.asarray(dt, dtype=expected_residuals.dtype), (T,))
    step_sizes = eqx.error_if(
        step_sizes,
        jnp.any(step_sizes < 0),
        "sde_kl_divergence: dt must be non-negative.",
    )
    # Δt_t e_t
    weighted_residuals = step_sizes * expected_residuals
    # ∑_t Δt_t e_t
    total = jnp.sum(weighted_residuals)
    # KL ≈ ½ ∑_t Δt_t e_t
    return 0.5 * total
