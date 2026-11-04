"""Inference sugar: expected log-likelihood, trace correction, cavity, Newton sites.

Sites use the positive-precision convention (Λμ_site, Λ_site): `cavity_distribution`
removes a site from the marginal, and `newton_update` builds one from derivatives
of the log-likelihood. Natural parameters are never written η, which is model noise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI, gaussian_log_prob
from gaussx._linalg._linalg import solve_rows, trace_product
from gaussx._primitives._inv import inv
from gaussx._primitives._trace import trace
from gaussx._strategies._base import AbstractSolverStrategy, AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


def log_marginal_likelihood(
    loc: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    y: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """GP log marginal likelihood, by way of `gaussx.gaussian_log_prob`.

        log p(y) = −½ (y − μ_y)ᵀ Σ_y⁻¹ (y − μ_y) − ½ log det Σ_y − (N / 2) log 2π

    Args:
        loc: Prior mean μ_y.
        cov_operator: Covariance Σ_y of the observations.
        y: Observations y.
        solver: Solver strategy for Σ_y; ``None`` uses structural dispatch.

    Returns:
        The log marginal likelihood log p(y).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> value = log_marginal_likelihood(jnp.zeros(2), covariance, jnp.zeros(2))
        >>> bool(jnp.allclose(value, -jnp.log(2.0 * jnp.pi)))
        True
    """
    return gaussian_log_prob(loc, cov_operator, y, solver=solver)


def gaussian_expected_log_lik(
    y: Float[Array, " N"],
    q_mu: Float[Array, " N"],
    q_cov: lx.AbstractLinearOperator,
    noise: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Expected Gaussian log-likelihood under q(x), the data term of an ELBO.

        𝔼_q[log 𝒩(y ∣ x, Σ_ε)] = log 𝒩(y ∣ μ_q, Σ_ε) − ½ tr(Σ_ε⁻¹ Σ_q)

    Algorithm:
        1. Solve Σ_ε⁻¹ r for the residual r = y − μ_q and form rᵀ Σ_ε⁻¹ r.
        2. Take log det Σ_ε and the trace correction tr(Σ_ε⁻¹ Σ_q).
        3. Add the terms and the constant.

    Args:
        y: Observations y.
        q_mu: Variational mean μ_q.
        q_cov: Variational covariance Σ_q.
        noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for the solve and log-determinant of Σ_ε;
            ``None`` uses structural dispatch.

    Returns:
        The expected log-likelihood.

    Examples:
        With y = μ_q and Σ_ε = Σ_q = I only the constant and the trace remain:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> unit = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> value = gaussian_expected_log_lik(jnp.zeros(2), jnp.zeros(2), unit, unit)
        >>> bool(jnp.allclose(value, -jnp.log(2.0 * jnp.pi) - 1.0))
        True
    """
    obs_size = y.shape[-1]
    # r = y − μ_q
    residual = y - q_mu
    # Σ_ε⁻¹ r
    precision_residual = dispatch_solve(noise, residual, solver)
    # rᵀ Σ_ε⁻¹ r
    quadratic = residual @ precision_residual
    # log det Σ_ε
    log_det_noise = dispatch_logdet(noise, solver)

    # Σ_ε⁻¹
    noise_precision = inv(noise)
    # tr(Σ_ε⁻¹ Σ_q)
    trace_term = trace_product(noise_precision, q_cov)

    # −½ (N log 2π + log det Σ_ε + rᵀ Σ_ε⁻¹ r + tr(Σ_ε⁻¹ Σ_q))
    return -0.5 * (obs_size * _LOG_2PI + log_det_noise + quadratic + trace_term)


def trace_correction(
    K_xx: lx.AbstractLinearOperator,
    K_xz: Float[Array, "N M"],
    K_zz: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, ""]:
    """Trace term of the Titsias collapsed ELBO, the Nyström approximation error.

        tr(K_XX − K_Xℐ K_ℐℐ⁻¹ K_ℐX) = tr(K_XX) − ∑ₙₘ [K_Xℐ]ₙₘ [K_Xℐ K_ℐℐ⁻¹]ₙₘ

    Algorithm:
        1. Take tr(K_XX).
        2. Solve K_Xℐ K_ℐℐ⁻¹ row by row.
        3. Contract it with K_Xℐ and subtract.

    Args:
        K_xx: Prior kernel matrix K_XX.
        K_xz: Cross-covariance K_Xℐ.
        K_zz: Inducing kernel matrix K_ℐℐ.
        solver: Solve strategy for K_ℐℐ; ``None`` uses structural dispatch.

    Returns:
        The trace correction.

    Examples:
        One inducing point explains one of two independent inputs:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> gram_train = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> gram_inducing = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> cross = jnp.array([[1.0], [0.0]])
        >>> value = trace_correction(gram_train, cross, gram_inducing)
        >>> bool(jnp.allclose(value, 1.0))
        True
    """
    # tr(K_XX)
    trace_full = trace(K_xx)

    # K_Xℐ K_ℐℐ⁻¹
    solved_rows = solve_rows(K_zz, K_xz, solver=solver)
    # jnp, not einx: einx cannot resolve an empty inducing set (N_ℐ = 0).
    # tr(K_Xℐ K_ℐℐ⁻¹ K_ℐX) = ∑ₙₘ [K_Xℐ]ₙₘ [K_Xℐ K_ℐℐ⁻¹]ₙₘ
    trace_nystrom = jnp.sum(K_xz * solved_rows)

    return trace_full - trace_nystrom


def cavity_distribution(
    post_mean: Float[Array, " N"],
    post_cov: lx.AbstractLinearOperator | Float[Array, " N"],
    site_nat1: Float[Array, " N"],
    site_nat2: lx.AbstractLinearOperator | Float[Array, " N"],
    power: float = 1.0,
    *,
    precision_floor: float | None = None,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator | Float[Array, " N"]]:
    """EP cavity distribution: remove a fraction α of a site from the marginal.

        Λ_cav = Σ_marg⁻¹ − α Λ_site
        Σ_cav = Λ_cav⁻¹
        μ_cav = Σ_cav (Σ_marg⁻¹ μ_marg − α Λμ_site)

    Two forms are dispatched on the argument types. Operators for ``post_cov``
    and ``site_nat2`` take the full-covariance path. ``(N,)`` arrays for both,
    the marginal variances σ²_marg and site precisions of N scalar latents as
    site-based EP over GPs represents them, take an elementwise path costing
    O(N) rather than the O(N²) of a `lineax.DiagonalLinearOperator`:

        1 / σ²_cav = 1 / σ²_marg − α Λ_site
        μ_cav = σ²_cav (μ_marg / σ²_marg − α Λμ_site)

    Both forms use the positive-precision convention of `gaussx.newton_update`.
    `gaussx.mean_cov_to_natural`, `gaussx.natural_to_mean_cov` and the default
    of `gaussx.blr_diag_update` and `gaussx.blr_full_update` use −½ Λ_x instead:
    multiply by −2, or call the BLR updates with ``convention="precision"``.
    `gaussx.damped_natural_update` works in either.

    Algorithm:
        1. Subtract α Λ_site from the marginal precision.
        2. On the diagonal path, floor the result if asked.
        3. Invert it and take the mean from the remaining information vector.

    Args:
        post_mean: Marginal mean μ_marg.
        post_cov: Marginal covariance Σ_marg as an operator, or marginal
            variances σ²_marg for the diagonal path.
        site_nat1: Site information vector Λμ_site.
        site_nat2: Site precision Λ_site as an operator, or per-site precisions
            for the diagonal path.
        power: Power-EP fraction α; 1 is standard EP.
        precision_floor: Lower bound on Λ_cav, on the diagonal path only. A site
            that has absorbed more precision than the marginal holds leaves a
            negative cavity precision, which EP loops guard against by clipping.
            ``None`` returns the raw cavity, whose negative variances
            `ep_tilted_moments` treats as invalid and passes through unchanged
            (a zero site update). The operator path ignores it, since flooring
            a matrix would need an eigendecomposition, the same split as
            `newton_update`.

    Returns:
        Tuple ``(cav_mean, cav_cov)`` holding μ_cav and Σ_cav. Σ_cav is an
        operator on the operator path and an array of variances σ²_cav on the
        diagonal path.

    Raises:
        TypeError: If ``post_cov`` and ``site_nat2`` are not both arrays or both
            operators.

    Examples:
        The elementwise path:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> mean, variance = jnp.array([0.0, 1.0]), jnp.array([0.5, 0.5])
        >>> site_info, site_precision = jnp.array([0.0, 1.0]), jnp.ones(2)
        >>> cav_mean, cav_var = cavity_distribution(
        ...     mean, variance, site_info, site_precision
        ... )
        >>> bool(jnp.allclose(cav_mean, mean)), bool(jnp.allclose(cav_var, 1.0))
        (True, True)

        The operator path agrees with it:

        >>> cav_mean, cav_cov = cavity_distribution(
        ...     mean,
        ...     lx.DiagonalLinearOperator(variance),
        ...     site_info,
        ...     lx.DiagonalLinearOperator(site_precision),
        ... )
        >>> bool(jnp.allclose(cav_mean, mean))
        True
        >>> bool(jnp.allclose(cav_cov.as_matrix(), jnp.eye(2)))
        True

        Mixing an array with an operator is rejected:

        >>> cavity_distribution(
        ...     mean, variance, site_info, lx.DiagonalLinearOperator(site_precision)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: post_cov and site_nat2 must both be arrays or both be operators
    """
    if isinstance(post_cov, jax.Array) and isinstance(site_nat2, jax.Array):
        # 1 / σ²_cav = 1 / σ²_marg − α Λ_site
        cavity_precision = 1.0 / post_cov - power * site_nat2
        if precision_floor is not None:
            cavity_precision = jnp.maximum(cavity_precision, precision_floor)
        # σ²_cav
        cavity_variance = 1.0 / cavity_precision
        # μ_cav = σ²_cav (μ_marg / σ²_marg − α Λμ_site)
        cavity_mean = cavity_variance * (post_mean / post_cov - power * site_nat1)
        return cavity_mean, cavity_variance
    if isinstance(post_cov, jax.Array) or isinstance(site_nat2, jax.Array):
        msg = "post_cov and site_nat2 must both be arrays or both be operators"
        raise TypeError(msg)

    # Λ_marg = Σ_marg⁻¹
    marginal_precision = inv(post_cov)
    # Λ_cav = Λ_marg − α Λ_site
    cavity_matrix = marginal_precision.as_matrix() - power * site_nat2.as_matrix()
    cavity_precision_op = lx.MatrixLinearOperator(cavity_matrix)
    # Σ_cav = Λ_cav⁻¹
    cavity_covariance = inv(cavity_precision_op)

    # Λ_marg μ_marg − α Λμ_site
    cavity_information = marginal_precision.mv(post_mean) - power * site_nat1
    # μ_cav = Σ_cav (Λ_marg μ_marg − α Λμ_site)
    cavity_mean = cavity_covariance.mv(cavity_information)

    return cavity_mean, cavity_covariance


def newton_update(
    mean: Float[Array, " N"],
    jacobian: Float[Array, " N"],
    hessian: Float[Array, "N N"] | Float[Array, " N"],
    *,
    precision_floor: float = 1e-6,
) -> tuple[Float[Array, " N"], Float[Array, "N N"] | Float[Array, " N"]]:
    """Convert a Newton step into the natural parameters of a site.

        Λ_site = −∇²_x
        Λμ_site = g + Λ_site x

    with g and ∇²_x the gradient and Hessian of the log-likelihood at x. Used in
    Laplace and Newton inference to turn function-space derivatives into a site.

    An ``(N,)`` array of per-site second derivatives h, the shape site-based EP
    and Laplace inference over N scalar latents actually has, takes an
    elementwise O(N) path instead of the (N, N) matrix product:

        Λ_site = max(−h, tol)
        Λμ_site = g + Λ_site ⊙ x

    Both forms use the positive-precision convention of
    `gaussx.cavity_distribution`. `gaussx.mean_cov_to_natural`,
    `gaussx.natural_to_mean_cov` and the default of `gaussx.blr_diag_update` and
    `gaussx.blr_full_update` use −½ Λ_x instead: multiply by −2, or call the BLR
    updates with ``convention="precision"``. `gaussx.damped_natural_update`
    works in either.

    Args:
        mean: Current point x.
        jacobian: Gradient g of the log-likelihood in x. The name is historical;
            it is a gradient, not a Jacobian matrix.
        hessian: Second derivative, negative definite where the likelihood is
            log-concave: the full ∇²_x or its diagonal h.
        precision_floor: Lower bound tol on the returned precision, on the
            diagonal path only. It keeps sites of a non-log-concave likelihood
            (positive entries of h) from producing a negative precision. The
            full-matrix path returns −∇²_x unmodified, since flooring it would
            need an eigendecomposition.

    Returns:
        Tuple ``(nat1, nat2)`` holding Λμ_site and Λ_site, which matches the
        shape of ``hessian``.

    Examples:
        The two paths agree on a diagonal Hessian:

        >>> import jax.numpy as jnp
        >>> mean, gradient = jnp.array([1.0, 2.0]), jnp.zeros(2)
        >>> info, precision = newton_update(mean, gradient, -2.0 * jnp.ones(2))
        >>> bool(jnp.allclose(info, 2.0 * mean)), bool(jnp.allclose(precision, 2.0))
        (True, True)
        >>> info, precision = newton_update(mean, gradient, -2.0 * jnp.eye(2))
        >>> bool(jnp.allclose(info, 2.0 * mean)), precision.shape
        (True, (2, 2))

        A positive second derivative is floored on the diagonal path:

        >>> _, precision = newton_update(mean, gradient, jnp.ones(2))
        >>> bool(jnp.all(precision > 0.0))
        True
    """
    if hessian.ndim == 1:
        # Λ_site = max(−h, tol)
        precision = jnp.maximum(-hessian, precision_floor)
        # Λμ_site = g + Λ_site ⊙ x
        return jacobian + precision * mean, precision

    # Λμ_site = g − ∇²_x x
    site_information = jacobian - hessian @ mean
    # Λ_site = −∇²_x
    site_precision = -hessian
    return site_information, site_precision
