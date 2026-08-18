"""Inference sugar: expected log-lik, trace correction, cavity, Newton update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI, gaussian_log_prob
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
    """GP log marginal likelihood.

    Computes:

        log p(y) = -0.5 * (y-mu)^T K^{-1} (y-mu) - 0.5 * log|K| - N/2 * log(2pi)

    Delegates to `gaussx.gaussian_log_prob`.

    Args:
        loc: Prior mean, shape ``(N,)``.
        cov_operator: Covariance operator K, shape ``(N, N)``.
        y: Observations, shape ``(N,)``.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar log marginal likelihood.
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
    r"""Expected log-likelihood ``E_q[log N(y | f, R)]``.

    Computes:

        E_q[log N(y|f,R)] = log N(y | q_mu, R) - 0.5 * tr(R^{-1} q_cov)

    Core to variational inference ELBO computation.

    Args:
        y: Observations, shape ``(N,)``.
        q_mu: Variational mean, shape ``(N,)``.
        q_cov: Variational covariance operator, shape ``(N, N)``.
        noise: Noise covariance operator R, shape ``(N, N)``.
        solver: Optional solver strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar expected log-likelihood.
    """
    N = y.shape[-1]
    residual = y - q_mu
    alpha = dispatch_solve(noise, residual, solver)
    quad = residual @ alpha
    ld = dispatch_logdet(noise, solver)

    # Trace correction: tr(R^{-1} q_cov)
    R_inv = inv(noise)
    from gaussx._linalg._linalg import trace_product

    tr_term = trace_product(R_inv, q_cov)

    return -0.5 * (N * _LOG_2PI + ld + quad + tr_term)


def trace_correction(
    K_xx: lx.AbstractLinearOperator,
    K_xz: Float[Array, "N M"],
    K_zz: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, ""]:
    """Trace term in Titsias collapsed ELBO.

    Computes:

        tr(K_xx) - tr(K_xz^T K_zz^{-1} K_xz)

    This is the "trace correction" that penalizes the Nystrom
    approximation error.

    Args:
        K_xx: Full covariance, shape ``(N, N)``.
        K_xz: Cross-covariance, shape ``(N, M)``.
        K_zz: Inducing covariance, shape ``(M, M)``.
        solver: Optional solve strategy. When ``None``, uses
            structural dispatch.

    Returns:
        Scalar trace correction.
    """
    tr_full = trace(K_xx)

    # tr(K_xz^T K_zz^{-1} K_xz) = sum_ij W_ij * K_xz_ij
    # where W = K_zz^{-1} K_xz^T reshaped, but easier:
    # tr(A^T B) = sum(A * B), so tr(K_xz^T W) where W_col = K_zz^{-1} K_xz_col
    from gaussx._linalg._linalg import solve_rows

    W = solve_rows(K_zz, K_xz, solver=solver)  # (N, M)
    tr_approx = jnp.sum(K_xz * W)

    return tr_full - tr_approx


def cavity_distribution(
    post_mean: Float[Array, " N"],
    post_cov: lx.AbstractLinearOperator | Float[Array, " N"],
    site_nat1: Float[Array, " N"],
    site_nat2: lx.AbstractLinearOperator | Float[Array, " N"],
    power: float = 1.0,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator | Float[Array, " N"]]:
    r"""Compute EP cavity distribution by removing a site.

    Computes:

        cav_prec = post_prec - power * site_nat2
        cav_cov  = inv(cav_prec)
        cav_mean = cav_cov @ (post_prec @ post_mean - power * site_nat1)

    Two forms are dispatched on the argument types. Passing ``post_cov``
    and ``site_nat2`` as operators takes the full-covariance path. Passing
    both as ``(N,)`` arrays — the marginal variances and per-site
    precisions of ``N`` scalar latents, as site-based EP over GPs
    represents them — takes an elementwise fast path costing ``O(N)``
    rather than the ``O(N²)`` of wrapping them in a
    `lineax.DiagonalLinearOperator`:

    $$
    v_{\mathrm{cav}}^{-1} = v^{-1} - \alpha \lambda_2, \qquad
    m_{\mathrm{cav}} = v_{\mathrm{cav}}
        \left( \frac{m}{v} - \alpha \lambda_1 \right).
    $$

    Note:
        Both forms use the ``nat2 = +Λ`` (positive precision) convention,
        matching `gaussx.newton_update` and `gaussx.damped_natural_update`.
        This differs from `gaussx.mean_cov_to_natural` /
        `gaussx.natural_to_mean_cov`, which use the exponential-family
        convention ``η₂ = −Λ/2``.

    Args:
        post_mean: Posterior mean, shape ``(N,)``.
        post_cov: Posterior covariance operator, or ``(N,)`` marginal
            variances for the diagonal path.
        site_nat1: Site natural parameter (precision-weighted mean),
            shape ``(N,)``.
        site_nat2: Site natural parameter (precision) as an operator, or
            ``(N,)`` per-site precisions for the diagonal path.
        power: Power EP fraction (default 1.0 for standard EP).

    Returns:
        Tuple ``(cav_mean, cav_cov)``. ``cav_cov`` is an operator for the
        operator path and an ``(N,)`` array of variances for the diagonal
        path.

    Raises:
        TypeError: If ``post_cov`` and ``site_nat2`` are not both arrays
            or both operators.
    """
    if isinstance(post_cov, jax.Array) and isinstance(site_nat2, jax.Array):
        cav_prec = 1.0 / post_cov - power * site_nat2
        cav_var = 1.0 / cav_prec
        cav_mean = cav_var * (post_mean / post_cov - power * site_nat1)
        return cav_mean, cav_var
    if isinstance(post_cov, jax.Array) or isinstance(site_nat2, jax.Array):
        msg = "post_cov and site_nat2 must both be arrays or both be operators"
        raise TypeError(msg)

    post_prec = inv(post_cov)
    cav_prec_mat = post_prec.as_matrix() - power * site_nat2.as_matrix()
    cav_prec = lx.MatrixLinearOperator(cav_prec_mat)
    cav_cov = inv(cav_prec)

    eta1_cav = post_prec.mv(post_mean) - power * site_nat1
    cav_mean = cav_cov.mv(eta1_cav)

    return cav_mean, cav_cov


def newton_update(
    mean: Float[Array, " N"],
    jacobian: Float[Array, " N"],
    hessian: Float[Array, "N N"] | Float[Array, " N"],
    *,
    precision_floor: float = 1e-6,
) -> tuple[Float[Array, " N"], Float[Array, "N N"] | Float[Array, " N"]]:
    r"""Convert a Newton step to natural pseudo-likelihood parameters.

    Computes:

        nat1 = jacobian - hessian @ mean
        nat2 = -hessian

    Used in Laplace/Newton-based approximate inference to convert
    function-space derivatives into site natural parameters.

    Passing ``hessian`` as an ``(N,)`` array of per-site second
    derivatives — the shape site-based EP / Laplace inference over ``N``
    scalar latents actually has — takes an elementwise ``O(N)`` path
    instead of forming the ``(N, N)`` matrix product:

    $$
    \Lambda = \max(-h, \varepsilon), \qquad
    \lambda_1 = g + \Lambda f, \qquad
    \lambda_2 = \Lambda.
    $$

    Note:
        Both forms use the ``nat2 = +Λ`` (positive precision) convention,
        matching `gaussx.cavity_distribution` and
        `gaussx.damped_natural_update`. This differs from
        `gaussx.mean_cov_to_natural` / `gaussx.natural_to_mean_cov`, which
        use the exponential-family convention ``η₂ = −Λ/2``.

    Args:
        mean: Current mean, shape ``(N,)`` or ``(D,)``.
        jacobian: First derivative of log-likelihood, shape ``(N,)``.
        hessian: Second derivative (negative definite), either the full
            ``(N, N)`` matrix or an ``(N,)`` diagonal.
        precision_floor: Lower bound on the returned precision, applied
            only on the diagonal path. Keeps sites from a non-log-concave
            likelihood (positive ``hessian`` entries) from producing a
            negative precision. The full-matrix path returns
            ``-hessian`` unmodified, since flooring it would require an
            eigendecomposition.

    Returns:
        Tuple ``(nat1, nat2)`` — site natural parameters. ``nat2`` matches
        the shape of ``hessian``.
    """
    if hessian.ndim == 1:
        precision = jnp.maximum(-hessian, precision_floor)
        return jacobian + precision * mean, precision

    nat1 = jacobian - hessian @ mean
    nat2 = -hessian
    return nat1, nat2
