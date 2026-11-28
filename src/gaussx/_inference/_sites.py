"""The Gaussian posterior of diagonal sites under a covariance-form prior.

Laplace, Gauss-Newton, posterior linearisation and expectation propagation
replace each likelihood term by a Gaussian site of precision λ_n and mean
μ_site,n, and differ only in how they choose the sites. The step they share,
from a prior 𝒩(μ_f, K_XX) and the sites to the posterior 𝒩(μ_f|y, Σ_f|y), is
`site_posterior`, through any solver strategy.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._gp._prediction_cache import (
    PredictionCache,
    build_prediction_cache,
    predict_variance,
)
from gaussx._primitives._cholesky import cholesky, warn_dense_fallback
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._dispatch import dispatch_solve


# Above this size an iterative solver's "auto" variance turns stochastic, the
# threshold `gaussx.diag_inv` uses between Cholesky and Hutchinson.
_EXACT_VARIANCE_MAX_SIZE = 2048


def _is_dense(solver: AbstractSolverStrategy | None) -> bool:
    """Whether the solver takes the dense Cholesky path.

    Args:
        solver: The solver strategy, or ``None``.

    Returns:
        ``True`` for ``None`` and `gaussx.DenseSolver`.
    """
    return solver is None or isinstance(solver, DenseSolver)


def _site_system(
    prior_cov: lx.AbstractLinearOperator,
    site_precision: Float[Array, " N"],
) -> lx.AbstractLinearOperator:
    """A = K_XX + diag(d), d = 1 / λ, PSD-tagged.

    Args:
        prior_cov: The prior covariance K_XX.
        site_precision: The site precisions λ.

    Returns:
        The lazy, PSD-tagged system A.
    """
    # d = 1 / λ, the site variances
    site_variance = 1.0 / site_precision
    # A = K_XX + diag(d)
    system = prior_cov + lx.DiagonalLinearOperator(site_variance)
    return lx.TaggedLinearOperator(system, lx.positive_semidefinite_tag)


class SitePosterior(eqx.Module):
    """𝒩(μ_f|y, Σ_f|y) for a covariance-form prior and diagonal sites.

        w = A⁻¹ (μ_site − μ_f),   μ_f|y = μ_f + K_XX w,
        Σ_f|y = (K_XX⁻¹ + Λ_site)⁻¹ = K_XX − K_XX A⁻¹ K_XX,
        A = K_XX + diag(d),   d = 1 / λ,   μ_site = (Λμ)_site / λ

    Built by `site_posterior`. The mean and the weights are computed at
    construction; the variances and the covariance only when asked for.

    Attributes:
        prior_mean: The prior mean μ_f.
        prior_cov: The prior covariance K_XX, jitter included.
        site_information: The site information vector (Λμ)_site = λ ⊙ μ_site.
        site_precision: The site precisions λ, all positive.
        weights: w = A⁻¹ (μ_site − μ_f), so that μ_f|y − μ_f = K_XX w and
            the prior quadratic form is wᵀ K_XX w.
        mean: The posterior mean μ_f|y.
        solver: The solver strategy every solve against A goes through;
            ``None`` and `gaussx.DenseSolver` take one dense Cholesky of A.
        factor: The lower Cholesky factor of A on the dense path, reused by
            `variance`; ``None`` for an iterative solver.
    """

    prior_mean: Float[Array, " N"]
    prior_cov: lx.AbstractLinearOperator
    site_information: Float[Array, " N"]
    site_precision: Float[Array, " N"]
    weights: Float[Array, " N"]
    mean: Float[Array, " N"]
    solver: AbstractSolverStrategy | None = None
    factor: lx.AbstractLinearOperator | None = None

    @property
    def system(self) -> lx.AbstractLinearOperator:
        """A = K_XX + diag(d), d = 1 / λ, PSD-tagged and lazy."""
        return _site_system(self.prior_cov, self.site_precision)

    def variance(
        self,
        *,
        method: Literal["auto", "exact", "stochastic"] = "auto",
        num_probes: int = 100,
        batch_size: int | None = None,
        key: jax.Array | None = None,
    ) -> Float[Array, " N"]:
        """The posterior marginal variances diag(Σ_f|y).

            diag(Σ_f|y) = diag(K_XX) − diag(K_XX A⁻¹ K_XX)

        This form stays accurate in float32 for weak sites. The shorter
        d − d ⊙ diag(A⁻¹) ⊙ d cancels catastrophically when λ is small and
        would save nothing, since the cost is the number of solves.

        Methods:

        - ``"exact"``: with a dense solver (``None`` or `gaussx.DenseSolver`),
          triangular solves with the Cholesky factor of A kept from
          construction, O(N³). With an iterative solver, N solves
          A v_n = K_XX e_n, ``batch_size`` at a time (one at a time when
          ``None``), so memory is O(N · batch_size) and K_XX is never
          densified.
        - ``"stochastic"``: a probe estimate of diag(K_XX A⁻¹ K_XX) with
          ``num_probes`` solves. Not available yet: it waits on the
          estimators of gaussx#827 and raises ``NotImplementedError``.
        - ``"auto"``: exact for a dense solver, and for an iterative one up
          to N = 2048 (the threshold of `gaussx.diag_inv`); stochastic
          above, which needs ``key``.

        Args:
            method: ``"auto"``, ``"exact"`` or ``"stochastic"``, as above.
            num_probes: Number of probe solves of the stochastic method.
            batch_size: Number of solves vectorised together by the exact
                iterative method; ``None`` runs them one at a time.
            key: PRNG key of the stochastic method.

        Returns:
            The marginal variances [Σ_f|y]_nn, one per point.

        Raises:
            ValueError: If ``method`` is not one of the three.
            NotImplementedError: If the stochastic method is chosen.
        """
        if method not in ("auto", "exact", "stochastic"):
            raise ValueError(
                f"method must be 'auto', 'exact' or 'stochastic', got {method!r}."
            )
        size = self.prior_mean.shape[0]
        dense = _is_dense(self.solver)
        if method == "auto":
            exact = dense or size <= _EXACT_VARIANCE_MAX_SIZE
            method = "exact" if exact else "stochastic"
        if method == "stochastic":
            del num_probes, key
            raise NotImplementedError(
                "site_posterior: the stochastic variance waits on the diagonal "
                "estimators of gaussx#827; use method='exact' (N solves, in "
                "batches of batch_size)."
            )
        prior_diagonal = _prior_diagonal(self.prior_cov)
        if self.factor is not None and prior_diagonal is not None:
            return _diagonal_variance(prior_diagonal, self.site_precision)
        if self.factor is not None:
            cache = PredictionCache(
                alpha=self.weights, operator=self.system, factor=self.factor
            )
            return _dense_variance(self.prior_cov, cache)
        return _iterative_variance(self.prior_cov, self.system, self.solver, batch_size)

    def covariance(self) -> lx.AbstractLinearOperator:
        """Σ_f|y = K_XX − K_XX A⁻¹ K_XX, as a lazy PSD operator.

        Each matvec costs two products with K_XX and one solve against A:
        two triangular solves with the cached factor on the dense path,
        ``solver`` otherwise.

        Returns:
            The posterior covariance as a `lineax.FunctionLinearOperator`.
        """
        prior_cov = self.prior_cov
        system = self.system
        solver = self.solver
        factor = self.factor

        def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
            # K_XX v
            prior_product = prior_cov.mv(vector)
            if factor is not None:
                # A⁻¹ K_XX v = L⁻ᵀ L⁻¹ K_XX v
                forward = solve(factor, prior_product)
                solved = solve(factor.transpose(), forward)
            else:
                # A⁻¹ K_XX v
                solved = dispatch_solve(system, prior_product, solver)
            # Σ_f|y v = K_XX v − K_XX A⁻¹ K_XX v
            return prior_product - prior_cov.mv(solved)

        return lx.FunctionLinearOperator(
            matvec, prior_cov.in_structure(), lx.positive_semidefinite_tag
        )


def _untagged(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """The operator under any tags.

    Args:
        operator: An operator, possibly tagged.

    Returns:
        The innermost untagged operator.
    """
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _is_dense_matrix(operator: lx.AbstractLinearOperator) -> bool:
    """Whether the prior already is a dense matrix, so densifying loses nothing.

    Args:
        operator: The prior covariance K_XX.

    Returns:
        ``True`` for a (tagged) `lineax.MatrixLinearOperator`.
    """
    return isinstance(_untagged(operator), lx.MatrixLinearOperator)


def _prior_diagonal(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, " N"] | None:
    """The diagonal k of a diagonal prior K_XX = diag(k), else ``None``.

    Args:
        operator: The prior covariance, looked at through tags.

    Returns:
        k for a (tagged) diagonal or identity operator, else ``None``.
    """
    inner = _untagged(operator)
    if isinstance(inner, lx.DiagonalLinearOperator | lx.IdentityLinearOperator):
        return lx.diagonal(inner)
    return None


def _diagonal_variance(
    prior_diagonal: Float[Array, " N"],
    site_precision: Float[Array, " N"],
) -> Float[Array, " N"]:
    """diag(K − K A⁻¹ K) for a diagonal K = diag(k), elementwise.

        [Σ_f|y]_nn = k_n − k_n² / (k_n + d_n),   d_n = 1 / λ_n

    Args:
        prior_diagonal: The prior variances k.
        site_precision: The site precisions λ.

    Returns:
        The marginal variances.
    """
    # k + d
    system_diagonal = prior_diagonal + 1.0 / site_precision
    # k − k² / (k + d), the same K − K A⁻¹ K form as the dense path
    return prior_diagonal - prior_diagonal**2 / system_diagonal


def _dense_variance(
    prior_cov: lx.AbstractLinearOperator,
    cache: PredictionCache,
) -> Float[Array, " N"]:
    """diag(K_XX − K_XX A⁻¹ K_XX) from the cached Cholesky factor L of A.

        [Σ_f|y]_nn = [K_XX]_nn − ∑_m [L⁻¹ K_XX]_mn²

    The GP predictive variance at the training points, so it reuses
    `gaussx.predict_variance` on the cache `site_posterior` built.

    Args:
        prior_cov: The prior covariance K_XX.
        cache: The prediction cache of A, holding its factor L.

    Returns:
        The marginal variances.
    """
    prior_matrix = prior_cov.as_matrix()
    # [K_XX]_nn
    prior_variance = jnp.diag(prior_matrix)
    # K_XX is symmetric, so K_*X = K_XX and [K_**]_nn = [K_XX]_nn
    return predict_variance(cache, prior_matrix, prior_variance)


def _iterative_variance(
    prior_cov: lx.AbstractLinearOperator,
    system: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    batch_size: int | None,
) -> Float[Array, " N"]:
    """diag(K_XX − K_XX A⁻¹ K_XX) by N solves, ``batch_size`` at a time.

        [Σ_f|y]_nn = [K_XX]_nn − k_nᵀ A⁻¹ k_n,   k_n = K_XX e_n

    Args:
        prior_cov: The prior covariance K_XX, applied by matvecs only.
        system: The system A.
        solver: The solver strategy for each solve.
        batch_size: Solves vectorised together; ``None`` for one at a time.

    Returns:
        The marginal variances.
    """
    structure = prior_cov.in_structure()
    size = structure.shape[0]

    def one_point(index: Array) -> Float[Array, ""]:
        # e_n
        unit = jnp.zeros(size, dtype=structure.dtype).at[index].set(1.0)
        # k_n = K_XX e_n
        column = prior_cov.mv(unit)
        # A⁻¹ k_n
        solved = dispatch_solve(system, column, solver)
        # [K_XX]_nn − k_nᵀ A⁻¹ k_n
        return column[index] - column @ solved

    return jax.lax.map(one_point, jnp.arange(size), batch_size=batch_size)


def site_posterior(
    prior_mean: Float[Array, " N"],
    prior_cov: lx.AbstractLinearOperator,
    site_information: Float[Array, " N"],
    site_precision: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> SitePosterior:
    """The Gaussian posterior of diagonal sites under a covariance-form prior.

        w = A⁻¹ (μ_site − μ_f),   μ_f|y = μ_f + K_XX w,
        A = K_XX + diag(d),   d = 1 / λ,   μ_site = (Λμ)_site / λ

    The step Laplace, Gauss-Newton, posterior linearisation and expectation
    propagation share: they differ only in how the sites (λ, (Λμ)_site) are
    chosen, for example by `gaussx.newton_update`. No step forms K_XX⁻¹,
    which is badly conditioned for a smooth kernel. Since
    μ_f|y − μ_f = K_XX w exactly, the prior quadratic form is wᵀ K_XX w,
    with no solve against K_XX.

    With a dense solver (``None`` or `gaussx.DenseSolver`) A is densified
    and factored once, and the factor is kept for `SitePosterior.variance`.
    A structured prior is densified with a `gaussx.DenseFallbackWarning`
    that names `gaussx.CGSolver`; a dense matrix prior is not warned about,
    and a diagonal prior stays diagonal (A = diag(k + d), O(N)).
    With an iterative solver (`gaussx.CGSolver`, preconditioned CG,
    `gaussx.BBMMSolver`, ...) every solve goes through it and a
    matrix-free K_XX is never densified.

    Contract: every λ_n > 0. A masked site (λ_n = 0) has no finite d_n, so
    the caller drops it before this call.

    Algorithm:
        1. Form μ_site = (Λμ)_site / λ and the system A = K_XX + diag(1 / λ).
        2. Solve A w = μ_site − μ_f with ``solver`` (on the dense path, by
           a Cholesky factor kept for the variances).
        3. Return μ_f|y = μ_f + K_XX w with w and the solver.

    Args:
        prior_mean: The prior mean μ_f at the N points.
        prior_cov: The prior covariance K_XX, an N × N PSD operator, jitter
            included.
        site_information: The site information (Λμ)_site = λ ⊙ μ_site, the
            first output of `gaussx.newton_update`.
        site_precision: The site precisions λ > 0, the second output of
            `gaussx.newton_update`.
        solver: Solver strategy for every solve against A. ``None`` and
            `gaussx.DenseSolver` use one dense Cholesky of A.

    Returns:
        The `SitePosterior`, with its weights w and mean μ_f|y.

    Examples:
        Gaussian sites λ = 1 / σ² at the observations give GP regression:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> kernel = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> prior_cov = lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag)
        >>> y = jnp.array([1.0, -1.0])
        >>> precision = jnp.full(2, 4.0)  # σ² = 0.25
        >>> posterior = gaussx.site_posterior(
        ...     jnp.zeros(2), prior_cov, precision * y, precision
        ... )
        >>> expected = kernel @ jnp.linalg.solve(kernel + 0.25 * jnp.eye(2), y)
        >>> bool(jnp.allclose(posterior.mean, expected, atol=1e-5))
        True
        >>> [round(float(v), 4) for v in posterior.variance()]
        [0.1905, 0.1905]

        The same posterior through conjugate gradients:

        >>> iterative = gaussx.site_posterior(
        ...     jnp.zeros(2), prior_cov, precision * y, precision,
        ...     solver=gaussx.CGSolver(),
        ... )
        >>> bool(jnp.allclose(iterative.mean, posterior.mean, atol=1e-4))
        True
        >>> bool(jnp.allclose(iterative.variance(), posterior.variance(), atol=1e-4))
        True

        One Newton step of a Laplace fit feeds it directly:

        >>> latent = jnp.zeros(2)
        >>> gradient, hessian = jnp.array([0.5, -0.5]), jnp.array([-0.25, -0.25])
        >>> information, precision = gaussx.newton_update(latent, gradient, hessian)
        >>> step = gaussx.site_posterior(
        ...     jnp.zeros(2), prior_cov, information, precision
        ... )
        >>> step.mean.shape, step.covariance().in_structure().shape
        ((2,), (2,))
    """
    # μ_site = (Λμ)_site / λ
    site_mean = site_information / site_precision
    # μ_site − μ_f
    residual = site_mean - prior_mean
    system = _site_system(prior_cov, site_precision)
    factor = None
    prior_diagonal = _prior_diagonal(prior_cov)
    if _is_dense(solver) and prior_diagonal is not None:
        # A = diag(k + d) is diagonal: nothing to densify or factor densely
        # L = diag(√(k + d))
        factor = cholesky(
            lx.DiagonalLinearOperator(prior_diagonal + 1.0 / site_precision)
        )
        # w = A⁻¹ (μ_site − μ_f) = L⁻ᵀ L⁻¹ (μ_site − μ_f)
        weights = solve(factor, solve(factor, residual))
    elif _is_dense(solver):
        if not _is_dense_matrix(prior_cov):
            warn_dense_fallback(
                "site_posterior: solver=None or DenseSolver densifies the "
                f"structured prior covariance {type(_untagged(prior_cov)).__name__} "
                "into an N × N matrix for one dense Cholesky of A = K + diag(1/λ). "
                "Pass solver=gaussx.CGSolver() (or another iterative strategy) to "
                "keep it matrix-free."
            )
        # one Cholesky of A, kept for the variances
        dense_system = lx.MatrixLinearOperator(
            system.as_matrix(), lx.positive_semidefinite_tag
        )
        cache = build_prediction_cache(dense_system, residual)
        weights = cache.alpha
        factor = cache.factor
    else:
        # w = A⁻¹ (μ_site − μ_f)
        weights = dispatch_solve(system, residual, solver)
    # μ_f|y = μ_f + K_XX w
    mean = prior_mean + prior_cov.mv(weights)
    return SitePosterior(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        site_information=site_information,
        site_precision=site_precision,
        weights=weights,
        mean=mean,
        solver=solver,
        factor=factor,
    )
