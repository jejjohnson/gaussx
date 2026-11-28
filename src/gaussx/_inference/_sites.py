"""The Gaussian posterior of diagonal sites under a covariance-form prior.

Laplace, Gauss-Newton, posterior linearisation and expectation propagation
replace each likelihood term by a Gaussian site of precision λ_n and mean
μ_site,n, and differ only in how they choose the sites. The step they share,
from a prior 𝒩(μ_f, K_XX) and the sites to the posterior 𝒩(μ_f|y, Σ_f|y), is
`site_posterior`, through any solver strategy, and its log evidence is
`site_log_evidence`.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI, gaussian_log_prob
from gaussx._distributions._sample import sample_mvn
from gaussx._gp._prediction_cache import (
    PredictionCache,
    build_prediction_cache,
    predict_variance,
)
from gaussx._gp._predictive_variance import sbpv_mean_and_error, spv_mean_and_error
from gaussx._operators import (
    BlockDiag,
    BlockTriDiag,
    Kronecker,
    KroneckerSum,
    LowRankUpdate,
    Toeplitz,
)
from gaussx._primitives._cholesky import cholesky, warn_dense_fallback
from gaussx._primitives._diag import matrix_free_diag
from gaussx._primitives._inv_quad_logdet import _has_exact_logdet
from gaussx._primitives._logdet import cholesky_logdet
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
    """A = K_XX + diag(d), d = 1 / λ, tagged symmetric and PSD.

    The symmetric tag lets a stochastic Lanczos log-determinant take A
    (gh-883).

    Args:
        prior_cov: The prior covariance K_XX.
        site_precision: The site precisions λ.

    Returns:
        The lazy, symmetric PSD-tagged system A.
    """
    # d = 1 / λ, the site variances
    site_variance = 1.0 / site_precision
    # A = K_XX + diag(d)
    system = prior_cov + lx.DiagonalLinearOperator(site_variance)
    return lx.TaggedLinearOperator(
        system, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


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
        estimator: Literal["sbpv", "spv"] = "sbpv",
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
        - ``"stochastic"``: an unbiased estimate of diag(K_XX A⁻¹ K_XX)
          from ``num_probes`` solves (gh-827), clamped at zero.
          ``estimator="sbpv"`` (the default, the more accurate) draws
          r ~ 𝒩(0, A) from the prior and the sites and averages
          (K_XX A⁻¹ r)², with `gaussx.predictive_variance_sbpv`; a prior
          `gaussx.sample_mvn` cannot sample by structure is densified once
          for its square root, with a `gaussx.DenseFallbackWarning`.
          ``estimator="spv"`` averages z ⊙ K_XX A⁻¹ K_XX z over sign probes
          z, with `gaussx.predictive_variance_spv`, and needs only products
          with K_XX. For the standard errors, and an estimate a caller can
          test against 1 / λ before clamping (EP cavities), use
          `variance_and_error`.
        - ``"auto"``: exact for a dense solver, and for an iterative one up
          to N = 2048 (the threshold of `gaussx.diag_inv`); stochastic
          above, which needs ``key``.

        Args:
            method: ``"auto"``, ``"exact"`` or ``"stochastic"``, as above.
            num_probes: Number of draws or probes of the stochastic method.
            batch_size: Number of solves vectorised together by the exact
                iterative method; ``None`` runs them one at a time.
            estimator: ``"sbpv"`` or ``"spv"``, for the stochastic method.
            key: PRNG key of the stochastic method.

        Returns:
            The marginal variances [Σ_f|y]_nn, one per point.

        Raises:
            ValueError: If ``method`` or ``estimator`` is not one of its
                options, or the stochastic method gets no ``key``.
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
            estimate, _ = self.variance_and_error(
                num_probes=num_probes, estimator=estimator, key=key
            )
            # clamped at zero, as base_conditional is (gh-363)
            return jnp.maximum(estimate, 0.0)
        prior_diagonal = _prior_diagonal(self.prior_cov)
        if self.factor is not None and prior_diagonal is not None:
            return _diagonal_variance(prior_diagonal, self.site_precision)
        if self.factor is not None:
            cache = PredictionCache(
                alpha=self.weights, operator=self.system, factor=self.factor
            )
            return _dense_variance(self.prior_cov, cache)
        return _iterative_variance(self.prior_cov, self.system, self.solver, batch_size)

    def variance_and_error(
        self,
        *,
        num_probes: int = 100,
        estimator: Literal["sbpv", "spv"] = "sbpv",
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """The stochastic marginal variances and their standard errors, unclamped.

            diag(Σ_f|y) ≈ diag(K_XX) − (1 / ℓ) ∑_i (K_XX A⁻¹ r^(i))²,
            r^(i) ~ 𝒩(0, A)   (SBPV), or
            diag(Σ_f|y) ≈ diag(K_XX) − (1 / ℓ) ∑_i z^(i) ⊙ K_XX A⁻¹ K_XX z^(i)
            (SPV)

        EP and posterior linearisation use these variances in cavities,
        1 / σ² − λ, where an estimate above 1 / λ gives a negative cavity
        precision; the estimate is returned as it is, with its standard
        error, so that the caller decides how to clamp it.

        Args:
            num_probes: Number of draws (SBPV) or sign probes (SPV) ℓ, one
                solve against A each.
            estimator: ``"sbpv"`` (the default) or ``"spv"``; see `variance`.
            key: PRNG key of the draws or probes.

        Returns:
            Tuple ``(estimate, standard_error)``, one entry per point.

        Raises:
            ValueError: If ``estimator`` is not ``"sbpv"`` or ``"spv"``, or
                ``key`` is ``None``.
        """
        if estimator not in ("sbpv", "spv"):
            raise ValueError(f"estimator must be 'sbpv' or 'spv', got {estimator!r}.")
        if key is None:
            raise ValueError(
                "The stochastic site variance needs a PRNG key for its draws."
            )
        prior_cov = self.prior_cov
        # diag(K_XX), exactly, by structure or N matvecs
        prior_diagonal = matrix_free_diag(prior_cov, estimate=False)
        solve = self._system_solve
        if estimator == "spv":

            def quadratic_fn(vector: Float[Array, " N"]) -> Float[Array, " N"]:
                # K_XX A⁻¹ K_XX v
                return prior_cov.mv(solve(prior_cov.mv(vector)))

            return spv_mean_and_error(
                key,
                deterministic_part=prior_diagonal,
                quadratic_fn=quadratic_fn,
                num_probes=num_probes,
            )
        _warn_if_prior_draws_densify(prior_cov)
        # d^½, the site standard deviations
        site_scale = 1.0 / jnp.sqrt(self.site_precision)
        zeros = jnp.zeros_like(self.prior_mean)

        def sample_fn(draw_key: jax.Array) -> Float[Array, " N"]:
            prior_key, site_key = jr.split(draw_key)
            # f ~ 𝒩(0, K_XX)
            prior_draw = sample_mvn(zeros, prior_cov, key=prior_key)[0]
            # ε ~ 𝒩(0, I)
            noise = jr.normal(site_key, zeros.shape, dtype=zeros.dtype)
            # r = f + d^½ ⊙ ε ~ 𝒩(0, K_XX + diag(d)) = 𝒩(0, A)
            return prior_draw + site_scale * noise

        def solve_fn(draw: Float[Array, " N"]) -> Float[Array, " N"]:
            # K_XX A⁻¹ r
            return prior_cov.mv(solve(draw))

        return sbpv_mean_and_error(
            key,
            deterministic_part=prior_diagonal,
            sample_fn=sample_fn,
            solve_fn=solve_fn,
            num_samples=num_probes,
        )

    def _system_solve(self, vector: Float[Array, " N"]) -> Float[Array, " N"]:
        """A⁻¹ v, by the cached factor on the dense path, ``solver`` otherwise.

        Args:
            vector: v.

        Returns:
            A⁻¹ v.
        """
        if self.factor is not None:
            # A⁻¹ v = L⁻ᵀ L⁻¹ v
            forward = solve(self.factor, vector)
            return solve(self.factor.transpose(), forward)
        return dispatch_solve(self.system, vector, self.solver)

    def covariance(self) -> lx.AbstractLinearOperator:
        """Σ_f|y = K_XX − K_XX A⁻¹ K_XX, as a lazy PSD operator.

        Each matvec costs two products with K_XX and one solve against A:
        two triangular solves with the cached factor on the dense path,
        ``solver`` otherwise.

        Returns:
            The posterior covariance as a `lineax.FunctionLinearOperator`.
        """
        prior_cov = self.prior_cov
        solve_system = self._system_solve

        def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
            # K_XX v
            prior_product = prior_cov.mv(vector)
            # A⁻¹ K_XX v, by the cached factor or ``solver``
            solved = solve_system(prior_product)
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


# The priors `gaussx.sample_mvn` samples by structure, with no dense root.
_STRUCTURED_DRAWS = (
    lx.DiagonalLinearOperator,
    lx.IdentityLinearOperator,
    BlockDiag,
    Kronecker,
    KroneckerSum,
    BlockTriDiag,
    Toeplitz,
    LowRankUpdate,
)


def _warn_if_prior_draws_densify(prior_cov: lx.AbstractLinearOperator) -> None:
    """Warn when SBPV's prior draws need a dense root of a matrix-free K_XX.

    Args:
        prior_cov: The prior covariance K_XX.
    """
    inner = _untagged(prior_cov)
    if _is_dense_matrix(prior_cov) or isinstance(inner, _STRUCTURED_DRAWS):
        return
    warn_dense_fallback(
        f"SitePosterior.variance(estimator='sbpv') draws from the prior "
        f"{type(inner).__name__} through gaussx.sample_mvn, which densifies it "
        "for its square root. estimator='spv' needs only products with K_XX."
    )


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


def site_evidence_correction(
    log_likelihood: Float[Array, ""],
    mean: Float[Array, " N"],
    site_information: Float[Array, " N"],
    site_precision: Float[Array, " N"],
    *,
    mask: Bool[Array, " N"] | None = None,
) -> Float[Array, ""]:
    """The correction from the pseudo-observation marginal to the site evidence.

        c = ∑_n log p(y_n ∣ m_n) − ∑_n o_n log 𝒩(μ_site,n; m_n, d_n),
        d_n = 1 / λ_n,   μ_site,n = (Λμ)_site,n / λ_n

    Added to the marginal log 𝒩(μ_site; μ_f, K_XX + diag(d)) of the
    pseudo-observations, it gives `site_log_evidence`. A caller with its own
    marginal of the pseudo-observations, such as a Kalman filter over a
    state-space prior, adds this correction to it. Both sums run over the
    observed sites only: ``log_likelihood`` is the caller's sum over them,
    and a site with ``mask`` false is left out of the second (its λ_n may be
    zero).

    Args:
        log_likelihood: ∑_n log p(y_n ∣ m_n) over the observed sites, at the
            posterior mean m.
        mean: The posterior mean m.
        site_information: The site information (Λμ)_site.
        site_precision: The site precisions λ, positive where observed.
        mask: The observed sites o_n; ``None`` means all.

    Returns:
        The scalar correction c.

    Examples:
        At m = μ_site the correction removes only the normalisers:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> precision = jnp.array([1.0, 4.0])
        >>> mean = jnp.array([0.5, -1.0])
        >>> correction = gaussx.site_evidence_correction(
        ...     jnp.asarray(-1.0), mean, precision * mean, precision
        ... )
        >>> expected = -1.0 + jnp.log(2.0 * jnp.pi) - 0.5 * jnp.log(4.0)
        >>> bool(jnp.allclose(correction, expected))
        True

        A masked site contributes nothing, whatever its (zero) precision:

        >>> masked = gaussx.site_evidence_correction(
        ...     jnp.asarray(-1.0),
        ...     mean,
        ...     jnp.array([0.5, 0.0]),
        ...     jnp.array([1.0, 0.0]),
        ...     mask=jnp.array([True, False]),
        ... )
        >>> bool(jnp.allclose(masked, -1.0 + 0.5 * jnp.log(2.0 * jnp.pi)))
        True
    """
    if mask is None:
        mask = jnp.ones(site_precision.shape, dtype=bool)
    # λ_n, with 1 on masked sites: a double where keeps their gradient finite
    safe_precision = jnp.where(mask, site_precision, 1.0)
    # μ_site,n = (Λμ)_site,n / λ_n
    site_mean = site_information / safe_precision
    # log 𝒩(μ_site,n; m_n, 1 / λ_n) = −½ (log 2π − log λ_n + λ_n (μ_site,n − m_n)²)
    per_site = -0.5 * (
        _LOG_2PI - jnp.log(safe_precision) + safe_precision * (site_mean - mean) ** 2
    )
    # ∑_n o_n log 𝒩(μ_site,n; m_n, d_n)
    pseudo_fit = jnp.sum(jnp.where(mask, per_site, 0.0))
    return log_likelihood - pseudo_fit


def site_log_evidence(
    posterior: SitePosterior,
    log_likelihood: Float[Array, ""],
    *,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """log Z of a Gaussian-site posterior, from its system A = K_XX + diag(d).

    Standard form, at m = μ_f|y = μ_f + K_XX w:

        log Z = ∑_n log p(y_n ∣ m_n) − ½ wᵀ K_XX w − ½ log det B,
        B = I + Λ_site^½ K_XX Λ_site^½

    Pseudo-observation form, which needs only rᵀ A⁻¹ r and log det A:

        log Z = log 𝒩(μ_site; μ_f, A)
            + ∑_n [log p(y_n ∣ m_n) − log 𝒩(μ_site,n; m_n, d_n)]

    The two are equal for any sites, not only at a Laplace mode. An exact
    solver (``None``, `gaussx.DenseSolver`, or any strategy whose
    log-determinant is exact) takes the standard form, with log det B from a
    Cholesky factor of B, whose eigenvalues are at least one; it stays
    accurate in float32 for weak sites, where the pseudo-observation form
    subtracts two large numbers. A stochastic log-determinant takes the
    pseudo-observation form, whose first term is `gaussx.gaussian_log_prob`
    on A: one shared pass of `gaussx.inv_quad_logdet` with a
    `gaussx.BBMMSolver`, and its rounding error is far below its sampling
    error. The standard form densifies a structured prior with a
    `gaussx.DenseFallbackWarning`, as `site_posterior`'s dense path does; a
    diagonal prior stays diagonal.

    **Meaning per method.** At a Laplace mode this is the Laplace evidence.
    For posterior linearisation and expectation propagation it is the
    Laplace-type approximation their classes report, not the EP evidence,
    which also needs the cavity normalisers.

    Algorithm:
        1. If the posterior's solver has an exact log-determinant: wᵀ K_XX w
           = wᵀ (m − μ_f), and log det B from one Cholesky of B.
        2. Otherwise: log 𝒩(μ_site; μ_f, A) through the posterior's solver
           with ``key``, plus `site_evidence_correction`.

    Args:
        posterior: The `SitePosterior` of `gaussx.site_posterior`.
        log_likelihood: ∑_n log p(y_n ∣ m_n), the log-likelihood at the
            posterior mean m = ``posterior.mean``.
        key: PRNG key for a stochastic log-determinant's probes; ignored on
            the exact route.

    Returns:
        The scalar log Z.

    Examples:
        Gaussian sites λ = 1 / σ² at y give the exact GP marginal likelihood:

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
        >>> residual = y - posterior.mean
        >>> fit = jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * 0.25) + residual**2 / 0.25))
        >>> evidence = gaussx.site_log_evidence(posterior, fit)
        >>> marginal = gaussx.gaussian_log_prob(
        ...     jnp.zeros(2),
        ...     lx.MatrixLinearOperator(
        ...         kernel + 0.25 * jnp.eye(2), lx.positive_semidefinite_tag
        ...     ),
        ...     y,
        ... )
        >>> bool(jnp.allclose(evidence, marginal, atol=1e-5))
        True

        Through BBMM it is a stochastic estimate, from one shared pass:

        >>> import jax.random as jr
        >>> stochastic = gaussx.site_posterior(
        ...     jnp.zeros(2), prior_cov, precision * y, precision,
        ...     solver=gaussx.BBMMSolver(num_probes=16),
        ... )
        >>> estimate = gaussx.site_log_evidence(stochastic, fit, key=jr.key(0))
        >>> bool(jnp.abs(estimate - marginal) < 0.5)
        True
    """
    if posterior.solver is None or _has_exact_logdet(
        posterior.solver, posterior.system
    ):
        return _standard_evidence(posterior, log_likelihood)
    return _pseudo_observation_evidence(posterior, log_likelihood, key)


def _standard_evidence(
    posterior: SitePosterior,
    log_likelihood: Float[Array, ""],
) -> Float[Array, ""]:
    """log Z = ∑_n log p(y_n ∣ m_n) − ½ wᵀ K_XX w − ½ log det B.

    Args:
        posterior: The site posterior.
        log_likelihood: ∑_n log p(y_n ∣ m_n) at its mean.

    Returns:
        The scalar log Z.
    """
    # m − μ_f = K_XX w, so wᵀ K_XX w needs no solve against K_XX
    shift = posterior.mean - posterior.prior_mean
    # wᵀ K_XX w
    quadratic = posterior.weights @ shift
    # log det(I + Λ^½ K_XX Λ^½)
    log_det = _whitened_logdet(posterior.prior_cov, posterior.site_precision)
    # ∑_n log p(y_n ∣ m_n) − ½ wᵀ K_XX w − ½ log det B
    return log_likelihood - 0.5 * quadratic - 0.5 * log_det


def _whitened_logdet(
    prior_cov: lx.AbstractLinearOperator,
    site_precision: Float[Array, " N"],
) -> Float[Array, ""]:
    """log det B, B = I + Λ^½ K_XX Λ^½, from a Cholesky factor of B.

        log det B = 2 ∑_n log L_nn,   L Lᵀ = B

    Every eigenvalue of B is at least one, so this is accurate in float32
    where log det A + ∑ log λ cancels.

    Args:
        prior_cov: The prior covariance K_XX.
        site_precision: The site precisions λ.

    Returns:
        The scalar log det B.
    """
    prior_diagonal = _prior_diagonal(prior_cov)
    if prior_diagonal is not None:
        # B = diag(1 + λ k): log det B = ∑_n log(1 + λ_n k_n)
        return jnp.sum(jnp.log1p(site_precision * prior_diagonal))
    if not _is_dense_matrix(prior_cov):
        warn_dense_fallback(
            "site_log_evidence: the exact route densifies the structured prior "
            f"covariance {type(_untagged(prior_cov)).__name__} into an N × N "
            "matrix for one Cholesky of B = I + Λ^½ K Λ^½. Use an iterative "
            "solver with a stochastic log-determinant (gaussx.BBMMSolver) to "
            "keep it matrix-free."
        )
    # Λ^½
    root = jnp.sqrt(site_precision)
    # Λ^½ K_XX Λ^½
    whitened = einx.multiply(
        "row, row col, col -> row col", root, prior_cov.as_matrix(), root
    )
    # B = I + Λ^½ K_XX Λ^½ = L Lᵀ
    factor = jnp.linalg.cholesky(jnp.eye(root.shape[0], dtype=root.dtype) + whitened)
    return cholesky_logdet(factor)


def _pseudo_observation_evidence(
    posterior: SitePosterior,
    log_likelihood: Float[Array, ""],
    key: jax.Array | None,
) -> Float[Array, ""]:
    """log Z = log 𝒩(μ_site; μ_f, A) + `site_evidence_correction`.

    Args:
        posterior: The site posterior.
        log_likelihood: ∑_n log p(y_n ∣ m_n) at its mean.
        key: PRNG key for the stochastic log-determinant.

    Returns:
        The scalar log Z.
    """
    # μ_site = (Λμ)_site / λ
    site_mean = posterior.site_information / posterior.site_precision
    # log 𝒩(μ_site; μ_f, A), one shared pass for BBMM
    marginal = gaussian_log_prob(
        posterior.prior_mean,
        posterior.system,
        site_mean,
        solver=posterior.solver,
        key=key,
    )
    correction = site_evidence_correction(
        log_likelihood,
        posterior.mean,
        posterior.site_information,
        posterior.site_precision,
    )
    return marginal + correction
