"""Gaussian in exponential-family form: log-partition, Fisher information and KL.

A Gaussian 𝒩(μ, Σ) on x ∈ ℝ^{D_x}, with precision Λ = Σ⁻¹, is held by its
natural parameters (Λ μ, −½ Λ), the fields ``eta1`` and ``eta2`` of
`GaussianExpFam`. Its sufficient statistics are (x, x xᵀ) and its
expectation parameters (𝔼[x], 𝔼[x xᵀ]) = (μ, Σ + μ μᵀ). The base measure is
one, so the normaliser (2π)^(−D_x/2) lives in the log-partition
log_partition(Λ μ, −½ Λ). For a KL divergence, q = 𝒩(μ_q, Σ_q) and
p = 𝒩(μ_p, Σ_p) with precisions Λ_q and Λ_p.
"""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum
from gaussx._expfam._natural import mean_cov_to_natural, natural_to_mean_cov
from gaussx._linalg._linalg import trace_product
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve


class GaussianExpFam(eqx.Module):
    """Gaussian 𝒩(μ, Σ) held by its natural parameters (Λ μ, −½ Λ).

        log 𝒩(x; μ, Σ) = xᵀ Λ μ + tr(−½ Λ x xᵀ) − log_partition(Λ μ, −½ Λ)
        log_partition(Λ μ, −½ Λ) = ½ μᵀ Λ μ − ½ log det Λ + ½ D_x log 2π

    with the precision Λ = Σ⁻¹ and the sufficient statistics (x, x xᵀ). The
    base measure is one, so the normaliser (2π)^(−D_x/2) lives in the
    log-partition, which `log_partition` returns.

    Attributes:
        eta1: Precision-weighted mean Λ μ, the first natural parameter.
        eta2: Scaled precision −½ Λ, the second natural parameter, as an
            operator.

    Examples:
        From a mean and a covariance Σ = 2 I:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> mean = jnp.array([1.0, -1.0])
        >>> covariance = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> expfam = gaussx.GaussianExpFam.from_mean_cov(mean, covariance)
        >>> bool(jnp.allclose(expfam.eta1, jnp.array([0.5, -0.5])))
        True
        >>> bool(jnp.allclose(expfam.eta2.as_matrix(), -0.25 * jnp.eye(2)))
        True

        From the precision Λ = Σ⁻¹ the natural parameters agree:

        >>> precision = lx.MatrixLinearOperator(
        ...     0.5 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> same = gaussx.GaussianExpFam.from_mean_prec(mean, precision)
        >>> bool(jnp.allclose(same.eta1, expfam.eta1))
        True
    """

    eta1: Float[Array, " N"]
    eta2: lx.AbstractLinearOperator

    @staticmethod
    def from_mean_cov(
        mu: Float[Array, " N"],
        Sigma: lx.AbstractLinearOperator,
    ) -> GaussianExpFam:
        """Natural-form Gaussian from its mean μ and covariance Σ.

            Λ μ = Σ⁻¹ μ,   −½ Λ = −½ Σ⁻¹

        Args:
            mu: Mean μ.
            Sigma: Covariance Σ.

        Returns:
            The `GaussianExpFam` with natural parameters (Λ μ, −½ Λ).

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
            >>> expfam = gaussx.GaussianExpFam.from_mean_cov(jnp.ones(2), covariance)
            >>> bool(jnp.allclose(expfam.eta1, jnp.array([0.5, 0.25])))
            True
        """
        eta1, eta2 = mean_cov_to_natural(mu, Sigma)
        return GaussianExpFam(eta1=eta1, eta2=eta2)

    @staticmethod
    def from_mean_prec(
        mu: Float[Array, " N"],
        Lambda: lx.AbstractLinearOperator,
    ) -> GaussianExpFam:
        """Natural-form Gaussian from its mean μ and precision Λ, with no solve.

            (Λ μ, −½ Λ)

        Args:
            mu: Mean μ.
            Lambda: Precision Λ = Σ⁻¹.

        Returns:
            The `GaussianExpFam` with natural parameters (Λ μ, −½ Λ).

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> precision = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
            >>> expfam = gaussx.GaussianExpFam.from_mean_prec(jnp.ones(2), precision)
            >>> bool(jnp.allclose(expfam.eta1, jnp.array([2.0, 4.0])))
            True
            >>> scaled_precision = jnp.diag(jnp.array([-1.0, -2.0]))
            >>> bool(jnp.allclose(expfam.eta2.as_matrix(), scaled_precision))
            True
        """
        # Λ μ
        eta1 = Lambda.mv(mu)
        # −½ Λ
        eta2 = -0.5 * Lambda
        return GaussianExpFam(eta1=eta1, eta2=eta2)


def to_mean_cov(
    expfam: GaussianExpFam,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Mean μ and covariance Σ of a Gaussian in natural form.

        Λ = −2 (−½ Λ),   μ = Λ⁻¹ (Λ μ),   Σ = Λ⁻¹

    The inverse of `GaussianExpFam.from_mean_cov`. These are **not** the
    expectation parameters (𝔼[x], 𝔼[x xᵀ]) = (μ, Σ + μ μᵀ): for those, use
    `gaussx.natural_to_expectation`.

    Args:
        expfam: Gaussian in natural form (Λ μ, −½ Λ).

    Returns:
        Tuple ``(mu, Sigma)`` holding the mean μ and the covariance Σ as a
        lazy operator.

    Examples:
        The conversion undoes `GaussianExpFam.from_mean_cov`:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> mean = jnp.array([1.0, -1.0])
        >>> matrix = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> covariance = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> expfam = gaussx.GaussianExpFam.from_mean_cov(mean, covariance)
        >>> mean_back, covariance_back = gaussx.to_mean_cov(expfam)
        >>> bool(jnp.allclose(mean_back, mean, atol=1e-5))
        True
        >>> bool(jnp.allclose(covariance_back.as_matrix(), matrix, atol=1e-5))
        True
    """
    return natural_to_mean_cov(expfam.eta1, expfam.eta2)


def to_expectation(
    expfam: GaussianExpFam,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Deprecated alias of `to_mean_cov`.

    Despite its name it returns the mean and covariance (μ, Σ), not the
    expectation parameters (μ, Σ + μ μᵀ) (see `gaussx.natural_to_expectation`).
    It will be removed in gaussx 0.7.0.

    Args:
        expfam: Gaussian in natural form (Λ μ, −½ Λ).

    Returns:
        Tuple ``(mu, Sigma)`` holding the mean μ and the covariance Σ.

    Examples:
        It warns and returns what `to_mean_cov` returns:

        >>> import warnings
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> expfam = gaussx.GaussianExpFam.from_mean_cov(jnp.ones(2), covariance)
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     mean, _ = gaussx.to_expectation(expfam)
        >>> caught[0].category.__name__
        'GaussxDeprecationWarning'
        >>> bool(jnp.allclose(mean, 1.0))
        True
    """
    warn_deprecated(
        "to_expectation is deprecated and will be removed in gaussx 0.7.0: it "
        "returns (mu, Sigma), not expectation "
        "parameters. Use gaussx.to_mean_cov (same result), or "
        "gaussx.natural_to_expectation for (mu, mu mu^T + Sigma)."
    )
    return to_mean_cov(expfam)


def to_natural(
    mu: Float[Array, " N"],
    Sigma: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Deprecated alias of `gaussx.mean_cov_to_natural`.

    It takes the mean and covariance (μ, Σ), not expectation parameters. It
    will be removed in gaussx 0.7.0.

    Args:
        mu: Mean μ.
        Sigma: Covariance Σ.

    Returns:
        Tuple ``(eta1, eta2)`` holding the natural parameters Λ μ and −½ Λ.

    Examples:
        It warns and returns what `gaussx.mean_cov_to_natural` returns:

        >>> import warnings
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     eta1, _ = gaussx.to_natural(jnp.ones(2), covariance)
        >>> caught[0].category.__name__
        'GaussxDeprecationWarning'
        >>> bool(jnp.allclose(eta1, jnp.array([0.5, 0.25])))
        True
    """
    warn_deprecated(
        "to_natural is deprecated and will be removed in gaussx 0.7.0: it takes "
        "(mu, Sigma), not expectation "
        "parameters. Use gaussx.mean_cov_to_natural (same result)."
    )
    return mean_cov_to_natural(mu, Sigma)


def log_partition(expfam: GaussianExpFam) -> Float[Array, ""]:
    """Log-partition of a Gaussian in natural form, normaliser included.

        log_partition(Λ μ, −½ Λ) = −¼ (Λ μ)ᵀ (−½ Λ)⁻¹ (Λ μ) − ½ log det Λ
                                   + ½ D_x log 2π
                                 = ½ μᵀ Λ μ − ½ log det Λ + ½ D_x log 2π

    The base measure is one (see `GaussianExpFam`), so
    xᵀ Λ μ + tr(−½ Λ x xᵀ) − log_partition(Λ μ, −½ Λ) is the normalised
    log-density log 𝒩(x; μ, Σ).

    Args:
        expfam: Gaussian in natural form (Λ μ, −½ Λ).

    Returns:
        The scalar log-partition.

    Examples:
        A standard normal in two dimensions has log-partition log 2π:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> identity = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> standard = gaussx.GaussianExpFam.from_mean_cov(jnp.zeros(2), identity)
        >>> round(float(gaussx.log_partition(standard)), 4)
        1.8379

        A mean μ adds ½ μᵀ Λ μ:

        >>> mean = jnp.array([2.0, 0.0])
        >>> shifted = gaussx.GaussianExpFam.from_mean_cov(mean, identity)
        >>> difference = gaussx.log_partition(shifted) - gaussx.log_partition(standard)
        >>> round(float(difference), 4)
        2.0
    """
    # Λ = −2 (−½ Λ)
    precision = -2.0 * expfam.eta2
    event_size = precision.in_size()

    # (−½ Λ)⁻¹ Λ μ = −2 μ
    minus_two_mean = solve(expfam.eta2, expfam.eta1)
    # −¼ (Λ μ)ᵀ (−½ Λ)⁻¹ (Λ μ) = ½ μᵀ Λ μ
    quadratic_term = -0.25 * (expfam.eta1 @ minus_two_mean)

    # −½ log det Λ
    logdet_term = -0.5 * logdet(precision)

    # log_partition = ½ μᵀ Λ μ − ½ log det Λ + ½ D_x log 2π
    return quadratic_term + logdet_term + 0.5 * event_size * _LOG_2PI


def fisher_info(
    expfam: GaussianExpFam,
) -> lx.AbstractLinearOperator:
    """Fisher information of the mean μ, which is the precision Λ = Σ⁻¹.

        𝔼[∇_μ log 𝒩(x; μ, Σ) ∇_μ log 𝒩(x; μ, Σ)ᵀ] = Λ = −2 (−½ Λ)

    This is the μ block that natural-gradient descent on a
    (μ, Σ)-parameterised objective uses. It is **not** the Fisher
    information in natural coordinates, the Hessian of the log-partition
    with respect to (Λ μ, −½ Λ): that is the (D_x + D_x²)-dimensional
    covariance of the sufficient statistics (x, x xᵀ), whose Λ μ block is
    Σ, not Λ.

    Args:
        expfam: Gaussian in natural form (Λ μ, −½ Λ).

    Returns:
        The precision Λ as an operator.

    Examples:
        A covariance of 2 I has Fisher information ½ I:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> expfam = gaussx.GaussianExpFam.from_mean_cov(jnp.ones(2), covariance)
        >>> information = gaussx.fisher_info(expfam)
        >>> bool(jnp.allclose(information.as_matrix(), 0.5 * jnp.eye(2)))
        True
    """
    # Λ = −2 (−½ Λ)
    return -2.0 * expfam.eta2


def sufficient_stats(
    x: Float[Array, "*batch N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
    """Sufficient statistics (x, x xᵀ) of a Gaussian, over any batch axes.

        (x, x xᵀ)

    Args:
        x: Points x, with any number of leading batch axes.

    Returns:
        Tuple ``(x, outer_product)`` holding x itself and x xᵀ, one D_x × D_x
        matrix per point.

    Examples:
        Four points in three dimensions:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> points = jnp.ones((4, 3))
        >>> first, second = gaussx.sufficient_stats(points)
        >>> first.shape, second.shape
        ((4, 3), (4, 3, 3))

        A single point gives its outer product:

        >>> _, outer = gaussx.sufficient_stats(jnp.array([1.0, 2.0]))
        >>> bool(jnp.allclose(outer, jnp.array([[1.0, 2.0], [2.0, 4.0]])))
        True
    """
    # x xᵀ
    outer_product = einsum(x, x, "... row, ... col -> ... row col")
    return x, outer_product


def kl_divergence(
    q: GaussianExpFam,
    p: GaussianExpFam,
) -> Float[Array, ""]:
    """KL(first ‖ second): KL[q ‖ p] in the Bregman form on natural parameters.

        KL[q ‖ p] = log_partition(p) − log_partition(q)
                    − (Λ_p μ_p − Λ_q μ_q)ᵀ 𝔼_q[x]
                    − tr((−½ Λ_p + ½ Λ_q) 𝔼_q[x xᵀ])

    with log_partition(p) and log_partition(q) taken at the natural
    parameters of p and q, and q's expectation parameters
    (𝔼_q[x], 𝔼_q[x xᵀ]) = (μ_q, Σ_q + μ_q μ_qᵀ), the gradient of the
    log-partition at q. The first argument is the distribution the
    expectation is taken under, whatever its parameter name:
    ``kl_divergence(a, b)`` is KL[a ‖ b]. Mathematically equal to
    `gaussian_kl` with its arguments in the same order; this form keeps the
    gradient in natural-parameter space, for use inside a natural-gradient
    loop.

    Algorithm:
        1. Evaluate the log-partitions of p and q.
        2. Recover μ_q and Σ_q from q's natural parameters (`to_mean_cov`).
        3. Contract Λ_p μ_p − Λ_q μ_q with 𝔼_q[x] = μ_q.
        4. Split the second-moment term into a quadratic form in μ_q, by
           operator matvecs, and two `gaussx.trace_product` calls against
           Σ_q, so neither −½ Λ nor Σ_q is materialised.
        5. Combine the terms.

    Args:
        q: First Gaussian q, the one the expectation is taken under.
        p: Second Gaussian p.

    Returns:
        The scalar KL divergence KL[q ‖ p].

    Examples:
        It agrees with `gaussian_kl` in mean and covariance form:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> q_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> p_cov = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> q_mean, p_mean = jnp.zeros(2), jnp.array([1.0, 0.0])
        >>> q = gaussx.GaussianExpFam.from_mean_cov(q_mean, q_cov)
        >>> p = gaussx.GaussianExpFam.from_mean_cov(p_mean, p_cov)
        >>> kl = gaussx.kl_divergence(q, p)
        >>> round(float(kl), 4)
        0.4431
        >>> reference = gaussx.gaussian_kl(q_mean, q_cov, p_mean, p_cov)
        >>> bool(jnp.allclose(kl, reference, atol=1e-5))
        True

        The divergence of a Gaussian from itself is zero:

        >>> bool(jnp.allclose(gaussx.kl_divergence(q, q), 0.0, atol=1e-5))
        True
    """
    # --- log-partitions ---
    # log_partition(p) = ½ μ_pᵀ Λ_p μ_p − ½ log det Λ_p + ½ D_x log 2π
    p_log_partition = log_partition(p)
    # log_partition(q) = ½ μ_qᵀ Λ_q μ_q − ½ log det Λ_q + ½ D_x log 2π
    q_log_partition = log_partition(q)

    # --- expectation parameters of q ---
    # (μ_q, Σ_q), so that 𝔼_q[x] = μ_q and 𝔼_q[x xᵀ] = Σ_q + μ_q μ_qᵀ
    q_mean, q_covariance = to_mean_cov(q)

    # --- first natural parameter ---
    # Λ_p μ_p − Λ_q μ_q
    precision_mean_difference = p.eta1 - q.eta1
    # (Λ_p μ_p − Λ_q μ_q)ᵀ μ_q
    linear_term = precision_mean_difference @ q_mean

    # --- second natural parameter ---
    # tr((−½ Λ_p + ½ Λ_q)(Σ_q + μ_q μ_qᵀ))
    #   = μ_qᵀ (−½ Λ_p + ½ Λ_q) μ_q + tr(−½ Λ_p Σ_q) − tr(−½ Λ_q Σ_q)
    # −½ Λ_p μ_q
    p_scaled_mean = p.eta2.mv(q_mean)
    # −½ Λ_q μ_q
    q_scaled_mean = q.eta2.mv(q_mean)
    # (−½ Λ_p + ½ Λ_q) μ_q
    scaled_mean_difference = p_scaled_mean - q_scaled_mean
    # μ_qᵀ (−½ Λ_p + ½ Λ_q) μ_q
    quadratic_term = q_mean @ scaled_mean_difference
    # tr(−½ Λ_p Σ_q)
    p_trace = trace_product(p.eta2, q_covariance)
    # tr(−½ Λ_q Σ_q)
    q_trace = trace_product(q.eta2, q_covariance)
    # tr((−½ Λ_p + ½ Λ_q) 𝔼_q[x xᵀ])
    second_moment_term = quadratic_term + p_trace - q_trace

    # KL[q ‖ p] = log_partition(p) − log_partition(q) − linear − second moment
    return p_log_partition - q_log_partition - linear_term - second_moment_term
