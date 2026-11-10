"""Gaussian expectations under any integrator: means, gradients, ELLs and the ELBO."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PyTree

from gaussx._einx import einsum
from gaussx._primitives._diag import diag
from gaussx._primitives._solve import solve
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._likelihood import AbstractLikelihood
from gaussx._quadrature._likelihoods import CallableLikelihood
from gaussx._quadrature._types import GaussianState


def mean_expectation(
    fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, " M"]:
    """Expectation of a map under a Gaussian: 𝔼[g(x)], x ~ 𝒩(μ_x, Σ_x).

        𝔼[g(x)] ≈ ∑ᵢ w_m,i g(χᵢ)

    For a point-based rule (one that implements ``points_and_weights``) this
    weighted sum is all that is computed: the output covariance that
    ``integrator.integrate`` would also assemble is never built, which matters
    for wide outputs (D_y = D_x² for a flattened outer product). Other
    integrators (e.g. Taylor) fall back to
    ``integrator.integrate(fn, state).state.mean``.

    Algorithm:
        1. Ask the rule for its points χᵢ and mean weights w_m,i.
        2. Without them, return the mean of ``integrator.integrate``.
        3. Otherwise evaluate g on the points and take the weighted sum.

    Args:
        fn: Map g from D_x inputs to D_y outputs.
        state: Input Gaussian 𝒩(μ_x, Σ_x).
        integrator: Integration method (Taylor, Unscented, MC, etc.).

    Returns:
        The expected output 𝔼[g(x)].

    Examples:
        The second moment of a unit Gaussian, 𝔼[x²] = μ_x² + 1:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.ones(2), covariance)
        >>> rule = gaussx.CubatureIntegrator()
        >>> expectation = mean_expectation(lambda x: x**2, state, rule)
        >>> bool(jnp.allclose(expectation, 2.0 * jnp.ones(2)))
        True

        A rule without points goes through ``integrate``:

        >>> taylor = gaussx.TaylorIntegrator()
        >>> mean_expectation(lambda x: 2.0 * x, state, taylor).shape
        (2,)
    """
    rule = _points_and_weights_or_none(integrator, state)
    if rule is None:
        result = integrator.integrate(fn, state)
        return result.state.mean
    points, mean_weights, _ = rule
    # yᵢ = g(χᵢ)
    outputs = jax.vmap(fn)(points)
    # 𝔼[g(x)] ≈ ∑ᵢ w_m,i yᵢ
    # The axis names stay: einx prints this pattern when D_y = 0 raises.
    return einsum(mean_weights, outputs, "p, p m -> m")


def _points_and_weights_or_none(
    integrator: AbstractIntegrator,
    state: GaussianState,
) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]] | None:
    """The rule's points and weights, or ``None`` if it is not point-based.

    Args:
        integrator: Integration method.
        state: Input Gaussian 𝒩(μ_x, Σ_x).

    Returns:
        Tuple ``(points, w_m, w_c)`` from ``integrator.points_and_weights``,
        or ``None`` if that raises `NotImplementedError`.
    """
    try:
        return integrator.points_and_weights(state)
    except NotImplementedError:
        return None


def gradient_expectation(
    fn: Callable[[Float[Array, " N"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, " N"]:
    """Expected gradient of a scalar map under a Gaussian, via Stein's lemma.

        𝔼[∇g(x)] = Σ_x⁻¹ Cov[x, g(x)],    x ~ 𝒩(μ_x, Σ_x)

    Algorithm:
        1. Lift g to a one-element output and integrate it for Σ_xy.
        2. Solve Σ_x against the single column of Σ_xy.

    Args:
        fn: Scalar-valued map g.
        state: Input Gaussian 𝒩(μ_x, Σ_x).
        integrator: Integration method, which must return Σ_xy.

    Returns:
        The expected gradient 𝔼[∇g(x)].

    Raises:
        AssertionError: If ``integrator`` returns no cross-covariance.

    Examples:
        A quadratic has 𝔼[∇g(x)] = 𝔼[2 x] = 2 μ_x:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> state = gaussx.GaussianState(jnp.array([1.0, -1.0]), covariance)
        >>> gradient = gradient_expectation(
        ...     lambda x: jnp.sum(x**2), state, gaussx.GaussHermiteIntegrator(order=5)
        ... )
        >>> bool(jnp.allclose(gradient, jnp.array([2.0, -2.0]), atol=1e-4))
        True
    """

    # The integrator expects an array output
    def vector_fn(x: Float[Array, " N"]) -> Float[Array, " 1"]:
        return jnp.atleast_1d(fn(x))

    result = integrator.integrate(vector_fn, state)
    # Σ_xy = Cov[x, g(x)]
    cross_covariance = result.cross_cov
    assert cross_covariance is not None, "Integrator must return cross_cov"
    # The one output column of Σ_xy
    cross_covariance_column = cross_covariance[:, 0]
    # 𝔼[∇g(x)] = Σ_x⁻¹ Σ_xy
    return solve(state.cov, cross_covariance_column)


def log_likelihood_expectation(
    likelihood_fn: Callable[[Float[Array, " N"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, ""]:
    """Expected log-likelihood of a callable under a Gaussian, by integration.

        𝔼[log p(y ∣ f(x))],    x ~ 𝒩(μ_x, Σ_x)

    For non-conjugate likelihoods (Bernoulli, Poisson, etc.) where the
    expectation has no closed form.

    Args:
        likelihood_fn: Maps latent values to the scalar log-likelihood.
        state: Input Gaussian 𝒩(μ_x, Σ_x).
        integrator: Integration method.

    Returns:
        The scalar expected log-likelihood.

    Examples:
        A unit Gaussian log-kernel has 𝔼[−x²/2] = −(μ_x² + 1)/2:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> state = gaussx.GaussianState(jnp.zeros(1), covariance)
        >>> expectation = log_likelihood_expectation(
        ...     lambda x: -0.5 * jnp.sum(x**2), state, gaussx.CubatureIntegrator()
        ... )
        >>> bool(jnp.allclose(expectation, -0.5))
        True
    """

    def vector_likelihood_fn(x: Float[Array, " N"]) -> Float[Array, " 1"]:
        return jnp.atleast_1d(likelihood_fn(x))

    expectation = mean_expectation(vector_likelihood_fn, state, integrator)
    return expectation[0]


def expected_log_likelihood(
    likelihood: AbstractLikelihood,
    state: GaussianState,
    integrator: AbstractIntegrator | None = None,
) -> Float[Array, ""]:
    """Unified expected log-likelihood with analytical dispatch.

        𝔼_q[log p(y ∣ f)],    q(f) = 𝒩(μ_f, Σ_f)

    If the likelihood has a closed-form expected log-likelihood (e.g.
    ``GaussianLikelihood``), it is used directly without an integrator.
    Otherwise, an ``integrator`` must be provided for numerical approximation.

    A `CallableLikelihood` factorises over sites, so its ELL depends on the
    marginals only and is integrated site by site,

        𝔼_q[log p(y ∣ f)] = ∑ᵢ 𝔼[ℓ(fᵢ, dᵢ)],    fᵢ ~ 𝒩(μ_f,i, Σ_f,ii)

    with the integrator applied to N one-dimensional Gaussians (vmapped)
    instead of one N-dimensional joint, so a tensor-product rule costs
    N × order evaluations rather than order^N. A `MonteCarloIntegrator`
    reuses its key at every site. Other likelihoods integrate the joint.

    Args:
        likelihood: Likelihood object with a ``log_prob`` method.
        state: Variational Gaussian q(f).
        integrator: Integration method. Required for non-conjugate likelihoods;
            ignored when the likelihood has an analytical fast path.

    Returns:
        The scalar expected log-likelihood.

    Raises:
        ValueError: If no integrator is provided and the likelihood has no
            analytical form.

    Examples:
        A Gaussian likelihood needs no integrator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> gaussian = gaussx.GaussianLikelihood(jnp.zeros(2), 1.0)
        >>> expected_log_likelihood(gaussian, state).shape
        ()

        A non-conjugate likelihood uses the integrator:

        >>> bernoulli = gaussx.BernoulliLikelihood(jnp.array([1.0, 0.0]))
        >>> rule = gaussx.CubatureIntegrator()
        >>> bool(expected_log_likelihood(bernoulli, state, rule) < 0.0)
        True

        And fails without one:

        >>> expected_log_likelihood(bernoulli, state)
        Traceback (most recent call last):
        ...
        ValueError: BernoulliLikelihood has no analytical ELL; provide an integrator
    """
    if likelihood.has_analytical_ell():
        return likelihood.analytical_expected_log_likelihood(state.mean, state.cov)
    if integrator is None:
        msg = (
            f"{type(likelihood).__name__} has no analytical ELL; provide an integrator"
        )
        raise ValueError(msg)
    if isinstance(likelihood, CallableLikelihood):
        return _site_log_likelihood_expectation(
            likelihood.log_density, likelihood.data, state, integrator
        )
    return log_likelihood_expectation(likelihood.log_prob, state, integrator)


def _site_log_likelihood_expectation(
    log_density: Callable[[Float[Array, ""], PyTree], Float[Array, ""]],
    data: PyTree,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, ""]:
    """Sum of per-site expectations of a factorised log density.

        ∑ᵢ 𝔼[ℓ(fᵢ, dᵢ)],    fᵢ ~ 𝒩(μ_f,i, Σ_f,ii)

    Args:
        log_density: Per-site log density ℓ(fᵢ, dᵢ).
        data: Per-site data, every leaf with a leading site axis of length N.
        state: Variational Gaussian q(f); only its marginals are used.
        integrator: Integration method, applied to each one-dimensional
            marginal.

    Returns:
        The scalar expected log-likelihood.
    """
    # Σ_f,ii, through the structural diagonal of Σ_f
    marginal_variances = diag(state.cov)

    def site_expectation(
        site_mean: Float[Array, ""],
        site_variance: Float[Array, ""],
        site_data: PyTree,
    ) -> Float[Array, ""]:
        # 𝒩(μ_f,i, Σ_f,ii) as a one-dimensional state
        site_state = GaussianState(
            mean=jnp.atleast_1d(site_mean),
            cov=lx.DiagonalLinearOperator(jnp.atleast_1d(site_variance)),
        )

        def site_log_density(site_latent: Float[Array, " 1"]) -> Float[Array, ""]:
            return log_density(site_latent[0], site_data)

        # 𝔼[ℓ(fᵢ, dᵢ)]
        return log_likelihood_expectation(site_log_density, site_state, integrator)

    site_expectations = jax.vmap(site_expectation)(state.mean, marginal_variances, data)
    # ∑ᵢ 𝔼[ℓ(fᵢ, dᵢ)]
    return jnp.sum(site_expectations)


def elbo(
    likelihood: AbstractLikelihood,
    state: GaussianState,
    kl: Float[Array, ""],
    integrator: AbstractIntegrator | None = None,
) -> Float[Array, ""]:
    """Evidence lower bound (ELBO) from a likelihood and a precomputed KL.

        ELBO = 𝔼_q[log p(y ∣ f)] − KL[q ‖ p]

    Dispatches to the analytical expected log-likelihood when available (e.g.
    Gaussian likelihood), or uses numerical integration.

    Args:
        likelihood: Likelihood object with a ``log_prob`` method.
        state: Variational Gaussian q(f).
        kl: The scalar divergence KL[q ‖ p].
        integrator: Integration method for non-conjugate likelihoods. Ignored
            when the likelihood has an analytical fast path.

    Returns:
        The scalar ELBO.

    Raises:
        ValueError: If no integrator is provided and the likelihood has no
            analytical form.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> likelihood = gaussx.GaussianLikelihood(jnp.zeros(2), 1.0)
        >>> bound = elbo(likelihood, state, jnp.array(0.5))
        >>> ell = expected_log_likelihood(likelihood, state)
        >>> bool(jnp.allclose(bound, ell - 0.5))
        True
    """
    # 𝔼_q[log p(y ∣ f)]
    expected_log_lik = expected_log_likelihood(likelihood, state, integrator)
    # ELBO = 𝔼_q[log p(y ∣ f)] − KL[q ‖ p]
    return expected_log_lik - kl


def cost_expectation(
    prediction_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    cost_fn: Callable[[Float[Array, " M"], Float[Array, " M"]], Float[Array, ""]],
    state: GaussianState,
    target: Float[Array, " M"],
    integrator: AbstractIntegrator,
) -> Float[Array, ""]:
    """Expected cost of a prediction under an uncertain state.

        𝔼[𝒥(g(x), y)],    x ~ 𝒩(μ_x, Σ_x)

    For model-based RL: expected cost of a policy under state uncertainty.

    Args:
        prediction_fn: Maps the state x to the prediction g(x).
        cost_fn: Scalar cost 𝒥 of a prediction against the target.
        state: Input Gaussian 𝒩(μ_x, Σ_x) over the uncertain state.
        target: Target value y.
        integrator: Integration method.

    Returns:
        The scalar expected cost.

    Examples:
        A squared error about the mean is the trace of Σ_x:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> cost = cost_expectation(
        ...     lambda x: x,
        ...     lambda prediction, target: jnp.sum((prediction - target) ** 2),
        ...     state,
        ...     jnp.zeros(2),
        ...     gaussx.CubatureIntegrator(),
        ... )
        >>> bool(jnp.allclose(cost, 2.0))
        True
    """

    def vector_cost_fn(x: Float[Array, " N"]) -> Float[Array, " 1"]:
        # g(x)
        prediction = prediction_fn(x)
        # 𝒥(g(x), y)
        return jnp.atleast_1d(cost_fn(prediction, target))

    expectation = mean_expectation(vector_cost_fn, state, integrator)
    return expectation[0]
