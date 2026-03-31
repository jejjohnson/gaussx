"""Gaussian expectation functions."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState


def mean_expectation(
    fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, " M"]:
    r"""Compute ``E[f(x)]`` where ``x ~ N(mu, Sigma)``.

    Args:
        fn: Function mapping ``(N,) -> (M,)``.
        state: Input Gaussian distribution.
        integrator: Integration method (Taylor, Unscented, MC, etc.).

    Returns:
        Expected function value, shape ``(M,)``.
    """
    result = integrator.integrate(fn, state)
    return result.state.mean


def gradient_expectation(
    fn: Callable[[Float[Array, " N"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, " N"]:
    r"""Compute ``E[nabla f(x)]`` via Stein's lemma.

    Uses the identity::

        E[nabla f(x)] = Sigma^{-1} Cov[x, f(x)]

    Args:
        fn: Scalar-valued function mapping ``(N,) -> ()``.
        state: Input Gaussian distribution.
        integrator: Integration method.

    Returns:
        Expected gradient, shape ``(N,)``.
    """
    from gaussx._primitives._solve import solve

    # Wrap scalar fn to return (1,) for the integrator
    def fn_vec(x: Float[Array, " N"]) -> Float[Array, " 1"]:
        return jnp.atleast_1d(fn(x))

    result = integrator.integrate(fn_vec, state)
    cross_cov = result.cross_cov
    assert cross_cov is not None, "Integrator must return cross_cov"
    # E[nabla f] = Sigma^{-1} @ Cov[x, f]  (N, 1) -> (N,)
    return solve(state.cov, cross_cov[:, 0])


def log_likelihood_expectation(
    likelihood_fn: Callable[[Float[Array, " N"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> Float[Array, ""]:
    r"""Compute ``E[log p(y_{obs} | f(x))]`` where ``x ~ N(mu, Sigma)``.

    For non-conjugate likelihoods (Bernoulli, Poisson, etc.) where
    the expectation has no closed form.

    Args:
        likelihood_fn: Function mapping latent values to scalar
            log-likelihood: ``(N,) -> ()``.
        state: Input Gaussian distribution.
        integrator: Integration method.

    Returns:
        Scalar expected log-likelihood.
    """
    return mean_expectation(
        lambda x: jnp.atleast_1d(likelihood_fn(x)),
        state,
        integrator,
    )[0]


def cost_expectation(
    prediction_fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    cost_fn: Callable[[Float[Array, " M"], Float[Array, " M"]], Float[Array, ""]],
    state: GaussianState,
    target: Float[Array, " M"],
    integrator: AbstractIntegrator,
) -> Float[Array, ""]:
    r"""Compute ``E[Cost(f(x), target)]`` where ``x ~ N(mu, Sigma)``.

    For model-based RL: expected cost of a policy under state
    uncertainty.

    Args:
        prediction_fn: Maps state to prediction, ``(N,) -> (M,)``.
        cost_fn: Cost function, ``(M,), (M,) -> ()``.
        state: Input Gaussian distribution (uncertain state).
        target: Target value, shape ``(M,)``.
        integrator: Integration method.

    Returns:
        Scalar expected cost.
    """

    def combined_fn(x: Float[Array, " N"]) -> Float[Array, " 1"]:
        pred = prediction_fn(x)
        return jnp.atleast_1d(cost_fn(pred, target))

    return mean_expectation(combined_fn, state, integrator)[0]
