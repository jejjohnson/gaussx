"""Tests for mixture_quantile and related helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from gaussx._primitives import (
    Chandrupatla,
    mixture_quantile,
    mixture_quantile_gaussian_approx,
)


def test_mixture_quantile_matches_ppf():
    q = jnp.array([0.025, 0.1, 0.5, 0.9, 0.975])
    mean = 0.3
    std = 1.7
    lower = mean - 8.0 * std
    upper = mean + 8.0 * std
    expected = jsp.stats.norm.ppf(q, loc=mean, scale=std)
    actual = mixture_quantile(
        lambda x: jsp.stats.norm.cdf(x, loc=mean, scale=std),
        q,
        lower,
        upper,
        solver=Chandrupatla(rtol=1e-6, atol=1e-9),
        max_steps=128,
    )
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_mixture_quantile_mixture_median_zero():
    sigma = 1.0
    lower = -6.0
    upper = 6.0
    q = jnp.array([0.5])

    def cdf(x):
        return 0.5 * (
            jsp.stats.norm.cdf(x, loc=-1.0, scale=sigma)
            + jsp.stats.norm.cdf(x, loc=1.0, scale=sigma)
        )

    median = mixture_quantile(cdf, q, lower, upper)[0]
    assert jnp.allclose(median, 0.0, atol=1e-3)


def test_mixture_quantile_gradients():
    q = jnp.array([0.95])
    lower = -10.0
    upper = 10.0

    def quantile(mean):
        cdf = lambda x: jsp.stats.norm.cdf(x, loc=mean, scale=1.0)
        return mixture_quantile(cdf, q, lower, upper)

    grad_fn = jax.grad(lambda m: quantile(m)[0])
    assert jnp.allclose(grad_fn(0.0), 1.0, atol=1e-3)


def test_gaussian_approximation_matches_closed_form():
    means = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    stds = jnp.array([[1.0, 1.0], [0.5, 1.5]])
    q = jnp.array([0.25, 0.75])
    approx = mixture_quantile_gaussian_approx(means, stds, q)

    mean = jnp.mean(means, axis=-1, keepdims=True)
    variance = jnp.mean(stds**2 + means**2, axis=-1, keepdims=True) - mean**2
    std = jnp.sqrt(jnp.maximum(variance, 0.0))
    expected = mean + std * jsp.special.ndtri(q)
    assert jnp.allclose(approx, expected)
