"""Tests for the simulation-based predictive variances SBPV and SPV (gh-827).

GP regression on 30 training and 40 test points, where the exact predictive
variance is diag(K_** − K_*X (K_XX + σ² I)⁻¹ K_X*), so Ω_det = diag(K_**),
C = K_X* and the precision draws come from K_XX + σ² I.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import reduce
from gaussx._gp._predictive_variance import sbpv_mean_and_error, spv_mean_and_error


_TRAIN = jnp.linspace(0.0, 1.0, 30)
# 40 test points, close enough to be correlated: SPV's error grows with the
# off-diagonal of Cᵀ M C and SBPV's does not (at 8 points SPV was slightly
# the more accurate, 0.113 against 0.130 RMS at 100; gh-827 PR)
_TEST = jnp.linspace(0.05, 0.95, 40)
_NOISE = 0.1
_PSD = lx.positive_semidefinite_tag


def _kernel(left, right, lengthscale=0.2):
    gaps = einx.subtract("i, j -> i j", left, right)
    return jnp.exp(-0.5 * gaps**2 / lengthscale**2)


def _gp(lengthscale=0.2):
    """The callables of both estimators and the exact variance."""
    system_matrix = _kernel(_TRAIN, _TRAIN, lengthscale) + _NOISE * jnp.eye(30)
    system = lx.MatrixLinearOperator(system_matrix, _PSD)
    cross = _kernel(_TEST, _TRAIN, lengthscale)
    prior = jnp.diag(_kernel(_TEST, _TEST, lengthscale))

    def sample_fn(key):
        # r ~ 𝒩(0, K_XX + σ² I)
        return gaussx.sample_mvn(jnp.zeros(30), system, key=key)[0]

    def solve_fn(draw):
        # K_*X (K_XX + σ² I)⁻¹ r
        return cross @ gaussx.solve(system, draw)

    def quadratic_fn(vector):
        # K_*X (K_XX + σ² I)⁻¹ K_X* v
        return cross @ gaussx.solve(system, cross.T @ vector)

    exact = prior - jnp.diag(cross @ jnp.linalg.solve(system_matrix, cross.T))
    return prior, sample_fn, solve_fn, quadratic_fn, exact


def _sbpv(key, num, lengthscale=0.2):
    prior, sample_fn, solve_fn, _, _ = _gp(lengthscale)
    return gaussx.predictive_variance_sbpv(
        key,
        deterministic_part=prior,
        sample_fn=sample_fn,
        solve_fn=solve_fn,
        num_samples=num,
    )


def _spv(key, num, lengthscale=0.2):
    prior, _, _, quadratic_fn, _ = _gp(lengthscale)
    return gaussx.predictive_variance_spv(
        key, deterministic_part=prior, quadratic_fn=quadratic_fn, num_probes=num
    )


_ESTIMATORS = {"sbpv": _sbpv, "spv": _spv}


def test_sbpv_smoke_within_its_standard_error(getkey):
    prior, sample_fn, solve_fn, _, exact = _gp()
    estimate, error = sbpv_mean_and_error(
        getkey(),
        deterministic_part=prior,
        sample_fn=sample_fn,
        solve_fn=solve_fn,
        num_samples=50,
    )
    # 7 of the estimator's own standard errors (gh-220), plus round-off
    assert jnp.all(jnp.abs(estimate - exact) <= 7.0 * error + 1e-4)
    assert jnp.all(_sbpv(getkey(), 50) >= 0.0)


def test_spv_smoke_within_its_standard_error(getkey):
    prior, _, _, quadratic_fn, exact = _gp()
    estimate, error = spv_mean_and_error(
        getkey(), deterministic_part=prior, quadratic_fn=quadratic_fn, num_probes=50
    )
    assert jnp.all(jnp.abs(estimate - exact) <= 7.0 * error + 1e-4)
    assert jnp.all(_spv(getkey(), 50) >= 0.0)


@pytest.mark.parametrize("name", list(_ESTIMATORS))
def test_variances_are_clamped_at_zero(getkey, name):
    """Ω_det = 0 makes every raw estimate negative; the result is zero."""
    _, sample_fn, solve_fn, quadratic_fn, _ = _gp()
    zeros = jnp.zeros(_TEST.shape[0])
    if name == "sbpv":
        raw, _ = sbpv_mean_and_error(
            getkey(),
            deterministic_part=zeros,
            sample_fn=sample_fn,
            solve_fn=solve_fn,
            num_samples=8,
        )
        clamped = gaussx.predictive_variance_sbpv(
            getkey(),
            deterministic_part=zeros,
            sample_fn=sample_fn,
            solve_fn=solve_fn,
            num_samples=8,
        )
    else:
        raw, _ = spv_mean_and_error(
            getkey(), deterministic_part=zeros, quadratic_fn=quadratic_fn, num_probes=8
        )
        clamped = gaussx.predictive_variance_spv(
            getkey(), deterministic_part=zeros, quadratic_fn=quadratic_fn, num_probes=8
        )
    assert jnp.any(raw < 0.0)
    assert jnp.array_equal(clamped, jnp.maximum(clamped, 0.0))
    assert jnp.all(clamped >= 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_ESTIMATORS))
def test_jit_vmap_over_keys_and_grad_in_the_lengthscale(name):
    estimate = _ESTIMATORS[name]
    keys = jr.split(jr.key(0), 4)
    batched = jax.jit(jax.vmap(lambda key: estimate(key, 20)))(keys)
    looped = jnp.stack([estimate(key, 20) for key in keys])
    assert jnp.allclose(batched, looped, rtol=1e-5, atol=1e-6)
    gradient = jax.grad(lambda ell: jnp.sum(estimate(jr.key(1), 20, ell)))(
        jnp.asarray(0.2)
    )
    assert jnp.isfinite(gradient)


def _raw(name, key, num):
    """The unclamped estimate, the one that is unbiased."""
    prior, sample_fn, solve_fn, quadratic_fn, _ = _gp()
    if name == "sbpv":
        estimate, _ = sbpv_mean_and_error(
            key,
            deterministic_part=prior,
            sample_fn=sample_fn,
            solve_fn=solve_fn,
            num_samples=num,
        )
        return estimate
    estimate, _ = spv_mean_and_error(
        key, deterministic_part=prior, quadratic_fn=quadratic_fn, num_probes=num
    )
    return estimate


def _standardised_errors(name, num, seeds):
    """(mean over seeds − exact) / its standard error, per test point."""
    estimate = jax.jit(jax.vmap(lambda key: _raw(name, key, num)))
    draws = estimate(jr.split(jr.key(2), seeds))
    exact = _gp()[-1]
    mean = reduce(draws, "seed point -> point", "mean")
    error = reduce(draws, "seed point -> point", "std") / jnp.sqrt(float(seeds))
    return (mean - exact) / error


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_ESTIMATORS))
def test_unbiased_over_fifty_seeds(name):
    """Mean over 50 seeds within 2 standard errors, in RMS over the 40 points.

    Unbiased, each standardised error is about 𝒩(0, 1), so their RMS is
    about one; a bias of a few standard errors would show. The estimate
    is the unclamped one: clamping at zero biases a small variance upward,
    which is the price of a non-negative result.
    """
    standardised = _standardised_errors(name, 100, 50)
    assert jnp.sqrt(jnp.mean(standardised**2)) <= 2.0, standardised


def _rms_error(name, num, seeds=10):
    estimate = jax.jit(jax.vmap(lambda key: _raw(name, key, num)))
    draws = estimate(jr.split(jr.key(3), seeds))
    return jnp.sqrt(jnp.mean((draws - _gp()[-1]) ** 2))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_ESTIMATORS))
def test_consistent_error_falls_with_the_count(name):
    errors = [_rms_error(name, num) for num in (25, 100, 400)]
    assert errors[0] > errors[1] > errors[2], errors


@pytest.mark.slow
def test_sbpv_at_least_as_accurate_as_spv():
    """At 100 draws and 100 probes on this GP (gh-827's ordering)."""
    assert _rms_error("sbpv", 100) <= _rms_error("spv", 100)
