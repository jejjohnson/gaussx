"""Tests for site_log_evidence and site_evidence_correction (gh-883).

The standard form (exact solvers) and the pseudo-observation form
(stochastic log-determinants) are checked against each other, against the
exact GP marginal likelihood, and against frozen values of the two
implementations they replace. The file is in the float32 lane (Makefile
``NO_X64_TESTS``); its float32 accuracy test takes a NumPy float64
reference, so it runs in both lanes.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._inference._sites import _pseudo_observation_evidence, _standard_evidence


_PSD = lx.positive_semidefinite_tag


def _x64() -> bool:
    return jnp.zeros(()).dtype == jnp.float64


def _matern32(inputs, lengthscale, variance=1.0):
    """Matérn-3/2 Gram matrix on 1-D inputs, with a 1e-6 jitter."""
    gaps = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))
    scaled = jnp.sqrt(3.0) * gaps / lengthscale
    jitter = 1e-6 * jnp.eye(inputs.shape[0], dtype=inputs.dtype)
    return variance * (1.0 + scaled) * jnp.exp(-scaled) + jitter


def _bernoulli(labels):
    """f ↦ ∑_n log p(y_n ∣ f_n) for Bernoulli labels with a logit link."""

    def log_likelihood(latent):
        return jnp.sum(labels * latent - jax.nn.softplus(latent))

    return log_likelihood


def _problem(key, size=16):
    """A prior, Bernoulli labels and random sites that are not a mode."""
    input_key, label_key, site_key, precision_key, mean_key = jr.split(key, 5)
    inputs = jnp.sort(jr.uniform(input_key, (size,)))
    kernel = _matern32(inputs, 0.3, variance=2.0)
    labels = jr.bernoulli(label_key, 0.5, (size,)).astype(kernel.dtype)
    precision = jr.uniform(precision_key, (size,), minval=0.1, maxval=0.25)
    information = precision * jr.normal(site_key, (size,))
    prior_mean = 0.1 * jr.normal(mean_key, (size,))
    return prior_mean, kernel, labels, information, precision


@pytest.mark.x64_only(reason="the issue's agreement bar is rtol 1e-10")
def test_both_forms_agree_for_sites_that_are_not_a_mode(getkey):
    prior_mean, kernel, labels, information, precision = _problem(getkey())
    posterior = gaussx.site_posterior(
        prior_mean,
        lx.MatrixLinearOperator(kernel, _PSD),
        information,
        precision,
        solver=gaussx.DenseSolver(),
    )
    fit = _bernoulli(labels)(posterior.mean)
    standard = _standard_evidence(posterior, fit)
    pseudo = _pseudo_observation_evidence(posterior, fit, None)
    assert jnp.allclose(standard, pseudo, rtol=1e-10, atol=0.0)
    assert jnp.array_equal(gaussx.site_log_evidence(posterior, fit), standard)


@pytest.mark.parametrize("form", ["standard", "pseudo_observation"])
def test_gaussian_sites_give_the_exact_marginal_likelihood(getkey, form):
    prior_mean, kernel, _, _, _ = _problem(getkey())
    size = kernel.shape[0]
    y = jr.normal(getkey(), (size,))
    noise_variance = 0.3
    # λ = σ⁻², μ_site = y: the Laplace approximation of a Gaussian is exact
    precision = jnp.full(size, 1.0 / noise_variance)
    posterior = gaussx.site_posterior(
        prior_mean, lx.MatrixLinearOperator(kernel, _PSD), precision * y, precision
    )
    residual = y - posterior.mean
    fit = jnp.sum(
        -0.5 * (jnp.log(2.0 * jnp.pi * noise_variance) + residual**2 / noise_variance)
    )
    if form == "standard":
        evidence = gaussx.site_log_evidence(posterior, fit)
    else:
        evidence = _pseudo_observation_evidence(posterior, fit, None)
    marginal = gaussx.log_marginal_likelihood(
        prior_mean,
        lx.MatrixLinearOperator(kernel + noise_variance * jnp.eye(size), _PSD),
        y,
    )
    rtol = 1e-10 if _x64() else 1e-4
    assert jnp.allclose(evidence, marginal, rtol=rtol, atol=rtol)


# Frozen from argminax.solve.laplace_mode_covariance at argminax cc32290
# (Bernoulli labels, Matérn-3/2 with variance 2 and lengthscale 0.3 on 12
# equispaced points in [0, 1], 1e-6 jitter, prior mean 0.1): its sites at the
# mode and its log_marginal.
_ARGMINAX_LABELS = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
_ARGMINAX_INFORMATION = [
    -0.4896406948797213,
    -0.49595973720652375,
    0.5000699133916947,
    -0.5003623326813352,
    0.4927063025050048,
    0.4824051564097804,
    0.48378984435551875,
    -0.5098355449476869,
    0.4905424680758806,
    0.4928142450456602,
    -0.5038035235362526,
    0.49616729305136464,
]
_ARGMINAX_PRECISION = [
    0.22561796049998228,
    0.23690195578796922,
    0.24911788906792848,
    0.24736075884731243,
    0.23064833374275862,
    0.21549394305652086,
    0.21729446369668293,
    0.2264358978332887,
    0.22703576878753184,
    0.23083769762962295,
    0.23741517549317515,
    0.23735132978008744,
]
_ARGMINAX_LOG_MARGINAL = -9.115879972734113


@pytest.mark.x64_only(reason="frozen float64 fixture, compared at rtol 1e-10")
def test_reproduces_argminax_laplace_mode_covariance():
    inputs = jnp.linspace(0.0, 1.0, 12)
    kernel = _matern32(inputs, 0.3, variance=2.0)
    posterior = gaussx.site_posterior(
        jnp.full(12, 0.1),
        lx.MatrixLinearOperator(kernel, _PSD),
        jnp.asarray(_ARGMINAX_INFORMATION),
        jnp.asarray(_ARGMINAX_PRECISION),
    )
    fit = _bernoulli(jnp.asarray(_ARGMINAX_LABELS))(posterior.mean)
    evidence = gaussx.site_log_evidence(posterior, fit)
    assert jnp.allclose(evidence, _ARGMINAX_LOG_MARGINAL, rtol=1e-10, atol=0.0)


@pytest.mark.x64_only(reason="frozen float64 fixture, compared at rtol 1e-10")
def test_reproduces_pyrox_ssm_site_corrected_log_marginal_with_a_mask():
    # Frozen from pyrox_ssm._inference_nongauss_markov._site_corrected_log_marginal
    # at pyrox 439b963, with a Kalman marginal of −3.25 and two masked sites
    # whose λ is zero.
    mean = jnp.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.8])
    observed = jnp.array([True, True, False, True, False, True])
    precision = jnp.array([0.24, 0.23, 0.0, 0.25, 0.0, 0.21])
    information = jnp.array([0.31, -0.42, 0.0, 0.27, 0.0, 0.44])
    labels = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    fit = jnp.sum(jnp.where(observed, labels * mean - jax.nn.softplus(mean), 0.0))
    corrected = -3.25 + gaussx.site_evidence_correction(
        fit, mean, information, precision, mask=observed
    )
    assert jnp.allclose(corrected, 1.8979188856443265, rtol=1e-10, atol=0.0)
    # the masked zero precisions leave the gradient finite
    gradient = jax.grad(
        lambda p: gaussx.site_evidence_correction(
            fit, mean, information, p, mask=observed
        )
    )(precision)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("site_precision", [0.25, 1e-3, 1e-6])
def test_float32_exact_route_is_accurate_for_weak_sites(site_precision):
    # The issue's table: N = 200, Matérn-3/2, lengthscale 0.3, unit variance,
    # zero means and fit, sites of equal precision λ; NumPy float64 reference.
    # The standard form measured at most 4.6e-6 there and 7.9e-6 here (the
    # pseudo-observation form 8.4e-5 to 7.5e-4 here).
    size = 200
    inputs = np.linspace(0.0, 1.0, size)
    scaled = np.sqrt(3.0) * np.abs(np.subtract.outer(inputs, inputs)) / 0.3
    kernel = (1.0 + scaled) * np.exp(-scaled) + 1e-6 * np.eye(size)
    whitened = np.eye(size) + site_precision * kernel
    reference = -0.5 * np.linalg.slogdet(whitened)[1]

    posterior = gaussx.site_posterior(
        jnp.zeros(size, dtype=jnp.float32),
        lx.MatrixLinearOperator(jnp.asarray(kernel, dtype=jnp.float32), _PSD),
        jnp.zeros(size, dtype=jnp.float32),
        jnp.full(size, site_precision, dtype=jnp.float32),
    )
    evidence = gaussx.site_log_evidence(posterior, jnp.zeros((), dtype=jnp.float32))

    assert evidence.dtype == jnp.float32
    # 2e-5: the measured 7.9e-6 with headroom, still below the other form
    assert abs(float(evidence) - reference) < 2e-5


def test_a_diagonal_prior_stays_diagonal(getkey):
    size = 10
    variance = jr.uniform(getkey(), (size,), minval=0.5, maxval=2.0)
    precision = jr.uniform(getkey(), (size,), minval=0.1, maxval=2.0)
    information = precision * jr.normal(getkey(), (size,))
    posterior = gaussx.site_posterior(
        jnp.zeros(size),
        lx.TaggedLinearOperator(lx.DiagonalLinearOperator(variance), _PSD),
        information,
        precision,
    )
    fit = jnp.asarray(-1.0)
    dense = gaussx.site_posterior(
        jnp.zeros(size),
        lx.MatrixLinearOperator(jnp.diag(variance), _PSD),
        information,
        precision,
    )
    rtol = 1e-10 if _x64() else 1e-5
    assert jnp.allclose(
        gaussx.site_log_evidence(posterior, fit),
        gaussx.site_log_evidence(dense, fit),
        rtol=rtol,
    )


def test_stochastic_solvers_take_the_pseudo_observation_form(getkey, monkeypatch):
    import gaussx._primitives._inv_quad_logdet as inv_quad_module

    prior_mean, kernel, labels, information, precision = _problem(getkey())
    calls = []
    original = inv_quad_module._shared_work_core

    def counting(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(inv_quad_module, "_shared_work_core", counting)
    tight = 1e-10 if _x64() else 1e-6
    posterior = gaussx.site_posterior(
        prior_mean,
        lx.MatrixLinearOperator(kernel, _PSD),
        information,
        precision,
        solver=gaussx.BBMMSolver(rtol=tight, atol=tight, num_probes=16),
    )
    fit = _bernoulli(labels)(posterior.mean)
    estimate = gaussx.site_log_evidence(posterior, fit, key=jr.key(0))
    # one shared pass for the evidence (the posterior's own solve is plain CG)
    assert len(calls) == 1
    exact = gaussx.site_log_evidence(
        gaussx.site_posterior(
            prior_mean, lx.MatrixLinearOperator(kernel, _PSD), information, precision
        ),
        fit,
    )
    assert jnp.abs(estimate - exact) < 1.0
    # an iterative solve with an exact logdet keeps the standard form
    composed = gaussx.site_posterior(
        prior_mean,
        lx.MatrixLinearOperator(kernel, _PSD),
        information,
        precision,
        solver=gaussx.compose_solver(
            gaussx.CGSolver(rtol=tight, atol=tight), logdet=gaussx.DenseLogdet()
        ),
    )
    rtol = 1e-6 if _x64() else 1e-3
    assert jnp.allclose(gaussx.site_log_evidence(composed, fit), exact, rtol=rtol)


@pytest.mark.slow
@pytest.mark.x64_only(reason="a spread comparison over keys")
def test_bbmm_preconditioner_cuts_the_spread_below_five_percent():
    # Bernoulli-like sites (λ in [0.1, 0.25]) on N = 300 Matérn-3/2 points;
    # the issue's FITC rank-100 P is replaced by the scaled partial Cholesky
    # of gh-878 at rank 100.
    size = 300
    keys = jr.split(jr.key(0), 4)
    inputs = jnp.sort(jr.uniform(keys[0], (size,)))
    kernel = _matern32(inputs, 0.1, variance=2.0)
    precision = jr.uniform(keys[1], (size,), minval=0.1, maxval=0.25)
    information = precision * jr.normal(keys[2], (size,))
    labels = jr.bernoulli(keys[3], 0.5, (size,)).astype(kernel.dtype)
    probe_keys = jr.split(jr.key(1), 20)

    def spread(preconditioner):
        solver = gaussx.BBMMSolver(num_probes=50, preconditioner=preconditioner)
        posterior = gaussx.site_posterior(
            jnp.zeros(size),
            lx.MatrixLinearOperator(kernel, _PSD),
            information,
            precision,
            solver=solver,
        )
        fit = _bernoulli(labels)(posterior.mean)
        estimates = jax.vmap(
            lambda key: gaussx.site_log_evidence(posterior, fit, key=key)
        )(probe_keys)
        return jnp.std(estimates)

    pre = gaussx.PartialCholeskyPreconditioner(rank=100, shift=None)
    # the issue's bar: below 5 % of the unpreconditioned spread
    assert spread(pre) < 0.05 * spread(None)


@pytest.mark.slow
@pytest.mark.x64_only(reason="central differences need float64")
def test_jit_vmap_and_grad_match_finite_differences(getkey):
    prior_mean, _, labels, information, precision = _problem(getkey())
    inputs = jnp.sort(jr.uniform(getkey(), (prior_mean.shape[0],)))
    log_likelihood = _bernoulli(labels)

    def evidence(lengthscale, precision):
        posterior = gaussx.site_posterior(
            prior_mean,
            lx.MatrixLinearOperator(_matern32(inputs, lengthscale, 2.0), _PSD),
            information,
            precision,
        )
        return gaussx.site_log_evidence(posterior, log_likelihood(posterior.mean))

    lengthscale = jnp.asarray(0.3)
    assert jnp.allclose(
        jax.jit(evidence)(lengthscale, precision), evidence(lengthscale, precision)
    )
    batch = einx.multiply("batch, site -> batch site", jnp.array([1.0, 1.5]), precision)
    batched = jax.vmap(evidence, in_axes=(None, 0))(lengthscale, batch)
    looped = jnp.stack([evidence(lengthscale, row) for row in batch])
    assert jnp.allclose(batched, looped, rtol=1e-12)
    gradient = jax.grad(evidence)(lengthscale, precision)
    step = 1e-6
    central = (
        evidence(lengthscale + step, precision)
        - evidence(lengthscale - step, precision)
    ) / (2.0 * step)
    assert jnp.allclose(gradient, central, rtol=1e-5)
