"""The public Newton step of `laplace_mode` (gh-816).

`laplace_model`, a Newton loop on `LaplaceModel.newton` and `laplace_result`
rebuild `laplace_mode` exactly, so a caller can own the loop (argminax#25) and
still return the `LaplaceResult` that `laplace_mode` and `vb_mean_correction`
use.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx as gx


pytestmark = pytest.mark.x64_only(reason="the rebuild is compared to 1e-12")


def _poisson_counts(n: int) -> gx.PoissonLikelihood:
    times = np.arange(n, dtype=np.float64)
    rates = np.exp(np.sin(times / 3.0))
    return gx.PoissonLikelihood(jnp.asarray(np.random.default_rng(0).poisson(rates)))


def _gaussian_prior(n: int) -> gx.GaussianMRF:
    # a tridiagonal (AR(1)-like) precision, so Λ_x|y is dense here
    precision = 2.0 * np.eye(n) - 0.9 * (np.eye(n, k=1) + np.eye(n, k=-1))
    operator = lx.MatrixLinearOperator(
        jnp.asarray(precision), lx.positive_semidefinite_tag
    )
    return gx.GaussianMRF(jnp.zeros(n), operator)


def _intrinsic_prior(n: int) -> gx.IntrinsicGMRF:
    # an RW2 field on the hard constraint Vᵀ x = 0 (V spans constants and lines)
    times = jnp.arange(n, dtype=jnp.float64)
    null_space = jnp.column_stack([jnp.ones(n), times - times.mean()])
    return gx.IntrinsicGMRF(jnp.zeros(n), 10.0, gx.rw2_structure(n), null_space)


def _rebuild(prior, likelihood, *, steps: int = 50) -> gx.LaplaceResult:
    """laplace_mode's loop, written by a caller on the public step."""
    model = gx.laplace_model(prior, likelihood)
    state = jnp.zeros(prior.event_shape[0])
    # x⁽⁰⁾ off the null space, as laplace_mode starts
    state = state - model.null_component(state)
    for _ in range(steps):
        # x⁽ʲ⁺¹⁾ = x̃, an undamped Newton step
        state = model.newton(state)
    return gx.laplace_result(model, prior, state, jnp.asarray(steps), jnp.asarray(True))


@pytest.mark.parametrize(
    "make_prior", [_gaussian_prior, _intrinsic_prior], ids=["gaussian", "intrinsic"]
)
def test_the_public_step_rebuilds_laplace_mode(make_prior) -> None:
    n = 20
    prior = make_prior(n)
    likelihood = _poisson_counts(n)
    fit = gx.laplace_mode(prior, likelihood)
    assert bool(fit.converged)
    rebuilt = _rebuild(prior, likelihood)
    assert jnp.allclose(rebuilt.mode, fit.mode, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(rebuilt.log_marginal, fit.log_marginal, rtol=1e-12)
    assert type(rebuilt.hessian) is type(fit.hessian)
    assert jnp.allclose(
        rebuilt.factor.solve(jnp.ones(n)), fit.factor.solve(jnp.ones(n)), rtol=1e-12
    )


def test_hessian_is_the_posterior_precision_at_the_mode() -> None:
    n = 12
    prior = _gaussian_prior(n)
    likelihood = _poisson_counts(n)
    model = gx.laplace_model(prior, likelihood)
    fit = gx.laplace_mode(prior, likelihood)
    _, hessian = model.hessian(fit.mode)
    assert jnp.allclose(hessian.as_matrix(), fit.hessian.as_matrix(), rtol=1e-12)
    # the ascent direction vanishes at the mode
    assert jnp.max(jnp.abs(model.grad(fit.mode))) <= 1e-10


def test_laplace_model_checks_its_arguments() -> None:
    prior = _gaussian_prior(4)
    likelihood = _poisson_counts(4)
    wrong = lx.MatrixLinearOperator(jnp.ones((4, 3)))
    with pytest.raises(ValueError, match="projector must have 4 columns"):
        gx.laplace_model(prior, likelihood, projector=wrong)
    with pytest.raises(TypeError, match="needs a GaussianMRF or IntrinsicGMRF"):
        gx.laplace_model(object(), likelihood)


def test_the_private_names_still_resolve() -> None:
    from gaussx._inference._laplace import LaplaceModel, _Model, _model

    assert _Model is LaplaceModel
    model = _model(_gaussian_prior(3), _poisson_counts(3), None, None)
    assert isinstance(model, gx.LaplaceModel)
