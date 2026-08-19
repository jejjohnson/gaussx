"""Tests for EP moment matching via the tilted log-normaliser."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from jax.scipy.special import logsumexp

from gaussx import (
    FifthOrderCubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    TaylorIntegrator,
    UnscentedIntegrator,
    moment_match,
)
from gaussx._testing import tree_allclose


# A fixed, well-conditioned linear-Gaussian site. The tilted normaliser and
# both its derivatives are available in closed form, so quadrature error is
# the only thing under test.
_COV = jnp.array([[1.0, 0.3], [0.3, 0.8]])
_GAIN = jnp.array([[1.0, 0.5], [0.0, 1.0]])
_NOISE = jnp.array([[1.5, 0.2], [0.2, 1.0]])
_MEAN = jnp.array([0.2, -0.1])
_OBS = jnp.array([0.5, 0.3])


def _gaussian_state(mean=_MEAN, cov=_COV):
    return GaussianState(
        mean=mean,
        cov=lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag),
    )


def _gaussian_log_lik(f):
    """log N(y | H f, R)."""
    residual = _OBS - _GAIN @ f
    return -0.5 * (
        residual @ jnp.linalg.solve(_NOISE, residual)
        + jnp.linalg.slogdet(_NOISE)[1]
        + _OBS.shape[0] * jnp.log(2 * jnp.pi)
    )


def _analytic_tilted(power):
    r"""Closed form for a Gaussian site raised to ``power``.

    ``N(y | Hf, R)^a = c_a N(y | Hf, R/a)``, so the tilted normaliser is
    ``c_a N(y | Hm, HCH^T + R/a)``.
    """
    dim = _OBS.shape[0]
    innov = _GAIN @ _COV @ _GAIN.T + _NOISE / power
    log_const = (
        (1 - power) * dim / 2 * jnp.log(2 * jnp.pi)
        + (1 - power) / 2 * jnp.linalg.slogdet(_NOISE)[1]
        - dim / 2 * jnp.log(power)
    )
    residual = _OBS - _GAIN @ _MEAN
    log_z = log_const - 0.5 * (
        residual @ jnp.linalg.solve(innov, residual)
        + jnp.linalg.slogdet(innov)[1]
        + dim * jnp.log(2 * jnp.pi)
    )
    d_log_z = _GAIN.T @ jnp.linalg.solve(innov, residual)
    d2_log_z = -_GAIN.T @ jnp.linalg.solve(innov, _GAIN)
    return log_z, d_log_z, d2_log_z


@pytest.mark.parametrize("power", [1.0, 0.5, 0.25])
def test_matches_analytic_gaussian_site(power):
    """Standard and fractional EP both match the closed-form tilted moments."""
    expected_log_z, expected_d, expected_d2 = _analytic_tilted(power)

    result = moment_match(
        _gaussian_log_lik,
        _gaussian_state(),
        GaussHermiteIntegrator(order=24),
        power=power,
    )

    assert tree_allclose(result.log_Z, expected_log_z, atol=1e-8)
    assert tree_allclose(result.d_log_Z, expected_d, atol=1e-8)
    assert tree_allclose(result.d2_log_Z, expected_d2, atol=1e-8)


def test_power_changes_the_result():
    """Guard against ``power`` being silently ignored."""
    state = _gaussian_state()
    integrator = GaussHermiteIntegrator(order=24)

    full = moment_match(_gaussian_log_lik, state, integrator, power=1.0)
    fractional = moment_match(_gaussian_log_lik, state, integrator, power=0.5)

    assert not bool(jnp.allclose(full.log_Z, fractional.log_Z))
    assert not bool(jnp.allclose(full.d2_log_Z, fractional.d2_log_Z))


def test_stein_derivatives_match_autodiff():
    """The Stein-lemma derivatives equal autodiff of the same quadrature sum.

    ``moment_match`` never differentiates through the integrator; this
    checks that the closed-form reweighting it uses instead lands on the
    same value that ``jax.grad`` / ``jax.hessian`` would.
    """
    integrator = GaussHermiteIntegrator(order=25)

    def log_z(mean):
        points, weights, _ = integrator.points_and_weights(_gaussian_state(mean))
        return logsumexp(jax.vmap(_bernoulli_log_lik)(points), b=weights)

    result = moment_match(_bernoulli_log_lik, _gaussian_state(), integrator)

    assert tree_allclose(result.log_Z, log_z(_MEAN), atol=1e-10)
    assert tree_allclose(result.d_log_Z, jax.grad(log_z)(_MEAN), atol=1e-10)
    assert tree_allclose(result.d2_log_Z, jax.hessian(log_z)(_MEAN), atol=1e-10)


def _bernoulli_log_lik(f):
    """GP classification site: y = 1 for both latents."""
    return jnp.sum(jax.nn.log_sigmoid(f))


def test_rules_agree_on_a_smooth_site():
    """Different rules converge to the same tilted moments on a smooth site.

    Tolerances differ by design: the degree-5 cubature resolves the
    curvature far better than the degree-3 unscented rule.
    """
    state = _gaussian_state(mean=jnp.zeros(2), cov=0.05 * jnp.eye(2))

    def smooth(f):
        return -0.25 * jnp.sum(f**2) + 0.3 * f[0]

    reference = moment_match(smooth, state, GaussHermiteIntegrator(order=25))
    cubature = moment_match(smooth, state, FifthOrderCubatureIntegrator())
    unscented = moment_match(smooth, state, UnscentedIntegrator(alpha=1.0))

    assert tree_allclose(cubature.log_Z, reference.log_Z, atol=1e-4)
    assert tree_allclose(cubature.d_log_Z, reference.d_log_Z, atol=1e-3)
    assert tree_allclose(cubature.d2_log_Z, reference.d2_log_Z, atol=5e-2)

    assert tree_allclose(unscented.log_Z, reference.log_Z, atol=1e-2)
    assert tree_allclose(unscented.d_log_Z, reference.d_log_Z, atol=1e-1)


def test_hessian_is_symmetric():
    """The returned Hessian is symmetrised before it leaves the primitive."""
    result = moment_match(
        _bernoulli_log_lik,
        _gaussian_state(),
        FifthOrderCubatureIntegrator(),
    )

    assert tree_allclose(result.d2_log_Z, result.d2_log_Z.T, atol=1e-14)


def test_survives_an_underflowing_likelihood():
    """The log-sum-exp shift keeps a tiny likelihood from underflowing."""
    offset = -800.0

    def tiny(f):
        return _bernoulli_log_lik(f) + offset

    state = _gaussian_state()
    integrator = FifthOrderCubatureIntegrator()

    shifted = moment_match(tiny, state, integrator)
    base = moment_match(_bernoulli_log_lik, state, integrator)

    assert bool(jnp.isfinite(shifted.log_Z))
    assert tree_allclose(shifted.log_Z, base.log_Z + offset, atol=1e-9)
    # Scaling the likelihood by a constant leaves the derivatives unchanged.
    assert tree_allclose(shifted.d_log_Z, base.d_log_Z, atol=1e-10)
    assert tree_allclose(shifted.d2_log_Z, base.d2_log_Z, atol=1e-10)


def test_rejects_non_point_based_integrator():
    """Taylor linearises rather than sampling, so it has no points."""
    with pytest.raises(NotImplementedError, match="not a point-based rule"):
        moment_match(_bernoulli_log_lik, _gaussian_state(), TaylorIntegrator())


def test_jit_vmap_grad():
    """The primitive is traceable and differentiable end to end."""

    def total(mean):
        result = moment_match(
            _bernoulli_log_lik,
            _gaussian_state(mean),
            FifthOrderCubatureIntegrator(),
        )
        return result.log_Z

    assert tree_allclose(jax.jit(total)(_MEAN), total(_MEAN))

    means = jnp.stack([_MEAN, _MEAN * 2, jnp.zeros(2)])
    assert jax.vmap(total)(means).shape == (3,)

    # Differentiating log_Z is a second, independent approximation of the
    # same derivative; the two agree in the converged limit, which
    # ``test_stein_derivatives_match_autodiff`` pins down. Here it is only
    # the tracing that is under test.
    grad = jax.grad(total)(_MEAN)
    assert grad.shape == _MEAN.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("power", [1.0, 0.5, 0.25])
def test_documented_site_conversion_is_exact(power):
    """The docstring's tilted-moment and site formulas recover the true site.

    For a Gaussian likelihood the tilted distribution is itself Gaussian, so
    the site is known in closed form. This pins both documented conversions
    and the warning attached to them: the naive ``(g, -H)`` pair is not the
    site.
    """
    result = moment_match(
        _gaussian_log_lik,
        _gaussian_state(),
        GaussHermiteIntegrator(order=30),
        power=power,
    )
    g, hess = result.d_log_Z, result.d2_log_Z

    # Documented tilted moments.
    tilt_mean = _MEAN + _COV @ g
    tilt_cov = _COV + _COV @ hess @ _COV

    # Exact tilted distribution: the conjugate posterior for the likelihood
    # raised to ``power``, i.e. with noise R / power.
    scaled_noise_inv = power * jnp.linalg.inv(_NOISE)
    exact_cov = jnp.linalg.inv(
        jnp.linalg.inv(_COV) + _GAIN.T @ scaled_noise_inv @ _GAIN
    )
    exact_mean = exact_cov @ (
        jnp.linalg.solve(_COV, _MEAN) + _GAIN.T @ scaled_noise_inv @ _OBS
    )

    assert tree_allclose(tilt_mean, exact_mean, atol=1e-8)
    assert tree_allclose(tilt_cov, exact_cov, atol=1e-8)

    # Documented site conversion, undoing the power scaling to recover the
    # site of p(y | f) itself.
    factor = jnp.linalg.inv(jnp.eye(_MEAN.shape[0]) + hess @ _COV)
    site_prec = -factor @ hess / power
    site_shift = factor @ (g - hess @ _MEAN) / power

    exact_prec = _GAIN.T @ jnp.linalg.solve(_NOISE, _GAIN)
    exact_shift = _GAIN.T @ jnp.linalg.solve(_NOISE, _OBS)

    assert tree_allclose(site_prec, exact_prec, atol=1e-8)
    assert tree_allclose(site_shift, exact_shift, atol=1e-8)

    # The pair the docstring warns against is materially wrong, not a
    # near-miss: dropping -H m and the (I + H C)^-1 factor matters.
    assert not bool(jnp.allclose(-hess / power, exact_prec, atol=1e-3))
    assert not bool(jnp.allclose(g / power, exact_shift, atol=1e-3))


def test_rejects_non_scalar_log_likelihood():
    """``log_lik_fn`` must return a scalar; a vector fails clearly, not deep
    inside the point weighting with an opaque broadcast error."""
    with pytest.raises(ValueError, match="must return a scalar per point"):
        moment_match(
            lambda f: jnp.atleast_1d(_bernoulli_log_lik(f)),
            _gaussian_state(),
            FifthOrderCubatureIntegrator(),
        )
