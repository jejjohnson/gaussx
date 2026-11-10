"""Tests for CallableLikelihood (gh-831) and the per-site ELL path."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import lineax as lx
import pytest

import gaussx


# float64 does not exist in the no-x64 lane (GAUSSX_TEST_X64=0).
DTYPES = [
    jnp.float32,
    pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case")),
]


def _tol(dtype):
    return 1e-5 if dtype == jnp.float32 else 1e-10


def _bernoulli_site(f_i, y_i):
    return jss.bernoulli.logpmf(y_i, jax.nn.sigmoid(f_i))


def _gaussian_site(noise_var):
    def log_density(f_i, y_i):
        return jss.norm.logpdf(y_i, f_i, jnp.sqrt(noise_var))

    return log_density


def _labels(dtype):
    return jnp.array([1.0, 0.0, 1.0, 1.0], dtype)


def _latents(dtype):
    return jnp.array([0.3, -1.2, 2.0, -0.4], dtype)


def _diagonal_state(dtype):
    variances = jnp.array([0.5, 1.0, 2.0, 0.25], dtype)
    return gaussx.GaussianState(_latents(dtype), lx.DiagonalLinearOperator(variances))


def _dense_state(dtype):
    a = jnp.array(
        [
            [1.0, 0.2, 0.0, 0.1],
            [0.3, 0.8, 0.1, 0.0],
            [0.0, 0.4, 1.1, 0.2],
            [0.1, 0.0, 0.3, 0.6],
        ],
        dtype,
    )
    covariance = lx.MatrixLinearOperator(a @ a.T, lx.positive_semidefinite_tag)
    return gaussx.GaussianState(_latents(dtype), covariance)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bernoulli_log_prob_and_site_derivatives(dtype):
    y = _labels(dtype)
    f = _latents(dtype)
    lik = gaussx.CallableLikelihood(_bernoulli_site, y)
    ref = gaussx.BernoulliLikelihood(y)

    log_prob = lik.log_prob(f)
    gradient, hessian_diagonal = lik.site_derivatives(f)
    ref_gradient, ref_hessian_diagonal = ref.site_derivatives(f)

    tol = _tol(dtype)
    assert log_prob.dtype == dtype
    assert gradient.dtype == dtype
    assert hessian_diagonal.dtype == dtype
    assert jnp.allclose(log_prob, ref.log_prob(f), rtol=tol, atol=tol)
    assert jnp.allclose(gradient, ref_gradient, rtol=tol, atol=tol)
    assert jnp.allclose(hessian_diagonal, ref_hessian_diagonal, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bernoulli_ell_matches_joint_rule_on_diagonal_covariance(dtype):
    # With a diagonal covariance the joint tensor-product Gauss-Hermite rule is
    # the product of the per-site rules, so the two ELLs agree to rounding.
    y = _labels(dtype)
    state = _diagonal_state(dtype)
    rule = gaussx.GaussHermiteIntegrator(order=6)
    lik = gaussx.CallableLikelihood(_bernoulli_site, y)

    ell = gaussx.expected_log_likelihood(lik, state, rule)
    ref = gaussx.expected_log_likelihood(gaussx.BernoulliLikelihood(y), state, rule)

    assert ell.dtype == dtype
    assert jnp.allclose(ell, ref, rtol=_tol(dtype), atol=_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_ell_uses_only_the_marginals(dtype):
    # A factorised ELL depends on Σ_f only through its diagonal: the per-site
    # path gives the same value for a dense Σ_f and for diag(Σ_f).
    y = _labels(dtype)
    dense = _dense_state(dtype)
    marginal = gaussx.GaussianState(
        dense.mean, lx.DiagonalLinearOperator(gaussx.diag(dense.cov))
    )
    rule = gaussx.GaussHermiteIntegrator(order=10)
    lik = gaussx.CallableLikelihood(_bernoulli_site, y)

    ell_dense = gaussx.expected_log_likelihood(lik, dense, rule)
    ell_marginal = gaussx.expected_log_likelihood(lik, marginal, rule)

    assert jnp.allclose(ell_dense, ell_marginal, rtol=_tol(dtype), atol=_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "rule",
    [gaussx.GaussHermiteIntegrator(order=5), gaussx.CubatureIntegrator()],
    ids=["gauss_hermite", "cubature"],
)
def test_gaussian_ell_matches_analytical(dtype, rule):
    # ℓ is quadratic in fᵢ, which both rules integrate exactly in one dimension.
    noise_var = jnp.asarray(0.3, dtype)
    y = jnp.array([0.5, -1.0, 1.5, 0.0], dtype)
    state = _dense_state(dtype)
    lik = gaussx.CallableLikelihood(_gaussian_site(noise_var), y)
    ref = gaussx.GaussianLikelihood(y, noise_var)

    ell = gaussx.expected_log_likelihood(lik, state, rule)
    analytical = gaussx.expected_log_likelihood(ref, state)

    tol = 1e-4 if dtype == jnp.float32 else 1e-10
    assert ell.dtype == dtype
    assert jnp.allclose(ell, analytical, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ell_grad_matches_gaussian_closed_form(dtype):
    y = jnp.array([0.5, -1.0, 1.5, 0.0], dtype)
    state = _dense_state(dtype)
    rule = gaussx.GaussHermiteIntegrator(order=5)

    def numerical(noise_var, mean):
        lik = gaussx.CallableLikelihood(_gaussian_site(noise_var), y)
        state_ = gaussx.GaussianState(mean, state.cov)
        return gaussx.expected_log_likelihood(lik, state_, rule)

    def analytical(noise_var, mean):
        lik = gaussx.GaussianLikelihood(y, noise_var)
        return gaussx.expected_log_likelihood(
            lik, gaussx.GaussianState(mean, state.cov)
        )

    noise_var = jnp.asarray(0.3, dtype)
    grads = jax.grad(numerical, argnums=(0, 1))(noise_var, state.mean)
    ref_grads = jax.grad(analytical, argnums=(0, 1))(noise_var, state.mean)

    tol = 1e-4 if dtype == jnp.float32 else 1e-10
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        assert grad.dtype == dtype
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(grad, ref_grad, rtol=tol, atol=tol)


class _ScaledGaussianSite(eqx.Module):
    log_noise_var: jax.Array

    def __call__(self, f_i, d_i):
        # σᵢ² = exp(θ) xᵢ: a per-site input scales the noise
        noise_var = jnp.exp(self.log_noise_var) * d_i["x"]
        return jss.norm.logpdf(d_i["y"], f_i, jnp.sqrt(noise_var))


def test_pytree_data_and_module_density_under_filter_jit_and_filter_grad():
    dtype = jnp.asarray(0.0).dtype
    data = {
        "y": jnp.array([0.5, -1.0, 1.5, 0.0], dtype),
        "x": jnp.array([1.0, 2.0, 0.5, 1.5], dtype),
    }
    state = _diagonal_state(dtype)
    rule = gaussx.GaussHermiteIntegrator(order=5)

    @eqx.filter_jit
    def ell(lik):
        return gaussx.expected_log_likelihood(lik, state, rule)

    lik = gaussx.CallableLikelihood(_ScaledGaussianSite(jnp.asarray(-1.0, dtype)), data)
    noise_vars = jnp.exp(lik.log_density.log_noise_var) * data["x"]
    variances = gaussx.diag(state.cov)
    # 𝔼[log 𝒩(yᵢ; fᵢ, σᵢ²)] = log 𝒩(yᵢ; mᵢ, σᵢ²) − Σᵢᵢ / (2 σᵢ²)
    expected = jnp.sum(
        jss.norm.logpdf(data["y"], state.mean, jnp.sqrt(noise_vars))
        - 0.5 * variances / noise_vars
    )
    assert jnp.allclose(ell(lik), expected, rtol=1e-5, atol=1e-5)

    grads = eqx.filter_grad(ell)(lik)
    assert grads.log_density.log_noise_var.dtype == dtype
    assert jnp.isfinite(grads.log_density.log_noise_var)


def test_jit_and_vmap_over_data():
    dtype = jnp.asarray(0.0).dtype
    labels = jnp.array(
        [[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype
    )
    state = _diagonal_state(dtype)
    rule = gaussx.GaussHermiteIntegrator(order=6)

    def ell(y):
        lik = gaussx.CallableLikelihood(_bernoulli_site, y)
        return gaussx.expected_log_likelihood(lik, state, rule)

    def log_prob(y):
        return gaussx.CallableLikelihood(_bernoulli_site, y).log_prob(state.mean)

    batched_ell = jax.jit(jax.vmap(ell))(labels)
    batched_log_prob = jax.jit(jax.vmap(log_prob))(labels)
    looped_ell = jnp.stack([ell(y) for y in labels])
    looped_log_prob = jnp.stack(
        [gaussx.BernoulliLikelihood(y).log_prob(state.mean) for y in labels]
    )

    assert batched_ell.shape == (3,)
    assert jnp.allclose(batched_ell, looped_ell, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(batched_log_prob, looped_log_prob, rtol=1e-5, atol=1e-5)


def test_has_no_analytical_ell_and_needs_an_integrator():
    lik = gaussx.CallableLikelihood(_bernoulli_site, _labels(jnp.float32))
    assert not lik.has_analytical_ell()
    with pytest.raises(ValueError, match="CallableLikelihood has no analytical ELL"):
        gaussx.expected_log_likelihood(lik, _diagonal_state(jnp.float32))
