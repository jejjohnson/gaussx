"""Rodgers information diagnostics, ``gaussx.information_diagnostics`` (#686).

In the prior-whitened coordinates the singular values σᵢ of
K̂ = L_ε⁻¹ K L_a = U Σ Vᵀ give the oracle for every output:

    Ŝ = L_a V (I + Σᵀ Σ)⁻¹ Vᵀ L_aᵀ,   A = L_a V Σᵀ Σ (I + Σᵀ Σ)⁻¹ Vᵀ L_a⁻¹,
    d_s = ∑ᵢ σᵢ² / (1 + σᵢ²),   H = ½ ∑ᵢ ln(1 + σᵢ²).
"""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._testing import (
    default_tolerances,
    random_kronecker_pd,
    random_pd_operator,
)


def _float():
    return jnp.result_type(float)


def _oracle(jacobian, noise_matrix, prior_matrix):
    n_state = jacobian.shape[1]
    noise_root = jnp.linalg.cholesky(noise_matrix)
    prior_root = jnp.linalg.cholesky(prior_matrix)
    whitened = jax.scipy.linalg.solve_triangular(noise_root, jacobian, lower=True)
    whitened = einsum(whitened, prior_root, "m k, k n -> m n")
    _, singular, right_h = jnp.linalg.svd(whitened, full_matrices=True)
    squared = jnp.zeros(n_state, jacobian.dtype)
    squared = squared.at[: singular.shape[0]].set(singular**2)
    # L_a V
    basis = einsum(prior_root, right_h, "i k, j k -> i j")
    posterior = einsum(basis / (1.0 + squared), basis, "i k, j k -> i j")
    # Vᵀ L_a⁻¹, from L_aᵀ⁻¹ V
    inverse_basis = jax.scipy.linalg.solve_triangular(
        prior_root, jnp.eye(n_state, dtype=jacobian.dtype), lower=True
    )
    inverse_basis = einsum(right_h, inverse_basis, "k i, i j -> k j")
    kernel = einsum(basis * squared / (1.0 + squared), inverse_basis, "i k, k j -> i j")
    return {
        "averaging_kernel": kernel,
        "dofs": jnp.sum(squared / (1.0 + squared)),
        "information_content": 0.5 * jnp.sum(jnp.log1p(squared)),
        "posterior_cov": posterior,
        "retrieval_noise_cov": einsum(kernel, posterior, "i k, k j -> i j"),
    }


def _problem(n_obs, n_state):
    jacobian = jr.normal(jr.key(0), (n_obs, n_state), _float())
    noise = lx.DiagonalLinearOperator(0.2 + jr.uniform(jr.key(1), (n_obs,), _float()))
    return jacobian, noise, random_pd_operator(jr.key(2), n_state)


@pytest.mark.parametrize(
    ("n_obs", "n_state"),
    [pytest.param(6, 3, id="m_gt_n"), pytest.param(2, 4, id="m_lt_n")],
)
def test_matches_the_singular_value_oracle(n_obs, n_state):
    jacobian, noise, prior = _problem(n_obs, n_state)
    actual = gaussx.information_diagnostics(jacobian, noise, prior)
    expected = _oracle(jacobian, noise.as_matrix(), prior.as_matrix())
    for name, value in expected.items():
        rtol, atol = default_tolerances(value)
        assert jnp.allclose(
            getattr(actual, name), value, rtol=1000 * rtol, atol=1000 * atol
        ), name


@pytest.mark.parametrize(
    ("n_obs", "n_state"),
    [pytest.param(6, 3, id="m_gt_n"), pytest.param(2, 4, id="m_lt_n")],
)
def test_dofs_are_bounded(n_obs, n_state):
    jacobian, noise, prior = _problem(n_obs, n_state)
    dofs = gaussx.information_diagnostics(jacobian, noise, prior).dofs
    assert 0.0 <= float(dofs) <= min(n_obs, n_state)


def test_information_is_the_entropy_difference():
    jacobian, noise, prior = _problem(6, 3)
    actual = gaussx.information_diagnostics(jacobian, noise, prior)
    posterior = lx.MatrixLinearOperator(
        actual.posterior_cov, lx.positive_semidefinite_tag
    )
    difference = gaussx.gaussian_entropy(prior) - gaussx.gaussian_entropy(posterior)
    rtol, atol = default_tolerances(difference)
    assert jnp.allclose(
        actual.information_content, difference, rtol=1000 * rtol, atol=1000 * atol
    )


def test_identity_of_the_averaging_kernel():
    """A = I − Ŝ S_a⁻¹."""
    jacobian, noise, prior = _problem(6, 3)
    actual = gaussx.information_diagnostics(jacobian, noise, prior)
    prior_inverse = jnp.linalg.inv(prior.as_matrix())
    expected = jnp.eye(3) - einsum(
        actual.posterior_cov, prior_inverse, "i k, k j -> i j"
    )
    rtol, atol = default_tolerances(expected)
    assert jnp.allclose(
        actual.averaging_kernel, expected, rtol=1000 * rtol, atol=1000 * atol
    )


def test_a_strongly_observed_retrieval_keeps_its_prior():
    """K = [1e4, 1e4] in float32: K̃ᵀ K̃ would round the unit prior away.

    With unit noise and prior, σ² = 2e8, so d_s ≈ 1, H = ½ ln(1 + 2e8) and Ŝ
    has diagonal ≈ ½.
    """
    identity = lx.DiagonalLinearOperator(jnp.ones(1, jnp.float32))
    prior = lx.DiagonalLinearOperator(jnp.ones(2, jnp.float32))
    actual = gaussx.information_diagnostics(
        jnp.array([[1e4, 1e4]], jnp.float32), identity, prior
    )
    assert actual.posterior_cov.dtype == jnp.float32
    assert jnp.allclose(jnp.diag(actual.posterior_cov), 0.5, rtol=1e-3)
    assert jnp.allclose(actual.dofs, 1.0, rtol=1e-3)
    assert jnp.allclose(actual.information_content, 9.557, rtol=1e-3)


def test_structured_covariances_are_not_materialised():
    """A diagonal S_ε and a Kronecker S_a match the dense result, undensified."""
    jacobian = jr.normal(jr.key(0), (8, 6), _float())
    noise = lx.DiagonalLinearOperator(jnp.full(8, 0.5, _float()))
    prior = random_kronecker_pd(jr.key(3), (2, 3))
    dense_prior = lx.MatrixLinearOperator(
        prior.as_matrix(), lx.positive_semidefinite_tag
    )
    reference = gaussx.information_diagnostics(jacobian, noise, dense_prior)
    with (
        mock.patch.object(
            gaussx.Kronecker, "as_matrix", side_effect=AssertionError("densified")
        ),
        mock.patch.object(
            lx.DiagonalLinearOperator,
            "as_matrix",
            side_effect=AssertionError("densified"),
        ),
    ):
        actual = gaussx.information_diagnostics(jacobian, noise, prior)
    rtol, atol = default_tolerances(reference.posterior_cov)
    assert jnp.allclose(
        actual.posterior_cov,
        reference.posterior_cov,
        rtol=1000 * rtol,
        atol=1000 * atol,
    )
    assert jnp.allclose(
        actual.information_content,
        reference.information_content,
        rtol=1000 * rtol,
        atol=1000 * atol,
    )


def test_gradient_in_the_jacobian():
    """∂d_s/∂K against the same quantity written with dense inverses."""
    jacobian, noise, prior = _problem(6, 3)
    noise_inverse = jnp.linalg.inv(noise.as_matrix())
    prior_inverse = jnp.linalg.inv(prior.as_matrix())

    def dense_dofs(matrix):
        data = einsum(matrix, noise_inverse @ matrix, "m i, m j -> i j")
        return jnp.trace(jnp.linalg.solve(data + prior_inverse, data))

    actual = jax.grad(
        lambda matrix: gaussx.information_diagnostics(matrix, noise, prior).dofs
    )(jacobian)
    expected = jax.grad(dense_dofs)(jacobian)
    rtol, atol = default_tolerances(expected)
    assert jnp.allclose(actual, expected, rtol=1000 * rtol, atol=1000 * atol)
