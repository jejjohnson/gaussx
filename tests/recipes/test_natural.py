"""Tests for natural <-> expectation parameter conversions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._recipes import expectation_to_natural, natural_to_expectation
from gaussx._testing import random_pd_matrix, tree_allclose


def test_roundtrip_nat_to_exp_to_nat(getkey):
    """natural -> expectation -> natural should be identity."""
    N = 4
    Sigma_mat = random_pd_matrix(getkey(), N)
    Sigma = lx.MatrixLinearOperator(Sigma_mat, lx.positive_semidefinite_tag)
    mu = jr.normal(getkey(), (N,))

    eta1, eta2 = expectation_to_natural(mu, Sigma)
    mu_recovered, Sigma_recovered = natural_to_expectation(eta1, eta2)

    assert tree_allclose(mu_recovered, mu, rtol=1e-4)
    assert tree_allclose(Sigma_recovered.as_matrix(), Sigma_mat, rtol=1e-4)


def test_roundtrip_exp_to_nat_to_exp(getkey):
    """expectation -> natural -> expectation should be identity."""
    N = 3
    Lambda_mat = random_pd_matrix(getkey(), N)
    mu = jr.normal(getkey(), (N,))

    # Natural params
    eta1 = Lambda_mat @ mu
    eta2_mat = -0.5 * Lambda_mat
    eta2 = lx.MatrixLinearOperator(eta2_mat)

    mu_rec, Sigma_rec = natural_to_expectation(eta1, eta2)
    eta1_rec, eta2_rec = expectation_to_natural(mu_rec, Sigma_rec)

    assert tree_allclose(eta1_rec, eta1, rtol=1e-4)
    assert tree_allclose(eta2_rec.as_matrix(), eta2_mat, rtol=1e-4)


def test_natural_to_expectation_diagonal(getkey):
    """With diagonal precision, mean recovery should be exact."""
    N = 5
    d = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
    mu = jr.normal(getkey(), (N,))

    # Lambda = diag(d), eta1 = Lambda mu, eta2 = -0.5 Lambda
    eta1 = d * mu
    eta2 = lx.DiagonalLinearOperator(-0.5 * d)

    mu_rec, _Sigma_rec = natural_to_expectation(eta1, eta2)
    assert tree_allclose(mu_rec, mu, rtol=1e-5)
