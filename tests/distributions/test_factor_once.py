"""gh-329: dense PSD log-densities, entropies and KLs factor each covariance once.

The counts walk the jaxpr, sub-jaxprs included: XLA does not CSE across
lineax's linear_solve primitive, so each counted factorisation really runs.
"""

from __future__ import annotations

import collections

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.stats
from jax.extend import core as jcore

import gaussx
from gaussx._testing import psd_operator, random_pd_matrix


def _count(f, *args) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts[eqn.primitive.name] += 1
            for value in eqn.params.values():
                for sub in value if isinstance(value, (list, tuple)) else [value]:
                    if isinstance(sub, jcore.ClosedJaxpr):
                        walk(sub.jaxpr)
                    elif isinstance(sub, jcore.Jaxpr):
                        walk(sub)

    walk(jax.make_jaxpr(f)(*args).jaxpr)
    return {name: counts[name] for name in ("cholesky", "lu")}


_S = random_pd_matrix(jr.key(0), 3, jitter=3)
_X = jnp.array([0.1, -0.2, 0.3])


def test_gaussian_log_prob_factors_once():
    f = lambda S, x: gaussx.gaussian_log_prob(jnp.zeros(3), psd_operator(S), x)
    assert _count(f, _S, _X) == {"cholesky": 1, "lu": 0}


def test_mvn_log_prob_factors_once():
    def f(S, x):
        return gaussx.MultivariateNormal(jnp.zeros(3), psd_operator(S)).log_prob(x)

    assert _count(f, _S, _X) == {"cholesky": 1, "lu": 0}


def test_entropy_factors_once():
    assert _count(lambda S: gaussx.gaussian_entropy(psd_operator(S)), _S) == {
        "cholesky": 1,
        "lu": 0,
    }


def test_kl_factors_each_covariance_once():
    def f(S, T):
        return gaussx.gaussian_kl(
            jnp.zeros(3), psd_operator(S), jnp.ones(3), psd_operator(T)
        )

    assert _count(f, _S, _S + jnp.eye(3)) == {"cholesky": 2, "lu": 0}


@pytest.mark.parametrize("n", [1, 3, 7])
def test_values_match_scipy_and_numpyro(n):
    """rtol=1e-10: float64 closed forms on well-conditioned SPD matrices."""
    nd = pytest.importorskip("numpyro.distributions")
    k1, k2, k3, k4 = jr.split(jr.key(n), 4)
    S, T = random_pd_matrix(k1, n, jitter=n), random_pd_matrix(k2, n, jitter=n)
    mu, x = jr.normal(k3, (n,)), jr.normal(k4, (n,))
    ref = scipy.stats.multivariate_normal(np.asarray(mu), np.asarray(S))
    assert np.allclose(
        gaussx.gaussian_log_prob(mu, psd_operator(S), x),
        ref.logpdf(np.asarray(x)),
        rtol=1e-10,
    )
    assert np.allclose(
        gaussx.gaussian_entropy(psd_operator(S)), ref.entropy(), rtol=1e-10
    )
    kl_ref = nd.kl_divergence(
        nd.MultivariateNormal(mu, covariance_matrix=S),
        nd.MultivariateNormal(x, covariance_matrix=T),
    )
    kl = gaussx.gaussian_kl(mu, psd_operator(S), x, psd_operator(T))
    assert jnp.allclose(kl, kl_ref, rtol=1e-10)


def test_log_prob_gradient_matches_the_two_factorisation_form():
    """d log N / d Sigma is unchanged by the factor-once path."""

    def old(S, x):
        alpha = jnp.linalg.solve(S, x)
        _, ld = jnp.linalg.slogdet(S)
        return -0.5 * (3 * jnp.log(2 * jnp.pi) + ld + x @ alpha)

    def new(S, x):
        return gaussx.gaussian_log_prob(jnp.zeros(3), psd_operator(S), x)

    assert jnp.allclose(jax.grad(new)(_S, _X), jax.grad(old)(_S, _X), rtol=1e-10)


def _structured():
    A = psd_operator(random_pd_matrix(jr.key(1), 2, jitter=2))
    B = psd_operator(random_pd_matrix(jr.key(2), 3, jitter=3))
    return {
        "kronecker": gaussx.Kronecker(A, B),
        "block_diag": gaussx.BlockDiag(A, B),
        "low_rank": gaussx.LowRankUpdate(
            lx.DiagonalLinearOperator(jnp.full(5, 2.0)), jnp.ones((5, 1))
        ),
        "diagonal": lx.DiagonalLinearOperator(jnp.full(5, 2.0)),
    }


@pytest.mark.parametrize("name", list(_structured()))
def test_structured_covariances_are_not_materialised(name, monkeypatch):
    cov = _structured()[name]
    n = cov.in_size()

    def refuse(self):
        raise AssertionError(f"{type(self).__name__}.as_matrix was called")

    monkeypatch.setattr(type(cov), "as_matrix", refuse)
    value = gaussx.gaussian_log_prob(jnp.zeros(n), cov, jnp.ones(n))
    assert jnp.isfinite(value)


def test_singular_covariance_stays_non_finite():
    """gh-302 holds on the new path: non-finite, no exception."""
    cov = psd_operator(jnp.ones((3, 3)))
    assert not jnp.isfinite(gaussx.gaussian_log_prob(jnp.zeros(3), cov, _X))
    assert not jnp.isfinite(gaussx.gaussian_entropy(cov))
