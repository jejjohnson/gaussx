"""Tests for AnalyticalPsiStatistics protocol and dispatch."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import (
    AnalyticalPsiStatistics,
    CubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    TaylorIntegrator,
    UnscentedIntegrator,
    compute_psi_statistics,
    kernel_expectations,
)
from gaussx._einx import rearrange


class MockAnalyticalKernel:
    """Mock kernel implementing AnalyticalPsiStatistics."""

    def psi0(self, state):
        return jnp.array(1.0)

    def psi1(self, state, X_train):
        M = X_train.shape[0]
        return jnp.ones((M,))

    def psi2(self, state, X_train):
        M = X_train.shape[0]
        return jnp.eye(M)


class TestAnalyticalPsiStatistics:
    def test_isinstance_check(self):
        """MockAnalyticalKernel passes isinstance check."""
        kernel = MockAnalyticalKernel()
        assert isinstance(kernel, AnalyticalPsiStatistics)

    def test_dispatch_to_analytical(self):
        """compute_psi_statistics dispatches to analytical methods."""
        kernel = MockAnalyticalKernel()
        mean = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3))
        state = GaussianState(mean=mean, cov=cov)
        X_train = jnp.ones((5, 3))

        psi0, psi1, psi2 = compute_psi_statistics(kernel, state, X_train)
        assert psi0.shape == ()
        assert psi1.shape == (5,)
        assert psi2.shape == (5, 5)

    def test_error_without_integrator(self):
        """Raises ValueError for non-analytical kernel with no integrator."""

        class PlainKernel:
            pass

        kernel = PlainKernel()
        mean = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3))
        state = GaussianState(mean=mean, cov=cov)
        X_train = jnp.ones((5, 3))

        with pytest.raises(ValueError, match="AnalyticalPsiStatistics"):
            compute_psi_statistics(kernel, state, X_train)

    def test_non_analytical_fails_isinstance(self):
        """Plain object does not pass isinstance check."""

        class NotAKernel:
            pass

        assert not isinstance(NotAKernel(), AnalyticalPsiStatistics)


# gh-323: one implementation (kernel_expectations), no dead (N², N²) covariance.


def _rbf(x, y):
    return jnp.exp(-0.5 * jnp.sum((x - y) ** 2))


def _psi_via_integrate(state, X, integrator):
    """The pre-gh-323 route: means of full `integrate` propagations."""

    def mean(fn):
        return integrator.integrate(fn, state).state.mean

    def row(x):
        return jax.vmap(lambda xi: _rbf(x, xi))(X)

    psi0 = mean(lambda x: jnp.atleast_1d(_rbf(x, x)))[0]
    psi1 = mean(row)
    flat = mean(lambda x: rearrange(jnp.outer(row(x), row(x)), "i j -> (i j)"))
    return psi0, psi1, rearrange(flat, "(i j) -> i j", i=X.shape[0])


_STATE_2D = GaussianState(
    mean=jnp.array([0.2, -0.3]),
    cov=lx.MatrixLinearOperator(
        jnp.array([[0.3, 0.05], [0.05, 0.2]]), lx.positive_semidefinite_tag
    ),
)


@pytest.mark.x64_only(reason="1e-14 agreement in float64")
@pytest.mark.parametrize(
    "integrator",
    [
        # Compile-bound (~3-4 s in CI) at any order; the other rules stay fast.
        pytest.param(GaussHermiteIntegrator(order=6), id="gh", marks=pytest.mark.slow),
        pytest.param(UnscentedIntegrator(alpha=1.0), id="unscented"),
        pytest.param(CubatureIntegrator(), id="cubature"),
        pytest.param(TaylorIntegrator(), id="taylor"),
    ],
)
def test_numerical_psi_unchanged(integrator):
    X = jr.normal(jr.key(0), (7, 2))
    ref = _psi_via_integrate(_STATE_2D, X, integrator)
    for got in (
        compute_psi_statistics(_rbf, _STATE_2D, X, integrator=integrator),
        kernel_expectations(_rbf, _STATE_2D, X, integrator),
    ):
        for g, r in zip(got, ref, strict=True):
            assert g.shape == r.shape
            assert jnp.allclose(g, r, rtol=1e-14, atol=1e-14)


def _sub_jaxprs(params):
    for v in params.values():
        for item in v if isinstance(v, (tuple, list)) else (v,):
            inner = getattr(item, "jaxpr", item)  # ClosedJaxpr -> Jaxpr
            if hasattr(inner, "eqns"):
                yield inner


def _largest_intermediate(fn, *args):
    """Most elements in any equation output, recursing into sub-jaxprs."""

    def walk(jaxpr):
        sizes = [math.prod(v.aval.shape) for e in jaxpr.eqns for v in e.outvars]
        sizes += [walk(sub) for e in jaxpr.eqns for sub in _sub_jaxprs(e.params)]
        return max(sizes, default=0)

    return walk(jax.make_jaxpr(fn)(*args).jaxpr)


@pytest.mark.parametrize("fn", ["kernel_expectations", "compute_psi_statistics"])
def test_no_n4_intermediate(fn):
    """GH-20 at N=10: nothing of N⁴ = 10⁴ elements is built (gh-323)."""
    state = GaussianState(
        mean=jnp.zeros(1),
        cov=lx.MatrixLinearOperator(0.1 * jnp.eye(1), lx.positive_semidefinite_tag),
    )
    X = rearrange(jnp.linspace(-1.0, 1.0, 10), "n -> n 1")
    integ = GaussHermiteIntegrator(order=20)
    if fn == "kernel_expectations":
        f = lambda X: kernel_expectations(_rbf, state, X, integ)
    else:
        f = lambda X: compute_psi_statistics(_rbf, state, X, integrator=integ)
    assert _largest_intermediate(f, X) < 10**4
