"""Tests for the control variate of stochastic log-determinant gradients (gh-826).

∂ log det A / ∂θ ≈ (1 / N_probe) ∑_i (a^(i) − b^(i)) + ∂ log det P / ∂θ, with
a^(i) = (A⁻¹ z)ᵀ ∂A (P⁻¹ z) and b^(i) = (P⁻¹ z)ᵀ ∂P (P⁻¹ z), through
`SLQLogdet(preconditioner=...)` and BBMM's shared pass. The uncontrolled
estimator is the same code with the zero-valued term patched out (to a
Python zero, through ``monkeypatch``).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._primitives._inv_quad_logdet as shared_pass
import gaussx._strategies._slq_logdet as slq_module
from gaussx._einx import reduce
from gaussx._strategies._control_variate import logdet_control_variate
from gaussx._strategies._whitening import whitened_operator


_TAGS = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _system(theta, n):
    """A = K_XX + diag(d), Matérn-3/2 with θ = (log variance, log ℓ)."""
    variance, lengthscale = jnp.exp(theta)
    inputs = jnp.linspace(0.0, 1.0, n, dtype=theta.dtype)
    gaps = jnp.abs(einx.subtract("i, j -> i j", inputs, inputs))
    # √3 r / ℓ
    scaled = jnp.sqrt(3.0) * gaps / lengthscale
    kernel = variance * (1.0 + scaled) * jnp.exp(-scaled)
    # d between 4 and 7, as in gh-826's prototype
    diagonal = jnp.linspace(4.0, 7.0, n, dtype=theta.dtype)
    operator = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel) + lx.DiagonalLinearOperator(diagonal), _TAGS
    )
    return operator, kernel + jnp.diag(diagonal)


def _theta(variance=2.0, lengthscale=0.3):
    return jnp.log(jnp.asarray([variance, lengthscale]))


def _estimate(route, rank, num_probes=50):
    """θ, key ↦ the stochastic log det A by ``route``."""
    pre = None if rank == 0 else gaussx.PartialCholeskyPreconditioner(rank, shift=None)

    def value(theta, key, n):
        operator, _ = _system(theta, n)
        if route == "slq":
            strategy = gaussx.SLQLogdet(num_probes=num_probes, preconditioner=pre)
            return strategy.logdet(operator, key=key)
        no_rhs = jnp.zeros((n, 0), dtype=theta.dtype)
        strategy = gaussx.BBMMSolver(num_probes=num_probes, preconditioner=pre)
        return gaussx.inv_quad_logdet(operator, no_rhs, strategy=strategy, key=key)[1]

    return value


def _exact_gradient(theta, n):
    return jax.grad(lambda t: jnp.linalg.slogdet(_system(t, n)[1])[1])(theta)


def test_term_is_zero_with_the_control_variate_as_gradient(getkey):
    """Value 0; gradient ∂ log det P − (1 / N_probe) ∑ wᵀ ∂P w, densely."""
    n, num_probes = 5, 3
    probes = jr.normal(getkey(), (n, num_probes))
    base = jr.normal(getkey(), (n, n))

    def matrix(scale):
        # P(s) = s B Bᵀ + I
        return scale * (base @ base.T) + jnp.eye(n)

    def term(scale):
        approximation = lx.MatrixLinearOperator(matrix(scale), _TAGS)
        return logdet_control_variate(
            jnp.linalg.slogdet(matrix(scale))[1], approximation, probes
        )

    scale = jnp.asarray(0.7)
    assert term(scale) == 0.0
    derivative = base @ base.T
    expected = jnp.trace(jnp.linalg.solve(matrix(scale), derivative)) - jnp.mean(
        einx.dot("n p, n m, m p -> p", probes, derivative, probes)
    )
    assert jnp.allclose(jax.grad(term)(scale), expected, rtol=1e-5)


@pytest.mark.parametrize("route", ["slq", "bbmm"])
def test_forward_value_is_unchanged(route, monkeypatch):
    """The term adds exactly zero to the estimate."""
    theta, key = _theta(), jr.key(0)
    controlled = _estimate(route, rank=4, num_probes=4)(theta, key, 16)
    monkeypatch.setattr(slq_module, "logdet_control_variate", lambda *a: 0.0)
    monkeypatch.setattr(shared_pass, "logdet_control_variate", lambda *a: 0.0)
    plain = _estimate(route, rank=4, num_probes=4)(theta, key, 16)
    assert controlled == plain


@pytest.mark.parametrize("route", ["slq", "bbmm"])
def test_smoke_gradient_is_finite_under_jit_and_vmap(route):
    """jit(vmap(grad)) over keys runs and stays near the exact gradient."""
    theta = _theta()
    value = _estimate(route, rank=4, num_probes=4)
    keys = jr.split(jr.key(1), 2)
    grads = jax.jit(
        jax.vmap(jax.grad(lambda t, k: value(t, k, 16)), in_axes=(None, 0))
    )(theta, keys)
    assert jnp.all(jnp.isfinite(grads))
    # a smoke test: rank 4 of 16 is a weak P and 4 probes are few, so only
    # the sign and the scale are checked (the slow tests bound the error)
    assert jnp.allclose(grads, _exact_gradient(theta, 16), rtol=0.5)


@pytest.mark.slow
def test_a_constant_preconditioner_leaves_the_raw_gradient():
    """P independent of θ: every b^(i) = 0, a finite gradient, the raw one.

    This is gh-826's Var(b) = 0 case; with a unit coefficient there is no
    ratio to guard.
    """
    theta, key = _theta(), jr.key(2)
    fixed = gaussx.JacobiPreconditioner(diagonal=jnp.linspace(4.0, 7.0, 40) + 2.0)

    def value(t):
        operator, _ = _system(t, 40)
        strategy = gaussx.SLQLogdet(num_probes=8, preconditioner=fixed)
        return strategy.logdet(operator, key=key)

    def raw(t):
        operator, _ = _system(t, 40)
        approximation = fixed.approximation(operator)
        whitened = whitened_operator(operator, approximation)
        plain = gaussx.SLQLogdet(num_probes=8, sampler="signs")
        return fixed.logdet(operator) + plain.logdet(whitened, key=key)

    got = jax.grad(value)(theta)
    assert jnp.all(jnp.isfinite(got))
    assert jnp.allclose(got, jax.grad(raw)(theta), rtol=1e-6)


def test_quadratic_gradient_does_not_see_the_preconditioner(getkey):
    """P's arrays get no gradient from the inverse quadratic."""
    theta = _theta()
    operator, _ = _system(theta, 16)
    rhs = jr.normal(getkey(), (16, 1))
    pre = gaussx.JacobiPreconditioner(diagonal=jnp.linspace(4.0, 7.0, 16) + 2.0)

    def quadratic(diagonal):
        strategy = gaussx.BBMMSolver(
            num_probes=8, preconditioner=gaussx.JacobiPreconditioner(diagonal=diagonal)
        )
        return gaussx.inv_quad_logdet(operator, rhs, strategy=strategy, key=getkey())[0]

    assert jnp.array_equal(
        jax.grad(quadratic)(pre.diagonal), jnp.zeros_like(pre.diagonal)
    )


@pytest.mark.slow
@pytest.mark.parametrize("route", ["slq", "bbmm"])
def test_a_preconditioner_built_from_theta_is_differentiated(route):
    """BBMMSolver(preconditioner=pre(θ)) under grad, as gh-826 writes it.

    Before gh-826 the shared pass took the strategy, and so P's arrays, as
    an argument that cannot carry a tangent, and raised.
    """
    n = 40
    inputs = jnp.linspace(0.0, 1.0, n)

    def value(theta, key):
        operator, _ = _system(theta, n)
        variance, lengthscale = jnp.exp(theta)

        def kernel(left, right):
            gaps = jnp.abs(einx.subtract("i 1, j 1 -> i j", left, right))
            scaled = jnp.sqrt(3.0) * gaps / lengthscale
            return variance * (1.0 + scaled) * jnp.exp(-scaled)

        column = einx.id("n -> n 1", inputs)
        pre = gaussx.FITCPreconditioner.from_kernel(kernel, column, column[::4])
        if route == "slq":
            strategy = gaussx.SLQLogdet(num_probes=8, preconditioner=pre)
            return strategy.logdet(operator, key=key)
        no_rhs = jnp.zeros((n, 0))
        strategy = gaussx.BBMMSolver(num_probes=8, preconditioner=pre)
        return gaussx.inv_quad_logdet(operator, no_rhs, strategy=strategy, key=key)[1]

    theta = _theta()
    got = jax.grad(value)(theta, jr.key(6))
    assert jnp.all(jnp.isfinite(got))
    assert jnp.allclose(got, _exact_gradient(theta, n), rtol=0.05)


@pytest.mark.slow
@pytest.mark.x64_only(reason="a 2-standard-error band of 200 draws needs float64")
@pytest.mark.parametrize("route", ["slq", "bbmm"])
def test_controlled_gradient_is_unbiased(route):
    """Mean over 200 keys within 2 standard errors of the exact gradient.

    A rank-3 preconditioner keeps some variance, so the band is not
    vacuous. The bound is the estimator's own standard error (gh-220).
    """
    theta, n = _theta(), 64
    value = _estimate(route, rank=3, num_probes=10)
    keys = jr.split(jr.key(3), 200)
    grads = jax.jit(jax.vmap(jax.grad(lambda t, k: value(t, k, n)), in_axes=(None, 0)))(
        theta, keys
    )
    mean = reduce(grads, "key param -> param", "mean")
    error = reduce(grads, "key param -> param", "std") / jnp.sqrt(200.0)
    assert jnp.all(jnp.abs(mean - _exact_gradient(theta, n)) <= 2.0 * error)


def _spread(route, rank, n, num_probes, theta, keys):
    value = _estimate(route, rank=rank, num_probes=num_probes)
    gradient = jax.jit(jax.grad(lambda t, k: value(t, k, n)))
    grads = jnp.stack([gradient(theta, k) for k in keys])
    exact = _exact_gradient(theta, n)
    squared = reduce((grads - exact) ** 2, "key param -> param", "mean")
    return jnp.sqrt(squared) / jnp.abs(exact)


@pytest.mark.slow
@pytest.mark.parametrize("route", ["slq", "bbmm"])
def test_variance_drops_below_the_uncontrolled_one(route, monkeypatch):
    """Matérn-3/2, n = 256: the controlled error is below the uncontrolled one."""
    theta, keys = _theta(), jr.split(jr.key(4), 10)
    controlled = _spread(route, 20, 256, 20, theta, keys)
    monkeypatch.setattr(slq_module, "logdet_control_variate", lambda *a: 0.0)
    monkeypatch.setattr(shared_pass, "logdet_control_variate", lambda *a: 0.0)
    raw = _spread(route, 20, 256, 20, theta, keys)
    assert jnp.all(controlled < raw), (controlled, raw)


@pytest.mark.slow
@pytest.mark.parametrize("route", ["slq", "bbmm"])
@pytest.mark.parametrize("setting", [(2.0, 0.3), (10.0, 0.1)])
def test_reproduces_the_gh_826_table(route, setting, monkeypatch):
    """n = 150, rank 100, 50 probes: below a quarter of the uncontrolled error."""
    theta, keys = _theta(*setting), jr.split(jr.key(5), 10)
    controlled = _spread(route, 100, 150, 50, theta, keys)
    monkeypatch.setattr(slq_module, "logdet_control_variate", lambda *a: 0.0)
    monkeypatch.setattr(shared_pass, "logdet_control_variate", lambda *a: 0.0)
    raw = _spread(route, 100, 150, 50, theta, keys)
    assert jnp.all(controlled < 0.25 * raw), (controlled, raw)
