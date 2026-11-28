"""gaussian_log_prob through BBMM's shared pass (gh-880).

With a `BBMMSolver` (or a `KeyedSolver` around one, or a `ComposedSolver`
of one BBMM twice) the quadratic form and the log-determinant come from one
`inv_quad_logdet` pass; every other strategy keeps its separate solve and
logdet. A counting spy on `_shared_work_core` checks the route. The file is
in the float32 lane (Makefile ``NO_X64_TESTS``).
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._primitives._inv_quad_logdet as inv_quad_module
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve


_SYMMETRIC_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)
_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


def _x64() -> bool:
    return jnp.zeros(()).dtype == jnp.float64


def _kernel(inputs, lengthscale):
    """Matérn-3/2 Gram matrix on 1-D inputs, with a 1e-6 jitter."""
    gaps = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))
    scaled = jnp.sqrt(3.0) * gaps / lengthscale
    jitter = 1e-6 * jnp.eye(inputs.shape[0], dtype=inputs.dtype)
    return (1.0 + scaled) * jnp.exp(-scaled) + jitter


def _problem(key, size=40, lengthscale=0.3):
    """Σ = K + diag(d) as a split system, its dense matrix, a mean and a y."""
    input_key, shift_key, mean_key, value_key = jr.split(key, 4)
    inputs = jnp.sort(jr.uniform(input_key, (size,)))
    kernel = _kernel(inputs, lengthscale)
    shift = 0.1 + 0.4 * jr.uniform(shift_key, (size,))
    system = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel) + lx.DiagonalLinearOperator(shift),
        _SYMMETRIC_PSD,
    )
    loc = 0.1 * jr.normal(mean_key, (size,))
    value = jr.normal(value_key, (size,))
    return system, kernel + jnp.diag(shift), loc, value


def _dense_log_prob(loc, matrix, value):
    residual = value - loc
    quadratic = residual @ jnp.linalg.solve(matrix, residual)
    logdet = jnp.linalg.slogdet(matrix)[1]
    return -0.5 * (residual.shape[0] * _LOG_2PI + logdet + quadratic)


class _CountingCore:
    """Counts the calls of `_shared_work_core`, the shared mBCG pass."""

    def __init__(self, monkeypatch):
        original = inv_quad_module._shared_work_core
        self.calls = 0

        def counting(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(inv_quad_module, "_shared_work_core", counting)


_BBMM = gaussx.BBMMSolver(num_probes=8)
_SHARED = {
    "bbmm": _BBMM,
    "keyed": gaussx.KeyedSolver(_BBMM, jr.key(3)),
    "composed": gaussx.ComposedSolver(_BBMM, _BBMM),
}


@pytest.mark.parametrize("name", list(_SHARED))
def test_one_shared_pass_per_call(getkey, monkeypatch, name):
    system, matrix, loc, value = _problem(getkey())
    counter = _CountingCore(monkeypatch)
    log_prob = gaussx.gaussian_log_prob(loc, system, value, solver=_SHARED[name])
    assert counter.calls == 1
    # the quadratic form is solved to rtol; the logdet is a probe estimate
    assert jnp.isfinite(log_prob)
    assert jnp.abs(log_prob - _dense_log_prob(loc, matrix, value)) < 5.0


def test_composed_of_two_different_strategies_stays_separate(getkey, monkeypatch):
    system, _, loc, value = _problem(getkey())
    counter = _CountingCore(monkeypatch)
    composed = gaussx.ComposedSolver(gaussx.CGSolver(), _BBMM)
    gaussx.gaussian_log_prob(loc, system, value, solver=composed)
    assert counter.calls == 0


@pytest.mark.parametrize(
    "solver",
    [
        gaussx.CGSolver(num_probes=4),
        gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.SLQLogdet(num_probes=4)),
        gaussx.ComposedSolver(gaussx.CGSolver(), gaussx.DenseLogdet()),
    ],
    ids=["cg", "cg_slq", "cg_dense_logdet"],
)
def test_other_strategies_keep_the_separate_route(getkey, solver):
    system, _, loc, value = _problem(getkey())
    key = getkey()
    log_prob = gaussx.gaussian_log_prob(loc, system, value, solver=solver, key=key)
    residual = value - loc
    # the route before this change: solve, then logdet
    quadratic = residual @ dispatch_solve(system, residual, solver)
    logdet = dispatch_logdet(system, solver, key=key)
    expected = -0.5 * (residual.shape[0] * _LOG_2PI + logdet + quadratic)
    assert jnp.array_equal(log_prob, expected)


@pytest.mark.slow
@pytest.mark.x64_only(reason="a spread comparison over keys, float64 references")
def test_value_within_three_standard_errors_and_preconditioning_narrows_it():
    system, matrix, loc, value = _problem(jr.key(0), size=120)
    exact = gaussx.gaussian_log_prob(loc, system, value, solver=gaussx.DenseSolver())
    keys = jr.split(jr.key(1), 20)

    def draws(solver):
        return jax.vmap(
            lambda key: gaussx.gaussian_log_prob(
                loc, system, value, solver=solver, key=key
            )
        )(keys)

    plain = draws(gaussx.BBMMSolver(num_probes=10))
    pre = gaussx.PartialCholeskyPreconditioner(rank=20, shift=None)
    preconditioned = draws(gaussx.BBMMSolver(num_probes=10, preconditioner=pre))
    # unbiased: the mean over 20 keys is within 3 standard errors
    standard_error = jnp.std(plain) / jnp.sqrt(20.0)
    assert jnp.abs(jnp.mean(plain) - exact) < 3.0 * standard_error
    assert jnp.std(preconditioned) < jnp.std(plain)
    assert jnp.allclose(exact, _dense_log_prob(loc, matrix, value))


def test_log_marginal_likelihood_takes_a_key(getkey, monkeypatch):
    system, _, loc, value = _problem(getkey())
    counter = _CountingCore(monkeypatch)
    first = gaussx.log_marginal_likelihood(
        loc, system, value, solver=_BBMM, key=jr.key(0)
    )
    second = gaussx.log_marginal_likelihood(
        loc, system, value, solver=_BBMM, key=jr.key(1)
    )
    again = gaussx.log_marginal_likelihood(
        loc, system, value, solver=_BBMM, key=jr.key(0)
    )
    assert counter.calls == 3
    assert not jnp.array_equal(first, second)
    assert jnp.array_equal(first, again)


def test_expected_log_lik_takes_the_shared_pass(getkey, monkeypatch):
    system, _, loc, value = _problem(getkey(), size=20)
    counter = _CountingCore(monkeypatch)
    q_cov = lx.DiagonalLinearOperator(jnp.full(20, 0.1))
    gaussx.gaussian_expected_log_lik(value, loc, q_cov, system, solver=_BBMM)
    assert counter.calls == 1


class _NoDensify(lx.FunctionLinearOperator):
    """A matrix-free kernel that fails the test if anything densifies it."""

    def as_matrix(self):
        raise AssertionError("as_matrix called on the matrix-free covariance")


def test_matrix_free_covariance_is_never_densified(getkey):
    _, matrix, loc, value = _problem(getkey())
    size = value.shape[0]
    structure = jax.ShapeDtypeStruct((size,), matrix.dtype)
    covariance = lx.TaggedLinearOperator(
        _NoDensify(lambda v: matrix @ v, structure, _SYMMETRIC_PSD), _SYMMETRIC_PSD
    )
    log_prob = gaussx.gaussian_log_prob(
        loc, covariance, value, solver=gaussx.BBMMSolver(num_probes=8)
    )
    assert jnp.isfinite(log_prob)


@pytest.mark.slow
def test_jit_vmap_and_grad_in_loc_value_and_lengthscale(getkey):
    size = 30
    inputs = jnp.sort(jr.uniform(getkey(), (size,)))
    shift = 0.1 + 0.4 * jr.uniform(getkey(), (size,))
    loc = 0.1 * jr.normal(getkey(), (size,))
    values = jr.normal(getkey(), (3, size))
    tight = 1e-10 if _x64() else 1e-6
    solver = gaussx.BBMMSolver(rtol=tight, atol=tight, num_probes=16)
    key = getkey()

    def log_prob(loc, value, lengthscale):
        covariance = lx.TaggedLinearOperator(
            lx.MatrixLinearOperator(_kernel(inputs, lengthscale))
            + lx.DiagonalLinearOperator(shift),
            _SYMMETRIC_PSD,
        )
        return gaussx.gaussian_log_prob(loc, covariance, value, solver=solver, key=key)

    def quadratic_only(loc, value, lengthscale):
        # −½ rᵀ Σ⁻¹ r: exact in the shared pass, so its gradient is too
        matrix = _kernel(inputs, lengthscale) + jnp.diag(shift)
        residual = value - loc
        return -0.5 * residual @ jnp.linalg.solve(matrix, residual)

    lengthscale = jnp.asarray(0.3)
    jitted = jax.jit(log_prob)(loc, values[0], lengthscale)
    assert jnp.allclose(jitted, log_prob(loc, values[0], lengthscale))
    batched = jax.vmap(log_prob, in_axes=(None, 0, None))(loc, values, lengthscale)
    looped = jnp.stack([log_prob(loc, row, lengthscale) for row in values])
    rtol = 1e-6 if _x64() else 1e-3
    assert jnp.allclose(batched, looped, rtol=rtol, atol=rtol)
    # the log-determinant does not depend on loc or y, so those gradients are
    # exactly the quadratic form's
    gradient = jax.grad(log_prob, argnums=(0, 1))(loc, values[0], lengthscale)
    expected = jax.grad(quadratic_only, argnums=(0, 1))(loc, values[0], lengthscale)
    for got, want in zip(gradient, expected, strict=True):
        assert jnp.allclose(got, want, rtol=rtol, atol=rtol)
    lengthscale_gradient = jax.grad(log_prob, argnums=2)(loc, values[0], lengthscale)
    assert jnp.isfinite(lengthscale_gradient)
