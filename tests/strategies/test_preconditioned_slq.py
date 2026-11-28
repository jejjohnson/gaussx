"""Tests for preconditioned stochastic Lanczos quadrature (gh-825).

log det A = log det P + log det(R⁻¹ A R⁻ᵀ), R Rᵀ = P, with SLQ on the
whitened operator. The system is a Matérn-3/2 kernel on N = 256 points plus
a per-point diagonal, so every preconditioner reads its shift from it.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import rearrange
from gaussx._strategies._whitening import whitened_operator


_N = 256
_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _matern32(lengthscale):
    def kernel(left, right):
        gaps = einx.subtract("n d, m d -> n m d", left, right)
        # √3 r / ℓ, with a floor under r² for a finite gradient at r = 0
        scaled = jnp.sqrt(3.0 * (einx.sum("n m [d]", gaps**2) + 1e-12)) / lengthscale
        return (1.0 + scaled) * jnp.exp(-scaled)

    return kernel


_X = rearrange(jnp.linspace(0.0, 1.0, _N), "n -> n 1")


def _system(lengthscale=0.3, n=_N):
    """``(A, dense A)`` with A = K_XX + diag(d), split-able, on n points."""
    inputs = _X[:: _N // n]
    # d_n between 0.1 and 0.5, as for Bernoulli sites
    diagonal = jnp.linspace(0.1, 0.5, n, dtype=inputs.dtype)
    K = _matern32(lengthscale)(inputs, inputs)
    operator = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(K) + lx.DiagonalLinearOperator(diagonal), _PSD
    )
    return operator, K + jnp.diag(diagonal)


_PRECONDITIONERS = {
    "jacobi": gaussx.JacobiPreconditioner,
    "nystrom": lambda: gaussx.NystromPreconditioner.lazy(40, key=jr.key(0)),
    "fitc": lambda: gaussx.FITCPreconditioner.from_kernel(
        _matern32(0.3), _X, _X[:: _N // 32]
    ),
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("jacobi", marks=pytest.mark.slow),
        "nystrom",
        pytest.param("fitc", marks=pytest.mark.slow),
    ],
)
def test_estimate_within_three_standard_errors(getkey, name):
    """The preconditioned estimate is unbiased: within 3 SEM of the exact one.

    The bound is the estimator's own standard error across probes (gh-220);
    30 Lanczos steps on the whitened operator leave no quadrature error at
    this size, so the probe variance is the only error.
    """
    operator, dense = _system()
    strategy = gaussx.SLQLogdet(num_probes=30, preconditioner=_PRECONDITIONERS[name]())
    estimate, error = strategy.logdet_and_error(operator, key=getkey())
    exact = jnp.linalg.slogdet(dense)[1]
    assert jnp.abs(estimate - exact) <= 3.0 * error + 1e-3 * jnp.abs(exact)


def test_whitening_matches_the_dense_log_determinant_difference():
    """log det(R⁻¹ A R⁻ᵀ) = log det A − log det P for each form of P."""
    operator, dense = _system(n=64)
    exact = jnp.linalg.slogdet(dense)[1]
    fitc = gaussx.FITCPreconditioner.from_kernel(_matern32(0.3), _X[::4], _X[::32])
    builders = {**_PRECONDITIONERS, "fitc": lambda: fitc}
    for name, make in builders.items():
        pre = make()
        approximation = pre.approximation(operator)
        whitened = whitened_operator(operator, approximation).as_matrix()
        assert jnp.allclose(whitened, whitened.T, atol=1e-6), name
        got = jnp.linalg.slogdet(whitened)[1] + pre.logdet(operator)
        # float32 round-off on a 256-point determinant is ~1e-4 relative
        rtol = 1e-8 if dense.dtype == jnp.float64 else 1e-3
        assert jnp.allclose(got, exact, rtol=rtol), name


def test_whitening_falls_back_to_the_contour_root():
    """A P with no closed-form factor is whitened by P^(−½)."""
    A = jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    P = jnp.array([[3.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 2.0]])
    whitened = whitened_operator(
        lx.MatrixLinearOperator(A, _PSD), lx.MatrixLinearOperator(P, _PSD)
    ).as_matrix()
    expected = jnp.linalg.slogdet(A)[1] - jnp.linalg.slogdet(P)[1]
    # contour quadrature to ~1e-5 relative (gaussx.sqrt_inv_matmul)
    assert jnp.allclose(jnp.linalg.slogdet(whitened)[1], expected, atol=1e-4)


def test_exact_preconditioner_has_no_probe_error():
    """P = A whitens to the identity: zero standard error."""
    diagonal = jnp.array([1.0, 2.0, 4.0, 8.0])
    A = lx.DiagonalLinearOperator(diagonal)
    pre = gaussx.JacobiPreconditioner(diagonal=diagonal)
    estimate, error = gaussx.SLQLogdet(preconditioner=pre).logdet_and_error(A)
    assert jnp.allclose(estimate, jnp.sum(jnp.log(diagonal)), rtol=1e-5)
    assert error < 1e-5


def test_preconditioner_without_p_raises():
    opaque = gaussx.OperatorPreconditioner(lambda v: v / 2.0)
    operator, _ = _system(n=64)
    with pytest.raises(ValueError, match="OperatorPreconditioner"):
        gaussx.SLQLogdet(preconditioner=opaque).logdet(operator)
    with pytest.raises(ValueError, match="supports_slq"):
        gaussx.SLQLogdet().with_preconditioner(opaque)


def test_none_keeps_todays_estimate(getkey):
    """``preconditioner=None`` is the plain estimator, bit for bit."""
    operator, _ = _system(n=64)
    key = getkey()
    plain = gaussx.SLQLogdet(num_probes=8).logdet(operator, key=key)
    explicit = gaussx.SLQLogdet(num_probes=8, preconditioner=None).logdet(
        operator, key=key
    )
    assert plain == explicit


@pytest.mark.slow
def test_fitc_reduces_the_spread_across_keys():
    """Sample std over 20 keys is smaller with FITC, at equal probes and order."""
    operator, _ = _system()
    keys = jr.split(jr.key(0), 20)
    plain = gaussx.SLQLogdet(num_probes=20)
    preconditioned = gaussx.SLQLogdet(
        num_probes=20, preconditioner=_PRECONDITIONERS["fitc"]()
    )
    plain_spread = jnp.std(jnp.stack([plain.logdet(operator, key=k) for k in keys]))
    spread = jnp.std(jnp.stack([preconditioned.logdet(operator, key=k) for k in keys]))
    assert spread < plain_spread, (spread, plain_spread)


@pytest.mark.slow
def test_jit_vmap_grad_in_the_lengthscale():
    """d log det A / dℓ through the whitened SLQ: finite and near the exact one."""
    pre = gaussx.NystromPreconditioner.lazy(60, key=jr.key(0))
    strategy = gaussx.SLQLogdet(num_probes=40, preconditioner=pre)

    def estimate(lengthscale, key):
        return strategy.logdet(_system(lengthscale)[0], key=key)

    def exact(lengthscale):
        return jnp.linalg.slogdet(_system(lengthscale)[1])[1]

    lengthscale = jnp.asarray(0.3, dtype=_X.dtype)
    got = jax.jit(jax.grad(estimate))(lengthscale, jr.key(1))
    expected = jax.grad(exact)(lengthscale)
    assert jnp.isfinite(got)
    # The gradient keeps a Hutchinson error a preconditioner does not remove
    # (gh-826 measured 1-5 % at 50 probes); 15 % is about 5 of its σ here.
    assert jnp.allclose(got, expected, rtol=0.15), (got, expected)
    values = jax.vmap(estimate, in_axes=(None, 0))(lengthscale, jr.split(jr.key(2), 3))
    assert jnp.all(jnp.abs(values - exact(lengthscale)) < 0.5)
