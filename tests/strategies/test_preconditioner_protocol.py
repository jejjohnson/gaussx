"""Tests for P itself on the preconditioners: logdet, sample, approximation (gh-823).

`as_operator` returns M⁻¹ = P⁻¹; `logdet`, `sample` and `approximation` are
about P ≈ A. Every comparison here is against a densified P or M⁻¹, in both
directions, so the sign convention is pinned. The file is in the float32 lane
(Makefile ``NO_X64_TESTS``); `test_preconditioners.py` is not, since 14 of
its older cases need float64.
"""

from __future__ import annotations

import contextlib

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import (
    JacobiPreconditioner,
    NystromPreconditioner,
    OperatorPreconditioner,
    PartialCholeskyPreconditioner,
)
from gaussx._testing import (
    assert_sample_moments,
    dense_logdet,
    random_pd_matrix,
    tree_allclose,
)


@contextlib.contextmanager
def _forbid_as_matrix(cls):
    """Fail if any instance of ``cls`` is materialised inside the block."""
    original = cls.as_matrix

    def _forbidden(self):
        raise AssertionError(f"{cls.__name__}.as_matrix called")

    cls.as_matrix = _forbidden
    try:
        yield
    finally:
        cls.as_matrix = original


def _matrix_and_operator(key, n):
    mat = random_pd_matrix(key, n)
    return mat, lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)


_NOISE = 0.3
_SLQ_CASES = [
    "jacobi_explicit",
    "jacobi_lazy",
    "partial_cholesky_built",
    "partial_cholesky_lazy",
    "nystrom",
]


def _slq_case(name, key, n=8, rank=3):
    """A preconditioner of A = K + σ² I and the system A it is used on."""
    kernel_key, sketch_key = jr.split(key)
    kernel = random_pd_matrix(kernel_key, n)
    psd = lx.positive_semidefinite_tag
    kernel_op = lx.MatrixLinearOperator(kernel, psd)
    system = lx.MatrixLinearOperator(kernel + _NOISE * jnp.eye(n), psd)
    if name == "jacobi_explicit":
        pre = JacobiPreconditioner(diagonal=jnp.diag(system.as_matrix()))
    elif name == "jacobi_lazy":
        pre = JacobiPreconditioner()
    elif name == "partial_cholesky_built":
        pre = PartialCholeskyPreconditioner.from_operator(kernel_op, rank, shift=_NOISE)
    elif name == "partial_cholesky_lazy":
        pre = PartialCholeskyPreconditioner(rank=rank, shift=_NOISE)
    elif name == "nystrom":
        pre = NystromPreconditioner.from_operator(
            kernel_op, rank, shift=_NOISE, key=sketch_key
        )
    else:
        raise ValueError(name)
    return pre, system


def _tolerance(dtype):
    """rtol for comparisons of P against M⁻¹: float32 rounding on an 8 × 8 P."""
    return 1e-4 if dtype == jnp.float32 else 1e-8


def test_logdet_is_the_preconditioner_not_its_inverse(getkey):
    mat, op = _matrix_and_operator(getkey(), 6)
    pre = JacobiPreconditioner(diagonal=jnp.diag(mat))
    # log det P = ∑ log A_nn, and −log det M⁻¹ is the same number
    assert tree_allclose(pre.logdet(op), jnp.sum(jnp.log(jnp.diag(mat))), rtol=1e-5)
    assert tree_allclose(pre.logdet(op), -dense_logdet(pre.as_operator(op)), rtol=1e-5)


@pytest.mark.parametrize("name", _SLQ_CASES)
def test_logdet_matches_the_dense_p_in_both_directions(getkey, name):
    pre, system = _slq_case(name, getkey())
    rtol = _tolerance(system.in_structure().dtype)
    logdet = pre.logdet(system)
    # log det P against the densified P
    assert tree_allclose(logdet, dense_logdet(pre.approximation(system)), rtol=rtol)
    # and against −log det M⁻¹, which pins the sign convention
    inverse = pre.as_operator(system)
    assert tree_allclose(logdet, -dense_logdet(inverse), rtol=rtol)


@pytest.mark.parametrize("name", _SLQ_CASES)
def test_approximation_is_the_inverse_of_as_operator(getkey, name):
    pre, system = _slq_case(name, getkey())
    size = system.in_size()
    dtype = system.in_structure().dtype
    approximation = pre.approximation(system)
    assert lx.is_positive_semidefinite(approximation)
    # M⁻¹ P = I
    product = pre.as_operator(system).as_matrix() @ approximation.as_matrix()
    tolerance = 1e-6 if dtype == jnp.float64 else 1e-4
    assert tree_allclose(product, jnp.eye(size, dtype=dtype), rtol=0, atol=tolerance)


@pytest.mark.parametrize("name", _SLQ_CASES)
def test_approximation_solve_and_logdet_never_densify(getkey, name):
    pre, system = _slq_case(name, getkey())
    approximation = pre.approximation(system)
    dense = approximation.as_matrix()
    rhs = jr.normal(getkey(), (system.in_size(),), dtype=dense.dtype)
    rtol = _tolerance(dense.dtype)
    with (
        _forbid_as_matrix(gaussx.LowRankUpdate),
        _forbid_as_matrix(lx.DiagonalLinearOperator),
    ):
        solution = gaussx.solve(approximation, rhs)
        logdet = gaussx.logdet(approximation)
    assert tree_allclose(solution, jnp.linalg.solve(dense, rhs), rtol=rtol)
    assert tree_allclose(logdet, pre.logdet(system), rtol=rtol)


@pytest.mark.parametrize("name", _SLQ_CASES)
def test_sample_shape_and_dtype(getkey, name):
    pre, system = _slq_case(name, getkey())
    dtype = system.in_structure().dtype
    assert pre.sample(getkey(), system).shape == (system.in_size(),)
    draws = pre.sample(getkey(), system, shape=(4, 3))
    assert draws.shape == (4, 3, system.in_size())
    assert draws.dtype == dtype


@pytest.mark.slow
@pytest.mark.parametrize("name", _SLQ_CASES)
def test_sample_moments_match_p(getkey, name):
    pre, system = _slq_case(name, getkey())
    covariance = pre.approximation(system).as_matrix()
    draws = pre.sample(getkey(), system, shape=(20_000,))
    # bound each moment by its own sampling distribution (gh-220)
    assert_sample_moments(draws, jnp.zeros(system.in_size()), covariance)


@pytest.mark.parametrize("name", _SLQ_CASES)
def test_supports_slq_for_the_three_known_concretions(getkey, name):
    pre, system = _slq_case(name, getkey())
    assert pre.supports_slq() is True
    assert pre.supports_slq(system) is True


def test_operator_preconditioner_does_not_know_p(getkey):
    _, op = _matrix_and_operator(getkey(), 4)
    pre = OperatorPreconditioner(lambda v: v / 2.0)
    assert pre.supports_slq(op) is False
    with pytest.raises(NotImplementedError, match="supports_slq"):
        pre.logdet(op)
    with pytest.raises(NotImplementedError, match="supports_slq"):
        pre.sample(getkey(), op)
    with pytest.raises(NotImplementedError, match="supports_slq"):
        pre.approximation(op)


def test_disabled_partial_cholesky_is_the_identity(getkey):
    _, op = _matrix_and_operator(getkey(), 4)
    pre = PartialCholeskyPreconditioner(rank=0)
    assert pre.as_operator(op) is None
    assert float(pre.logdet(op)) == 0.0
    assert tree_allclose(pre.approximation(op).as_matrix(), jnp.eye(4))
    assert pre.sample(getkey(), op, shape=(2,)).shape == (2, 4)
    with pytest.raises(ValueError, match="size"):
        pre.approximation()


def _kernel_system(lengthscale, n=12):
    """A = K(ℓ) + σ² I for a squared-exponential K on a fixed grid."""
    inputs = jnp.linspace(0.0, 3.0, n, dtype=lengthscale.dtype)
    gaps = einx.subtract("row, col -> row col", inputs, inputs)
    kernel = jnp.exp(-0.5 * (gaps / lengthscale) ** 2)
    psd = lx.positive_semidefinite_tag
    noise = _NOISE * jnp.eye(n, dtype=kernel.dtype)
    return (
        lx.MatrixLinearOperator(kernel, psd),
        lx.MatrixLinearOperator(kernel + noise, psd),
    )


def _preconditioner_of(name, lengthscale):
    kernel_op, system = _kernel_system(lengthscale)
    if name == "jacobi":
        return JacobiPreconditioner(), system
    if name == "partial_cholesky":
        built = PartialCholeskyPreconditioner.from_operator(kernel_op, 4, shift=_NOISE)
        return built, system
    return NystromPreconditioner.from_operator(
        kernel_op, 4, shift=_NOISE, key=jr.key(0)
    ), system


@pytest.mark.slow
@pytest.mark.parametrize("name", ["jacobi", "partial_cholesky", "nystrom"])
def test_logdet_jit_grad_vmap_in_the_lengthscale(name):
    def structured(lengthscale):
        pre, system = _preconditioner_of(name, lengthscale)
        return pre.logdet(system)

    def dense(lengthscale):
        pre, system = _preconditioner_of(name, lengthscale)
        # log det of the densified P, by LU
        return jnp.linalg.slogdet(pre.approximation(system).as_matrix())[1]

    lengthscale = jnp.asarray(0.7)
    rtol = _tolerance(lengthscale.dtype)
    assert tree_allclose(
        jax.jit(structured)(lengthscale), dense(lengthscale), rtol=rtol
    )
    gradient = jax.jit(jax.grad(structured))(lengthscale)
    assert bool(jnp.isfinite(gradient))
    assert tree_allclose(gradient, jax.grad(dense)(lengthscale), rtol=rtol, atol=rtol)
    lengthscales = jnp.array([0.5, 0.7, 1.1])
    batched = jax.vmap(structured)(lengthscales)
    looped = jnp.stack([structured(value) for value in lengthscales])
    assert tree_allclose(batched, looped, rtol=rtol)
