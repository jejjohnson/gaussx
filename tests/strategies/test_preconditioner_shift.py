"""Per-point shifts by symmetric scaling (gh-878).

A system A = K + diag(d) is scaled to B = D^(−½) A D^(−½) = K_B + I, and a
unit-shift preconditioner P_B of B is pulled back to P = D^½ P_B D^½. Every
check is against a densified P, A or log det A. The file is in the float32
lane (Makefile ``NO_X64_TESTS``); the efficacy and accuracy cases of the
issue are float64 measurements and are marked ``x64_only``.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import NystromPreconditioner, PartialCholeskyPreconditioner
from gaussx._einx import rearrange
from gaussx._testing import tree_allclose


_PSD = lx.positive_semidefinite_tag
_SYMMETRIC_PSD = (lx.symmetric_tag, _PSD)


def _x64() -> bool:
    return jnp.zeros(()).dtype == jnp.float64


def _matern32(inputs, variance, lengthscale):
    """Matérn-3/2 Gram matrix of points ``(n, dim)``, with a 1e-6 jitter."""
    gaps = einx.subtract("row dim, col dim -> row col dim", inputs, inputs)
    distance = jnp.sqrt(einx.sum("row col dim -> row col", gaps**2) + 1e-30)
    scaled = jnp.sqrt(3.0) * distance / lengthscale
    jitter = 1e-6 * jnp.eye(inputs.shape[0], dtype=inputs.dtype)
    return variance * (1.0 + scaled) * jnp.exp(-scaled) + jitter


def _site_system(kernel, shift):
    """A = K + diag(d), tagged symmetric PSD, split by `split_diagonal`."""
    return lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel) + lx.DiagonalLinearOperator(shift),
        _SYMMETRIC_PSD,
    )


def _problem(key, size=40, decades=3.0):
    """A rough Matérn-3/2 kernel on [0, 1] and d spanning ``decades``."""
    input_key, shift_key = jr.split(key)
    points = jnp.sort(jr.uniform(input_key, (size,)))
    inputs = rearrange(points, "point -> point dim", dim=1)
    kernel = _matern32(inputs, 10.0, 0.1)
    exponent = jr.uniform(shift_key, (size,), minval=-1.0, maxval=decades - 1.0)
    shift = 10.0**exponent
    return kernel, shift


def _preconditioner(name, kernel, shift, rank, key):
    kernel_op = lx.MatrixLinearOperator(kernel, _PSD)
    if name == "partial_cholesky_built":
        return PartialCholeskyPreconditioner.from_operator(kernel_op, rank, shift=shift)
    if name == "partial_cholesky_vector":
        return PartialCholeskyPreconditioner(rank=rank, shift=shift)
    if name == "partial_cholesky_none":
        return PartialCholeskyPreconditioner(rank=rank, shift=None)
    if name == "nystrom_built":
        return NystromPreconditioner.from_operator(
            kernel_op, rank, shift=shift, key=key
        )
    if name == "nystrom_lazy":
        return NystromPreconditioner.lazy(rank, key=key)
    raise ValueError(name)


_NAMES = [
    "partial_cholesky_built",
    "partial_cholesky_vector",
    "partial_cholesky_none",
    "nystrom_built",
    "nystrom_lazy",
]


@pytest.mark.parametrize("name", _NAMES)
def test_inverse_matches_the_dense_inverse_of_p(getkey, name):
    kernel, shift = _problem(getkey())
    system = _site_system(kernel, shift)
    pre = _preconditioner(name, kernel, shift, 10, getkey())
    vector = jr.normal(getkey(), (kernel.shape[0],))
    dense = pre.approximation(system).as_matrix()
    # P⁻¹ v against the dense solve with P, P spanning three decades of d
    applied = pre.as_operator(system).mv(vector)
    rtol = 1e-6 if _x64() else 1e-3
    assert tree_allclose(applied, jnp.linalg.solve(dense, vector), rtol=rtol)
    # log det P against the dense P
    assert tree_allclose(
        pre.logdet(system), jnp.linalg.slogdet(dense)[1], rtol=rtol, atol=rtol
    )


@pytest.mark.x64_only(reason="the issue's identity check is at 1e-6")
@pytest.mark.parametrize("name", _NAMES)
def test_full_rank_preconditions_to_a_scaled_identity(getkey, name):
    kernel, shift = _problem(getkey(), size=24)
    size = kernel.shape[0]
    system = _site_system(kernel, shift)
    pre = _preconditioner(name, kernel, shift, size, getkey())
    # P⁻¹ A, the identity for partial Cholesky and (γ̂_k + 1) I for Nyström
    product = pre.as_operator(system).as_matrix() @ system.as_matrix()
    scale = jnp.trace(product) / size
    assert tree_allclose(product / scale, jnp.eye(size), rtol=0.0, atol=1e-6)
    if name.startswith("partial_cholesky"):
        assert tree_allclose(scale, jnp.asarray(1.0), rtol=1e-6)
    solver = gaussx.CGSolver(preconditioner=pre, rtol=1e-8, atol=1e-8)
    _, info = gaussx.linear_solve_with_info(system, jnp.ones(size), solver=solver)
    assert int(info.iterations) <= 2


def _rough_case(key):
    """The issue's rough case: N = 400 in the unit square, variance 10,
    lengthscale 0.1, d over the measured range [4, 71] of a Bernoulli
    Laplace fit."""
    input_key, shift_key = jr.split(key)
    inputs = jr.uniform(input_key, (400, 2))
    kernel = _matern32(inputs, 10.0, 0.1)
    # log-uniform on [4, 71]
    shift = 4.0 * (71.0 / 4.0) ** jr.uniform(shift_key, (400,))
    return kernel, shift


@pytest.mark.slow
@pytest.mark.x64_only(reason="the issue's measurements are float64")
def test_scaled_partial_cholesky_beats_the_minimum_shift():
    kernel, shift = _rough_case(jr.key(0))
    system = _site_system(kernel, shift)
    rhs = jnp.ones(400)

    def steps(preconditioner):
        solver = gaussx.CGSolver(preconditioner=preconditioner, rtol=1e-6, atol=1e-6)
        _, info = gaussx.linear_solve_with_info(system, rhs, solver=solver)
        return int(info.iterations)

    scalar = steps(PartialCholeskyPreconditioner(rank=150, shift=float(shift.min())))
    scaled = steps(PartialCholeskyPreconditioner(rank=150, shift=None))
    # the issue's bar: at most 60 % (measured there: 8 against 29)
    assert scaled <= 0.6 * scalar


@pytest.mark.slow
@pytest.mark.x64_only(reason="the issue's measurements are float64")
def test_nystrom_logdet_shift_none_on_the_smooth_case():
    # The smooth case: variance 1, lengthscale 0.3, d over [4, 10.9]
    input_key, shift_key = jr.split(jr.key(1))
    inputs = jr.uniform(input_key, (400, 2))
    kernel = _matern32(inputs, 1.0, 0.3)
    shift = 4.0 + 6.9 * jr.uniform(shift_key, (400,))
    system = _site_system(kernel, shift)
    exact = jnp.linalg.slogdet(system.as_matrix())[1]
    strategy = gaussx.NystromLogdet(shift=None, rank=150, num_probes=50)
    keys = jr.split(jr.key(2), 10)
    estimates = jax.vmap(lambda key: strategy.logdet(system, key=key))(keys)
    rms = jnp.sqrt(jnp.mean((estimates - exact) ** 2))
    # the issue's bar: 0.05 nats (measured there: 0.009 at rank 150)
    assert float(rms) < 0.05


def test_nystrom_logdet_vector_and_none_agree(getkey):
    kernel, shift = _problem(getkey(), size=30, decades=2.0)
    system = _site_system(kernel, shift)
    key = getkey()
    from_system = gaussx.NystromLogdet(shift=None, rank=30).logdet(system, key=key)
    given = gaussx.NystromLogdet(shift=shift, rank=30).logdet(system, key=key)
    assert tree_allclose(from_system, given)
    # full rank: the remainder is log det of a near-identity, so it is exact
    exact = jnp.linalg.slogdet(system.as_matrix())[1]
    rtol = 1e-6 if _x64() else 1e-3
    assert tree_allclose(from_system, exact, rtol=rtol)


def test_nystrom_logdet_scalar_shift_with_noise_inside_the_kernel_part(getkey):
    # A = (K + 0.5 I) + diag(1e-6): the shift 0.5 exceeds the diagonal that
    # split_diagonal sees, yet A − 0.5 I = K + 1e-6 I is PSD, so it must run.
    kernel, _ = _problem(getkey(), size=20)
    size = kernel.shape[0]
    system = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel + 0.5 * jnp.eye(size))
        + lx.DiagonalLinearOperator(jnp.full((size,), 1e-6, kernel.dtype)),
        _SYMMETRIC_PSD,
    )
    estimate = gaussx.NystromLogdet(shift=0.5, rank=size).logdet(system, key=getkey())
    exact = jnp.linalg.slogdet(system.as_matrix())[1]
    rtol = 1e-6 if _x64() else 1e-3
    assert tree_allclose(estimate, exact, rtol=rtol)


@pytest.mark.slow
def test_nystrom_logdet_non_finite_estimate_raises(getkey):
    # a dense system that does not split: only the result can be checked
    kernel, _ = _problem(getkey(), size=20)
    system = lx.MatrixLinearOperator(kernel + 0.1 * jnp.eye(20), _SYMMETRIC_PSD)
    with pytest.raises(Exception, match="not finite"):
        gaussx.NystromLogdet(shift=50.0, rank=5).logdet(system)


@pytest.mark.parametrize(
    "build",
    [
        lambda shift: PartialCholeskyPreconditioner(rank=4, shift=shift),
        lambda shift: NystromPreconditioner.from_operator(
            lx.MatrixLinearOperator(jnp.eye(6), _PSD), 4, shift=shift
        ),
        lambda shift: gaussx.NystromLogdet(shift=shift, rank=4),
    ],
    ids=["partial_cholesky", "nystrom", "nystrom_logdet"],
)
def test_non_positive_shift_raises(getkey, build):
    kernel, _ = _problem(getkey(), size=6)
    shift = jnp.array([1.0, 0.5, 0.0, 2.0, 1.0, 1.0])
    system = _site_system(kernel, shift)
    with pytest.raises(Exception, match="must be positive"):
        target = build(shift)
        if isinstance(target, gaussx.NystromLogdet):
            target.logdet(system)
        else:
            target.as_operator(system)


def test_shift_none_on_a_system_that_does_not_split_raises(getkey):
    kernel, _ = _problem(getkey(), size=8)
    system = lx.MatrixLinearOperator(kernel + jnp.eye(8), _SYMMETRIC_PSD)
    with pytest.raises(ValueError, match="split_diagonal"):
        PartialCholeskyPreconditioner(rank=4, shift=None).as_operator(system)
    with pytest.raises(ValueError, match="split_diagonal"):
        NystromPreconditioner.lazy(4).as_operator(system)
    with pytest.raises(ValueError, match="split_diagonal"):
        gaussx.NystromLogdet(shift=None, rank=4).logdet(system)


def test_lazy_nystrom_needs_a_system():
    with pytest.raises(ValueError, match="system operator"):
        NystromPreconditioner.lazy(4).as_operator()
    with pytest.raises(ValueError, match="rank"):
        NystromPreconditioner.lazy(0)


@pytest.mark.parametrize("name", ["partial_cholesky", "nystrom"])
def test_constant_vector_shift_matches_the_scalar_shift(getkey, name):
    # d_n = σ² for all n: the scaled form is the scalar form, in exact
    # arithmetic
    kernel, _ = _problem(getkey(), size=20)
    noise = 0.3
    system = lx.MatrixLinearOperator(kernel + noise * jnp.eye(20), _SYMMETRIC_PSD)
    vector = jnp.full(20, noise)
    kernel_op = lx.MatrixLinearOperator(kernel, _PSD)
    key = getkey()
    if name == "partial_cholesky":
        scalar = PartialCholeskyPreconditioner.from_operator(kernel_op, 6, shift=noise)
        scaled = PartialCholeskyPreconditioner.from_operator(kernel_op, 6, shift=vector)
    else:
        scalar = NystromPreconditioner.from_operator(kernel_op, 6, shift=noise, key=key)
        scaled = NystromPreconditioner.from_operator(
            kernel_op, 6, shift=vector, key=key
        )
    rtol = 1e-8 if _x64() else 1e-3
    # Nyström's P is A only up to a scale: (γ̂_k + σ²) for the scalar form
    # and (γ̂_k + σ²) / σ² for the scaled one, so their inverses differ by σ²
    scale = 1.0 if name == "partial_cholesky" else noise
    assert tree_allclose(
        scale * scaled.as_operator(system).as_matrix(),
        scalar.as_operator(system).as_matrix(),
        rtol=rtol,
        atol=rtol,
    )
    if name == "partial_cholesky":
        # the same M, so the same log det M
        assert tree_allclose(scaled.logdet(system), scalar.logdet(system), rtol=rtol)


def test_scalar_shift_keeps_todays_code_path(getkey):
    # A scalar shift never builds the scaled wrapper, and a Python float and
    # a 0-d array give identical bits.
    kernel, _ = _problem(getkey(), size=20)
    system = lx.MatrixLinearOperator(kernel + 0.3 * jnp.eye(20), _SYMMETRIC_PSD)
    vector = jr.normal(getkey(), (20,))
    as_float = PartialCholeskyPreconditioner(rank=5, shift=0.3)
    as_array = PartialCholeskyPreconditioner(rank=5, shift=jnp.asarray(0.3))
    assert as_float._scaled(system) is None
    assert bool(
        jnp.all(
            as_float.as_operator(system).mv(vector)
            == as_array.as_operator(system).mv(vector)
        )
    )


def _lengthscale_system(lengthscale, shift):
    inputs = jnp.linspace(0.0, 1.0, shift.shape[0], dtype=shift.dtype)
    kernel = _matern32(rearrange(inputs, "point -> point dim", dim=1), 1.0, lengthscale)
    return _site_system(kernel, shift), kernel


@pytest.mark.parametrize(
    "name", [pytest.param("partial_cholesky", marks=pytest.mark.slow), "nystrom"]
)
def test_jit_vmap_grad_in_d_and_the_lengthscale(getkey, name):
    size = 16
    shift = jr.uniform(getkey(), (size,), minval=0.1, maxval=2.0)
    rhs = jr.normal(getkey(), (size,))
    if name == "partial_cholesky":
        pre = PartialCholeskyPreconditioner(rank=5, shift=None)
    else:
        pre = NystromPreconditioner.lazy(5, key=getkey())
    tight = 1e-10 if _x64() else 1e-6
    solver = gaussx.CGSolver(preconditioner=pre, rtol=tight, atol=tight)

    def objective(shift, lengthscale):
        system, _ = _lengthscale_system(lengthscale, shift)
        return jnp.sum(solver.solve(system, rhs))

    def dense_objective(shift, lengthscale):
        _, kernel = _lengthscale_system(lengthscale, shift)
        return jnp.sum(jnp.linalg.solve(kernel + jnp.diag(shift), rhs))

    lengthscale = jnp.asarray(0.3)
    rtol = 1e-6 if _x64() else 2e-3
    value = jax.jit(objective)(shift, lengthscale)
    assert tree_allclose(value, dense_objective(shift, lengthscale), rtol=rtol)
    # the preconditioner's own gradient is stopped, so the gradient is the
    # exact one of A⁻¹ b
    gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(shift, lengthscale)
    expected = jax.grad(dense_objective, argnums=(0, 1))(shift, lengthscale)
    assert tree_allclose(gradient, expected, rtol=rtol, atol=rtol)
    shifts = einx.multiply("batch, point -> batch point", jnp.array([1.0, 2.0]), shift)
    batched = jax.vmap(objective, in_axes=(0, None))(shifts, lengthscale)
    looped = jnp.stack([objective(row, lengthscale) for row in shifts])
    assert tree_allclose(batched, looped, rtol=rtol)


@pytest.mark.slow
def test_nystrom_logdet_grad_in_d(getkey):
    size = 16
    shift = jr.uniform(getkey(), (size,), minval=0.1, maxval=2.0)
    key = getkey()

    def estimate(shift):
        system, _ = _lengthscale_system(jnp.asarray(0.3, dtype=shift.dtype), shift)
        return gaussx.NystromLogdet(shift=None, rank=size).logdet(system, key=key)

    def exact(shift):
        system, _ = _lengthscale_system(jnp.asarray(0.3, dtype=shift.dtype), shift)
        return jnp.linalg.slogdet(system.as_matrix())[1]

    gradient = jax.jit(jax.grad(estimate))(shift)
    rtol = 1e-4 if _x64() else 5e-2
    # full rank: log det B is exact, so its gradient is too
    assert tree_allclose(gradient, jax.grad(exact)(shift), rtol=rtol, atol=rtol)


def test_preconditioned_cg_solver_takes_a_per_point_shift(getkey):
    kernel, shift = _problem(getkey(), size=30)
    system = _site_system(kernel, shift)
    rhs = jr.normal(getkey(), (30,))
    expected = jnp.linalg.solve(kernel + jnp.diag(shift), rhs)
    rtol = 1e-4 if _x64() else 1e-2
    for value in (None, shift):
        solver = gaussx.PreconditionedCGSolver(preconditioner_rank=8, shift=value)
        assert tree_allclose(solver.solve(system, rhs), expected, rtol=rtol, atol=rtol)
