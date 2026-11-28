"""Tests for site_posterior: the Gaussian posterior of diagonal sites (gh-882).

Every case is checked against the dense formulas

    w = A⁻¹ (μ_site − μ_f),   μ_f|y = μ_f + K w,   Σ_f|y = K − K A⁻¹ K,

with A = K + diag(1 / λ). The file is in the float32 lane (Makefile
``NO_X64_TESTS``); its float32 accuracy test takes a NumPy float64 reference,
so it runs in both lanes.
"""

from __future__ import annotations

import warnings

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._testing import tree_allclose


_N = 12
_PSD = lx.positive_semidefinite_tag


def _matern32(inputs, lengthscale):
    """Matérn-3/2 Gram matrix, unit variance, with a 1e-6 jitter."""
    gaps = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))
    scaled = jnp.sqrt(3.0) * gaps / lengthscale
    jitter = 1e-6 * jnp.eye(inputs.shape[0], dtype=inputs.dtype)
    return (1.0 + scaled) * jnp.exp(-scaled) + jitter


def _problem(key, n=_N, lengthscale=0.4):
    """A prior (μ_f, K) on a sorted grid and positive sites (Λμ, λ)."""
    input_key, mean_key, site_key, precision_key = jr.split(key, 4)
    inputs = jnp.sort(jr.uniform(input_key, (n,)))
    kernel = _matern32(inputs, lengthscale)
    prior_mean = 0.1 * jr.normal(mean_key, (n,))
    site_mean = jr.normal(site_key, (n,))
    precision = jr.uniform(precision_key, (n,), minval=0.5, maxval=4.0)
    return prior_mean, kernel, precision * site_mean, precision


def _dense_reference(prior_mean, kernel, information, precision):
    """The dense formulas: weights, mean and marginal variances."""
    system = kernel + jnp.diag(1.0 / precision)
    # w = A⁻¹ (μ_site − μ_f)
    weights = jnp.linalg.solve(system, information / precision - prior_mean)
    # μ_f|y = μ_f + K w
    mean = prior_mean + kernel @ weights
    # diag(K − K A⁻¹ K)
    variance = jnp.diag(kernel - kernel @ jnp.linalg.solve(system, kernel))
    return weights, mean, variance


def _solvers():
    """Each solver path, with the tolerance its result is compared at."""
    x64 = jnp.zeros(()).dtype == jnp.float64
    # dense: rtol 1e-8 in float64 (the issue's bar), float32 rounding otherwise
    exact = 1e-8 if x64 else 1e-4
    # iterative: solved to 1e-10 (float64) or 1e-6 (float32), compared at the
    # accuracy that buys on these systems of condition number ~1e2
    tight = 1e-10 if x64 else 1e-6
    loose = 1e-6 if x64 else 2e-3
    jacobi = gaussx.JacobiPreconditioner()
    return {
        "none": (None, exact),
        "dense": (gaussx.DenseSolver(), exact),
        "cg": (gaussx.CGSolver(rtol=tight, atol=tight), loose),
        "preconditioned_cg": (
            gaussx.PreconditionedCGSolver(
                preconditioner=jacobi, rtol=tight, atol=tight
            ),
            loose,
        ),
        "bbmm": (gaussx.BBMMSolver(rtol=tight, atol=tight), loose),
    }


_SOLVER_NAMES = [
    "none",
    "dense",
    # ~2 s warm; the fast tier keeps CG through the matrix-free test
    pytest.param("cg", marks=pytest.mark.slow),
    "preconditioned_cg",
    "bbmm",
]


@pytest.mark.parametrize("name", _SOLVER_NAMES)
def test_matches_the_dense_formulas(getkey, name):
    prior_mean, kernel, information, precision = _problem(getkey())
    solver, rtol = _solvers()[name]
    prior_cov = lx.MatrixLinearOperator(kernel, _PSD)

    posterior = gaussx.site_posterior(
        prior_mean, prior_cov, information, precision, solver=solver
    )

    weights, mean, variance = _dense_reference(
        prior_mean, kernel, information, precision
    )
    assert tree_allclose(posterior.weights, weights, rtol=rtol, atol=rtol)
    assert tree_allclose(posterior.mean, mean, rtol=rtol, atol=rtol)
    batched = posterior.variance(method="exact", batch_size=5)
    assert tree_allclose(batched, variance, rtol=rtol, atol=rtol)
    assert tree_allclose(posterior.variance(), variance, rtol=rtol, atol=rtol)
    # the lazy covariance has the same diagonal
    covariance = posterior.covariance().as_matrix()
    assert tree_allclose(jnp.diag(covariance), variance, rtol=rtol, atol=rtol)
    assert lx.is_positive_semidefinite(posterior.system)


def test_gaussian_sites_are_exact_gp_regression(getkey):
    prior_mean, kernel, _, _ = _problem(getkey())
    y = jr.normal(getkey(), (_N,))
    noise_variance = 0.2
    # λ = σ⁻², μ_site = y
    precision = jnp.full(_N, 1.0 / noise_variance)
    posterior = gaussx.site_posterior(
        prior_mean, lx.MatrixLinearOperator(kernel, _PSD), precision * y, precision
    )
    # GP regression: μ + K (K + σ² I)⁻¹ (y − μ), diag(K − K (K + σ² I)⁻¹ K)
    regression = kernel + noise_variance * jnp.eye(_N)
    mean = prior_mean + kernel @ jnp.linalg.solve(regression, y - prior_mean)
    variance = jnp.diag(kernel - kernel @ jnp.linalg.solve(regression, kernel))
    assert tree_allclose(posterior.mean, mean)
    assert tree_allclose(posterior.variance(), variance)


def test_variances_lie_between_zero_and_the_prior(getkey):
    prior_mean, kernel, information, precision = _problem(getkey())
    posterior = gaussx.site_posterior(
        prior_mean, lx.MatrixLinearOperator(kernel, _PSD), information, precision
    )
    variance = posterior.variance()
    assert bool(jnp.all(variance >= 0.0))
    assert bool(jnp.all(variance <= jnp.diag(kernel)))


def test_permuting_the_points_permutes_the_result(getkey):
    prior_mean, kernel, information, precision = _problem(getkey())
    order = jr.permutation(getkey(), _N)
    posterior = gaussx.site_posterior(
        prior_mean, lx.MatrixLinearOperator(kernel, _PSD), information, precision
    )
    permuted_kernel = kernel[order][:, order]
    permuted = gaussx.site_posterior(
        prior_mean[order],
        lx.MatrixLinearOperator(permuted_kernel, _PSD),
        information[order],
        precision[order],
    )
    assert tree_allclose(permuted.mean, posterior.mean[order])
    assert tree_allclose(permuted.variance(), posterior.variance()[order])


def test_prior_quadratic_form_needs_no_solve_against_k(getkey):
    prior_mean, kernel, information, precision = _problem(getkey())
    posterior = gaussx.site_posterior(
        prior_mean, lx.MatrixLinearOperator(kernel, _PSD), information, precision
    )
    shift = posterior.mean - prior_mean
    # (μ_f|y − μ_f)ᵀ K⁻¹ (μ_f|y − μ_f) = wᵀ K w
    by_solve = shift @ jnp.linalg.solve(kernel, shift)
    by_weights = posterior.weights @ (kernel @ posterior.weights)
    assert tree_allclose(by_weights, by_solve, rtol=1e-3)


class _NoDensify(lx.FunctionLinearOperator):
    """A matrix-free prior that fails the test if anything densifies it."""

    def as_matrix(self):
        raise AssertionError("as_matrix called on the matrix-free prior")


def _matrix_free_prior(key, n):
    """K = U Uᵀ + 0.1 I applied by matvecs, with no N × N array anywhere."""
    factor = jr.normal(key, (n, 3))

    def matvec(vector):
        # K v = U (Uᵀ v) + 0.1 v
        coefficients = einx.dot("row rank, row -> rank", factor, vector)
        return einx.dot("row rank, rank -> row", factor, coefficients) + 0.1 * vector

    structure = jax.ShapeDtypeStruct((n,), factor.dtype)
    dense = factor @ factor.T + 0.1 * jnp.eye(n, dtype=factor.dtype)
    return _NoDensify(matvec, structure, _PSD), dense


def _shapes_in(jaxpr):
    """Every variable shape in a closed jaxpr, sub-jaxprs included."""
    shapes = set()
    for equation in jaxpr.eqns:
        for variable in (*equation.invars, *equation.outvars):
            shape = getattr(getattr(variable, "aval", None), "shape", None)
            if shape is not None:
                shapes.add(tuple(shape))
        for parameter in equation.params.values():
            for inner in (
                parameter if isinstance(parameter, list | tuple) else [parameter]
            ):
                if isinstance(inner, jax.extend.core.ClosedJaxpr):
                    shapes |= _shapes_in(inner.jaxpr)
                elif isinstance(inner, jax.extend.core.Jaxpr):
                    shapes |= _shapes_in(inner)
    return shapes


@pytest.mark.parametrize("name", ["cg", "bbmm"])
def test_matrix_free_prior_is_never_densified(getkey, name):
    size = 40
    prior_cov, dense = _matrix_free_prior(getkey(), size)
    precision = jr.uniform(getkey(), (size,), minval=0.5, maxval=4.0)
    information = precision * jr.normal(getkey(), (size,))
    prior_mean = jnp.zeros(size)
    solver, rtol = _solvers()[name]

    def mean(information, precision):
        return gaussx.site_posterior(
            prior_mean, prior_cov, information, precision, solver=solver
        ).mean

    jaxpr = jax.make_jaxpr(mean)(information, precision)
    assert (size, size) not in _shapes_in(jaxpr.jaxpr)

    posterior = gaussx.site_posterior(
        prior_mean, prior_cov, information, precision, solver=solver
    )
    variance = posterior.variance(method="exact", batch_size=8)
    _, expected_mean, expected_variance = _dense_reference(
        prior_mean, dense, information, precision
    )
    assert tree_allclose(posterior.mean, expected_mean, rtol=rtol, atol=rtol)
    assert tree_allclose(variance, expected_variance, rtol=rtol, atol=rtol)


@pytest.mark.parametrize("site_precision", [0.25, 1e-3, 1e-6])
def test_float32_variance_is_accurate_for_weak_sites(site_precision):
    # The issue's table: N = 200, Matérn-3/2, lengthscale 0.3, unit variance,
    # float64 reference. The K − K A⁻¹ K form measured ≤ 3.9e-7 there.
    size = 200
    inputs = np.linspace(0.0, 1.0, size)
    scaled = np.sqrt(3.0) * np.abs(np.subtract.outer(inputs, inputs)) / 0.3
    kernel = (1.0 + scaled) * np.exp(-scaled) + 1e-6 * np.eye(size)
    site_variance = np.full(size, 1.0 / site_precision)
    system = kernel + np.diag(site_variance)
    reference = np.diag(kernel - kernel @ np.linalg.solve(system, kernel))

    posterior = gaussx.site_posterior(
        jnp.zeros(size, dtype=jnp.float32),
        lx.MatrixLinearOperator(jnp.asarray(kernel, dtype=jnp.float32), _PSD),
        jnp.zeros(size, dtype=jnp.float32),
        jnp.full(size, site_precision, dtype=jnp.float32),
    )
    variance = posterior.variance()

    assert variance.dtype == jnp.float32
    error = np.max(np.abs(np.asarray(variance, dtype=np.float64) - reference))
    # 1e-6: the first form's measured 3.9e-7 with headroom; the form
    # d − d diag(A⁻¹) d measured 4.4e-4 at λ = 1e-3 and would fail
    assert error < 1e-6


def test_jit_and_vmap_over_sites(getkey):
    prior_mean, kernel, _, _ = _problem(getkey())
    prior_cov = lx.MatrixLinearOperator(kernel, _PSD)
    precisions = jr.uniform(getkey(), (3, _N), minval=0.5, maxval=4.0)
    informations = precisions * jr.normal(getkey(), (3, _N))

    @eqx.filter_jit
    def moments(information, precision):
        posterior = gaussx.site_posterior(prior_mean, prior_cov, information, precision)
        return posterior.mean, posterior.variance()

    batched_mean, batched_variance = jax.vmap(moments)(informations, precisions)
    for row in range(3):
        mean, variance = moments(informations[row], precisions[row])
        assert tree_allclose(batched_mean[row], mean)
        assert tree_allclose(batched_variance[row], variance)


@pytest.mark.parametrize("name", ["dense", "cg"])
def test_grad_in_the_sites_and_the_lengthscale(getkey, name):
    prior_mean, _, information, precision = _problem(getkey())
    inputs = jnp.sort(jr.uniform(getkey(), (_N,)))
    solver, _ = _solvers()[name]

    def objective(information, precision, lengthscale):
        kernel = _matern32(inputs, lengthscale)
        posterior = gaussx.site_posterior(
            prior_mean,
            lx.MatrixLinearOperator(kernel, _PSD),
            information,
            precision,
            solver=solver,
        )
        return jnp.sum(posterior.mean) + jnp.sum(posterior.variance())

    def dense_objective(information, precision, lengthscale):
        kernel = _matern32(inputs, lengthscale)
        _, mean, variance = _dense_reference(prior_mean, kernel, information, precision)
        return jnp.sum(mean) + jnp.sum(variance)

    arguments = (information, precision, jnp.asarray(0.4))
    gradient = jax.grad(objective, argnums=(0, 1, 2))(*arguments)
    expected = jax.grad(dense_objective, argnums=(0, 1, 2))(*arguments)
    rtol = 1e-6 if information.dtype == jnp.float64 else 2e-3
    assert tree_allclose(gradient, expected, rtol=rtol, atol=rtol)


def test_stochastic_variance_waits_on_827(getkey):
    prior_mean, kernel, information, precision = _problem(getkey())
    posterior = gaussx.site_posterior(
        prior_mean,
        lx.MatrixLinearOperator(kernel, _PSD),
        information,
        precision,
        solver=gaussx.CGSolver(),
    )
    with pytest.raises(NotImplementedError, match="827"):
        posterior.variance(method="stochastic", key=jr.key(0))
    with pytest.raises(ValueError, match="method"):
        posterior.variance(method="hutchinson")  # type: ignore[arg-type]


def test_dense_path_warns_when_it_densifies_a_structured_prior(getkey):
    size = 10
    prior_cov, dense = _matrix_free_prior(getkey(), size)
    # a structured prior that only the warning path may densify
    structured = lx.FunctionLinearOperator(prior_cov.fn, prior_cov.in_structure(), _PSD)
    precision = jr.uniform(getkey(), (size,), minval=0.5, maxval=4.0)
    information = precision * jr.normal(getkey(), (size,))
    prior_mean = jnp.zeros(size)

    with pytest.warns(gaussx.DenseFallbackWarning, match="CGSolver"):
        posterior = gaussx.site_posterior(
            prior_mean, structured, information, precision
        )

    _, mean, _ = _dense_reference(prior_mean, dense, information, precision)
    assert tree_allclose(posterior.mean, mean)


def test_dense_matrix_prior_does_not_warn(getkey):
    prior_mean, kernel, information, precision = _problem(getkey())
    with warnings.catch_warnings():
        warnings.simplefilter("error", gaussx.DenseFallbackWarning)
        posterior = gaussx.site_posterior(
            prior_mean, lx.MatrixLinearOperator(kernel, _PSD), information, precision
        )
        posterior.variance()


class _NoDensifyDiagonal(lx.DiagonalLinearOperator):
    """A diagonal prior that fails the test if anything densifies it."""

    def as_matrix(self):
        raise AssertionError("as_matrix called on the diagonal prior")


@pytest.mark.parametrize("solver", [None, gaussx.DenseSolver()])
def test_diagonal_prior_stays_diagonal(getkey, solver):
    prior_variance = jr.uniform(getkey(), (_N,), minval=0.5, maxval=2.0)
    precision = jr.uniform(getkey(), (_N,), minval=1e-3, maxval=4.0)
    information = precision * jr.normal(getkey(), (_N,))
    prior_mean = 0.1 * jr.normal(getkey(), (_N,))
    prior_cov = lx.TaggedLinearOperator(_NoDensifyDiagonal(prior_variance), _PSD)

    with warnings.catch_warnings():
        warnings.simplefilter("error", gaussx.DenseFallbackWarning)
        posterior = gaussx.site_posterior(
            prior_mean, prior_cov, information, precision, solver=solver
        )
        variance = posterior.variance()
        covariance_diagonal = jax.vmap(posterior.covariance().mv)(
            jnp.eye(_N, dtype=prior_variance.dtype)
        )

    weights, mean, expected = _dense_reference(
        prior_mean, jnp.diag(prior_variance), information, precision
    )
    assert tree_allclose(posterior.weights, weights)
    assert tree_allclose(posterior.mean, mean)
    assert tree_allclose(variance, expected)
    assert tree_allclose(jnp.diag(covariance_diagonal), expected)
