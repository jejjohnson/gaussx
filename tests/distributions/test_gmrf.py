"""Tests for GaussianMRF, IntrinsicGMRF and ConstrainedGMRF (G6)."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest


pytest.importorskip("numpyro")

import numpyro
from jax.scipy.stats import multivariate_normal

import gaussx as gx
from gaussx._einx import einsum, rearrange
from gaussx._testing import assert_sample_moments


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def path_laplacian(n: int) -> np.ndarray:
    L = np.diag(np.r_[1.0, 2.0 * np.ones(n - 2), 1.0])
    return L - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)


def grid_laplacian(side: int) -> np.ndarray:
    L, eye = path_laplacian(side), np.eye(side)
    return np.kron(L, eye) + np.kron(eye, L)


def sparse_from_dense(M: np.ndarray) -> gx.SparseOperator:
    rows, cols = np.nonzero(np.tril(M))
    return gx.SparseOperator.from_coo(
        rows, cols, jnp.asarray(M[rows, cols]), M.shape, symmetric=True, tags=_PSD
    )


def banded_from_dense(M: np.ndarray, d: int) -> gx.BlockTriDiag:
    blocks = rearrange(jnp.asarray(M), "(N a) (K b) -> N K a b", a=d, b=d)
    num = M.shape[0] // d
    diagonal = blocks[jnp.arange(num), jnp.arange(num)]
    sub = blocks[jnp.arange(1, num), jnp.arange(num - 1)]
    return gx.BlockTriDiag(diagonal, sub, tags=_PSD)


def grid_incidence(side: int) -> gx.SparseOperator:
    """Signed incidence of the 4-neighbour grid: ``BᵀB`` is its Laplacian."""
    index = np.asarray(rearrange(np.arange(side * side), "(h w) -> h w", h=side))
    edges = np.r_[
        np.c_[index[:, :-1].ravel(), index[:, 1:].ravel()],
        np.c_[index[:-1, :].ravel(), index[1:, :].ravel()],
    ]
    m = edges.shape[0]
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[edges[:, 0], edges[:, 1]]
    values = jnp.r_[jnp.ones(m), -jnp.ones(m)]
    return gx.SparseOperator.from_coo(rows, cols, values, (m, side * side))


def kronecker_sum(side: int, shift: float) -> gx.KroneckerSum:
    half = path_laplacian(side) + 0.5 * shift * np.eye(side)
    factor = lx.MatrixLinearOperator(jnp.asarray(half), lx.symmetric_tag)
    return gx.KroneckerSum(factor, factor)


def spectral(side: int, shift: float) -> gx.SpectralFunction:
    return gx.SpectralFunction(kronecker_sum(side, 0.0), lambda lam: shift + lam)


def proper_precision(kind: str, side: int, kappa: float):
    """``L ⊕ L + κI`` on a ``side × side`` grid, with the requested structure."""
    dense = grid_laplacian(side) + kappa * np.eye(side * side)
    factors = None
    solver = None
    if kind == "sparse":
        op = sparse_from_dense(dense)
        solver = gx.SparseCholeskySolver()
    elif kind == "banded":
        op = banded_from_dense(dense, side)
    elif kind == "dense":
        op = lx.MatrixLinearOperator(jnp.asarray(dense), _PSD)
    elif kind == "kronecker_sum":
        op = kronecker_sum(side, kappa)
    elif kind == "spectral":
        op = spectral(side, kappa)
    elif kind == "cg":
        op = sparse_from_dense(dense)
        n = side * side
        factors = (
            grid_incidence(side),
            lx.DiagonalLinearOperator(jnp.full(n, math.sqrt(kappa))),
        )
    else:  # pragma: no cover
        raise ValueError(kind)
    return op, factors, solver, dense


def intrinsic_structure(kind: str, side: int):
    """The grid Laplacian ``L ⊕ L`` (null space: constants)."""
    dense = grid_laplacian(side)
    factors = None
    if kind == "sparse":
        op = sparse_from_dense(dense)
    elif kind == "banded":
        op = banded_from_dense(dense, side)
    elif kind == "dense":
        op = lx.MatrixLinearOperator(jnp.asarray(dense), _PSD)
    elif kind == "kronecker_sum":
        op = kronecker_sum(side, 0.0)
    elif kind == "cg":
        op = sparse_from_dense(dense)
        factors = (grid_incidence(side),)
    else:  # pragma: no cover
        raise ValueError(kind)
    n = side * side
    null = jnp.ones((n, 1)) / jnp.sqrt(n)
    return op, factors, null, dense


def path_structure(n: int) -> gx.SparseOperator:
    return sparse_from_dense(path_laplacian(n))


def weighted_gram(A, w):
    """``Aᵀ diag(w) A``, densely."""
    return einsum(einx.multiply("k i, k -> k i", A, w), A, "k i, k j -> i j")


def factor_gram(F: lx.AbstractLinearOperator):
    """``FᵀF``, densely."""
    matrix = F.as_matrix()
    return einsum(matrix, matrix, "k i, k j -> i j")


def dense_conditional(Q, A, e, mu):
    """Mean and covariance of ``x | A x = e`` for ``x ~ N(mu, Q⁻¹)``, densely."""
    Sigma = np.linalg.inv(Q)
    cross = einsum(Sigma, A, "i j, c j -> i c")  # Σ Aᵀ
    S = einsum(A, cross, "c i, i d -> c d")
    gain = einsum(cross, np.linalg.inv(S), "i c, c d -> i d")
    excess = einsum(A, mu, "c n, n -> c") - e
    mean = mu - einsum(gain, excess, "i c, c -> i")
    return np.asarray(mean), np.asarray(Sigma - einsum(gain, cross, "i c, j c -> i j"))


PROPER_KINDS = ["sparse", "banded", "dense", "kronecker_sum", "spectral", "cg"]
INTRINSIC_KINDS = ["sparse", "banded", "dense", "kronecker_sum", "cg"]


# ---------------------------------------------------------------------------
# GaussianMRF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_log_prob_matches_dense_mvn(kind):
    op, factors, solver, dense = proper_precision(kind, 3, 0.7)
    n = dense.shape[0]
    loc = jnp.linspace(-1.0, 1.0, n)
    d = gx.GaussianMRF(loc, op, factors, solver=solver)
    x = jr.normal(jr.key(0), (4, n))
    expected = multivariate_normal.logpdf(x, loc, jnp.linalg.inv(dense))
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-10, atol=1e-10)
    assert d.log_prob(x[0]).shape == ()


def test_log_det_precision_overrides_the_strategy():
    op, _, _, dense = proper_precision("sparse", 3, 0.7)
    n = dense.shape[0]
    x = jr.normal(jr.key(1), (n,))
    exact = jnp.linalg.slogdet(dense)[1]
    reference = gx.GaussianMRF(jnp.zeros(n), op).log_prob(x)
    known = gx.GaussianMRF(jnp.zeros(n), op, log_det_precision=exact)
    np.testing.assert_allclose(known.log_prob(x), reference, rtol=1e-12)
    shifted = gx.GaussianMRF(jnp.zeros(n), op, log_det_precision=exact + 2.0)
    np.testing.assert_allclose(shifted.log_prob(x), reference + 1.0, rtol=1e-12)


def test_logdet_strategy_path():
    op, _, _, dense = proper_precision("dense", 3, 0.7)
    n = dense.shape[0]
    x = jr.normal(jr.key(2), (n,))
    d = gx.GaussianMRF(jnp.zeros(n), op, logdet_strategy=gx.DenseSolver())
    expected = multivariate_normal.logpdf(x, jnp.zeros(n), jnp.linalg.inv(dense))
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-10)


@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_marginal_variances_match_dense(kind):
    op, factors, solver, dense = proper_precision(kind, 3, 0.7)
    d = gx.GaussianMRF(jnp.zeros(dense.shape[0]), op, factors, solver=solver)
    expected = np.diag(np.linalg.inv(dense))
    np.testing.assert_allclose(d.marginal_variances(), expected, rtol=1e-8)
    np.testing.assert_allclose(d.variance, expected, rtol=1e-8)


@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_sample_shapes_and_determinism(kind):
    op, factors, solver, dense = proper_precision(kind, 3, 0.7)
    n = dense.shape[0]
    d = gx.GaussianMRF(jnp.ones(n), op, factors, solver=solver)
    draws = d.sample(jr.key(3), (2, 5))
    assert draws.shape == (2, 5, n)
    assert d.sample(jr.key(3)).shape == (n,)
    assert jnp.all(jnp.isfinite(draws))
    np.testing.assert_array_equal(d.sample(jr.key(3), (2, 5)), draws)
    # key=None is PRNGKey(0)
    np.testing.assert_array_equal(d.sample(None), d.sample(jr.PRNGKey(0)))


@pytest.mark.slow
@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_each_sampling_branch_has_the_right_covariance(kind):
    # 10 × 10 grid; bound by the estimator's own sampling distribution
    # (assert_sample_moments, 7 sigma).
    op, factors, solver, dense = proper_precision(kind, 10, 0.5)
    n = dense.shape[0]
    loc = jnp.linspace(-1.0, 1.0, n)
    d = gx.GaussianMRF(loc, op, factors, solver=solver)
    draws = d.sample(jr.key(4), (4000,))
    assert_sample_moments(draws, loc, jnp.linalg.inv(dense))


def test_condition_on_observations_sparse_matches_dense():
    op, _, _, dense = proper_precision("sparse", 4, 0.3)
    n = dense.shape[0]
    loc = jnp.linspace(0.0, 1.0, n)
    idx = np.array([0, 5, 9, 15])
    A = gx.SparseOperator.from_coo(np.arange(4), idx, jnp.ones(4), (4, n))
    y = jnp.array([1.0, -1.0, 0.5, 2.0])
    lam = jnp.array([4.0, 4.0, 25.0, 1.0])
    post = gx.GaussianMRF(loc, op).condition_on_observations(A, lam, y)
    assert isinstance(post.precision, gx.SparseOperator)
    Ad = np.asarray(A.as_matrix())
    Q_post = dense + weighted_gram(Ad, lam)
    residual = lam * (y - einsum(Ad, loc, "k n, n -> k"))
    mean = loc + np.linalg.solve(Q_post, einsum(Ad, residual, "k n, k -> n"))
    np.testing.assert_allclose(post.precision.as_matrix(), Q_post, atol=1e-12)
    np.testing.assert_allclose(post.loc, mean, rtol=1e-10)
    np.testing.assert_allclose(
        post.marginal_variances(), np.diag(np.linalg.inv(Q_post)), rtol=1e-8
    )


def test_condition_on_observations_dense_fallback_and_factors():
    op, factors, _, dense = proper_precision("cg", 3, 0.5)
    n = dense.shape[0]
    A = jr.normal(jr.key(5), (2, n))
    y = jnp.array([0.3, -0.2])
    prior = gx.GaussianMRF(jnp.zeros(n), op, factors)
    post = prior.condition_on_observations(A, 2.0, y)
    Q_post = dense + weighted_gram(A, jnp.full(2, 2.0))
    np.testing.assert_allclose(post.precision.as_matrix(), Q_post, atol=1e-12)
    np.testing.assert_allclose(
        post.loc,
        np.linalg.solve(Q_post, 2.0 * einsum(A, y, "k n, k -> n")),
        rtol=1e-10,
    )
    # The factors gain Λ^{1/2} A, so Σ FᵀF is still the posterior precision.
    assert post.precision_factors is not None
    gram = sum(factor_gram(F) for F in post.precision_factors)
    np.testing.assert_allclose(gram, Q_post, atol=1e-12)


@pytest.mark.parametrize("kind", ["spectral", "kronecker_sum", "banded"])
def test_condition_on_observations_structured_prior_stays_matrix_free(kind):
    op, _, _, dense = proper_precision(kind, 4, 0.5)
    n = dense.shape[0]
    idx = np.array([1, 6, 10])
    A = gx.SparseOperator.from_coo(np.arange(3), idx, jnp.ones(3), (3, n))
    y = jnp.array([0.5, -1.0, 2.0])
    post = gx.GaussianMRF(jnp.zeros(n), op).condition_on_observations(A, 9.0, y)
    assert not isinstance(post.precision, lx.MatrixLinearOperator)
    Ad = np.asarray(A.as_matrix())
    Q_post = dense + weighted_gram(Ad, jnp.full(3, 9.0))
    np.testing.assert_allclose(
        post.loc,
        np.linalg.solve(Q_post, 9.0 * einsum(Ad, y, "k n, k -> n")),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        post.marginal_variances(), np.diag(np.linalg.inv(Q_post)), rtol=1e-8
    )
    if kind == "banded":
        assert post.precision_factors is None
    else:
        # (Q^{1/2}, Λ^{1/2}A): the factors reproduce the posterior precision.
        assert post.precision_factors is not None
        gram = sum(factor_gram(F) for F in post.precision_factors)
        np.testing.assert_allclose(gram, Q_post, atol=1e-10)


@pytest.mark.slow
def test_grid_posterior_samples_by_perturbation_optimisation():
    # Gap-filling on a 10 × 10 SPDE-like grid prior; bound by the estimator's
    # own sampling distribution (assert_sample_moments, 7 sigma).
    op, _, _, dense = proper_precision("spectral", 10, 0.5)
    n = dense.shape[0]
    idx = np.arange(0, n, 7)
    A = gx.SparseOperator.from_coo(
        np.arange(idx.size), idx, jnp.ones(idx.size), (idx.size, n)
    )
    y = jnp.sin(jnp.arange(idx.size, dtype=float))
    post = gx.GaussianMRF(jnp.zeros(n), op).condition_on_observations(A, 4.0, y)
    Ad = np.asarray(A.as_matrix())
    Q_post = dense + weighted_gram(Ad, jnp.full(idx.size, 4.0))
    draws = post.sample(jr.key(22), (4000,))
    assert_sample_moments(draws, post.loc, jnp.linalg.inv(Q_post))


def test_loc_shape_is_checked():
    op, _, _, _ = proper_precision("dense", 3, 0.5)
    with pytest.raises(ValueError, match="loc must have shape"):
        gx.GaussianMRF(jnp.zeros(4), op)


# ---------------------------------------------------------------------------
# ConstrainedGMRF
# ---------------------------------------------------------------------------


def _constrained(kind: str):
    op, factors, solver, dense = proper_precision(kind, 3, 0.4)
    n = dense.shape[0]
    loc = jnp.linspace(-1.0, 2.0, n)
    A = jnp.stack([jnp.ones(n), jnp.arange(n, dtype=float)])
    e = jnp.array([1.0, -2.0])
    field = gx.GaussianMRF(loc, op, factors, solver=solver).condition_on_constraints(
        A, e
    )
    return field, dense, np.asarray(A), np.asarray(e), np.asarray(loc)


@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_hard_constrained_samples_satisfy_the_constraint(kind):
    field, _, A, e, _ = _constrained(kind)
    draws = field.sample(jr.key(6), (50,))
    residual = einsum(draws, A, "s n, c n -> s c") - e
    scale = jnp.max(jnp.abs(draws)) * jnp.max(jnp.abs(A))
    assert jnp.max(jnp.abs(residual)) < 50 * jnp.finfo(draws.dtype).eps * scale


@pytest.mark.parametrize("kind", PROPER_KINDS)
def test_constrained_mean_and_marginal_variances_equal_dense(kind):
    field, dense, A, e, loc = _constrained(kind)
    mean, cov = dense_conditional(dense, A, e, loc)
    np.testing.assert_allclose(field.mean, mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(field.marginal_variances(), np.diag(cov), atol=1e-8)
    np.testing.assert_allclose(field.variance, np.diag(cov), atol=1e-8)


def test_constrained_log_prob_is_the_density_on_the_subspace():
    field, dense, A, e, loc = _constrained("sparse")
    mean, cov = dense_conditional(dense, A, e, loc)
    # Orthonormal coordinates w of the subspace {A x = e}: x = mean + B w.
    B = np.linalg.svd(A)[2][A.shape[0] :].T
    cov_w = einsum(B, einsum(cov, B, "i j, j k -> i k"), "i l, i k -> l k")
    w = jr.normal(jr.key(7), (3, B.shape[1]))
    x = mean + einsum(jnp.asarray(B), w, "n k, s k -> s n")
    expected = multivariate_normal.logpdf(w, jnp.zeros(B.shape[1]), cov_w)
    np.testing.assert_allclose(field.log_prob(x), expected, rtol=1e-8)


@pytest.mark.slow
def test_constrained_sample_moments():
    # Bound by the estimator's own sampling distribution (7 sigma).
    field, dense, A, e, loc = _constrained("sparse")
    mean, cov = dense_conditional(dense, A, e, loc)
    draws = field.sample(jr.key(8), (4000,))
    assert_sample_moments(draws, jnp.asarray(mean), jnp.asarray(cov))


def test_constraint_shapes_are_checked():
    op, _, _, dense = proper_precision("dense", 2, 0.5)
    d = gx.GaussianMRF(jnp.zeros(dense.shape[0]), op)
    with pytest.raises(ValueError, match="constraint_matrix"):
        d.condition_on_constraints(jnp.ones((1, 3)), jnp.zeros(1))
    single = d.condition_on_constraints(jnp.ones(4), 0.0)
    assert single.constraint_matrix.shape == (1, 4)


# ---------------------------------------------------------------------------
# IntrinsicGMRF
# ---------------------------------------------------------------------------


def _intrinsic_closed_form(x, mu, tau, R, rank, normalizer=False):
    r = x - mu
    quad = einsum(r, R, r, "i, i j, j ->")
    value = 0.5 * rank * jnp.log(tau) - 0.5 * tau * quad
    if normalizer:
        eigenvalues = np.linalg.eigvalsh(R)
        positive = eigenvalues[eigenvalues > 1e-9 * eigenvalues.max()]
        value = value + 0.5 * np.sum(np.log(positive)) - 0.5 * rank * _LOG_2PI
    return value


@pytest.mark.parametrize("tau", [0.3, 2.0])
@pytest.mark.parametrize("normalizer", [False, True])
def test_intrinsic_log_prob_path_graph_closed_form(tau, normalizer):
    n = 8
    R = path_structure(n)
    mu = jnp.linspace(0.0, 1.0, n)
    null = jnp.ones(n) / jnp.sqrt(n)
    d = gx.IntrinsicGMRF(mu, tau, R, null, include_normalizer=normalizer)
    x = jr.normal(jr.key(9), (n,))
    expected = _intrinsic_closed_form(x, mu, tau, path_laplacian(n), n - 1, normalizer)
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-10)
    # (N − c)/2 · log τ: doubling τ at the null-space point x = μ adds
    # (N − 1)/2 · log 2.
    d2 = gx.IntrinsicGMRF(mu, 2 * tau, R, null, include_normalizer=normalizer)
    np.testing.assert_allclose(
        d2.log_prob(mu) - d.log_prob(mu), 0.5 * (n - 1) * math.log(2.0), rtol=1e-10
    )
    # Invariant along the null space.
    np.testing.assert_allclose(d.log_prob(x + 3.0), d.log_prob(x), rtol=1e-10)


def test_intrinsic_log_prob_two_components():
    # Two disjoint paths: c = 2, rank N − 2.
    L = np.zeros((9, 9))
    L[:4, :4], L[4:, 4:] = path_laplacian(4), path_laplacian(5)
    null = np.zeros((9, 2))
    null[:4, 0], null[4:, 1] = 0.5, 1 / np.sqrt(5)
    d = gx.IntrinsicGMRF(
        jnp.zeros(9), 1.7, sparse_from_dense(L), null, include_normalizer=True
    )
    x = jr.normal(jr.key(10), (2, 9))
    expected = jax.vmap(
        lambda v: _intrinsic_closed_form(v, 0.0, 1.7, L, 7, normalizer=True)
    )(x)
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-10)


@pytest.mark.parametrize("kind", INTRINSIC_KINDS)
def test_intrinsic_log_prob_structures_agree(kind):
    op, factors, null, dense = intrinsic_structure(kind, 3)
    n = dense.shape[0]
    d = gx.IntrinsicGMRF(jnp.zeros(n), 1.3, op, null, factors, include_normalizer=True)
    x = jr.normal(jr.key(11), (n,))
    expected = _intrinsic_closed_form(x, 0.0, 1.3, dense, n - 1, normalizer=True)
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-9)


@pytest.mark.parametrize("kind", INTRINSIC_KINDS)
@pytest.mark.parametrize("constraint", ["hard", "none", "soft"])
def test_intrinsic_samples_orthogonal_to_null_space(kind, constraint):
    op, factors, null, dense = intrinsic_structure(kind, 4)
    n = dense.shape[0]
    loc = jnp.linspace(0.0, 2.0, n)
    d = gx.IntrinsicGMRF(loc, 2.0, op, null, factors, constraint=constraint)
    draws = d.sample(jr.key(12), (20,))
    assert draws.shape == (20, n)
    coeffs = einsum(draws - d.mean, null, "s n, n c -> s c")
    scale = jnp.max(jnp.abs(draws))
    if constraint == "soft":
        # V is orthonormal, so Vᵀx ~ N(0, s²): within 7 sd.
        assert jnp.max(jnp.abs(coeffs)) < 7 * d.soft_constraint_scale
    else:
        assert jnp.max(jnp.abs(coeffs)) < 100 * jnp.finfo(draws.dtype).eps * scale
    if constraint == "hard":
        hard = einsum(draws, null, "s n, n c -> s c")
        assert jnp.max(jnp.abs(hard)) < 100 * jnp.finfo(draws.dtype).eps * scale


@pytest.mark.parametrize("kind", INTRINSIC_KINDS)
def test_intrinsic_hard_marginal_variances_equal_pinv(kind):
    op, factors, null, dense = intrinsic_structure(kind, 4)
    n = dense.shape[0]
    d = gx.IntrinsicGMRF(jnp.zeros(n), 2.0, op, null, factors)
    expected = np.diag(np.linalg.pinv(2.0 * dense))
    # The ε-shift biases the kriged path by ~ε/λ_min ≈ 1e-7 (relative).
    np.testing.assert_allclose(d.marginal_variances(), expected, rtol=1e-6)


def test_intrinsic_soft_constraint_density_and_variances():
    n = 6
    R = path_structure(n)
    null = jnp.ones((n, 1)) / jnp.sqrt(n)
    s = 0.05
    d = gx.IntrinsicGMRF(
        jnp.zeros(n),
        1.5,
        R,
        null,
        constraint="soft",
        soft_constraint_scale=s,
        include_normalizer=True,
    )
    # With an orthonormal null space the soft prior is the proper Gaussian
    # with precision τR + VVᵀ/s².
    Q = 1.5 * path_laplacian(n) + einsum(null, null, "i c, j c -> i j") / s**2
    x = jr.normal(jr.key(13), (3, n))
    expected = multivariate_normal.logpdf(x, jnp.zeros(n), jnp.linalg.inv(Q))
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-9)
    np.testing.assert_allclose(
        d.marginal_variances(), np.diag(np.linalg.inv(Q)), rtol=1e-6
    )


def test_intrinsic_odd_rw2_padding_node():
    # rw2_structure(7) is 8 × 8: node 7 is a decoupled unit-precision pad.
    n, tau = 7, 3.0
    R = gx.rw2_structure(n)
    assert R.in_size() == n + 1
    t = jnp.arange(n, dtype=float)
    null = jnp.zeros((n + 1, 2)).at[:n, 0].set(1.0).at[:n, 1].set(t - t.mean())
    d = gx.IntrinsicGMRF(jnp.zeros(n + 1), tau, R, null, include_normalizer=True)
    # N − c = n − 1 = rank(R_padded): the RW2 density on the first n nodes
    # (rank n − 2) times the pad's own N(0, 1/τ).
    D = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    D[rows, rows], D[rows, rows + 1], D[rows, rows + 2] = 1.0, -2.0, 1.0
    R_n = np.asarray(einsum(D, D, "k i, k j -> i j"))
    x = jr.normal(jr.key(14), (n + 1,))
    expected = _intrinsic_closed_form(
        x[:n], 0.0, tau, R_n, n - 2, normalizer=True
    ) + multivariate_normal.logpdf(x[n:], jnp.zeros(1), jnp.eye(1) / tau)
    np.testing.assert_allclose(d.log_prob(x), expected, rtol=1e-9)
    # Samples: the field part is orthogonal to {1, t}; the pad has var 1/τ.
    draws = d.sample(jr.key(15), (10,))
    coeffs = einsum(draws, null, "s n, n c -> s c")
    assert jnp.max(jnp.abs(coeffs)) < 1e-10
    variances = d.marginal_variances()
    pinv = np.linalg.pinv(tau * R_n)
    np.testing.assert_allclose(variances[:n], np.diag(pinv), rtol=1e-6)
    np.testing.assert_allclose(variances[n], 1 / tau, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("kind", INTRINSIC_KINDS)
@pytest.mark.parametrize("constraint", ["hard", "soft"])
def test_intrinsic_sampling_branches_covariance(kind, constraint):
    # 10 × 10 grid; bound by the estimator's own sampling distribution
    # (assert_sample_moments, 7 sigma). The soft constraint adds s² VVᵀ.
    op, factors, null, dense = intrinsic_structure(kind, 10)
    n = dense.shape[0]
    loc = jnp.linspace(-1.0, 1.0, n)
    scale = 0.1
    d = gx.IntrinsicGMRF(
        loc, 2.0, op, null, factors, constraint, soft_constraint_scale=scale
    )
    outer = einsum(null, null, "i c, j c -> i j")
    cov = np.linalg.pinv(2.0 * dense)
    if constraint == "soft":
        cov = cov + scale**2 * outer
    draws = d.sample(jr.key(16), (4000,))
    mean = loc - einsum(outer, loc, "i j, j -> i")
    assert_sample_moments(draws, mean, jnp.asarray(cov))


def test_intrinsic_validation():
    R = path_structure(4)
    null = jnp.ones(4) / 2
    with pytest.raises(ValueError, match="constraint must be"):
        gx.IntrinsicGMRF(jnp.zeros(4), 1.0, R, null, constraint="bogus")
    with pytest.raises(ValueError, match="null_space must have"):
        gx.IntrinsicGMRF(jnp.zeros(4), 1.0, R, jnp.ones(3))
    with pytest.raises(ValueError, match="loc must have shape"):
        gx.IntrinsicGMRF(jnp.zeros(3), 1.0, R, null)
    d = gx.IntrinsicGMRF(jnp.ones(4), 1.0, R, null, constraint="none")
    np.testing.assert_array_equal(d.mean, jnp.ones(4))


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_grad_log_prob_wrt_factor_scale():
    # Leroux: Q(rho) = rho BᵀB + (1 − rho) I, factored; log|Q| through the sparse
    # Cholesky's exact VJP, against jax.grad through the dense density.
    side = 3
    n = side * side
    B = grid_incidence(side)
    lap = sparse_from_dense(grid_laplacian(side))
    x = jr.normal(jr.key(17), (n,))

    def sparse_lp(rho):
        Q = gx.SparseOperator(rho * lap.values, lap.pattern, tags=_PSD)
        Q = Q.add_diagonal(jnp.full(n, 1.0 - rho), tags=_PSD)
        factors = (
            jnp.sqrt(rho) * B,
            lx.DiagonalLinearOperator(jnp.full(n, jnp.sqrt(1.0 - rho))),
        )
        return gx.GaussianMRF(jnp.zeros(n), Q, factors).log_prob(x)

    def dense_lp(rho):
        Q = rho * jnp.asarray(grid_laplacian(side)) + (1.0 - rho) * jnp.eye(n)
        return multivariate_normal.logpdf(x, jnp.zeros(n), jnp.linalg.inv(Q))

    np.testing.assert_allclose(
        jax.grad(sparse_lp)(0.6), jax.grad(dense_lp)(0.6), rtol=1e-8
    )


def test_grad_intrinsic_log_prob_wrt_log_tau():
    n = 6
    R = path_structure(n)
    null = jnp.ones(n) / jnp.sqrt(n)
    x = jr.normal(jr.key(18), (n,))
    quad = einsum(x, path_laplacian(n), x, "i, i j, j ->")

    def lp(log_tau):
        return gx.IntrinsicGMRF(jnp.zeros(n), jnp.exp(log_tau), R, null).log_prob(x)

    # d/d log τ of (N − 1)/2 log τ − τ q/2 is (N − 1)/2 − τ q/2.
    expected = 0.5 * (n - 1) - 0.5 * math.exp(0.4) * quad
    np.testing.assert_allclose(jax.grad(lp)(0.4), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# NumPyro integration
# ---------------------------------------------------------------------------


def _distributions():
    op, factors, _, dense = proper_precision("cg", 3, 0.5)
    n = dense.shape[0]
    proper = gx.GaussianMRF(jnp.zeros(n), op, factors)
    R, _, null, _ = intrinsic_structure("sparse", 3)
    return {
        "proper": proper,
        "constrained": proper.condition_on_constraints(jnp.ones(n), 0.0),
        "intrinsic": gx.IntrinsicGMRF(jnp.zeros(n), 1.5, R, null),
        "soft": gx.IntrinsicGMRF(jnp.zeros(n), 1.5, R, null, constraint="soft"),
    }


@pytest.mark.parametrize("name", ["proper", "constrained", "intrinsic", "soft"])
def test_numpyro_seed_and_trace(name):
    d = _distributions()[name]

    def model():
        numpyro.sample("x", d)

    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    site = trace["x"]
    assert site["value"].shape == d.event_shape
    np.testing.assert_allclose(
        site["fn"].log_prob(site["value"]), d.log_prob(site["value"])
    )
    # A re-seeded run reproduces the draw.
    again = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    np.testing.assert_array_equal(again["x"]["value"], site["value"])


@pytest.mark.parametrize("name", ["proper", "intrinsic"])
def test_pytree_round_trip_under_jit(name):
    d = _distributions()[name]
    x = jr.normal(jr.key(19), d.event_shape)
    leaves, treedef = jax.tree_util.tree_flatten(d)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(
        jax.jit(lambda dist_, v: dist_.log_prob(v))(rebuilt, x), d.log_prob(x)
    )


# ---------------------------------------------------------------------------
# MultivariateNormalPrecision through the sparse factor
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mvn_precision_samples_through_sparse_factor():
    # Bound by the estimator's own sampling distribution (7 sigma).
    op, _, _, dense = proper_precision("sparse", 4, 0.5)
    n = dense.shape[0]
    d = gx.MultivariateNormalPrecision(jnp.ones(n), op)
    draws = d.sample(jr.key(20), (4000,))
    assert_sample_moments(draws, jnp.ones(n), jnp.linalg.inv(dense))


def test_mvn_precision_sparse_sample_is_the_factor_draw():
    op, _, _, _ = proper_precision("sparse", 3, 0.5)
    n = op.in_size()
    d = gx.MultivariateNormalPrecision(jnp.zeros(n), op)
    z = jr.normal(jr.key(21), (n,))
    expected = gx.sparse_cholesky(op).solve_lower_transpose(z)
    np.testing.assert_allclose(d.sample(jr.key(21)), expected, rtol=1e-12)
