"""Laplace fits whose posterior precision is singular off the constraint (gh-867).

Λ_x + Aᵀ Λ_site A is positive definite on {Vᵀ x = 0} but not on the whole space when
two intrinsic parts cancel in A x, or when no row of A reads one. The reference is the
same model with the constraint removed by hand: x = N w for an orthonormal basis N of
the complement of V, a proper dense Gaussian prior on w and the projector A N.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx as gx
from gaussx._einx import einsum
from gaussx._testing import assert_sample_moments

from ._laplace_fit import laplace_fit


def walk(n: int, order: int = 1) -> np.ndarray:
    # rows of Dᵀ: the differences of each unit vector
    d = np.diff(np.eye(n), order)
    return np.einsum("ik,jk->ij", d, d)


def sparse(matrix: np.ndarray, symmetric: bool = False) -> gx.SparseOperator:
    rows, cols = np.nonzero(np.tril(matrix) if symmetric else matrix)
    values = jnp.asarray(matrix[rows, cols])
    return gx.SparseOperator.from_coo(
        rows, cols, values, matrix.shape, symmetric=symmetric
    )


class Case(eqx.Module):
    """A prior structure R, its kernel basis V, the constraint, and a projector A."""

    structure: np.ndarray
    kernel: np.ndarray
    constraint: np.ndarray
    projector: np.ndarray

    def operator(self, storage: str) -> lx.AbstractLinearOperator:
        if storage == "dense":
            return lx.MatrixLinearOperator(
                jnp.asarray(self.structure), lx.positive_semidefinite_tag
            )
        if storage == "banded":
            size = self.structure.shape[0]
            blocks = jnp.stack([jnp.asarray(self.structure)])
            # one block: the banded path with no sub-diagonal
            empty = jnp.zeros((0, size, size))
            return gx.BlockTriDiag(blocks, empty, symmetric=True)
        return sparse(self.structure, symmetric=True)

    def prior(self, tau=1.0, storage="sparse"):
        n = self.structure.shape[0]
        return gx.IntrinsicGMRF(
            jnp.zeros(n),
            tau,
            self.operator(storage),
            jnp.asarray(self.kernel),
            include_normalizer=True,
            constraint_basis=jnp.asarray(self.constraint),
        )

    def complement(self) -> np.ndarray:
        num = self.constraint.shape[1]
        return np.linalg.svd(self.constraint)[0][:, num:]

    def reference(self, tau=1.0):
        """The proper prior on w and the projector A N."""
        basis = self.complement()
        dense = einsum(basis, self.structure, basis, "i a, i j, j b -> a b")
        precision = lx.MatrixLinearOperator(
            tau * jnp.asarray(dense), lx.positive_semidefinite_tag
        )
        projector = lx.MatrixLinearOperator(jnp.asarray(self.projector @ basis))
        return gx.GaussianMRF(jnp.zeros(basis.shape[1]), precision), projector


def _block(*matrices):
    size = sum(m.shape[0] for m in matrices)
    out, at = np.zeros((size, size)), 0
    for m in matrices:
        out[at : at + m.shape[0], at : at + m.shape[0]] = m
        at += m.shape[0]
    return out


def two_walks() -> Case:
    """η = u[i] + v[j], every node read: the two constants cancel."""
    n, m = 7, 4
    V = np.zeros((n + m, 2))
    V[:n, 0], V[n:, 1] = 1.0, 1.0
    i, j = np.tile(np.arange(n), m), np.repeat(np.arange(m), n)
    A = np.zeros((i.size, n + m))
    A[np.arange(i.size), i] = 1.0
    A[np.arange(i.size), n + j] = 1.0
    return Case(_block(walk(n), walk(m)), V, V, A)


def unread_part() -> Case:
    """Two replicates of a walk; the second is not observed."""
    n = 6
    V = np.zeros((2 * n, 2))
    V[:n, 0], V[n:, 1] = 1.0, 1.0
    A = np.zeros((n, 2 * n))
    A[np.arange(n), np.arange(n)] = 1.0
    return Case(_block(walk(n), walk(n)), V, V, A)


def one_observation() -> Case:
    """An RW2 under both constraints, read at two of its nodes' mean."""
    n = 9
    t = np.arange(n) - (n - 1) / 2
    V = np.stack([np.ones(n), t]).T
    A = np.zeros((1, n))
    A[0, 1], A[0, 5] = 0.5, 0.5
    return Case(walk(n, 2), V, V, A)


def free_trend() -> Case:
    """An RW2 whose trend is free (R-INLA's) plus a walk."""
    n, m = 9, 5
    kernel = np.zeros((n + m, 3))
    kernel[:n, 0], kernel[n:, 1] = 1.0, 1.0
    kernel[:n, 2] = np.arange(n) - (n - 1) / 2
    i, j = np.tile(np.arange(n), m), np.repeat(np.arange(m), n)
    A = np.zeros((i.size, n + m))
    A[np.arange(i.size), i] = 1.0
    A[np.arange(i.size), n + j] = 1.0
    return Case(_block(walk(n, 2), walk(m)), kernel, kernel[:, :2], A)


PROPER = {
    "two_walks": two_walks,
    "unread_part": unread_part,
    "one_observation": one_observation,
}
ALL = {**PROPER, "free_trend": free_trend}


def likelihoods(case: Case):
    num = case.projector.shape[0]
    rng = np.random.default_rng(0)
    return {
        "gaussian": gx.GaussianLikelihood(jnp.asarray(rng.normal(size=num)), 0.1),
        "poisson": gx.PoissonLikelihood(jnp.asarray(rng.poisson(2.0, num) * 1.0)),
    }


def constrained_covariance(fit, V):
    """Dense Σ = G − G V (Vᵀ G V)⁻¹ Vᵀ G, G the fit's inverse."""
    inverse = jax.vmap(fit.factor.solve)(jnp.eye(V.shape[0]))
    solved = einsum(inverse, V, "i k, k c -> i c")
    schur = einsum(V, solved, "i a, i b -> a b")
    gain = jnp.linalg.solve(schur, solved.T)
    return inverse - einsum(solved, gain, "i c, c j -> i j")


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("storage", ["sparse", "banded", "dense"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_mode_and_covariance_match_the_reduced_model(name, storage, family):
    case = ALL[name]()
    likelihood = likelihoods(case)[family]
    N = jnp.asarray(case.complement())
    reduced_prior, reduced_projector = case.reference(1.3)
    reduced = laplace_fit(reduced_prior, likelihood, projector=reduced_projector)
    fit = laplace_fit(
        case.prior(1.3, storage), likelihood, projector=sparse(case.projector)
    )
    assert bool(fit.converged)
    np.testing.assert_allclose(
        fit.mode, einsum(N, reduced.mode, "i a, a -> i"), atol=1e-8
    )
    V = jnp.asarray(case.constraint)
    cov = einsum(
        N, jnp.linalg.inv(reduced.hessian.as_matrix()), N, "i a, a b, j b -> i j"
    )
    np.testing.assert_allclose(constrained_covariance(fit, V), cov, atol=1e-8)


@pytest.mark.parametrize("name", PROPER)
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_log_marginal_and_its_gradient_match_the_reduced_model(name, family):
    case = PROPER[name]()
    likelihood = likelihoods(case)[family]

    def full(log_tau):
        return laplace_fit(
            case.prior(jnp.exp(log_tau)), likelihood, projector=sparse(case.projector)
        ).log_marginal

    def reduced(log_tau):
        prior, projector = case.reference(jnp.exp(log_tau))
        return laplace_fit(prior, likelihood, projector=projector).log_marginal

    value, slope = jax.value_and_grad(full)(0.4)
    expected, expected_slope = jax.value_and_grad(reduced)(0.4)
    np.testing.assert_allclose(value, expected, rtol=1e-9)
    np.testing.assert_allclose(slope, expected_slope, rtol=1e-6)


def test_free_trend_log_marginal_differences():
    # The free trend makes the prior improper, so only differences in θ that leave
    # its constant alone are comparable: the two-walk parts of the data scale.
    case = free_trend()
    likelihood = likelihoods(case)["gaussian"]
    fits = [
        laplace_fit(
            case.prior(tau), likelihood, projector=sparse(case.projector)
        ).log_marginal
        for tau in (0.5, 4.0)
    ]
    N = case.complement()
    H = [
        tau * (N.T @ case.structure @ N)
        + N.T @ case.projector.T @ case.projector @ N / 0.1
        for tau in (0.5, 4.0)
    ]
    y = np.asarray(likelihood.y)
    b = N.T @ case.projector.T @ y / 0.1
    rank = case.structure.shape[0] - 3
    expected = [
        0.5 * rank * np.log(tau)
        - 0.5 * np.linalg.slogdet(h)[1]
        + 0.5 * b @ np.linalg.solve(h, b)
        for tau, h in zip((0.5, 4.0), H, strict=True)
    ]
    np.testing.assert_allclose(fits[1] - fits[0], expected[1] - expected[0], rtol=1e-9)


@pytest.mark.parametrize("name", PROPER)
def test_vb_mean_correction_matches_the_reduced_model(name):
    case = PROPER[name]()
    likelihood = likelihoods(case)["poisson"]
    prior, projector = case.prior(1.3), sparse(case.projector)
    reduced_prior, reduced_projector = case.reference(1.3)
    N = jnp.asarray(case.complement())
    subspace = jnp.eye(N.shape[0])[:, :3]
    fit = laplace_fit(prior, likelihood, projector=projector)
    reduced = laplace_fit(reduced_prior, likelihood, projector=reduced_projector)
    mean = gx.vb_mean_correction(
        fit, prior, likelihood, subspace=subspace, projector=projector
    )
    reduced_mean = gx.vb_mean_correction(
        reduced,
        reduced_prior,
        likelihood,
        subspace=einsum(N, subspace, "i a, i p -> a p"),
        projector=reduced_projector,
    )
    np.testing.assert_allclose(mean, einsum(N, reduced_mean, "i a, a -> i"), atol=1e-6)


@pytest.mark.parametrize("name", PROPER)
@pytest.mark.parametrize("storage", ["sparse", "dense"])
def test_posterior_field_matches_the_reduced_model(name, storage):
    # The same Gaussian as a ConstrainedGMRF: variances, density and draws.
    case = PROPER[name]()
    likelihood = likelihoods(case)["gaussian"]
    prior = case.prior(1.3, storage)
    fit = laplace_fit(prior, likelihood, projector=sparse(case.projector))
    num = case.constraint.shape[1]
    field = gx.GaussianMRF(fit.mode, fit.hessian).condition_on_constraints(
        jnp.asarray(case.constraint).T, jnp.zeros(num), null_space=prior.null_space
    )
    N = jnp.asarray(case.complement())
    reduced_prior, reduced_projector = case.reference(1.3)
    reduced = laplace_fit(reduced_prior, likelihood, projector=reduced_projector)
    hessian = reduced.hessian.as_matrix()
    cov = einsum(N, jnp.linalg.inv(hessian), N, "i a, a b, j b -> i j")
    np.testing.assert_allclose(field.mean, fit.mode, atol=1e-9)
    np.testing.assert_allclose(field.marginal_variances(), jnp.diag(cov), atol=1e-9)
    draws = field.sample(jr.key(0), (4000,))
    coeffs = einsum(draws, jnp.asarray(case.constraint), "s i, i c -> s c")
    assert float(jnp.max(jnp.abs(coeffs))) < 1e-8
    assert_sample_moments(draws, fit.mode, cov)
    reduced_field = gx.GaussianMRF(
        reduced.mode, lx.MatrixLinearOperator(hessian, lx.positive_semidefinite_tag)
    )
    reduced_draws = einsum(draws[:5], N, "s i, i a -> s a")
    np.testing.assert_allclose(
        field.log_prob(draws[:5]), reduced_field.log_prob(reduced_draws), rtol=1e-9
    )


def test_constrained_value_off_zero():
    # A_c x = e with e ≠ 0, on a singular precision: the walk shifted by its mean.
    n = 6
    precision = sparse(walk(n), symmetric=True)
    field = gx.GaussianMRF(jnp.zeros(n), precision).condition_on_constraints(
        jnp.ones(n), 3.0, null_space=jnp.ones(n)
    )
    np.testing.assert_allclose(field.mean, jnp.full(n, 0.5), atol=1e-12)
    pinv = np.linalg.pinv(walk(n), hermitian=True)
    np.testing.assert_allclose(field.marginal_variances(), np.diag(pinv), atol=1e-12)
    x = field.sample(jr.key(1), (3,))
    # the density of the walk on its subspace, at the mean-free part
    intrinsic = gx.IntrinsicGMRF(
        jnp.zeros(n),
        1.0,
        precision,
        jnp.ones(n) / jnp.sqrt(n),
        include_normalizer=True,
    )
    np.testing.assert_allclose(
        field.log_prob(x), intrinsic.log_prob(x - 0.5), rtol=1e-9
    )


def test_every_node_read_keeps_the_plain_factor():
    n = 8
    prior = gx.IntrinsicGMRF(
        jnp.zeros(n), 1.0, sparse(walk(n), symmetric=True), jnp.ones(n)
    )
    likelihood = gx.PoissonLikelihood(jnp.ones(n))
    assert gx.laplace_model(prior, likelihood).ground_nodes is None
    # every node read once, node 0 twice
    selection = sparse(np.eye(n)[[*range(n), 0]])
    model = gx.laplace_model(
        prior, gx.PoissonLikelihood(jnp.ones(n + 1)), projector=selection
    )
    assert model.ground_nodes is None
    partial = sparse(np.eye(n)[:-1])
    model = gx.laplace_model(
        prior, gx.PoissonLikelihood(jnp.ones(n - 1)), projector=partial
    )
    assert model.ground_nodes is not None


def test_validation():
    n = 4
    R = sparse(walk(n), symmetric=True)
    with pytest.raises(ValueError, match="hard constraint"):
        gx.IntrinsicGMRF(
            jnp.zeros(n),
            1.0,
            R,
            jnp.ones(n),
            constraint="none",
            constraint_basis=jnp.ones(n),
        )
    with pytest.raises(ValueError, match="constraint_basis must have"):
        gx.IntrinsicGMRF(
            jnp.zeros(n), 1.0, R, jnp.ones(n), constraint_basis=jnp.ones(3)
        )
    field = gx.GaussianMRF(jnp.zeros(n), R)
    with pytest.raises(ValueError, match="null_space must have"):
        field.condition_on_constraints(jnp.ones(n), 0.0, null_space=jnp.ones(3))
    grid = gx.GaussianMRF(
        jnp.zeros(4),
        gx.KroneckerSum(
            lx.MatrixLinearOperator(jnp.asarray(walk(2)), lx.symmetric_tag),
            lx.MatrixLinearOperator(jnp.asarray(walk(2)), lx.symmetric_tag),
        ),
    )
    with pytest.raises(TypeError, match="singular precision"):
        grid.condition_on_constraints(jnp.ones(4), 0.0, null_space=jnp.ones(4))
