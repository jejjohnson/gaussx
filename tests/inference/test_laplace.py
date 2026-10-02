"""Tests for `laplace_mode`: precision-form Newton, implicit gradients, log-marginal.

Every check is against a dense reference: Newton on the dense matrices (in an
orthonormal basis of ``{Vᵀx = 0}`` for hard-constrained intrinsic priors, so
the constrained Laplace approximation is an ordinary one there). Its
derivatives come from JAX through a few unrolled Newton steps started at the
mode: Newton's iteration map has zero Jacobian at its fixed point, so two
steps already carry exact first and second derivatives of the mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.stats

import gaussx as gx
from gaussx._einx import einsum


# ---------------------------------------------------------------------------
# Problems and the dense reference
# ---------------------------------------------------------------------------


def path_laplacian(n: int) -> gx.SparseOperator:
    """Symmetric path-graph Laplacian 0 - 1 - ... - (n-1)."""
    degree = np.r_[1.0, 2.0 * np.ones(n - 2), 1.0]
    return gx.SparseOperator.from_coo(
        np.r_[np.arange(n), np.arange(1, n)],
        np.r_[np.arange(n), np.arange(n - 1)],
        jnp.asarray(np.r_[degree, -np.ones(n - 1)]),
        (n, n),
        symmetric=True,
    )


def two_per_row(m: int, n: int, seed: int = 0) -> gx.SparseOperator:
    """A general sparse projector: two entries per row (not a row selection)."""
    rng = np.random.default_rng(seed)
    cols = np.stack([rng.choice(n, 2, replace=False) for _ in range(m)])
    values = rng.uniform(0.5, 1.5, (m, 2))
    return gx.SparseOperator.from_coo(
        np.repeat(np.arange(m), 2), cols.ravel(), jnp.asarray(values.ravel()), (m, n)
    )


def complement_basis(V: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal basis of ``{x : Vᵀx = 0}``."""
    return jnp.linalg.svd(V, full_matrices=True)[0][:, V.shape[1] :]


@eqx.filter_jit
def dense_laplace(Q, mu, lik, A=None, offset=0.0, B=None, init=None, iters=30):
    """Dense Laplace in ``x = B z``: ``(mode, log_marginal)``.

    ``log_marginal = f(Ax̂ + o) − ½(x̂−μ)ᵀQ(x̂−μ) + ½log|BᵀQB| − ½log|BᵀHB|``,
    which is the proper GMRF's formula for ``B = I`` and the hard-constrained
    one (with ``include_normalizer=True``) for ``B`` orthonormal on
    ``{Vᵀx = 0}``.
    """
    n = Q.shape[0]
    A = jnp.eye(n) if A is None else A
    B = jnp.eye(n) if B is None else B
    Qz = einsum(B, einsum(Q, B, "i j, j k -> i k"), "i a, i k -> a k")
    AB = A @ B

    def derivatives(z):
        eta = AB @ z + offset
        g = jax.grad(lik.log_prob)(eta)
        w = -jnp.diag(jax.hessian(lik.log_prob)(eta))
        return eta, g, w

    def step(_, z):
        _, g, w = derivatives(z)
        grad = einsum(AB, g, "m k, m -> k") - einsum(B, Q @ (B @ z - mu), "n k, n -> k")
        Hz = Qz + einsum(AB, einx.multiply("m b, m -> m b", AB, w), "m a, m b -> a b")
        return z + jnp.linalg.solve(Hz, grad)

    z = jnp.zeros(B.shape[1]) if init is None else einsum(B, init, "n k, n -> k")
    z = jax.lax.fori_loop(0, iters, step, z)
    eta, _, w = derivatives(z)
    Hz = Qz + einsum(AB, einx.multiply("m b, m -> m b", AB, w), "m a, m b -> a b")
    x = B @ z
    r = x - mu
    log_marginal = (
        lik.log_prob(eta)
        - 0.5 * r @ Q @ r
        + 0.5 * jnp.linalg.slogdet(Qz)[1]
        - 0.5 * jnp.linalg.slogdet(Hz)[1]
    )
    return x, log_marginal


def rw2_counts(n: int, seed: int = 0) -> jnp.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.poisson(np.exp(1.0 + np.sin(t / 3.0))), dtype=float)


def rw2_null(n: int) -> jnp.ndarray:
    t = jnp.arange(n, dtype=float)
    return jnp.column_stack([jnp.ones(n), t - t.mean()])


# ---------------------------------------------------------------------------
# Gaussian likelihood: exact conjugate result
# ---------------------------------------------------------------------------


class TestGaussianConjugate:
    @pytest.mark.parametrize("structure", ["sparse", "banded", "dense"])
    def test_mode_and_log_marginal_are_exact(self, structure):
        n, m, noise = 10, 7, 0.3
        rng = np.random.default_rng(1)
        L = path_laplacian(n)
        if structure == "sparse":
            Q = L.add_diagonal(jnp.full(n, 0.5), tags=lx.positive_semidefinite_tag)
        elif structure == "banded":
            Q = gx.ar1_precision(n, rho=0.7, tau=2.0)
        else:
            Q = lx.MatrixLinearOperator(
                L.as_matrix() + 0.5 * jnp.eye(n), lx.positive_semidefinite_tag
            )
        mu = jnp.asarray(rng.normal(size=n))
        A = two_per_row(m, n)
        offset = jnp.asarray(rng.normal(size=m))
        y = jnp.asarray(rng.normal(size=m))
        prior = gx.GaussianMRF(mu, Q)
        result = gx.laplace_mode(
            prior, gx.GaussianLikelihood(y, noise), projector=A, offset=offset
        )

        Qd, Ad = Q.as_matrix(), A.as_matrix()
        H = Qd + einsum(Ad, Ad, "m i, m j -> i j") / noise
        rhs = Qd @ mu + einsum(Ad, y - offset, "m i, m -> i") / noise
        mean = jnp.linalg.solve(H, rhs)
        cov = Ad @ jnp.linalg.solve(Qd, Ad.T) + noise * jnp.eye(m)
        evidence = scipy.stats.multivariate_normal(Ad @ mu + offset, cov).logpdf(y)
        assert result.converged
        assert result.n_iter <= 2  # one step, and one that confirms it
        assert jnp.allclose(result.mode, mean, atol=1e-10)
        assert jnp.allclose(result.log_marginal, evidence, atol=1e-9)
        assert jnp.allclose(result.hessian.as_matrix(), H, atol=1e-10)

    def test_banded_prior_with_identity_projector_stays_banded(self):
        n = 8
        y = jnp.linspace(-1.0, 1.0, n)
        prior = gx.GaussianMRF(jnp.zeros(n), gx.ar1_precision(n, rho=0.5, tau=1.0))
        result = gx.laplace_mode(prior, gx.GaussianLikelihood(y, 0.5))
        assert isinstance(result.hessian, gx.BlockTriDiag)
        Q = prior.precision.as_matrix()
        H = Q + 2.0 * jnp.eye(n)
        cov = jnp.linalg.inv(Q) + 0.5 * jnp.eye(n)
        evidence = scipy.stats.multivariate_normal(jnp.zeros(n), cov).logpdf(y)
        assert jnp.allclose(result.mode, jnp.linalg.solve(H, 2.0 * y), atol=1e-10)
        assert jnp.allclose(result.log_marginal, evidence, atol=1e-9)


# ---------------------------------------------------------------------------
# Modes against dense Newton
# ---------------------------------------------------------------------------


class TestModes:
    def test_poisson_rw2_matches_dense_newton(self):
        n = 16
        counts = rw2_counts(n)
        V = rw2_null(n)
        R = gx.rw2_structure(n)
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 3.0, R, V, include_normalizer=True)
        result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts))
        mode, log_marginal = dense_laplace(
            3.0 * R.as_matrix(),
            jnp.zeros(n),
            gx.PoissonLikelihood(counts),
            B=complement_basis(V),
        )
        assert isinstance(result.hessian, gx.BlockTriDiag)
        assert result.converged
        assert jnp.allclose(result.mode, mode, atol=1e-10)
        assert jnp.allclose(einsum(V, result.mode, "n c, n -> c"), 0.0, atol=1e-10)
        assert jnp.allclose(result.log_marginal, log_marginal, atol=1e-9)

    @pytest.mark.slow
    def test_binomial_rw2_matches_dense_newton(self):
        n = 14
        rng = np.random.default_rng(2)
        trials = jnp.asarray(rng.integers(1, 10, n), dtype=float)
        p = 1.0 / (1.0 + np.exp(-np.cos(np.arange(n) / 2.0)))
        successes = jnp.asarray(rng.binomial(np.asarray(trials, int), p), dtype=float)
        lik = gx.BinomialLikelihood(successes, trials)
        V = rw2_null(n)
        R = gx.rw2_structure(n)
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 5.0, R, V)
        result = gx.laplace_mode(prior, lik)
        mode, _ = dense_laplace(
            5.0 * R.as_matrix(), jnp.zeros(n), lik, B=complement_basis(V)
        )
        assert result.converged
        assert jnp.allclose(result.mode, mode, atol=1e-10)

    @pytest.mark.slow
    def test_odd_rw2_with_padding_node(self):
        """Odd ``n``: ``rw2_structure(n)`` has a padding node at index ``n``.

        Observe the first ``n`` nodes through a row selection (so ``H`` stays
        `BlockTriDiag`), with the null space zero on the padding node; the
        padding node then has mode 0 and the rest equal the size-``n`` model.
        """
        n = 13
        counts = rw2_counts(n, seed=3)
        V_n = rw2_null(n)
        V = jnp.zeros((n + 1, 2)).at[:n].set(V_n)
        select = gx.SparseOperator.from_coo(
            np.arange(n), np.arange(n), jnp.ones(n), (n, n + 1)
        )
        R = gx.rw2_structure(n)
        assert R.in_size() == n + 1
        prior = gx.IntrinsicGMRF(jnp.zeros(n + 1), 4.0, R, V)
        result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts), projector=select)
        mode, _ = dense_laplace(
            4.0 * R.as_matrix()[:n, :n],
            jnp.zeros(n),
            gx.PoissonLikelihood(counts),
            B=complement_basis(V_n),
        )
        assert isinstance(result.hessian, gx.BlockTriDiag)
        assert jnp.allclose(result.mode[:n], mode, atol=1e-10)
        assert jnp.allclose(result.mode[n], 0.0, atol=1e-12)

    @pytest.mark.slow
    def test_negative_binomial_bym2_matches_dense_newton(self):
        n = 9
        laplacian = path_laplacian(n)
        # Close the path into a cycle with two chords.
        senders, receivers = np.array([n - 1, 5, 7]), np.array([0, 1, 3])
        chords = gx.SparseOperator.from_coo(
            np.r_[senders, receivers, senders],
            np.r_[senders, receivers, receivers],
            jnp.r_[jnp.ones(6), -jnp.ones(3)],
            (n, n),
            symmetric=True,
        )
        R = gx.besag_structure(laplacian.union(chords))
        s = gx.generalized_variance_scale(R, jnp.ones(n))
        Q_bym2 = gx.bym2_precision(s * R, tau=2.0, phi=0.6)
        # Latent (b, u*, β₀): BYM2 plus an intercept with precision 0.01.
        size = 2 * n + 1
        pattern = Q_bym2.pattern
        Q = gx.SparseOperator.from_coo(
            np.r_[pattern.rows, 2 * n],
            np.r_[pattern.cols, 2 * n],
            jnp.r_[Q_bym2.values, 0.01],
            (size, size),
            symmetric=True,
        )
        A = gx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(n)],
            np.r_[np.arange(n), np.full(n, 2 * n)],
            jnp.ones(2 * n),
            (n, size),
        )
        V = jnp.zeros((size, 1)).at[n : 2 * n, 0].set(1.0)
        rng = np.random.default_rng(4)
        y = jnp.asarray(rng.negative_binomial(3, 0.3, n), dtype=float)
        lik = gx.NegativeBinomialLikelihood(y, 3.0)
        offset = jnp.asarray(rng.normal(scale=0.2, size=n))
        prior = gx.IntrinsicGMRF(jnp.zeros(size), 1.0, Q, V)
        result = gx.laplace_mode(prior, lik, projector=A, offset=offset)
        mode, _ = dense_laplace(
            Q.as_matrix(),
            jnp.zeros(size),
            lik,
            A=A.as_matrix(),
            offset=offset,
            B=complement_basis(V),
        )
        assert isinstance(result.hessian, gx.SparseOperator)
        assert result.converged
        assert jnp.allclose(result.mode, mode, atol=1e-9)
        assert jnp.allclose(jnp.sum(result.mode[n : 2 * n]), 0.0, atol=1e-10)


# The structure paths that the fast lane covers; the rest of the grid is slow.
_FAST_CASES = {("banded", "sparse"), ("kronecker_sum", "selection"), ("dense", "dense")}


class TestStructures:
    """Each structure path of ``H`` gives the dense result."""

    @staticmethod
    def _priors(n):
        L = path_laplacian(n)
        dense = L.as_matrix() + 0.7 * jnp.eye(n)
        grid = gx.KroneckerSum(
            lx.MatrixLinearOperator(path_laplacian(2).as_matrix() + 0.5 * jnp.eye(2)),
            lx.MatrixLinearOperator(path_laplacian(n // 2).as_matrix()),
        )
        return {
            "sparse": L.add_diagonal(jnp.full(n, 0.7)),
            "banded": gx.ar1_precision(n, rho=0.6, tau=1.5),
            "diagonal": lx.DiagonalLinearOperator(jnp.full(n, 1.3)),
            "dense": lx.MatrixLinearOperator(dense, lx.positive_semidefinite_tag),
            "kronecker_sum": grid,
        }

    @pytest.mark.parametrize(
        ("structure", "projector"),
        [
            pytest.param(
                structure,
                projector,
                marks=() if (structure, projector) in _FAST_CASES else pytest.mark.slow,
            )
            for structure in ["sparse", "banded", "diagonal", "dense", "kronecker_sum"]
            for projector in ["identity", "selection", "sparse", "dense"]
        ],
    )
    def test_matches_dense(self, structure, projector):
        n, m = 8, 6
        Q = self._priors(n)[structure]
        rng = np.random.default_rng(5)
        if projector == "identity":
            A, m = None, n
        elif projector == "selection":
            A = gx.SparseOperator.from_coo(
                np.arange(m), rng.choice(n, m, replace=False), jnp.full(m, 2.0), (m, n)
            )
        elif projector == "sparse":
            A = two_per_row(m, n, seed=6)
        else:
            A = lx.MatrixLinearOperator(jnp.asarray(rng.normal(size=(m, n))) / 3.0)
        counts = jnp.asarray(rng.poisson(2.0, m), dtype=float)
        mu = jnp.asarray(rng.normal(size=n)) / 4.0
        lik = gx.PoissonLikelihood(counts)
        result = gx.laplace_mode(gx.GaussianMRF(mu, Q), lik, projector=A)
        mode, log_marginal = dense_laplace(
            Q.as_matrix(), mu, lik, A=None if A is None else A.as_matrix()
        )
        assert result.converged
        assert jnp.allclose(result.mode, mode, atol=1e-10)
        assert jnp.allclose(result.log_marginal, log_marginal, atol=1e-9)
        assert jnp.allclose(
            result.factor.solve(mode),
            jnp.linalg.solve(result.hessian.as_matrix(), mode),
            atol=1e-9,
        )

    def test_structure_classes(self):
        n = 8
        priors = self._priors(n)
        lik = gx.PoissonLikelihood(jnp.ones(n))
        cases = {
            ("banded", None): gx.BlockTriDiag,
            ("sparse", None): gx.SparseOperator,
            ("diagonal", None): lx.DiagonalLinearOperator,
            ("banded", "sparse"): gx.SparseOperator,
            ("diagonal", "sparse"): gx.SparseOperator,
            ("dense", "sparse"): lx.MatrixLinearOperator,
            ("kronecker_sum", "sparse"): lx.TaggedLinearOperator,
        }
        A = two_per_row(n, n)
        for (structure, projector), expected in cases.items():
            prior = gx.GaussianMRF(jnp.zeros(n), priors[structure])
            result = gx.laplace_mode(
                prior, lik, projector=None if projector is None else A
            )
            assert isinstance(result.hessian, expected), (structure, projector)
        sparse = gx.laplace_mode(gx.GaussianMRF(jnp.zeros(n), priors["sparse"]), lik)
        assert isinstance(sparse.factor, gx.SparseCholeskyFactor)


# ---------------------------------------------------------------------------
# Golden: R-INLA on the Scotland lip cancer data
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).resolve().parents[2] / "scripts" / "golden" / "inla"


@pytest.mark.slow
def test_scotland_bym2_poisson_mode_matches_r_inla():
    """Golden: R-INLA's latent mode at fixed ``(τ, φ)``.

    Generated offline by scripts/golden/inla/scotland_bym2.R, with R-INLA's
    VB correction, stabilising diagonal and predictor noise switched off so
    that its mean is the exact constrained mode (agreement ~5e-7). The graph
    is the one of scotland_scale.json.
    """
    fixture = json.loads((GOLDEN / "scotland_bym2.json").read_text())
    graph = json.loads((GOLDEN / "scotland_scale.json").read_text())
    n = fixture["n"]
    senders, receivers = np.asarray(graph["senders"]), np.asarray(graph["receivers"])
    degree = np.bincount(np.r_[senders, receivers], minlength=n).astype(float)
    laplacian = gx.SparseOperator.from_coo(
        np.r_[np.arange(n), senders],
        np.r_[np.arange(n), receivers],
        jnp.asarray(np.r_[degree, -np.ones(senders.shape[0])]),
        (n, n),
        symmetric=True,
    )
    R = gx.besag_structure(laplacian)
    s = gx.generalized_variance_scale(R, jnp.ones(n))
    Q_bym2 = gx.bym2_precision(s * R, tau=fixture["tau"], phi=fixture["phi"])
    # Latent (b, u*, β₀, β₁) with N(0, 1/prec_fixed) fixed effects.
    size = 2 * n + 2
    Q = gx.SparseOperator.from_coo(
        np.r_[Q_bym2.pattern.rows, 2 * n, 2 * n + 1],
        np.r_[Q_bym2.pattern.cols, 2 * n, 2 * n + 1],
        jnp.r_[Q_bym2.values, fixture["prec_fixed"], fixture["prec_fixed"]],
        (size, size),
        symmetric=True,
    )
    region = np.asarray(fixture["region"])
    covariate = jnp.asarray(fixture["covariate"], dtype=float)
    A = gx.SparseOperator.from_coo(
        np.r_[np.arange(n), np.arange(n), np.arange(n)],
        np.r_[region, np.full(n, 2 * n), np.full(n, 2 * n + 1)],
        jnp.r_[jnp.ones(2 * n), covariate],
        (n, size),
    )
    V = jnp.zeros((size, 1)).at[n : 2 * n, 0].set(1.0)
    prior = gx.IntrinsicGMRF(jnp.zeros(size), 1.0, Q, V)
    lik = gx.PoissonLikelihood(jnp.asarray(fixture["counts"], dtype=float))
    offset = jnp.log(jnp.asarray(fixture["expected"]))
    result = gx.laplace_mode(prior, lik, projector=A, offset=offset)
    expected = jnp.r_[
        jnp.asarray(fixture["mode_b"]),
        jnp.asarray(fixture["mode_u"]),
        fixture["mode_intercept"],
        fixture["mode_slope"],
    ]
    assert result.converged
    assert isinstance(result.hessian, gx.SparseOperator)
    assert jnp.allclose(result.mode, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Gradients: implicit mode, log-marginal, reverse-over-reverse Hessian
# ---------------------------------------------------------------------------


def rw2_log_marginal(counts, n):
    R = gx.rw2_structure(n)
    V = rw2_null(n)

    def log_marginal(log_tau):
        prior = gx.IntrinsicGMRF(
            jnp.zeros(n), jnp.exp(log_tau), R, V, include_normalizer=True
        )
        return gx.laplace_mode(prior, gx.PoissonLikelihood(counts)).log_marginal

    def reference(log_tau, init):
        return dense_laplace(
            jnp.exp(log_tau) * R.as_matrix(),
            jnp.zeros(n),
            gx.PoissonLikelihood(counts),
            B=complement_basis(V),
            init=init,
            iters=2,
        )

    return log_marginal, reference


def sparse_problem(n=10):
    """Poisson on a sparse GMRF ``Q(θ) = τ (L + κ I)``, ``θ = (log τ, log κ)``."""
    L = path_laplacian(n)
    counts = rw2_counts(n, seed=7)

    def precision(theta):
        Q = L.add_diagonal(jnp.exp(theta[1]) * jnp.ones(n))
        return eqx.tree_at(lambda op: op.values, Q, jnp.exp(theta[0]) * Q.values)

    def log_marginal(theta):
        prior = gx.GaussianMRF(jnp.zeros(n), precision(theta))
        return gx.laplace_mode(prior, gx.PoissonLikelihood(counts)).log_marginal

    def reference(theta, init):
        Q = precision(theta).as_matrix()
        lik = gx.PoissonLikelihood(counts)
        return dense_laplace(Q, jnp.zeros(n), lik, init=init, iters=2)

    return log_marginal, reference, precision, counts


def _mode_rw2(counts, n, log_tau):
    prior = gx.IntrinsicGMRF(
        jnp.zeros(n), jnp.exp(log_tau), gx.rw2_structure(n), rw2_null(n)
    )
    return gx.laplace_mode(prior, gx.PoissonLikelihood(counts)).mode


class TestGradients:
    @pytest.mark.slow
    def test_poisson_rw2_grad_log_tau_matches_finite_differences(self):
        n = 12
        counts = rw2_counts(n)
        log_marginal, _ = rw2_log_marginal(counts, n)
        theta, h = 0.4, 1e-5
        grad = jax.grad(log_marginal)(theta)
        fd = (log_marginal(theta + h) - log_marginal(theta - h)) / (2 * h)
        assert jnp.allclose(grad, fd, rtol=1e-6, atol=1e-7)

    @pytest.mark.slow
    def test_mode_jacobian_matches_dense(self):
        n = 12
        counts = rw2_counts(n)
        _, reference = rw2_log_marginal(counts, n)
        theta = 0.4
        mode = _mode_rw2(counts, n, theta)
        jac = jax.jacrev(lambda a: _mode_rw2(counts, n, a))(theta)
        dense = jax.jacrev(lambda a: reference(a, mode)[0])(theta)
        fd = (
            _mode_rw2(counts, n, theta + 1e-6) - _mode_rw2(counts, n, theta - 1e-6)
        ) / 2e-6
        assert jnp.allclose(jac, dense, atol=1e-10)
        assert jnp.allclose(jac, fd, atol=1e-7)

    @pytest.mark.slow
    def test_sparse_grad_matches_dense_and_finite_differences(self):
        log_marginal, reference, precision, counts = sparse_problem()
        theta = jnp.array([0.3, -0.5])
        prior = gx.GaussianMRF(jnp.zeros(10), precision(theta))
        mode = gx.laplace_mode(prior, gx.PoissonLikelihood(counts)).mode
        grad = jax.grad(log_marginal)(theta)
        dense = jax.grad(lambda t: reference(t, mode)[1])(theta)
        h = 1e-5
        fd = jnp.array(
            [
                (log_marginal(theta + h * e) - log_marginal(theta - h * e)) / (2 * h)
                for e in jnp.eye(2)
            ]
        )
        assert jnp.allclose(grad, dense, atol=1e-10)
        assert jnp.allclose(grad, fd, atol=1e-6)

    @pytest.mark.slow
    def test_negative_binomial_concentration_gradient(self):
        n = 8
        rng = np.random.default_rng(8)
        y = jnp.asarray(rng.negative_binomial(2, 0.4, n), dtype=float)
        prior = gx.GaussianMRF(jnp.zeros(n), gx.ar1_precision(n, rho=0.5, tau=1.0))

        def log_marginal(log_r):
            lik = gx.NegativeBinomialLikelihood(y, jnp.exp(log_r))
            return gx.laplace_mode(prior, lik).log_marginal

        def reference(log_r):
            lik = gx.NegativeBinomialLikelihood(y, jnp.exp(log_r))
            return dense_laplace(prior.precision.as_matrix(), jnp.zeros(n), lik)[1]

        theta, h = 0.7, 1e-5
        fd = (log_marginal(theta + h) - log_marginal(theta - h)) / (2 * h)
        assert jnp.allclose(log_marginal(theta), reference(theta), atol=1e-9)
        assert jnp.allclose(jax.grad(log_marginal)(theta), fd, atol=1e-6)

    @pytest.mark.slow
    def test_rw2_reverse_over_reverse_hessian_matches_dense(self):
        """`theta_design`'s default Hessian (P8 depends on it), banded path."""
        n = 12
        counts = rw2_counts(n)
        log_marginal, reference = rw2_log_marginal(counts, n)
        theta = 0.4
        mode = _mode_rw2(counts, n, theta)
        hess = jax.jacrev(jax.jacrev(log_marginal))(theta)
        dense = jax.jacrev(jax.jacrev(lambda a: reference(a, mode)[1]))(theta)
        assert jnp.allclose(hess, dense, rtol=1e-8, atol=1e-9)

    @pytest.mark.slow
    def test_sparse_reverse_over_reverse_hessian_matches_dense(self):
        """`theta_design`'s default Hessian through the sparse Cholesky VJPs."""
        log_marginal, reference, precision, counts = sparse_problem()
        theta = jnp.array([0.3, -0.5])
        prior = gx.GaussianMRF(jnp.zeros(10), precision(theta))
        mode = gx.laplace_mode(prior, gx.PoissonLikelihood(counts)).mode
        hess = jax.jacrev(jax.jacrev(log_marginal))(theta)
        dense = jax.jacrev(jax.jacrev(lambda t: reference(t, mode)[1]))(theta)
        assert jnp.allclose(hess, hess.T, atol=1e-10)
        assert jnp.allclose(hess, dense, rtol=1e-8, atol=1e-9)

    @pytest.mark.slow
    def test_jit_and_vmap_over_theta(self):
        n = 12
        counts = rw2_counts(n)
        log_marginal, _ = rw2_log_marginal(counts, n)
        thetas = jnp.array([0.0, 0.5, 1.0])
        batched = jax.jit(jax.vmap(log_marginal))(thetas)
        looped = jnp.array([log_marginal(t) for t in thetas])
        assert jnp.allclose(batched, looped, atol=1e-10)


# ---------------------------------------------------------------------------
# Options and errors
# ---------------------------------------------------------------------------


class TestOptions:
    def test_non_converged_is_reported(self):
        n = 12
        counts = rw2_counts(n) * 10.0
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 1.0, gx.rw2_structure(n), rw2_null(n))
        result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts), max_iter=2)
        assert result.n_iter == 2
        assert not result.converged

    @pytest.mark.slow
    def test_damping_and_init_reach_the_same_mode(self):
        n = 12
        counts = rw2_counts(n)
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 2.0, gx.rw2_structure(n), rw2_null(n))
        lik = gx.PoissonLikelihood(counts)
        full = gx.laplace_mode(prior, lik)
        damped = gx.laplace_mode(
            prior, lik, damping=0.7, init=jnp.ones(n), max_iter=200
        )
        assert damped.converged
        assert damped.n_iter > full.n_iter
        assert jnp.allclose(damped.mode, full.mode, atol=1e-7)

    @pytest.mark.slow
    def test_float32_stays_float32(self):
        n = 12
        counts = rw2_counts(n).astype(jnp.float32)
        R = gx.rw2_structure(n)
        R32 = gx.BlockTriDiag(
            R.diagonal.astype(jnp.float32), R.sub_diagonal.astype(jnp.float32)
        )
        V = rw2_null(n).astype(jnp.float32)
        prior = gx.IntrinsicGMRF(jnp.zeros(n, jnp.float32), 2.0, R32, V)
        result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts))
        assert result.mode.dtype == jnp.float32
        assert result.log_marginal.dtype == jnp.float32
        assert result.converged

    def test_soft_constraint_raises(self):
        prior = gx.IntrinsicGMRF(
            jnp.zeros(4), 1.0, gx.rw1_structure(4), jnp.ones(4), constraint="soft"
        )
        with pytest.raises(NotImplementedError, match="soft"):
            gx.laplace_mode(prior, gx.PoissonLikelihood(jnp.ones(4)))

    def test_bad_prior_and_projector_raise(self):
        mvn = gx.MultivariateNormalPrecision(
            jnp.zeros(3), lx.MatrixLinearOperator(jnp.eye(3))
        )
        with pytest.raises(TypeError, match="GaussianMRF"):
            gx.laplace_mode(mvn, gx.PoissonLikelihood(jnp.ones(3)))
        prior = gx.GaussianMRF(jnp.zeros(3), lx.DiagonalLinearOperator(jnp.ones(3)))
        A = lx.MatrixLinearOperator(jnp.ones((2, 4)))
        with pytest.raises(ValueError, match="columns"):
            gx.laplace_mode(prior, gx.PoissonLikelihood(jnp.ones(2)), projector=A)

    @pytest.mark.slow
    def test_none_constraint_runs_unconstrained(self):
        n = 10
        counts = rw2_counts(n)
        R = gx.rw2_structure(n)
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 2.0, R, rw2_null(n), constraint="none")
        result = gx.laplace_mode(prior, gx.PoissonLikelihood(counts))
        mode, _ = dense_laplace(
            2.0 * R.as_matrix(), jnp.zeros(n), gx.PoissonLikelihood(counts)
        )
        assert jnp.allclose(result.mode, mode, atol=1e-10)
