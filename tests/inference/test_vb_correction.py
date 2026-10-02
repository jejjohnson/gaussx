"""Tests for `vb_mean_correction`: the low-rank VB mean shift of the Laplace mode.

The reference is a dense implementation of the same objective,
``L(δ) = Σ_k w_k log p(y | m(δ) + √v z_k) − ½ (x̄ − μ)ᵀ Q (x̄ − μ)`` with
``x̄ = x̂ + Σ S δ``, ``Σ`` the (constrained) inverse of the dense Hessian at the
mode and ``(z_k, w_k)`` the Gauss-Hermite rule, maximised by Newton with JAX's
own gradient and Hessian. The posterior-mean references are a 2-D grid
quadrature (exact to ~1e-9) and importance sampling (slow, bounded by its own
standard error); the golden is R-INLA's ``control.vb`` mean.
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

import gaussx as gx
from gaussx._einx import einsum, rearrange, reduce
from gaussx._inference._vb_correction import _pair_plan, _unconstrained_variances


GOLDEN = Path(__file__).parents[2] / "scripts" / "golden" / "inla"


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


def proper_rw2(n: int, tau: float = 4.0, shift: float = 0.05) -> gx.BlockTriDiag:
    """``τ R₂ + shift I`` as a `BlockTriDiag` (even ``n``)."""
    R = gx.rw2_structure(n)
    return gx.BlockTriDiag(tau * R.diagonal + shift * jnp.eye(2), tau * R.sub_diagonal)


def interpolation(m: int, n: int, seed: int = 0) -> gx.SparseOperator:
    """1-D linear (FEM) interpolation: row ``k`` touches neighbours ``j, j+1``."""
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n - 1, m)
    s = rng.uniform(0.0, 1.0, m)
    return gx.SparseOperator.from_coo(
        np.repeat(np.arange(m), 2),
        np.column_stack([left, left + 1]).ravel(),
        jnp.asarray(np.column_stack([1.0 - s, s]).ravel()),
        (m, n),
    )


def two_per_row(m: int, n: int, seed: int = 0) -> gx.SparseOperator:
    """Two entries per row on random, generally non-adjacent nodes."""
    rng = np.random.default_rng(seed)
    cols = np.stack([rng.choice(n, 2, replace=False) for _ in range(m)])
    values = rng.uniform(0.5, 1.5, (m, 2))
    return gx.SparseOperator.from_coo(
        np.repeat(np.arange(m), 2), cols.ravel(), jnp.asarray(values.ravel()), (m, n)
    )


def counts(n: int, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.poisson(np.exp(-0.5 + np.sin(np.arange(n) / 3.0))), float)


def detections(n: int, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(1.5 - 1.5 * np.sin(np.arange(n) / 4.0)))
    return jnp.asarray(rng.random(n) < p, float)


def dense_covariance(Q, lik, mode, A, offset, V=None):
    """``Σ`` of the Laplace approximation at ``mode``, densely."""
    eta = A @ mode + offset
    w = -jnp.diag(jax.hessian(lik.log_prob)(eta))
    H = Q + einsum(A, einx.multiply("m j, m -> m j", A, w), "m i, m j -> i j")
    sigma = jnp.linalg.inv(H)
    if V is not None:
        SV = sigma @ V
        gram = einsum(V, SV, "n c, n d -> c d")
        sigma = sigma - einsum(
            SV, jnp.linalg.solve(gram, rearrange(SV, "n c -> c n")), "i c, c j -> i j"
        )
    return sigma


def dense_vb(Q, mu, lik, mode, S, A=None, offset=0.0, V=None):
    """Maximise the VB objective densely, with JAX's gradient and Hessian."""
    n = Q.shape[0]
    A = jnp.eye(n) if A is None else A
    return _dense_vb(Q, mu, lik, mode, S, A, jnp.asarray(offset), V)


@eqx.filter_jit
def _dense_vb(Q, mu, lik, mode, S, A, offset, V, order=20, iters=30):
    sigma = dense_covariance(Q, lik, mode, A, offset, V)
    sd = jnp.sqrt(einsum(A, A @ sigma, "m i, m i -> m"))
    U = sigma @ S
    z, w = np.polynomial.hermite_e.hermegauss(order)
    z, w = jnp.asarray(z), jnp.asarray(w / w.sum())

    def objective(delta):
        x = mode + U @ delta
        m = A @ x + offset
        ell = w @ jax.vmap(lambda zk: lik.log_prob(m + sd * zk))(z)
        r = x - mu
        return ell - 0.5 * r @ Q @ r

    def step(_, delta):
        return delta - jnp.linalg.solve(
            jax.hessian(objective)(delta), jax.grad(objective)(delta)
        )

    delta = jax.lax.fori_loop(0, iters, step, jnp.zeros(S.shape[1]))
    return mode + U @ delta


# ---------------------------------------------------------------------------
# Gaussian likelihood: the mode is the posterior mean, so no correction
# ---------------------------------------------------------------------------


class TestGaussianLikelihood:
    @pytest.mark.parametrize(
        "case",
        [
            "banded",
            pytest.param("sparse_projector", marks=pytest.mark.slow),
            pytest.param("intrinsic", marks=pytest.mark.slow),
        ],
    )
    def test_correction_is_zero(self, case):
        n = 20
        y = jnp.asarray(np.random.default_rng(0).normal(size=n))
        if case == "banded":
            prior, A = gx.GaussianMRF(jnp.zeros(n), proper_rw2(n)), None
            lik = gx.GaussianLikelihood(y, 0.3)
        elif case == "sparse_projector":
            Q = path_laplacian(n).add_diagonal(jnp.full(n, 0.5))
            prior, A = gx.GaussianMRF(jnp.linspace(-1, 1, n), Q), interpolation(30, n)
            lik = gx.GaussianLikelihood(jnp.r_[y, y[:10]], 0.3)
        else:
            t = jnp.arange(n, dtype=float)
            V = jnp.column_stack([jnp.ones(n), t - t.mean()])
            prior = gx.IntrinsicGMRF(jnp.zeros(n), 3.0, gx.rw2_structure(n), V)
            A, lik = None, gx.GaussianLikelihood(y, 0.3)
        res = gx.laplace_mode(prior, lik, projector=A)
        mean = gx.vb_mean_correction(
            res, prior, lik, projector=A, subspace=jnp.arange(n)
        )
        # Zero up to rounding: the ELBO's gradient at δ = 0 is Σ Aᵀ∇l(x̂) = 0.
        np.testing.assert_allclose(mean, res.mode, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# Against the dense VB objective
# ---------------------------------------------------------------------------


class TestDenseReference:
    def test_poisson_banded_identity(self):
        n = 24
        Q = proper_rw2(n)
        prior, lik = gx.GaussianMRF(jnp.zeros(n), Q), gx.PoissonLikelihood(counts(n))
        res = gx.laplace_mode(prior, lik)
        assert isinstance(res.hessian, gx.BlockTriDiag)
        mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(n))
        ref = dense_vb(Q.as_matrix(), prior.loc, lik, res.mode, jnp.eye(n))
        np.testing.assert_allclose(mean, ref, atol=1e-9)
        assert jnp.max(jnp.abs(mean - res.mode)) > 1e-2  # a real correction

    @pytest.mark.slow
    def test_bernoulli_index_subspace_and_offset(self):
        n = 24
        Q = proper_rw2(n)
        prior, lik = (
            gx.GaussianMRF(jnp.zeros(n), Q),
            gx.BernoulliLikelihood(detections(n)),
        )
        offset = jnp.linspace(-0.5, 0.5, n)
        res = gx.laplace_mode(prior, lik, offset=offset)
        idx = jnp.array([0, 5, 11, 23])
        mean = gx.vb_mean_correction(res, prior, lik, offset=offset, subspace=idx)
        S = jnp.eye(n)[:, idx]
        ref = dense_vb(Q.as_matrix(), prior.loc, lik, res.mode, S, offset=offset)
        np.testing.assert_allclose(mean, ref, atol=1e-9)

    @pytest.mark.slow
    @pytest.mark.parametrize("projector", ["interpolation", "two_per_row"])
    def test_poisson_sparse_projector(self, projector):
        n, m = 16, 30
        Q = path_laplacian(n).add_diagonal(jnp.full(n, 0.3))
        A = interpolation(m, n) if projector == "interpolation" else two_per_row(m, n)
        prior, lik = gx.GaussianMRF(jnp.zeros(n), Q), gx.PoissonLikelihood(counts(m))
        res = gx.laplace_mode(prior, lik, projector=A)
        assert isinstance(res.hessian, gx.SparseOperator)
        plan = _pair_plan(A.pattern, res.factor.symbolic.inverse_plan[0])
        # Each row of A is a clique of AᵀWA ⊆ pattern(H): always Takahashi.
        assert plan is not None
        basis = jnp.asarray(np.random.default_rng(1).normal(size=(n, 3)))
        mean = gx.vb_mean_correction(res, prior, lik, projector=A, subspace=basis)
        ref = dense_vb(Q.as_matrix(), prior.loc, lik, res.mode, basis, A=A.as_matrix())
        np.testing.assert_allclose(mean, ref, atol=1e-9)

    @pytest.mark.slow
    def test_selection_projector_and_hard_constraint(self):
        n, m = 20, 14
        t = jnp.arange(n, dtype=float)
        V = jnp.column_stack([jnp.ones(n), t - t.mean()])
        prior = gx.IntrinsicGMRF(jnp.zeros(n), 2.0, gx.rw2_structure(n), V)
        obs = np.sort(np.random.default_rng(2).choice(n, m, replace=False))
        A = gx.SparseOperator.from_coo(np.arange(m), obs, jnp.full(m, 1.5), (m, n))
        lik = gx.PoissonLikelihood(counts(m) + 1.0)
        offset = 0.5
        res = gx.laplace_mode(prior, lik, projector=A, offset=offset)
        assert isinstance(res.hessian, gx.BlockTriDiag)
        mean = gx.vb_mean_correction(
            res, prior, lik, projector=A, offset=offset, subspace=jnp.arange(n)
        )
        Q = 2.0 * gx.rw2_structure(n).as_matrix()
        # The same span as S = I (Σ's range, {Vᵀx = 0}) with a full-rank basis.
        basis = jnp.linalg.svd(V, full_matrices=True)[0][:, 2:]
        ref = dense_vb(
            Q, prior.loc, lik, res.mode, basis, A=A.as_matrix(), offset=offset, V=V
        )
        np.testing.assert_allclose(mean, ref, atol=1e-9)
        np.testing.assert_allclose(einsum(V, mean, "n c, n -> c"), 0.0, atol=1e-10)

    def test_dense_hessian(self):
        n = 6
        rng = np.random.default_rng(3)
        B = rng.normal(size=(n, n))
        Q = jnp.asarray(einsum(B, B, "i k, j k -> i j") + n * np.eye(n))
        prior = gx.GaussianMRF(jnp.zeros(n), lx.MatrixLinearOperator(Q))
        lik = gx.PoissonLikelihood(counts(n))
        res = gx.laplace_mode(prior, lik)
        mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(n))
        np.testing.assert_allclose(
            mean, dense_vb(Q, prior.loc, lik, res.mode, jnp.eye(n)), atol=1e-9
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", ["banded", "selection", "interpolation", "two_per_row", "dense"]
)
def test_predictor_variances_equal_dense(case):
    """``diag(A Σ Aᵀ)`` on each path: block, selection, Takahashi pairs, solves."""
    from gaussx._inference._laplace import _model

    n = 16
    A = None
    lik = gx.PoissonLikelihood(counts(n))
    Q = path_laplacian(n).add_diagonal(jnp.full(n, 0.3))
    if case == "banded":
        Q = proper_rw2(n)
    elif case == "selection":
        A = gx.SparseOperator.from_coo(
            np.arange(8), np.arange(0, n, 2), 2.0 * jnp.ones(8), (8, n)
        )
        lik = gx.PoissonLikelihood(counts(8))
    elif case in ("interpolation", "two_per_row"):
        A = interpolation(20, n) if case == "interpolation" else two_per_row(20, n)
        lik = gx.PoissonLikelihood(counts(20))
    elif case == "dense":
        dense = np.random.default_rng(4).uniform(0.0, 0.3, (10, n))
        A = lx.MatrixLinearOperator(jnp.asarray(dense))
        lik = gx.PoissonLikelihood(counts(10))
    prior = gx.GaussianMRF(jnp.zeros(n), Q)
    res = gx.laplace_mode(prior, lik, projector=A)
    dense_A = jnp.eye(n) if A is None else A.as_matrix()
    sigma = dense_covariance(Q.as_matrix(), lik, res.mode, dense_A, 0.0)
    v = _unconstrained_variances(_model(prior, lik, A, None), res)
    expected = einsum(dense_A, dense_A @ sigma, "m i, m i -> m")
    np.testing.assert_allclose(v, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Towards the posterior mean
# ---------------------------------------------------------------------------


def test_moves_towards_quadrature_posterior_mean():
    """Two latent nodes, small Poisson counts: exact mean by grid quadrature."""
    Q = jnp.array([[2.0, -0.8], [-0.8, 1.5]])
    prior = gx.GaussianMRF(jnp.zeros(2), lx.MatrixLinearOperator(Q))
    lik = gx.PoissonLikelihood(jnp.array([0.0, 1.0]))
    res = gx.laplace_mode(prior, lik)
    mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(2))

    # Trapezoid grid over ±10 Laplace sds: the tails beyond are < e⁻⁵⁰.
    sd = jnp.sqrt(jnp.diag(jnp.linalg.inv(res.hessian.as_matrix())))
    axes = [
        jnp.linspace(c - 10 * s, c + 10 * s, 801)
        for c, s in zip(res.mode, sd, strict=True)
    ]
    x0, x1 = jnp.meshgrid(*axes, indexing="ij")
    pts = jnp.column_stack(
        [rearrange(x0, "a b -> (a b)"), rearrange(x1, "a b -> (a b)")]
    )
    log_post = jax.vmap(lambda x: lik.log_prob(x) - 0.5 * x @ Q @ x)(pts)
    w = jnp.exp(log_post - log_post.max())
    exact = einsum(w, pts, "k, k d -> d") / jnp.sum(w)

    laplace_error = jnp.linalg.norm(res.mode - exact)
    vb_error = jnp.linalg.norm(mean - exact)
    assert laplace_error > 0.05
    assert vb_error < 0.2 * laplace_error


@pytest.mark.slow
def test_moves_towards_importance_sampled_mean():
    """Bernoulli / proper RW2, 24 nodes: self-normalised IS from a Student-t.

    The bound is the estimator's own: 5 standard errors (delta method,
    ``SE_i² = Σ_k w̃_k² (x_ki − x̄_i)²``) in the norm.
    """
    n = 24
    Q = proper_rw2(n)
    prior, lik = (
        gx.GaussianMRF(jnp.zeros(n), Q),
        gx.BernoulliLikelihood(detections(n, 1)),
    )
    res = gx.laplace_mode(prior, lik)
    mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(n))

    sigma = jnp.linalg.inv(res.hessian.as_matrix())
    chol = jnp.linalg.cholesky(sigma)
    df, size = 6.0, 400_000
    k1, k2 = jax.random.split(jax.random.key(0))
    z = jax.random.normal(k1, (size, n))
    g = jax.random.chisquare(k2, df, (size,)) / df
    x = res.mode + einx.divide(
        "k n, k -> k n", einsum(z, chol, "k j, n j -> k n"), jnp.sqrt(g)
    )
    Qd = Q.as_matrix()
    log_target = jax.vmap(lambda xi: lik.log_prob(xi) - 0.5 * xi @ Qd @ xi)(x)
    white = jax.scipy.linalg.solve_triangular(
        chol, rearrange(x - res.mode, "k n -> n k"), lower=True
    )
    maha = reduce(jnp.square(white), "n k -> k", "sum")
    log_proposal = -0.5 * (df + n) * jnp.log1p(maha / df)
    w = jnp.exp(log_target - log_proposal - jnp.max(log_target - log_proposal))
    w = w / jnp.sum(w)
    ref = einsum(w, x, "k, k n -> n")
    se = jnp.sqrt(einsum(jnp.square(w), jnp.square(x - ref), "k, k n -> n"))

    tol = 5.0 * jnp.linalg.norm(se)
    laplace_error = jnp.linalg.norm(res.mode - ref)
    vb_error = jnp.linalg.norm(mean - ref)
    assert laplace_error > 10 * tol  # the Laplace bias is resolved
    assert vb_error < 0.2 * laplace_error + tol


# ---------------------------------------------------------------------------
# Golden: R-INLA's VB-corrected means
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "subspace"),
    [
        ("mean_vb_fixed", "fixed"),
        pytest.param("mean_vb_sparse", "sparse", marks=pytest.mark.slow),
        pytest.param("mean_vb_all", "all", marks=pytest.mark.slow),
    ],
)
def test_bernoulli_rw2_matches_r_inla_vb(key, subspace):
    """Golden: R-INLA's ``control.vb`` mean on a Bernoulli / RW2 + intercept.

    Generated offline by scripts/golden/inla/bernoulli_rw2_vb.R. Latent
    ``(u, β₀)`` with ``u`` an RW2 under ``Σu = Σ(t − t̄)u = 0`` and the
    intercept entering every site, ``A = [I | 1]``: rows touch ``u_i`` and
    ``β₀``, adjacent in ``H``, so the variances come from Takahashi. R-INLA
    stops its VB iterations at ``max |Δx| / sd ≈ 0.01``, so agreement is a
    few 1e-3 sd, against a correction of over one sd.
    """
    fixture = json.loads((GOLDEN / "bernoulli_rw2_vb.json").read_text())
    n = fixture["n"]
    size = n + 1
    R = np.asarray(gx.rw2_structure(n).as_matrix())
    rows, cols = np.nonzero(np.tril(R))
    Q = gx.SparseOperator.from_coo(
        np.r_[rows, n],
        np.r_[cols, n],
        jnp.r_[fixture["tau"] * jnp.asarray(R[rows, cols]), fixture["prec_fixed"]],
        (size, size),
        symmetric=True,
    )
    t = jnp.arange(1, n + 1, dtype=float)
    V = jnp.zeros((size, 2)).at[:n, 0].set(1.0).at[:n, 1].set(t - t.mean())
    prior = gx.IntrinsicGMRF(jnp.zeros(size), 1.0, Q, V)
    A = gx.SparseOperator.from_coo(
        np.r_[np.arange(n), np.arange(n)],
        np.r_[np.arange(n), np.full(n, n)],
        jnp.ones(2 * n),
        (n, size),
    )
    lik = gx.BernoulliLikelihood(jnp.asarray(fixture["y"], dtype=float))
    res = gx.laplace_mode(prior, lik, projector=A)
    np.testing.assert_allclose(res.mode, fixture["mean_laplace"], atol=1e-6)
    assert _pair_plan(A.pattern, res.factor.symbolic.inverse_plan[0]) is not None

    index = {
        "fixed": jnp.array([n]),
        "sparse": jnp.asarray(fixture["subspace_sparse"]),
        "all": jnp.arange(size),
    }[subspace]
    mean = gx.vb_mean_correction(res, prior, lik, projector=A, subspace=index)
    expected = jnp.asarray(fixture[key])
    sd = jnp.asarray(fixture["sd_laplace"])
    laplace = jnp.asarray(fixture["mean_laplace"])
    assert jnp.max(jnp.abs(expected - laplace) / sd) > 1.0  # a large correction
    assert jnp.max(jnp.abs(mean - expected) / sd) < 5e-3
    # Towards R-INLA's VB mean and away from the Gaussian approximation.
    assert jnp.linalg.norm(mean - expected) < 0.01 * jnp.linalg.norm(laplace - expected)


# ---------------------------------------------------------------------------
# Gradients, dtype and validation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reverse_mode_gradient_matches_finite_differences():
    n = 12
    lik = gx.PoissonLikelihood(counts(n))
    weights = jnp.linspace(-1.0, 1.0, n)

    def f(log_tau):
        prior = gx.GaussianMRF(jnp.zeros(n), proper_rw2(n, tau=jnp.exp(log_tau)))
        res = gx.laplace_mode(prior, lik)
        mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(n))
        return weights @ mean

    h = 1e-5
    fd = (f(0.5 + h) - f(0.5 - h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(f)(0.5), fd, rtol=1e-6)


@pytest.mark.slow
def test_float32_stays_float32():
    n = 8
    Q = proper_rw2(n)
    Q32 = gx.BlockTriDiag(
        Q.diagonal.astype(jnp.float32), Q.sub_diagonal.astype(jnp.float32)
    )
    prior = gx.GaussianMRF(jnp.zeros(n, jnp.float32), Q32)
    lik = gx.PoissonLikelihood(counts(n).astype(jnp.float32))
    res = gx.laplace_mode(prior, lik)
    mean = gx.vb_mean_correction(res, prior, lik, subspace=jnp.arange(n))
    assert mean.dtype == jnp.float32


def test_subspace_shape_is_checked():
    n = 8
    prior = gx.GaussianMRF(jnp.zeros(n), proper_rw2(n))
    lik = gx.PoissonLikelihood(counts(n))
    res = gx.laplace_mode(prior, lik)
    with pytest.raises(ValueError, match="dense subspace"):
        gx.vb_mean_correction(res, prior, lik, subspace=jnp.ones((n + 1, 2)))
    with pytest.raises(ValueError, match="1-D"):
        gx.vb_mean_correction(res, prior, lik, subspace=jnp.zeros((2, 2), int))
