"""Tests for base_conditional — Gaussian conditional via Schur complement."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsla
import lineax as lx
import pytest

from gaussx._gp._base_conditional import base_conditional
from gaussx._gp._svgp import whitened_svgp_predict
from gaussx._testing import tree_allclose


def _make_pd(key, M):
    A = jr.normal(key, (M, M))
    return A @ A.T + 0.1 * jnp.eye(M)


# ---------------------------------------------------------------------------
# Prior conditional (no q_sqrt)
# ---------------------------------------------------------------------------


class TestPriorConditional:
    def test_mean_shape(self, getkey):
        M, N, R = 5, 8, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
        f = jr.normal(getkey(), (M, R))
        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

    def test_mean_matches_dense(self, getkey):
        """Mean should be K_nm K_mm^{-1} f."""
        M, N, R = 5, 8, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        f = jr.normal(getkey(), (M, R))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1

        mean, _ = base_conditional(K_mm, K_mn, K_nn_diag, f)
        expected = K_mn.T @ jnp.linalg.solve(K_mm, f)
        assert tree_allclose(mean, expected, rtol=1e-4)

    def test_var_diagonal_knn(self):
        """Variance with diagonal K_nn."""
        M, N = 4, 6
        # Slice a PD joint so the true conditional variances are positive
        # (a random K_mn gives negative ones, which are now clipped).
        joint = _make_pd(jr.key(0), M + N)
        K_mm, K_mn = joint[:M, :M], joint[:M, M:]
        K_nn_diag = jnp.diag(joint[M:, M:])
        f = jr.normal(jr.key(1), (M, 1))

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f)

        # Expected: K_nn_diag - diag(K_nm K_mm^{-1} K_mn)
        A = jnp.linalg.solve(K_mm, K_mn)  # (M, N)
        schur_diag = jnp.sum(K_mn * A, axis=0)
        expected = K_nn_diag - schur_diag
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)

    def test_var_full_knn(self, getkey):
        """Variance with full K_nn."""
        M, N = 4, 6
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn = _make_pd(getkey(), N)
        f = jr.normal(getkey(), (M, 1))

        _, var = base_conditional(K_mm, K_mn, K_nn, f)
        assert var.shape == (N, N, 1)

        A = jnp.linalg.solve(K_mm, K_mn)
        expected = K_nn - K_mn.T @ A
        assert tree_allclose(var[:, :, 0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Whitened parameterization
# ---------------------------------------------------------------------------


class TestWhitened:
    def test_mean_whitened(self, getkey):
        """Whitened: mean = A^T f where A = L^{-1} K_mn."""
        M, N, R = 5, 8, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1
        f = jr.normal(getkey(), (M, R))

        mean, _ = base_conditional(K_mm, K_mn, K_nn_diag, f, white=True)
        L = jnp.linalg.cholesky(K_mm)
        A = jsla.solve_triangular(L, K_mn, lower=True)
        expected = A.T @ f
        assert tree_allclose(mean, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# With q_sqrt (variational posterior)
# ---------------------------------------------------------------------------


class TestVariational:
    def test_diagonal_q_sqrt(self, getkey):
        M, N, R = 5, 8, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

        # Verify variance is adjusted from prior conditional
        _, var_prior = base_conditional(K_mm, K_mn, K_nn_diag, f)
        # Variational variance should differ from prior (unless q_sqrt=0)
        assert not jnp.allclose(var, var_prior)

    def test_full_q_sqrt(self, getkey):
        M, N, R = 4, 6, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))

        q_sqrt_list = []
        for _ in range(R):
            L = jnp.tril(jr.normal(getkey(), (M, M)))
            L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
            q_sqrt_list.append(L)
        q_sqrt = jnp.stack(q_sqrt_list, axis=0)  # (R, M, M)

        mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_sqrt)
        assert mean.shape == (N, R)
        assert var.shape == (N, R)

    def test_full_q_sqrt_full_knn(self, getkey):
        """Full q_sqrt with full K_nn should give (N, N, R) variance."""
        M, N, R = 4, 6, 2
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn = _make_pd(getkey(), N)
        f = jr.normal(getkey(), (M, R))

        q_sqrt_list = []
        for _ in range(R):
            L = jnp.tril(jr.normal(getkey(), (M, M)))
            L = L.at[jnp.diag_indices(M)].set(jnp.abs(jnp.diag(L)) + 0.1)
            q_sqrt_list.append(L)
        q_sqrt = jnp.stack(q_sqrt_list, axis=0)

        mean, var = base_conditional(K_mm, K_mn, K_nn, f, q_sqrt=q_sqrt)
        assert mean.shape == (N, R)
        assert var.shape == (N, N, R)

    def test_variance_formula_diagonal(self, getkey):
        """Verify the variance formula with diagonal q_sqrt."""
        M, N, R = 4, 6, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)

        Kmm_inv = jnp.linalg.inv(K_mm)
        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        schur = K_mn.T @ Kmm_inv @ K_mn
        var_adj = jnp.diag(K_mn.T @ Kmm_inv @ q_cov @ Kmm_inv @ K_mn)
        expected = K_nn_diag - jnp.diag(schur) + var_adj
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)

    def test_variance_formula_diagonal_nonwhite(self, getkey):
        """Non-whitened q_sqrt should include the prior solve."""
        M, N, R = 4, 6, 1
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 1.0
        f = jr.normal(getkey(), (M, R))
        q_diag = jnp.abs(jr.normal(getkey(), (M, R))) + 0.1

        _, var = base_conditional(K_mm, K_mn, K_nn_diag, f, q_sqrt=q_diag)

        Kmm_inv = jnp.linalg.inv(K_mm)
        q_cov = jnp.diag(q_diag[:, 0] ** 2)
        schur = K_mn.T @ Kmm_inv @ K_mn
        var_adj = jnp.diag(K_mn.T @ Kmm_inv @ q_cov @ Kmm_inv @ K_mn)
        expected = K_nn_diag - jnp.diag(schur) + var_adj
        assert tree_allclose(var[:, 0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


class TestGradient:
    def test_grad_through_f(self, getkey):
        M, N = 5, 8
        K_mm = _make_pd(getkey(), M)
        K_mn = jr.normal(getkey(), (M, N))
        K_nn_diag = jnp.abs(jr.normal(getkey(), (N,))) + 0.1

        def loss(f):
            mean, var = base_conditional(K_mm, K_mn, K_nn_diag, f)
            return jnp.sum(mean**2) + jnp.sum(var)

        f = jr.normal(getkey(), (M, 1))
        g = jax.grad(loss)(f)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == (M, 1)


# ---------------------------------------------------------------------------
# gh-363: 1-D f, shape validation and non-negative diagonal variances
# ---------------------------------------------------------------------------


def _ill_conditioned_float32():
    """RBF on a fine grid without jitter: K_nn - Q_nn rounds below 0."""

    def rbf(a, b):
        return jnp.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / 0.5**2)

    Z = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    X = jnp.linspace(-0.1, 1.1, 50, dtype=jnp.float32)
    return rbf(Z, Z), rbf(Z, X), jnp.ones(50, dtype=jnp.float32)


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("q", ["none", "diag", "full"])
def test_single_output_f_matches_two_d(white, q):
    K_mm, K_mn, K_nn = _ill_conditioned_float32()
    K_mm = K_mm.astype(jnp.float64) + 1e-6 * jnp.eye(8)
    K_mn, K_nn = K_mn.astype(jnp.float64), K_nn.astype(jnp.float64)
    u = jnp.linspace(-1.0, 1.0, 8)
    q_1d = {"none": None, "diag": 0.1 * jnp.ones(8), "full": 0.1 * jnp.eye(8)}[q]
    q_2d = {
        "none": None,
        "diag": 0.1 * jnp.ones((8, 1)),
        "full": 0.1 * jnp.eye(8)[None],
    }[q]
    for knn in (K_nn, jnp.diag(K_nn)):
        mean, var = base_conditional(K_mm, K_mn, knn, u, q_sqrt=q_1d, white=white)
        mean2, var2 = base_conditional(
            K_mm, K_mn, knn, u[:, None], q_sqrt=q_2d, white=white
        )
        assert mean.shape == (50,)
        assert jnp.array_equal(mean, mean2[:, 0])
        assert jnp.array_equal(var, var2[..., 0])


@pytest.mark.parametrize(
    ("f_shape", "q_shape", "match"),
    [
        ((8, 2, 1), None, "f must have shape"),
        ((7, 1), None, "f must have shape"),
        ((8, 2), (8, 3), "q_sqrt must have shape"),
        ((8, 2), (3, 8, 8), "q_sqrt must have shape"),
    ],
)
def test_shape_validation(f_shape, q_shape, match):
    K_mm, K_mn, K_nn = _ill_conditioned_float32()
    q_sqrt = None if q_shape is None else jnp.ones(q_shape)
    with pytest.raises(ValueError, match=match):
        base_conditional(K_mm, K_mn, K_nn, jnp.zeros(f_shape), q_sqrt=q_sqrt)


def test_diagonal_variance_clipped_like_whitened_svgp_predict():
    K_mm, K_mn, K_nn = _ill_conditioned_float32()
    u = jnp.zeros(8, dtype=jnp.float32)
    _, var_bc = base_conditional(K_mm, K_mn, K_nn, u, white=True)
    _, var_sv = whitened_svgp_predict(
        lx.MatrixLinearOperator(K_mm, lx.positive_semidefinite_tag),
        K_mn.T,
        u,
        jnp.zeros((8, 8), dtype=jnp.float32),
        K_nn,
    )
    assert var_bc.dtype == jnp.float32
    assert jnp.min(var_bc) >= 0.0
    assert jnp.allclose(var_bc, var_sv, atol=1e-6)


def test_full_covariance_is_not_clipped():
    K_mm, K_mn, K_nn = _ill_conditioned_float32()
    u = jnp.zeros((8, 1), dtype=jnp.float32)
    _, var = base_conditional(K_mm, K_mn, jnp.diag(K_nn), u, white=True)
    L = jnp.linalg.cholesky(K_mm)
    A = jsla.solve_triangular(L, K_mn, lower=True)
    assert jnp.allclose(var[..., 0], jnp.diag(K_nn) - A.T @ A, atol=1e-6)
    # The same round-off the diagonal branch clips is left in place here.
    assert jnp.min(jnp.diagonal(var[..., 0])) < 0.0
