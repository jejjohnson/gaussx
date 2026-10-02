"""Tests for besag_structure, generalized_variance_scale and bym2_precision."""

import json
from pathlib import Path

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "golden"
    / "inla"
    / "scotland_scale.json"
)


def graph_laplacian(n: int, extra_edges: int = 20, seed: int = 0):
    """Laplacian of a connected random graph (a path plus random chords)."""
    rng = np.random.default_rng(seed)
    edges = {(i + 1, i) for i in range(n - 1)}
    target = min(n - 1 + extra_edges, n * (n - 1) // 2)
    while len(edges) < target:
        i, j = rng.choice(n, 2, replace=False)
        edges.add((max(i, j), min(i, j)))
    senders, receivers = (np.array(side) for side in zip(*sorted(edges), strict=True))
    return laplacian_from_edges(n, senders, receivers)


def laplacian_from_edges(n, senders, receivers):
    weights = np.ones(senders.shape[0])
    degree = np.bincount(senders, weights, n) + np.bincount(receivers, weights, n)
    return gaussx.SparseOperator.from_coo(
        np.r_[np.arange(n), senders],
        np.r_[np.arange(n), receivers],
        jnp.asarray(np.r_[degree, -weights]),
        (n, n),
        symmetric=True,
    )


def dense_scale(R: np.ndarray) -> float:
    """exp(mean(log(diag(R⁺)))), the dense definition for a connected graph."""
    return float(np.exp(np.mean(np.log(np.diag(np.linalg.pinv(R))))))


class TestBesagStructure:
    def test_tags_and_keeps_type(self):
        R = gaussx.besag_structure(graph_laplacian(10))
        assert isinstance(R, gaussx.SparseOperator)
        assert lx.is_positive_semidefinite(R)

    def test_dense_operator_is_wrapped(self):
        R = graph_laplacian(6).as_matrix()
        tagged = gaussx.besag_structure(lx.MatrixLinearOperator(R, lx.symmetric_tag))
        assert lx.is_positive_semidefinite(tagged)
        assert jnp.allclose(tagged.as_matrix(), R)

    def test_nonzero_row_sums_raise(self):
        R = graph_laplacian(6)
        shifted = R.add_diagonal(jnp.ones(6))
        with pytest.raises(ValueError, match="sum"):
            gaussx.besag_structure(shifted)

    def test_non_symmetric_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            gaussx.besag_structure(lx.MatrixLinearOperator(jnp.ones((3, 3))))

    def test_traced_values_skip_the_check(self):
        R = graph_laplacian(6)
        out = jax.jit(lambda v: gaussx.besag_structure(eqx_values(R, v)).values)(
            R.values + 1.0
        )
        assert out.shape == R.values.shape


def eqx_values(R, values):
    return gaussx.SparseOperator(values, R.pattern, tags=R.tags)


class TestGeneralizedVarianceScale:
    def test_graph_matches_dense_definition(self):
        R = graph_laplacian(30)
        s = gaussx.generalized_variance_scale(R, jnp.ones(30))
        assert jnp.allclose(s, dense_scale(np.asarray(R.as_matrix())), rtol=1e-6)

    def test_scaled_structure_has_unit_generalized_variance(self):
        R = graph_laplacian(30)
        s = gaussx.generalized_variance_scale(R, jnp.ones(30))
        scaled = np.asarray(s * R.as_matrix())
        assert np.isclose(dense_scale(scaled), 1.0, rtol=1e-6)

    # The default ridge ε = √eps · max diag(R) (R-INLA's) biases the variances
    # by about ε / λ_min(range R): ~1e-6 for these random walks.
    def test_rw1_block_tridiagonal(self):
        R = gaussx.rw1_structure(25)
        s = gaussx.generalized_variance_scale(R, jnp.ones(25))
        assert jnp.allclose(s, dense_scale(np.asarray(R.as_matrix())), rtol=1e-5)

    @pytest.mark.parametrize("n", [11, 12])
    def test_rw2_with_padding(self, n):
        R = gaussx.rw2_structure(n)
        t = jnp.arange(n, dtype=float)
        null_space = einx.id("k n -> n k", jnp.stack([jnp.ones(n), t]))
        s = gaussx.generalized_variance_scale(R, null_space)
        expected = dense_scale(np.asarray(R.as_matrix())[:n, :n])
        assert jnp.allclose(s, expected, rtol=1e-5)

    def test_grid_kronecker_sum_is_exact(self):
        L = gaussx.rw1_structure(5).as_matrix()
        M = gaussx.rw1_structure(6).as_matrix()
        grid = gaussx.KroneckerSum(
            lx.MatrixLinearOperator(L, lx.symmetric_tag),
            lx.MatrixLinearOperator(M, lx.symmetric_tag),
        )
        s = gaussx.generalized_variance_scale(grid, jnp.ones(30))
        assert jnp.allclose(s, dense_scale(np.asarray(grid.as_matrix())), rtol=1e-10)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="null_space"):
            gaussx.generalized_variance_scale(graph_laplacian(10), jnp.ones(5))

    def test_scotland_matches_r_inla(self):
        """Golden: R-INLA's inla.scale.model on the Scotland graph.

        Generated offline by scripts/golden/inla/scotland_scale.R; the same
        ridge ε as R-INLA makes the agreement ~1e-10, far tighter than the
        ε-bias against the exact pseudo-inverse (~5e-7 here).
        """
        fixture = json.loads(GOLDEN.read_text())
        n = fixture["n"]
        senders = np.asarray(fixture["senders"])
        receivers = np.asarray(fixture["receivers"])
        R = laplacian_from_edges(n, senders, receivers)
        s = gaussx.generalized_variance_scale(R, jnp.ones(n))
        assert jnp.allclose(s, fixture["scale"], rtol=1e-8)
        # R-INLA's scaled structure is s · R
        scaled_diag = s * gaussx.diag(R)
        assert jnp.allclose(scaled_diag, jnp.asarray(fixture["scaled_diagonal"]))


class TestBYM2:
    n = 30

    def _structure(self):
        R = graph_laplacian(self.n)
        s = gaussx.generalized_variance_scale(R, jnp.ones(self.n))
        return s * R

    def test_block_form(self):
        R_star = self._structure()
        tau, phi = 1.5, 0.7
        Q = gaussx.bym2_precision(R_star, tau, phi)
        n = self.n
        I = jnp.eye(n)
        c = -jnp.sqrt(tau * phi) / (1 - phi)
        expected = jnp.block(
            [
                [tau / (1 - phi) * I, c * I],
                [c * I, R_star.as_matrix() + phi / (1 - phi) * I],
            ]
        )
        assert jnp.allclose(Q.as_matrix(), expected)
        assert lx.is_positive_semidefinite(Q)

    def test_marginal_of_b_has_bym2_covariance(self):
        """Dense check at n = 30 under u*'s sum-to-zero constraint."""
        R_star = self._structure()
        tau, phi = 1.5, 0.7
        Q = np.asarray(gaussx.bym2_precision(R_star, tau, phi).as_matrix())
        n = self.n
        # Constrained covariance by kriging on Q + εI, ε → 0 (float64).
        eps = 1e-9
        S = np.linalg.inv(Q + eps * np.eye(2 * n))
        a = np.r_[np.zeros(n), np.ones(n)]
        Sa = S @ a
        constrained = S - einx.multiply("i, j -> i j", Sa, Sa) / (a @ Sa)
        expected = (
            (1 - phi) * np.eye(n) + phi * np.linalg.pinv(R_star.as_matrix())
        ) / tau
        assert np.allclose(constrained[:n, :n], expected, atol=1e-6)

    def test_pattern_fixed_across_parameters(self):
        R_star = self._structure()
        Q1 = gaussx.bym2_precision(R_star, 1.0, 0.2)
        Q2 = gaussx.bym2_precision(R_star, 3.0, 0.9)
        assert Q1.pattern == Q2.pattern

        @jax.jit
        def total(tau):
            return jnp.sum(gaussx.bym2_precision(R_star, tau, 0.5).values)

        assert jnp.isfinite(jax.grad(total)(2.0))

    def test_requires_sparse_structure(self):
        with pytest.raises(TypeError, match="SparseOperator"):
            gaussx.bym2_precision(
                lx.MatrixLinearOperator(jnp.eye(3), lx.symmetric_tag), 1.0, 0.5
            )
