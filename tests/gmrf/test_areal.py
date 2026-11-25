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
from gaussx._deprecation import GaussxDeprecationWarning


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
    # Every dense pseudo-inverse of a singular structure in the tests takes
    # `hermitian=True, rtol=None`: pinv's legacy default cutoff (rcond=1e-15
    # x sigma_max) sits below the round-off of the null singular value, and
    # numpy 2.5's SVD puts that value at ~3 eps x sigma_max, so the default
    # inverts the null space (#523). rtol=None cuts at max(M, N) eps instead.
    return float(
        np.exp(np.mean(np.log(np.diag(np.linalg.pinv(R, hermitian=True, rtol=None)))))
    )


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

    def test_float32_laplacian_is_accepted(self):
        # gh-760: rounding leaves row sums of 1.5e-8, above a fixed 1e-8.
        weights = np.array([0.1, 0.2, 0.3])
        lower, upper = np.array([1, 2, 3]), np.array([0, 1, 2])
        degree = np.bincount(lower, weights, 4) + np.bincount(upper, weights, 4)
        values = jnp.asarray(np.r_[degree, -weights], dtype=jnp.float32)
        R = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(4), lower],
            np.r_[np.arange(4), upper],
            values,
            (4, 4),
            symmetric=True,
        )
        assert gaussx.besag_structure(R).values.dtype == jnp.float32
        # a row sum of 1e-4 of the diagonal is still an error
        with pytest.raises(ValueError, match="sum to zero"):
            gaussx.besag_structure(eqx_values(R, values.at[0].add(1e-4)))

    def test_non_symmetric_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            gaussx.besag_structure(lx.MatrixLinearOperator(jnp.ones((3, 3))))

    def test_traced_values_skip_the_check(self):
        R = graph_laplacian(6)
        out = jax.jit(lambda v: gaussx.besag_structure(eqx_values(R, v)).values)(
            R.values + 1.0
        )
        assert out.shape == R.values.shape


def rw2_scale(n: int) -> float:
    """Exact scale of a second-order random walk, from the banded inverse.

    Both end nodes are fixed, so the rest is D₂ᵀD₂ without them, the square of the
    Dirichlet path Laplacian, whose inverse is min(i, j) (n − 1 − max(i, j)) / (n − 1).
    """
    i = np.arange(1, n - 1, dtype=float)
    low, high = np.minimum.outer(i, i), np.maximum.outer(i, i)
    inverse = low * (n - 1 - high) / (n - 1)
    grounded = np.zeros((n, n))
    grounded[1:-1, 1:-1] = np.einsum("ik,kj->ij", inverse, inverse)
    t = np.arange(n) - (n - 1) / 2
    basis = np.stack([np.ones(n) / np.sqrt(n), t / np.sqrt(np.sum(t**2))])
    projector = np.eye(n) - np.einsum("ki,kj->ij", basis, basis)
    variances = np.einsum("ij,jk,ki->i", projector, grounded, projector)
    return float(np.exp(np.mean(np.log(variances))))


def eqx_values(R, values):
    return gaussx.SparseOperator(values, R.pattern, tags=R.tags)


class TestGeneralizedVarianceScale:
    @pytest.mark.slow
    def test_graph_matches_dense_definition(self):
        R = graph_laplacian(30)
        s = gaussx.generalized_variance_scale(R, jnp.ones(30))
        assert jnp.allclose(s, dense_scale(np.asarray(R.as_matrix())), rtol=1e-6)

    def test_scaled_structure_has_unit_generalized_variance(self):
        R = graph_laplacian(30)
        s = gaussx.generalized_variance_scale(R, jnp.ones(30))
        scaled = np.asarray(s * R.as_matrix())
        assert np.isclose(dense_scale(scaled), 1.0, rtol=1e-6)

    @pytest.mark.slow
    def test_rw1_block_tridiagonal(self):
        R = gaussx.rw1_structure(25)
        s = gaussx.generalized_variance_scale(R, jnp.ones(25))
        assert jnp.allclose(s, dense_scale(np.asarray(R.as_matrix())), rtol=1e-9)

    # A grounded path is conditioned like N² (RW1) and N⁴ (RW2), so each bound is
    # ~100 machine epsilons times that, capped by what the test is for.
    @pytest.mark.parametrize("n", [100, pytest.param(500, marks=pytest.mark.slow)])
    @pytest.mark.parametrize(
        "dtype",
        [
            jnp.float32,
            pytest.param(
                jnp.float64, marks=pytest.mark.x64_only(reason="float64 case")
            ),
        ],
    )
    def test_long_rw1_is_exact(self, n, dtype):
        # gh-762: the ridge gave 0.74 of the scale at n = 100 in float32.
        R = gaussx.rw1_structure(n)
        R = gaussx.BlockTriDiag(
            R.diagonal.astype(dtype), R.sub_diagonal.astype(dtype), symmetric=True
        )
        s = gaussx.generalized_variance_scale(R, jnp.ones(n, dtype))
        assert s.dtype == dtype
        # diag(R⁺) of a path, closed form: (Σ_k<i k² + Σ_k>i (n − k)²) / n².
        i = np.arange(n, dtype=float)
        left = i * (i + 1) * (2 * i + 1) / 6
        right = (n - 1 - i) * (n - i) * (2 * (n - 1 - i) + 1) / 6
        expected = np.exp(np.mean(np.log((left + right) / n**2)))
        rtol = 2e-3 if dtype == jnp.float32 else 1e-10
        assert np.isclose(float(s), expected, rtol=rtol)

    @pytest.mark.x64_only(reason="a grounded RW2 is conditioned like N⁴")
    @pytest.mark.parametrize("n", [100, pytest.param(500, marks=pytest.mark.slow)])
    def test_long_rw2_is_exact(self, n):
        # gh-866: the ridge gave 0.99 of the scale at n = 100 and 0.19 at 500.
        t = jnp.arange(n, dtype=float)
        null_space = einx.id("k n -> n k", jnp.stack([jnp.ones(n), t]))
        s = gaussx.generalized_variance_scale(gaussx.rw2_structure(n), null_space)
        assert np.isclose(float(s), rw2_scale(n), rtol=1e-6)

    def test_sparse_and_banded_agree(self):
        n = 40
        banded = gaussx.rw2_structure(n)
        dense = np.asarray(banded.as_matrix())
        rows, cols = np.nonzero(np.tril(dense))
        sparse = gaussx.SparseOperator.from_coo(
            rows, cols, jnp.asarray(dense[rows, cols]), (n, n), symmetric=True
        )
        t = jnp.arange(n, dtype=float)
        null_space = einx.id("k n -> n k", jnp.stack([jnp.ones(n), t]))
        s_banded = gaussx.generalized_variance_scale(banded, null_space)
        s_sparse = gaussx.generalized_variance_scale(sparse, null_space)
        rtol = 1e-2 if s_banded.dtype == jnp.float32 else 1e-8
        assert jnp.allclose(s_banded, s_sparse, rtol=rtol)

    @pytest.mark.x64_only(reason="finite differences")
    def test_gradient_in_the_weights(self):
        n = 12

        def scale(log_spacing):
            R = gaussx.rw1_structure(n, spacing=jnp.exp(log_spacing))
            return gaussx.generalized_variance_scale(R, jnp.ones(n))

        point = jnp.linspace(-0.3, 0.4, n - 1)
        direction = jnp.cos(jnp.arange(n - 1.0))
        step = 1e-5
        upper, lower = scale(point + step * direction), scale(point - step * direction)
        slope = jnp.vdot(jax.grad(scale)(point), direction)
        assert jnp.allclose(slope, (upper - lower) / (2 * step), rtol=1e-7)

    @pytest.mark.x64_only(reason="promotion to float64 needs x64")
    def test_integer_null_space_keeps_float32(self):
        # gh-761
        R = graph_laplacian(10)
        R = eqx_values(R, R.values.astype(jnp.float32))
        s = gaussx.generalized_variance_scale(R, np.ones(10, int))
        assert s.dtype == jnp.float32

    def test_eps_is_deprecated_and_ignored(self):
        R = gaussx.rw1_structure(10)
        with pytest.warns(GaussxDeprecationWarning, match="0.9.0"):
            s = gaussx.generalized_variance_scale(R, jnp.ones(10), eps=1.0)
        assert s == gaussx.generalized_variance_scale(R, jnp.ones(10))

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [11, 12])
    def test_rw2_with_padding(self, n):
        R = gaussx.rw2_structure(n)
        t = jnp.arange(n, dtype=float)
        null_space = einx.id("k n -> n k", jnp.stack([jnp.ones(n), t]))
        s = gaussx.generalized_variance_scale(R, null_space)
        expected = dense_scale(np.asarray(R.as_matrix())[:n, :n])
        assert jnp.allclose(s, expected, rtol=1e-9)

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_scotland_matches_r_inla(self):
        """Golden: R-INLA's inla.scale.model on the Scotland graph.

        Generated offline by scripts/golden/inla/scotland_scale.R. R-INLA adds a
        ridge of √eps times the diagonal, which biases its scale by 5e-7 here;
        the exact pseudo-inverse is the reference below.
        """
        fixture = json.loads(GOLDEN.read_text())
        n = fixture["n"]
        senders = np.asarray(fixture["senders"])
        receivers = np.asarray(fixture["receivers"])
        R = laplacian_from_edges(n, senders, receivers)
        s = gaussx.generalized_variance_scale(R, jnp.ones(n))
        assert jnp.allclose(s, fixture["scale"], rtol=1e-6)
        assert jnp.allclose(s, dense_scale(np.asarray(R.as_matrix())), rtol=1e-10)
        # R-INLA's scaled structure is s · R
        scaled_diag = s * gaussx.diag(R)
        assert jnp.allclose(scaled_diag, jnp.asarray(fixture["scaled_diagonal"]))


class TestBYM2:
    n = 30

    def _structure(self):
        R = graph_laplacian(self.n)
        s = gaussx.generalized_variance_scale(R, jnp.ones(self.n))
        return s * R

    @pytest.mark.slow
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
            (1 - phi) * np.eye(n)
            + phi * np.linalg.pinv(R_star.as_matrix(), hermitian=True, rtol=None)
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

    @pytest.mark.x64_only(reason="promotion to float64 needs x64")
    @pytest.mark.parametrize("tau, phi", [(2, 0), (2.0, 0.0), (np.int32(2), 0.3)])
    def test_integer_parameters_keep_float32(self, tau, phi):
        # gh-761
        R = graph_laplacian(5)
        R = eqx_values(R, R.values.astype(jnp.float32))
        assert gaussx.bym2_precision(R, tau, phi).values.dtype == jnp.float32

    def test_requires_sparse_structure(self):
        with pytest.raises(TypeError, match="SparseOperator"):
            gaussx.bym2_precision(
                lx.MatrixLinearOperator(jnp.eye(3), lx.symmetric_tag), 1.0, 0.5
            )
