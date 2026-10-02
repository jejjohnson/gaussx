"""Tests for SparseOperator and SparsityPattern (G1)."""

from __future__ import annotations

import time

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx import SparseOperator, SparsityPattern
from gaussx._einx import einsum, rearrange
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._slq_logdet import SLQLogdet


PSD = lx.positive_semidefinite_tag


def _grid_edges(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """4-neighbour edges of an ``height x width`` grid, each edge once."""
    i = np.arange(height * width)
    right = i[(i % width) != width - 1]
    down = i[i < (height - 1) * width]
    senders = np.concatenate([right, down])
    receivers = np.concatenate([right + 1, down + width])
    return senders, receivers


def _laplacian(
    senders: np.ndarray,
    receivers: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    shift: float = 0.0,
    tags: frozenset[object] = frozenset(),
) -> SparseOperator:
    """Graph Laplacian plus ``shift * I``, built edge-once (the ICAR example)."""
    deg = np.bincount(senders, weights, n) + np.bincount(receivers, weights, n)
    return SparseOperator.from_coo(
        np.r_[np.arange(n), senders],
        np.r_[np.arange(n), receivers],
        jnp.asarray(np.r_[deg + shift, -weights]),
        (n, n),
        symmetric=True,
        tags=tags,
    )


def _congruence_dense(A: jax.Array, w: jax.Array) -> jax.Array:
    """Dense ``Aᵀ diag(w) A``."""
    return einsum(A, einx.multiply("k j, k -> k j", A, w), "k i, k j -> i j")


@pytest.fixture
def small_laplacian() -> SparseOperator:
    senders, receivers = _grid_edges(3, 4)
    weights = np.linspace(0.5, 2.0, senders.size)
    return _laplacian(senders, receivers, weights, 12, shift=0.3, tags=frozenset({PSD}))


@pytest.fixture
def rectangular() -> SparseOperator:
    rows = np.array([0, 0, 1, 2, 2, 4])
    cols = np.array([0, 3, 1, 2, 5, 4])
    values = jnp.array([1.0, -2.0, 3.0, 0.5, 4.0, -1.5])
    return SparseOperator.from_coo(rows, cols, values, (5, 6))


# ---------------------------------------------------------------------------
# SparsityPattern
# ---------------------------------------------------------------------------


class TestSparsityPattern:
    def test_canonical_order_and_diagonal(self):
        p = SparsityPattern(np.array([2, 0, 2]), np.array([0, 2, 0]), (3, 3))
        np.testing.assert_array_equal(p.rows, [0, 0, 1, 2, 2])
        np.testing.assert_array_equal(p.cols, [0, 2, 1, 0, 2])
        assert p.rows.dtype == np.int32
        assert p.nnz == 5

    def test_symmetric_stores_lower_triangle(self):
        p = SparsityPattern(np.array([0, 1]), np.array([1, 2]), (3, 3), symmetric=True)
        assert np.all(p.rows >= p.cols)
        np.testing.assert_array_equal(p.rows, [0, 1, 1, 2, 2])
        np.testing.assert_array_equal(p.cols, [0, 0, 1, 1, 2])

    def test_rectangular_has_no_forced_diagonal(self):
        p = SparsityPattern(np.array([1]), np.array([3]), (2, 4))
        assert p.nnz == 1

    def test_equal_patterns_hash_equal(self):
        a = SparsityPattern(np.array([1, 0]), np.array([0, 1]), (2, 2))
        b = SparsityPattern(np.array([0, 1]), np.array([1, 0]), (2, 2))
        assert a == b
        assert hash(a) == hash(b)
        assert a != SparsityPattern(np.array([1]), np.array([0]), (2, 2))
        assert a != SparsityPattern(
            np.array([1]), np.array([0]), (2, 2), symmetric=True
        )

    def test_hash_is_stable_across_processes(self):
        # A content hash: this golden digest must never depend on the process
        # (e.g. PYTHONHASHSEED) or the platform. It changes only if the
        # canonical layout or the hashing scheme does.
        p = SparsityPattern(np.array([1, 2]), np.array([0, 1]), (3, 3), symmetric=True)
        assert p.digest == (
            "a5f5f80f35928c49ecb9cc1287f3ac29b6fdc6de12616a0434f75b62ea5de42a"
        )
        assert hash(p) == int(p.digest[:15], 16)

    def test_immutable(self):
        p = SparsityPattern(np.array([1]), np.array([0]), (2, 2))
        with pytest.raises(AttributeError):
            p.shape = (3, 3)  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError):
            p.rows[0] = 1

    @pytest.mark.parametrize(
        ("rows", "cols", "shape", "symmetric", "error"),
        [
            (np.array([0, 1]), np.array([0]), (2, 2), False, ValueError),
            (np.array([2]), np.array([0]), (2, 2), False, ValueError),
            (np.array([0]), np.array([-1]), (2, 2), False, ValueError),
            (np.array([0]), np.array([1]), (2, 3), True, ValueError),
            (np.array([0.0]), np.array([1.0]), (2, 2), False, TypeError),
        ],
    )
    def test_invalid(self, rows, cols, shape, symmetric, error):
        with pytest.raises(error):
            SparsityPattern(rows, cols, shape, symmetric=symmetric)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_symmetric_mirrors_off_diagonals(self):
        # Edge-once storage; an upper-triangle pair is mirrored too.
        op = SparseOperator.from_coo(
            np.array([0, 1, 2, 0]),
            np.array([0, 1, 2, 2]),
            jnp.array([2.0, 3.0, 4.0, -1.0]),
            (3, 3),
            symmetric=True,
        )
        expected = jnp.array([[2.0, 0.0, -1.0], [0.0, 3.0, 0.0], [-1.0, 0.0, 4.0]])
        assert jnp.allclose(op.as_matrix(), expected)
        assert lx.symmetric_tag in op.tags
        assert lx.is_symmetric(op)

    def test_general_matches_symmetric(self, small_laplacian):
        dense = small_laplacian.as_matrix()
        rows, cols = np.nonzero(np.asarray(dense))
        full = SparseOperator.from_coo(rows, cols, dense[rows, cols], dense.shape)
        assert not full.pattern.symmetric
        assert jnp.allclose(full.as_matrix(), dense)

    def test_duplicates_are_summed(self):
        op = SparseOperator.from_coo(
            np.array([0, 0, 1]), np.array([1, 1, 0]), jnp.array([1.0, 2.0, 5.0]), (2, 2)
        )
        assert jnp.allclose(op.as_matrix(), jnp.array([[0.0, 3.0], [5.0, 0.0]]))

    def test_integer_values_promoted(self):
        op = SparseOperator.from_coo(
            np.array([0]), np.array([0]), jnp.array([2]), (1, 1)
        )
        assert jnp.issubdtype(op.values.dtype, jnp.floating)

    def test_values_shape_checked(self, small_laplacian):
        with pytest.raises(ValueError, match="shape"):
            SparseOperator(jnp.ones(3), small_laplacian.pattern)
        with pytest.raises(ValueError, match="shape"):
            SparseOperator.from_coo(np.array([0]), np.array([0]), jnp.ones(2), (1, 1))

    def test_traced_indices_rejected(self):
        def build(rows):
            return SparseOperator.from_coo(rows, rows, jnp.ones(2), (2, 2)).values

        with pytest.raises(TypeError, match="concrete"):
            jax.jit(build)(jnp.arange(2))

    def test_float32_preserved(self, small_laplacian):
        op = SparseOperator(
            small_laplacian.values.astype(jnp.float32), small_laplacian.pattern
        )
        x = jnp.ones(12, dtype=jnp.float32)
        assert op.mv(x).dtype == jnp.float32
        assert op.as_matrix().dtype == jnp.float32
        assert gaussx.diag(op).dtype == jnp.float32


# ---------------------------------------------------------------------------
# Lineax interface against dense
# ---------------------------------------------------------------------------


class TestAgainstDense:
    @pytest.mark.parametrize("name", ["small_laplacian", "rectangular"])
    def test_mv_as_matrix_transpose(self, name, request):
        op = request.getfixturevalue(name)
        dense = op.as_matrix()
        m, n = op.pattern.shape
        assert (op.out_size(), op.in_size()) == (m, n)
        x = jr.normal(jr.key(0), (n,))
        y = jr.normal(jr.key(1), (m,))
        assert jnp.allclose(op.mv(x), dense @ x)
        assert jnp.allclose(op.T.mv(y), einsum(dense, y, "i j, i -> j"))
        assert jnp.allclose(op.T.as_matrix(), rearrange(dense, "i j -> j i"))
        assert jnp.allclose(op.to_bcoo().todense(), dense)
        assert jnp.allclose(op.to_bcoo() @ x, dense @ x)

    def test_dense_reference(self, rectangular):
        expected = np.zeros((5, 6))
        expected[[0, 0, 1, 2, 2, 4], [0, 3, 1, 2, 5, 4]] = [
            1.0,
            -2.0,
            3.0,
            0.5,
            4.0,
            -1.5,
        ]
        assert jnp.allclose(rectangular.as_matrix(), expected)

    def test_symmetric_transpose_is_self(self, small_laplacian):
        assert small_laplacian.T is small_laplacian

    def test_transpose_tags(self):
        op = SparseOperator.from_coo(
            np.array([1]),
            np.array([0]),
            jnp.array([1.0]),
            (2, 2),
            tags=lx.lower_triangular_tag,
        )
        assert lx.is_lower_triangular(op)
        assert lx.is_upper_triangular(op.T)
        assert jnp.allclose(op.T.T.as_matrix(), op.as_matrix())

    @pytest.mark.parametrize("name", ["small_laplacian", "rectangular"])
    def test_diag(self, name, request):
        op = request.getfixturevalue(name)
        expected = jnp.diag(op.as_matrix())
        assert jnp.allclose(gaussx.diag(op), expected)
        if name == "small_laplacian":
            assert jnp.allclose(lx.diagonal(op), expected)

    def test_tag_queries(self, small_laplacian, rectangular):
        assert lx.is_positive_semidefinite(small_laplacian)
        assert not lx.is_negative_semidefinite(small_laplacian)
        assert not lx.is_diagonal(small_laplacian)
        assert not lx.is_symmetric(rectangular)
        assert not lx.is_positive_semidefinite(rectangular)
        assert not lx.has_unit_diagonal(rectangular)


# ---------------------------------------------------------------------------
# Pattern algebra
# ---------------------------------------------------------------------------


class TestPatternAlgebra:
    def test_add_diagonal_preserves_pattern(self, small_laplacian):
        d = jnp.linspace(1.0, 2.0, 12)
        out = small_laplacian.add_diagonal(d)
        assert out.pattern is small_laplacian.pattern
        assert jnp.allclose(out.as_matrix(), small_laplacian.as_matrix() + jnp.diag(d))
        # An arbitrary shift can break definiteness: only symmetry is kept.
        assert out.tags == frozenset({lx.symmetric_tag})
        assert PSD in small_laplacian.add_diagonal(d, tags=PSD).tags

    def test_union_symmetric(self, small_laplacian):
        other = SparseOperator.from_coo(
            np.array([11, 5]),
            np.array([0, 0]),
            jnp.array([2.0, -1.0]),
            (12, 12),
            symmetric=True,
            tags=PSD,
        )
        out = small_laplacian.union(other)
        assert out.pattern.symmetric
        assert out.pattern.nnz == small_laplacian.pattern.nnz + 2
        assert jnp.allclose(
            out.as_matrix(), small_laplacian.as_matrix() + other.as_matrix()
        )
        assert out.tags == frozenset({lx.symmetric_tag, PSD})

    def test_union_mixed_storage(self, small_laplacian):
        other = SparseOperator.from_coo(
            np.array([3, 0]), np.array([7, 9]), jnp.array([1.0, 2.0]), (12, 12)
        )
        out = small_laplacian.union(other)
        assert not out.pattern.symmetric
        assert jnp.allclose(
            out.as_matrix(), small_laplacian.as_matrix() + other.as_matrix()
        )
        with pytest.raises(ValueError, match="Shapes"):
            small_laplacian.union(
                SparseOperator(
                    jnp.ones(2), SparsityPattern(np.array([0]), np.array([0]), (2, 2))
                )
            )

    def test_congruence_against_dense(self, small_laplacian, rectangular):
        A = SparseOperator.from_coo(
            np.array([0, 0, 1, 1, 1, 2]),
            np.array([0, 5, 3, 4, 11, 7]),
            jnp.array([1.0, 2.0, -1.0, 0.5, 3.0, 2.0]),
            (3, 12),
        )
        w = jnp.array([0.5, 2.0, -1.0])
        out = small_laplacian.congruence(A, w)
        Ad = A.as_matrix()
        expected = _congruence_dense(Ad, w)
        assert out.pattern.symmetric
        assert jnp.allclose(out.as_matrix(), expected)
        # The result already carries Q's pattern, so the union is aligned.
        H = small_laplacian.union(out)
        assert H.pattern == out.pattern
        assert jnp.allclose(H.as_matrix(), small_laplacian.as_matrix() + expected)

    def test_congruence_preserves_pattern(self, small_laplacian):
        # A projector whose rows touch grid neighbours only (like a FEM
        # projector on a mesh precision) adds no entry to Q's pattern.
        A = SparseOperator.from_coo(
            np.array([0, 0, 1, 1, 2]),
            np.array([0, 1, 5, 9, 11]),
            jnp.array([0.3, 0.7, 0.5, 0.5, 1.0]),
            (3, 12),
        )
        out = small_laplacian.congruence(A, jnp.array([1.0, 2.0, 3.0]))
        assert out.pattern == small_laplacian.pattern
        assert small_laplacian.union(out).pattern == small_laplacian.pattern

    def test_congruence_general_storage(self, rectangular):
        base = SparseOperator(
            jnp.zeros(6), SparsityPattern(np.array([0]), np.array([0]), (6, 6))
        )
        w = jnp.linspace(0.5, 1.5, 5)
        out = base.congruence(rectangular, w)
        Ad = rectangular.as_matrix()
        assert not out.pattern.symmetric
        assert jnp.allclose(out.as_matrix(), _congruence_dense(Ad, w))

    def test_congruence_checks_shapes(self, small_laplacian, rectangular):
        with pytest.raises(ValueError, match="columns"):
            small_laplacian.congruence(rectangular, jnp.ones(5))
        with pytest.raises(ValueError, match="square"):
            rectangular.congruence(rectangular, jnp.ones(5))

    def test_newton_rebuild_example(self):
        # The roadmap's ICAR / Newton-rebuild example, end to end.
        senders, receivers = _grid_edges(3, 3)
        w = np.ones(senders.size)
        R = _laplacian(senders, receivers, w, 9, tags=frozenset({PSD}))
        Q = eqx.tree_at(lambda op: op.values, R, 2.0 * R.values)
        A = SparseOperator.from_coo(
            np.array([0, 1]), np.array([4, 8]), jnp.ones(2), (2, 9)
        )
        w_t = jnp.array([0.5, 1.5])
        H = Q.union(Q.congruence(A, w_t))  # Q + Aᵀ diag(w_t) A
        Ad = A.as_matrix()
        expected = 2.0 * R.as_matrix() + _congruence_dense(Ad, w_t)
        assert jnp.allclose(H.as_matrix(), expected)
        assert H.pattern == Q.pattern


# ---------------------------------------------------------------------------
# Transformations over values
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_jit_does_not_retrace_on_new_values(self, small_laplacian):
        traces = []

        @eqx.filter_jit
        def apply(op, x):
            traces.append(None)
            return op.mv(x)

        x = jnp.ones(12)
        apply(small_laplacian, x)
        rescaled = eqx.tree_at(
            lambda op: op.values, small_laplacian, 3.0 * small_laplacian.values
        )
        # A pattern rebuilt from the same indices is equal, so no retrace either.
        rebuilt = SparseOperator(
            small_laplacian.values,
            SparsityPattern(
                small_laplacian.pattern.rows,
                small_laplacian.pattern.cols,
                (12, 12),
                symmetric=True,
            ),
            tags=small_laplacian.tags,
        )
        assert jnp.allclose(apply(rescaled, x), 3.0 * small_laplacian.mv(x))
        apply(rebuilt, x)
        assert len(traces) == 1

    def test_grad_through_values(self, small_laplacian):
        x = jr.normal(jr.key(0), (12,))
        pattern = small_laplacian.pattern

        def quad(values):
            return x @ SparseOperator(values, pattern).mv(x)

        grad = jax.jit(jax.grad(quad))(small_laplacian.values)
        # d(xᵀ A x)/dA_ij = x_i x_j, counted twice for a mirrored off-diagonal.
        r, c = pattern.rows, pattern.cols
        expected = jnp.where(r == c, 1.0, 2.0) * x[r] * x[c]
        assert jnp.allclose(grad, expected)

    def test_grad_through_solve(self, small_laplacian):
        b = jnp.ones(12)
        pattern = small_laplacian.pattern

        def loss(values):
            return jnp.sum(gaussx.solve(SparseOperator(values, pattern, tags=PSD), b))

        def loss_dense(values):
            dense = SparseOperator(values, pattern).as_matrix()
            return jnp.sum(jnp.linalg.solve(dense, b))

        assert jnp.allclose(
            jax.grad(loss)(small_laplacian.values),
            jax.grad(loss_dense)(small_laplacian.values),
        )

    def test_vmap_over_values(self, small_laplacian):
        pattern = small_laplacian.pattern
        batch = einx.multiply(
            "b, n -> b n", jnp.arange(1.0, 4.0), small_laplacian.values
        )
        x = jr.normal(jr.key(0), (12,))
        out = jax.vmap(lambda v: SparseOperator(v, pattern).mv(x))(batch)
        for k in range(3):
            assert jnp.allclose(out[k], (k + 1.0) * small_laplacian.mv(x))

        ops = jax.vmap(lambda v: SparseOperator(v, pattern))(batch)
        diags = eqx.filter_vmap(gaussx.diag)(ops)
        assert jnp.allclose(diags[1], 2.0 * gaussx.diag(small_laplacian))

    def test_vmap_congruence_over_weights(self, small_laplacian):
        A = SparseOperator.from_coo(
            np.array([0, 1]), np.array([2, 6]), jnp.ones(2), (2, 12)
        )
        ws = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        H = jax.vmap(
            lambda w: small_laplacian.union(small_laplacian.congruence(A, w)).values
        )(ws)
        single = small_laplacian.union(small_laplacian.congruence(A, ws[1]))
        assert jnp.allclose(H[1], single.values)


# ---------------------------------------------------------------------------
# Primitive dispatch
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_small_solve_logdet_cholesky_dense(self, small_laplacian):
        dense = small_laplacian.as_matrix()
        b = jnp.arange(12.0)
        assert jnp.allclose(
            gaussx.solve(small_laplacian, b), jnp.linalg.solve(dense, b)
        )
        assert jnp.allclose(
            gaussx.logdet(small_laplacian), jnp.linalg.slogdet(dense)[1]
        )
        L = gaussx.cholesky(small_laplacian).as_matrix()
        assert jnp.allclose(L @ rearrange(L, "i j -> j i"), dense)

    def test_cg_solve_against_dense(self, small_laplacian):
        # PSD Laplacian plus a shift, through the CG strategy and lineax CG.
        b = jr.normal(jr.key(0), (12,))
        expected = jnp.linalg.solve(small_laplacian.as_matrix(), b)
        x = gaussx.CGSolver(rtol=1e-10, atol=1e-10).solve(small_laplacian, b)
        assert jnp.allclose(x, expected, atol=1e-8)
        cg = lx.CG(rtol=1e-10, atol=1e-10)
        x = gaussx.solve(small_laplacian, b, solver=cg)
        assert jnp.allclose(x, expected, atol=1e-8)

    def test_diag_inv_auto_avoids_refused_cholesky(self):
        # Between AutoSolver's threshold and diag_inv's dense limit (2048),
        # "auto" must not pick the Cholesky that cholesky(SparseOperator) refuses.
        n = AutoSolver().size_threshold + 1
        d = jnp.linspace(1.0, 2.0, n)
        op = SparseOperator.from_coo(
            np.arange(n), np.arange(n), d, (n, n), tags=frozenset({PSD})
        )
        out = gaussx.diag_inv(op)
        assert out.shape == (n,)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.slow
    def test_cg_solve_large_psd(self):
        # Above AutoSolver's threshold a PSD operator goes to CG.
        side = 33
        n = side * side
        assert n > AutoSolver().size_threshold
        senders, receivers = _grid_edges(side, side)
        op = _laplacian(
            senders,
            receivers,
            np.ones(senders.size),
            n,
            shift=0.5,
            tags=frozenset({PSD}),
        )
        b = jr.normal(jr.key(0), (n,))
        x = gaussx.solve(op, b)
        expected = jnp.linalg.solve(op.as_matrix(), b)
        # CG stops at rtol = 1e-5 on the residual; cond(op) <= 8.5 / 0.5.
        assert jnp.linalg.norm(x - expected) <= 1e-3 * jnp.linalg.norm(expected)

    def test_cholesky_refuses_above_threshold(self):
        n = AutoSolver().size_threshold + 1
        op = SparseOperator.from_coo(
            np.arange(n), np.arange(n), jnp.ones(n), (n, n), tags=PSD
        )
        with pytest.raises(NotImplementedError, match="sparse Cholesky"):
            gaussx.cholesky(op)

    def test_lanczos_eig_against_dense(self, small_laplacian):
        dense = small_laplacian.as_matrix()
        # Full-rank Lanczos recovers the dense spectrum through the matvec ...
        vals, vecs = gaussx.eig(small_laplacian, rank=12, key=jr.key(0))
        assert jnp.allclose(jnp.sort(vals), jnp.linalg.eigvalsh(dense), atol=1e-8)
        assert jnp.allclose(
            small_laplacian.mv(vecs[:, 0]), vals[0] * vecs[:, 0], atol=1e-8
        )
        # ... and a partial one matches Lanczos on the dense operator.
        partial, _ = gaussx.eig(small_laplacian, rank=4, key=jr.key(1))
        dense_op = lx.MatrixLinearOperator(dense, lx.symmetric_tag)
        dense_partial, _ = gaussx.eig(dense_op, rank=4, key=jr.key(1))
        assert jnp.allclose(partial, dense_partial)

    def test_jacobi_preconditioned_cg(self, small_laplacian):
        b = jnp.ones(12)
        solver = gaussx.CGSolver(
            preconditioner=gaussx.JacobiPreconditioner(), rtol=1e-10, atol=1e-10
        )
        x = solver.solve(small_laplacian, b)
        assert jnp.allclose(
            x, jnp.linalg.solve(small_laplacian.as_matrix(), b), atol=1e-6
        )

    @pytest.mark.slow
    def test_slq_logdet_within_error_bound(self):
        side = 33
        n = side * side
        senders, receivers = _grid_edges(side, side)
        op = _laplacian(
            senders,
            receivers,
            np.ones(senders.size),
            n,
            shift=0.5,
            tags=frozenset({PSD}),
        )
        estimate = gaussx.logdet(op)  # above the threshold: SLQ
        point, sem = SLQLogdet().logdet_and_error(op)
        exact = jnp.linalg.slogdet(op.as_matrix())[1]
        assert jnp.allclose(estimate, point)
        # Bounded by the estimator's own standard error (Monte-Carlo over 20
        # probes); 6 sigma, plus a small allowance for the Lanczos bias.
        assert jnp.abs(estimate - exact) <= 6.0 * sem + 1e-3 * jnp.abs(exact)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_matvec_benchmark():
    """``segment_sum`` (the internal matvec) against the BCOO matvec.

    On CPU (and in the G1 benchmark at 10³ to 10⁶ nodes) ``segment_sum`` was
    10 to 25 % faster than BCOO, so `SparseOperator.mv` uses it. The bound here
    only guards against a large regression.
    """
    side = 316
    n = side * side
    senders, receivers = _grid_edges(side, side)
    op = _laplacian(senders, receivers, np.ones(senders.size), n, shift=1.0)
    x = jr.normal(jr.key(0), (n,))
    seg = jax.jit(lambda o, v: o.mv(v))
    bcoo = jax.jit(lambda o, v: o.to_bcoo() @ v)

    def timed(fn) -> float:
        fn(op, x).block_until_ready()
        best = np.inf
        for _ in range(20):
            start = time.perf_counter()
            fn(op, x).block_until_ready()
            best = min(best, time.perf_counter() - start)
        return best

    assert jnp.allclose(seg(op, x), bcoo(op, x))
    t_seg, t_bcoo = timed(seg), timed(bcoo)
    print(
        f"\nmatvec n={n}: segment_sum {t_seg * 1e6:.0f} us, BCOO {t_bcoo * 1e6:.0f} us"
    )
    assert t_seg <= 3.0 * t_bcoo
