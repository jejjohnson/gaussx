"""Tests for gaussx.pseudo_logdet (G5)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._testing import default_tolerances


def _dense_pseudo_logdet(M, rtol=1e-9):
    evals = np.linalg.eigvalsh(np.asarray(M, dtype=np.float64))
    return float(np.sum(np.log(evals[evals > rtol * evals.max()])))


def _laplacian(n, edges, weights, *, symmetric=True):
    """Weighted graph Laplacian as a SparseOperator (lower or full storage)."""
    edges = np.array(edges, dtype=int)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    degree = jnp.zeros(n).at[edges[:, 0]].add(weights).at[edges[:, 1]].add(weights)
    lo, hi = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
    if symmetric:
        rows, cols = np.r_[np.arange(n), hi], np.r_[np.arange(n), lo]
        values = jnp.concatenate([degree, -weights])
    else:
        rows = np.r_[np.arange(n), hi, lo]
        cols = np.r_[np.arange(n), lo, hi]
        values = jnp.concatenate([degree, -weights, -weights])
    return gaussx.SparseOperator.from_coo(
        rows, cols, values, (n, n), symmetric=symmetric
    )


def _path_edges(n, offset=0):
    return [(offset + i, offset + i + 1) for i in range(n - 1)]


def _grid():
    A = gaussx.rw1_structure(5).as_matrix()
    B = gaussx.rw1_structure(4, spacing=jnp.array([1.0, 0.5, 2.0])).as_matrix()
    op = gaussx.KroneckerSum(
        lx.MatrixLinearOperator(A, lx.symmetric_tag),
        lx.MatrixLinearOperator(B, lx.symmetric_tag),
    )
    return op, op.as_matrix()


# ---------------------------------------------------------------------------
# Grid Laplacian: dense vs KroneckerSum vs null_space (+ SLQ)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_grid_kronecker_sum_and_null_space_match_dense():
    op, M = _grid()
    expected = _dense_pseudo_logdet(M)
    ones = jnp.ones(M.shape[0]) / jnp.sqrt(M.shape[0])
    dense = gaussx.pseudo_logdet(lx.MatrixLinearOperator(M, lx.symmetric_tag))
    assert jnp.allclose(dense, expected, rtol=1e-10)
    assert jnp.allclose(gaussx.pseudo_logdet(op), expected, rtol=1e-10)
    assert jnp.allclose(gaussx.pseudo_logdet(op, null_space=ones), expected)
    # A grid Laplacian with structure="laplacian" takes the eigenvalue path.
    laplacian = gaussx.pseudo_logdet(op, structure="laplacian")
    assert jnp.allclose(laplacian, expected, rtol=1e-10)


@pytest.mark.slow
def test_grid_null_space_slq_within_its_standard_error():
    op, M = _grid()
    n = M.shape[0]
    ones = jnp.ones(n) / jnp.sqrt(n)
    slq = gaussx.SLQLogdet(num_probes=200, lanczos_order=n)
    estimate = gaussx.pseudo_logdet(op, null_space=ones, strategy=slq)
    shifted = lx.MatrixLinearOperator(
        M + einsum(ones, ones, "i, j -> i j"),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    reference, sem = slq.logdet_and_error(shifted)  # the same probes
    assert jnp.allclose(estimate, reference)
    assert abs(float(estimate) - _dense_pseudo_logdet(M)) < 4 * float(sem)


# ---------------------------------------------------------------------------
# Two connected components and basis invariance
# ---------------------------------------------------------------------------


def _two_components():
    n1, n2 = 4, 5
    edges = _path_edges(n1) + _path_edges(n2, offset=n1) + [(n1, n1 + 2)]
    weights = jnp.linspace(0.5, 2.0, len(edges))
    L = _laplacian(n1 + n2, edges, weights)
    N = np.zeros((n1 + n2, 2))
    N[:n1, 0] = 1 / np.sqrt(n1)
    N[n1:, 1] = 1 / np.sqrt(n2)
    return L, jnp.asarray(N)


@pytest.mark.slow
def test_two_components():
    L, N = _two_components()
    expected = _dense_pseudo_logdet(L.as_matrix())
    assert jnp.allclose(gaussx.pseudo_logdet(L), expected, rtol=1e-10)
    assert jnp.allclose(gaussx.pseudo_logdet(L, null_space=N), expected)
    assert jnp.allclose(gaussx.pseudo_logdet(L, structure="laplacian"), expected)


def test_null_space_basis_invariance():
    L, N = _two_components()
    expected = gaussx.pseudo_logdet(L, null_space=N)
    rotation, _ = jnp.linalg.qr(jr.normal(jr.key(0), (2, 2)))
    rotated = einsum(N, rotation, "n c, c d -> n d")
    # Any basis of the kernel, not only an orthonormal one.
    mixed = einsum(N, jnp.array([[3.0, 1.0], [0.5, 2.0]]), "n c, c d -> n d")
    assert jnp.allclose(gaussx.pseudo_logdet(L, null_space=rotated), expected)
    assert jnp.allclose(gaussx.pseudo_logdet(L, null_space=mixed), expected)


# ---------------------------------------------------------------------------
# Laplacian cofactor path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
def test_cofactor_connected_random_graph(symmetric):
    rng = np.random.default_rng(0)
    n = 12
    tree = [(i, int(rng.integers(i))) for i in range(1, n)]  # spanning tree
    extra = [(int(i), int(j)) for i, j in rng.integers(n, size=(10, 2)) if i != j]
    edges = list({tuple(sorted(e)) for e in tree + extra})
    weights = jnp.asarray(rng.uniform(0.2, 3.0, len(edges)))
    L = _laplacian(n, edges, weights, symmetric=symmetric)
    expected = _dense_pseudo_logdet(L.as_matrix())
    assert jnp.allclose(gaussx.pseudo_logdet(L, structure="laplacian"), expected)


def test_cofactor_disconnected_graph_with_isolated_node():
    # Components {0..3}, {4, 5, 6} (a triangle), {7} (isolated), {8, 9}.
    edges = [*_path_edges(4), (4, 5), (5, 6), (4, 6), (8, 9)]
    weights = jnp.arange(1.0, len(edges) + 1.0)
    L = _laplacian(10, edges, weights)
    expected = _dense_pseudo_logdet(L.as_matrix())
    assert jnp.allclose(gaussx.pseudo_logdet(L, structure="laplacian"), expected)


def test_cofactor_is_exact_under_jit_and_grad():
    L, _ = _two_components()
    rank = L.in_size() - 2

    def f(log_tau):
        scaled = eqx.tree_at(lambda op: op.values, L, jnp.exp(log_tau) * L.values)
        return gaussx.pseudo_logdet(scaled, structure="laplacian")

    value, slope = jax.jit(jax.value_and_grad(f))(jnp.log(2.0))
    # log|τR|₊ = rank log τ + log|R|₊
    assert jnp.allclose(value, rank * jnp.log(2.0) + f(0.0))
    assert jnp.allclose(slope, rank)


def test_cofactor_with_explicit_strategy():
    L, _ = _two_components()
    expected = _dense_pseudo_logdet(L.as_matrix())
    got = gaussx.pseudo_logdet(L, structure="laplacian", strategy=gaussx.DenseSolver())
    assert jnp.allclose(got, expected)


# ---------------------------------------------------------------------------
# RW1 and RW2
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 7, 30])
def test_rw1_closed_form(n):
    R = gaussx.rw1_structure(n)
    k = np.arange(1, n)
    expected = float(np.sum(np.log(2 - 2 * np.cos(np.pi * k / n))))
    assert jnp.allclose(gaussx.pseudo_logdet(R), expected)
    assert jnp.allclose(gaussx.pseudo_logdet(R, structure="laplacian"), expected)
    ones = jnp.ones(n) / jnp.sqrt(n)
    assert jnp.allclose(gaussx.pseudo_logdet(R, null_space=ones), expected)
    # The closed form is pdet = n (a path has one spanning tree).
    assert jnp.allclose(expected, np.log(n))


def test_rw1_irregular_spacing_laplacian_matches_dense():
    spacing = jnp.array([0.5, 1.0, 2.0, 0.25, 1.5])
    R = gaussx.rw1_structure(6, spacing=spacing)
    expected = _dense_pseudo_logdet(R.as_matrix())
    assert jnp.allclose(gaussx.pseudo_logdet(R, structure="laplacian"), expected)


def test_rw1_cyclic_laplacian_matches_dense():
    R = gaussx.rw1_structure(8, cyclic=True)
    expected = _dense_pseudo_logdet(R.as_matrix())
    assert jnp.allclose(gaussx.pseudo_logdet(R, structure="laplacian"), expected)


@pytest.mark.parametrize("n", [6, 7, 10, 11])
def test_rw2_against_dense(n):
    # Unpadded D₂ᵀD₂; for odd n the operator has a decoupled unit-precision
    # padding node, whose eigenvalue 1 adds log 1 = 0.
    D = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    D[rows, rows], D[rows, rows + 1], D[rows, rows + 2] = 1.0, -2.0, 1.0
    expected = _dense_pseudo_logdet(einsum(D, D, "k i, k j -> i j"))

    R = gaussx.rw2_structure(n)
    assert R.in_size() == n + n % 2
    # RW2's condition number grows like n^4, so a float32 run (the no-x64
    # lane) is only good to ~1e-5 relative against this float64 reference.
    result = gaussx.pseudo_logdet(R)
    rtol, atol = default_tolerances(result)
    assert jnp.allclose(result, expected, rtol=rtol, atol=atol)

    t = np.arange(n, dtype=float)
    basis = np.zeros((R.in_size(), 2))
    basis[:n, 0], basis[:n, 1] = 1.0, t  # not orthonormal: any basis works
    result = gaussx.pseudo_logdet(R, null_space=basis)
    assert jnp.allclose(result, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# dtype, rcond and errors
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_float32_stays_float32():
    R = gaussx.SparseOperator.from_coo(
        np.array([0, 1, 2, 1, 2]),
        np.array([0, 1, 2, 0, 1]),
        jnp.array([1.0, 2.0, 1.0, -1.0, -1.0], dtype=jnp.float32),
        (3, 3),
        symmetric=True,
    )
    for kwargs in (
        {},
        {"structure": "laplacian"},
        {"null_space": jnp.ones(3, dtype=jnp.float32) / jnp.sqrt(3.0)},
    ):
        got = gaussx.pseudo_logdet(R, **kwargs)
        assert got.dtype == jnp.float32
        assert jnp.allclose(got, jnp.log(3.0), rtol=1e-5)


def test_rcond_drops_small_eigenvalues():
    M = jnp.diag(jnp.array([0.0, 1e-6, 1.0, 2.0]))
    op = lx.MatrixLinearOperator(M, lx.symmetric_tag)
    assert jnp.allclose(gaussx.pseudo_logdet(op), jnp.log(1e-6 * 2.0))
    assert jnp.allclose(gaussx.pseudo_logdet(op, rcond=1e-3), jnp.log(2.0))


def test_errors():
    L, N = _two_components()
    with pytest.raises(ValueError, match="not both"):
        gaussx.pseudo_logdet(L, null_space=N, structure="laplacian")
    with pytest.raises(ValueError, match="structure must be"):
        gaussx.pseudo_logdet(L, structure="graph")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="strategy"):
        gaussx.pseudo_logdet(L, strategy=gaussx.DenseSolver())
    with pytest.raises(ValueError, match="shape"):
        gaussx.pseudo_logdet(L, null_space=jnp.ones((3, 1)))
    with pytest.raises(TypeError, match="laplacian"):
        gaussx.pseudo_logdet(
            lx.MatrixLinearOperator(L.as_matrix()), structure="laplacian"
        )
    with pytest.raises(ValueError, match="1 x 1"):
        gaussx.pseudo_logdet(gaussx.rw2_structure(6), structure="laplacian")
