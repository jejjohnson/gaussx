"""Tests for `DiagonalisedOperator`, `Circulant` and diagonalised Kronecker sums.

Covers gh-261 (DiagonalisedOperator), gh-262 (Circulant) and the operator-level
part of gh-263 (Kronecker sums of non-symmetric diagonalisable factors).

Every reference is a dense materialisation of the same operator; inputs come
from pinned keys (the randomness is incidental), so tolerances are round-off
bounds for small well-conditioned systems.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

import gaussx
from gaussx import DiagonalisedOperator


# ---------------------------------------------------------------------------
# Fixtures: a periodic (FFT) and a Dirichlet (DST-I) Laplacian
# ---------------------------------------------------------------------------


def _fd_symbol(n: int) -> jnp.ndarray:
    """Eigenvalues of the periodic second difference, numpy.fft order."""
    return 2.0 * jnp.cos(2.0 * jnp.pi * jnp.fft.fftfreq(n)) - 2.0


class _Apply:
    """Identity-hashed ``x ↦ M x`` (a stand-in for a DST routine)."""

    def __init__(self, m):
        self.m = m

    def __call__(self, x):
        return self.m @ x


def _dst1_laplacian(n: int) -> DiagonalisedOperator:
    """Dirichlet second difference via the orthonormal (symmetric) DST-I."""
    j = np.arange(1, n + 1)
    S = np.sqrt(2.0 / (n + 1)) * np.sin(np.pi * np.outer(j, j) / (n + 1))
    lam = 2.0 * np.cos(np.pi * j / (n + 1)) - 2.0
    dst = _Apply(jnp.asarray(S))
    return DiagonalisedOperator(lam, dst, dst, (n,), normal=True)


def _dense_second_difference(n: int, periodic: bool) -> np.ndarray:
    A = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    if periodic:
        A[0, -1] = A[-1, 0] = 1.0
    return A


@pytest.fixture
def periodic_2d():
    """16×16 periodic five-point Laplacian as a block circulant."""
    sym = _fd_symbol(16)[:, None] + _fd_symbol(16)[None, :]
    return gaussx.circulant_from_symbol(sym)


@pytest.fixture
def periodic_2d_dense():
    A1 = _dense_second_difference(16, periodic=True)
    return jnp.asarray(np.kron(A1, np.eye(16)) + np.kron(np.eye(16), A1))


# ---------------------------------------------------------------------------
# DiagonalisedOperator basics
# ---------------------------------------------------------------------------


def test_circulant_symbol_matches_dense_laplacian(periodic_2d, periodic_2d_dense):
    assert jnp.allclose(periodic_2d.as_matrix(), periodic_2d_dense, atol=1e-12)
    assert lx.is_symmetric(periodic_2d)


def test_dst_operator_matches_dense_laplacian():
    op = _dst1_laplacian(16)
    dense = _dense_second_difference(16, periodic=False)
    assert jnp.allclose(op.as_matrix(), dense, atol=1e-12)


def test_eigenvalue_size_mismatch_raises():
    with pytest.raises(ValueError, match="eigenvalues"):
        DiagonalisedOperator(jnp.ones(5), lambda x: x, lambda x: x, (4,))


@pytest.mark.parametrize("lam", [0.5, 2.0])
def test_shift_stays_diagonalised_and_solves(periodic_2d, periodic_2d_dense, lam):
    """gh-261: ``A − λI`` never materialises; solve matches dense."""
    shifted = periodic_2d - lam * lx.IdentityLinearOperator(periodic_2d.in_structure())
    assert isinstance(shifted, DiagonalisedOperator)
    b = jr.normal(jr.key(0), (256,))
    x = gaussx.solve(shifted, b)
    expected = jnp.linalg.solve(periodic_2d_dense - lam * jnp.eye(256), b)
    assert jnp.allclose(x, expected, atol=1e-12)


def test_scalar_algebra_stays_closed(periodic_2d):
    identity = lx.IdentityLinearOperator(periodic_2d.in_structure())
    for op in (
        2.0 * periodic_2d,
        periodic_2d * 2.0,
        periodic_2d / 2.0,
        -periodic_2d,
        periodic_2d + identity,
        periodic_2d - identity,
        periodic_2d + periodic_2d,
        periodic_2d - 3.0 * periodic_2d,
    ):
        assert isinstance(op, DiagonalisedOperator)
    assert jnp.allclose(
        (periodic_2d - 3.0 * periodic_2d).as_matrix(),
        -2.0 * periodic_2d.as_matrix(),
        atol=1e-12,
    )


def test_shift_drops_definiteness_tags():
    """A PSD tag must not survive ``A − λI`` or ``−A``."""
    op = gaussx.Circulant(
        jnp.array([2.0, -1.0, 0.0, -1.0]), tags=lx.positive_semidefinite_tag
    )
    assert lx.is_positive_semidefinite(op)
    shifted = op - 5.0 * lx.IdentityLinearOperator(op.in_structure())
    assert not lx.is_positive_semidefinite(shifted)
    assert not lx.is_positive_semidefinite(-op)
    assert lx.is_symmetric(shifted)  # symmetry is re-derived


def test_singular_solve_is_pseudo_inverse(periodic_2d, periodic_2d_dense):
    """Zero eigenvalue (constant mode) → minimum-norm solution."""
    b = jr.normal(jr.key(1), (256,))
    b = b - b.mean()  # compatible right-hand side
    x = gaussx.solve(periodic_2d, b)
    assert jnp.allclose(x, jnp.linalg.pinv(periodic_2d_dense) @ b, atol=1e-10)


@pytest.mark.parametrize("lam", [0.3, 1.0])
def test_logdet_inv_trace_match_dense(periodic_2d, periodic_2d_dense, lam):
    op = periodic_2d - lam * lx.IdentityLinearOperator(periodic_2d.in_structure())
    dense = periodic_2d_dense - lam * jnp.eye(256)
    assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(dense)[1], atol=1e-10)
    inv = gaussx.inv(op)
    assert isinstance(inv, DiagonalisedOperator)
    assert jnp.allclose(inv.as_matrix(), jnp.linalg.inv(dense), atol=1e-12)
    assert jnp.allclose(gaussx.trace(op), jnp.trace(dense), atol=1e-10)


def test_sqrt_squares_to_operator():
    """sqrt(−Δ + I): S @ S = A, elementwise √λ, no materialisation."""
    op = -_dst1_laplacian(16) + lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((16,), jnp.float64)
    )
    S = gaussx.sqrt(op)
    assert isinstance(S, DiagonalisedOperator)
    assert jnp.allclose(S.as_matrix() @ S.as_matrix(), op.as_matrix(), atol=1e-12)


def test_diag_falls_back_correctly():
    op = _dst1_laplacian(12)
    assert jnp.allclose(gaussx.diag(op), jnp.diag(op.as_matrix()), atol=1e-12)


def test_inv_quad_logdet_is_exact(periodic_2d, periodic_2d_dense):
    op = -periodic_2d + lx.IdentityLinearOperator(periodic_2d.in_structure())
    dense = -periodic_2d_dense + jnp.eye(256)
    R = jr.normal(jr.key(2), (256, 3))
    inv_quad, ld = gaussx.inv_quad_logdet(op, R)
    assert jnp.allclose(inv_quad, jnp.sum(R * jnp.linalg.solve(dense, R)), atol=1e-10)
    assert jnp.allclose(ld, jnp.linalg.slogdet(dense)[1], atol=1e-10)


def test_transpose_normal_and_via_pair():
    op = gaussx.Circulant(jnp.array([1.0, 2.0, 0.0, 0.0, -1.0]))  # non-symmetric
    assert jnp.allclose(op.T.as_matrix(), op.as_matrix().T, atol=1e-12)
    M = _nonsymmetric(jr.key(3), 6)
    mat_op = DiagonalisedOperator.from_eigen_factorization(
        gaussx.EigenFactorization.from_matrix(M)
    )
    assert jnp.allclose(mat_op.as_matrix(), M, atol=1e-12)
    assert jnp.allclose(mat_op.T.as_matrix(), M.T, atol=1e-12)


def test_transpose_without_pair_raises():
    op = DiagonalisedOperator(jnp.ones(3), lambda x: x, lambda x: x, (3,))
    with pytest.raises(NotImplementedError):
        op.transpose()


# ---------------------------------------------------------------------------
# JAX transformations
# ---------------------------------------------------------------------------


def test_jit_vmap_grad(periodic_2d):
    op = periodic_2d - lx.IdentityLinearOperator(periodic_2d.in_structure())
    B = jr.normal(jr.key(4), (5, 256))

    solve = eqx.filter_jit(lambda o, b: gaussx.solve(o, b))
    assert jnp.allclose(solve(op, B[0]), gaussx.solve(op, B[0]), atol=1e-12)
    batched = jax.vmap(lambda b: gaussx.solve(op, b))(B)
    assert jnp.allclose(batched[2], gaussx.solve(op, B[2]), atol=1e-12)

    # d/db  ‖A⁻¹ b‖² / 2 = A⁻ᵀ A⁻¹ b
    g = jax.grad(lambda b: 0.5 * jnp.sum(gaussx.solve(op, b) ** 2))(B[0])
    Ad = op.as_matrix()
    assert jnp.allclose(
        g, jnp.linalg.solve(Ad.T, jnp.linalg.solve(Ad, B[0])), atol=1e-10
    )

    # d/dΛ logdet = 1/Λ
    lam = op.eigenvalues
    g_lam = jax.grad(lambda e: gaussx.logdet(op.with_eigenvalues(e)))(lam)
    assert jnp.allclose(g_lam, 1.0 / lam, atol=1e-12)


# ---------------------------------------------------------------------------
# Circulant (gh-262)
# ---------------------------------------------------------------------------


def test_circulant_1d_matches_scipy():
    c = jr.normal(jr.key(5), (32,))
    op = gaussx.Circulant(c)
    dense = jnp.asarray(scipy.linalg.circulant(np.asarray(c)))
    assert jnp.allclose(op.as_matrix(), dense, atol=1e-12)
    b = jr.normal(jr.key(6), (32,))
    assert jnp.allclose(gaussx.solve(op, b), jnp.linalg.solve(dense, b), atol=1e-10)
    assert jnp.allclose(gaussx.logdet(op), jnp.linalg.slogdet(dense)[1], atol=1e-10)
    assert jnp.allclose(gaussx.trace(op), jnp.trace(dense), atol=1e-12)
    assert jnp.allclose(gaussx.inv(op).as_matrix(), jnp.linalg.inv(dense), atol=1e-10)


def test_circulant_2d_block_circulant():
    c = jnp.zeros((8, 8)).at[0, 0].set(4.0).at[0, 1].set(-1.0).at[1, 0].set(-1.0)
    c = c.at[0, -1].set(-1.0).at[-1, 0].set(-1.0)  # symmetric 5-point kernel
    op = gaussx.Circulant(c, tags=lx.positive_semidefinite_tag)
    dense = op.as_matrix()
    assert jnp.allclose(dense, dense.T, atol=1e-12)
    assert jnp.allclose(dense[0].reshape(8, 8), c, atol=1e-12)  # first row = kernel
    S = gaussx.sqrt(op)
    assert jnp.allclose(S.as_matrix() @ S.as_matrix(), dense, atol=1e-10)


# ---------------------------------------------------------------------------
# Kronecker sums of diagonalised factors (gh-263, operator level)
# ---------------------------------------------------------------------------


@pytest.fixture
def no_eigh(monkeypatch):
    """Fail if any dense eigendecomposition runs (the structured path needs none)."""

    def boom(*args, **kwargs):
        raise AssertionError("dense eigendecomposition called")

    monkeypatch.setattr(jnp.linalg, "eigh", boom)
    monkeypatch.setattr(jnp.linalg, "eig", boom)


def _dense_kron_sum(*mats):
    total = 0
    for i, M in enumerate(mats):
        left = np.eye(int(np.prod([m.shape[0] for m in mats[:i]])))
        right = np.eye(int(np.prod([m.shape[0] for m in mats[i + 1 :]])))
        total = total + np.kron(np.kron(left, np.asarray(M)), right)
    return jnp.asarray(total)


def test_kronecker_sum_of_diagonalised_solves_without_eigh(no_eigh):
    """Periodic (FFT, complex) × Dirichlet (DST, real) factors."""
    a = gaussx.circulant_from_symbol(_fd_symbol(10)) - 0.7 * lx.IdentityLinearOperator(
        jax.ShapeDtypeStruct((10,), jnp.float64)
    )
    b = _dst1_laplacian(12)
    ks = gaussx.KroneckerSum(a, b)
    dense = _dense_kron_sum(a.as_matrix(), b.as_matrix())
    rhs = jr.normal(jr.key(7), (120,))
    assert jnp.allclose(gaussx.solve(ks, rhs), jnp.linalg.solve(dense, rhs), atol=1e-12)
    assert jnp.allclose(gaussx.logdet(ks), jnp.linalg.slogdet(dense)[1], atol=1e-10)


def test_kronecker_sum_three_factors(no_eigh):
    ops = [_dst1_laplacian(n) for n in (4, 5, 6)]
    ops[0] = ops[0] - 0.5 * lx.IdentityLinearOperator(ops[0].in_structure())
    ks = gaussx.KroneckerSum(gaussx.KroneckerSum(ops[0], ops[1]), ops[2])
    dense = _dense_kron_sum(*(o.as_matrix() for o in ops))
    rhs = jr.normal(jr.key(8), (120,))
    assert jnp.allclose(gaussx.solve(ks, rhs), jnp.linalg.solve(dense, rhs), atol=1e-12)
    assert isinstance(gaussx.as_diagonalised(ks), DiagonalisedOperator)


def _nonsymmetric(key, n: int):
    """V diag(λ) V⁻¹ with real negative λ and well-conditioned V."""
    k1, k2 = jr.split(key)
    V = jnp.eye(n) + 0.3 * jr.normal(k1, (n, n)) / jnp.sqrt(n)
    lam = jr.uniform(k2, (n,), minval=-6.0, maxval=-1.0)
    return (V * lam) @ jnp.linalg.inv(V)


def test_kronecker_sum_nonsymmetric_factors_with_shift():
    """gh-263: ``solve(KroneckerSum(A − σI, B))`` with non-symmetric factors."""
    A = _nonsymmetric(jr.key(9), 7)
    B = _nonsymmetric(jr.key(10), 6)
    fa = DiagonalisedOperator.from_eigen_factorization(
        gaussx.EigenFactorization.from_matrix(A)
    )
    fb = DiagonalisedOperator.from_eigen_factorization(
        gaussx.EigenFactorization.from_matrix(B)
    )
    sigma = 1.3
    ks = gaussx.KroneckerSum(
        fa - sigma * lx.IdentityLinearOperator(fa.in_structure()), fb
    )
    dense = _dense_kron_sum(A - sigma * jnp.eye(7), B)
    rhs = jr.normal(jr.key(11), (42,))
    assert jnp.allclose(gaussx.solve(ks, rhs), jnp.linalg.solve(dense, rhs), atol=1e-10)
