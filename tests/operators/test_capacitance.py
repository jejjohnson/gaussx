"""Tests for the generic capacitance-matrix solver."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gaussx import CapacitanceSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_capacitance_enforces_boundary_constraint(getkey):
    """The solution is zero at the constrained (boundary) indices."""
    n = 12
    b_mat = random_pd_matrix(getkey(), n)
    b_inv = jnp.linalg.inv(b_mat)
    boundary = jnp.array([1, 4, 7])

    solver = CapacitanceSolver(lambda f: b_inv @ f, boundary, n)
    rhs = jr.normal(getkey(), (n,))
    x = solver(rhs)

    assert tree_allclose(x[boundary], jnp.zeros(boundary.shape[0]), atol=1e-6)


def test_capacitance_residual_supported_on_boundary(getkey):
    """``B x - f`` vanishes away from the constrained indices.

    The capacitance method yields ``x = B^{-1}(f - sum_k alpha_k e_{b_k})``, so
    the residual ``B x - f`` is a combination of point sources located only at
    the constrained indices.
    """
    n = 16
    b_mat = random_pd_matrix(getkey(), n)
    b_inv = jnp.linalg.inv(b_mat)
    boundary = jnp.array([2, 9, 13])

    solver = CapacitanceSolver(lambda f: b_inv @ f, boundary, n)
    rhs = jr.normal(getkey(), (n,))
    x = solver(rhs)

    residual = b_mat @ x - rhs
    interior_mask = jnp.ones(n).at[boundary].set(0.0)
    assert tree_allclose(residual * interior_mask, jnp.zeros(n), atol=1e-5)


def test_capacitance_is_linear(getkey):
    """The solver is a linear map of the right-hand side."""
    n = 10
    b_mat = random_pd_matrix(getkey(), n)
    b_inv = jnp.linalg.inv(b_mat)
    boundary = jnp.array([0, 5])
    solver = CapacitanceSolver(lambda f: b_inv @ f, boundary, n)

    f1 = jr.normal(getkey(), (n,))
    f2 = jr.normal(getkey(), (n,))
    combined = solver(2.0 * f1 + 3.0 * f2)
    separate = 2.0 * solver(f1) + 3.0 * solver(f2)
    assert tree_allclose(combined, separate, rtol=1e-5)


# ---------------------------------------------------------------------------
# Masked-domain Poisson on a periodic grid (gh-259, gh-260).
#
# The grid, mask and RHS are fixed (pinned key): the randomness is incidental,
# so the tolerances are round-off bounds for a well-conditioned 24×20 problem.
# ---------------------------------------------------------------------------

NY, NX = 20, 24


def _periodic_laplacian(u):
    """Five-point periodic Laplacian (unit spacing) on a (NY, NX) field."""
    return (
        jnp.roll(u, 1, 0)
        + jnp.roll(u, -1, 0)
        + jnp.roll(u, 1, 1)
        + jnp.roll(u, -1, 1)
        - 4.0 * u
    )


def _periodic_poisson_pinv(f_flat):
    """FFT pseudo-inverse of the periodic Laplacian: zero mode projected out."""
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(NY)
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(NX)
    eig = (2.0 * jnp.cos(ky)[:, None] - 2.0) + (2.0 * jnp.cos(kx)[None, :] - 2.0)
    f_hat = jnp.fft.fft2(f_flat.reshape(NY, NX))
    u_hat = jnp.where(eig == 0.0, 0.0, f_hat / jnp.where(eig == 0.0, 1.0, eig))
    return jnp.fft.ifft2(u_hat).real.reshape(-1)


def _ring_indices():
    """Flat indices of a one-cell ring around a disc of radius 6."""
    j, i = np.mgrid[:NY, :NX]
    r = np.hypot(j - NY / 2 + 0.5, i - NX / 2 + 0.5)
    ring = (r >= 6.0) & (r < 7.0)
    return jnp.asarray(np.flatnonzero(ring.ravel()))


@pytest.fixture
def masked_problem():
    boundary = _ring_indices()
    rhs = jr.normal(jr.key(0), (NY * NX,))
    free = jnp.ones(NY * NX, dtype=bool).at[boundary].set(False)
    return boundary, rhs, free


def _residual(x, rhs):
    return _periodic_laplacian(x.reshape(NY, NX)).reshape(-1) - rhs


def test_singular_base_without_null_vector_leaves_constant_residual(masked_problem):
    """Documents the gh-260 defect: a pseudo-inverse base alone gives a
    uniform nonzero residual at the unconstrained indices."""
    boundary, rhs, free = masked_problem
    solver = CapacitanceSolver(_periodic_poisson_pinv, boundary, NY * NX)
    res = _residual(solver(rhs), rhs)[free]
    assert float(jnp.abs(res).max()) > 1e-6
    assert float(jnp.std(res)) < 1e-10  # the error is a constant


def test_singular_base_with_null_vector_solves_the_pde(masked_problem):
    """gh-260: with the null vector the PDE holds away from the constraints."""
    boundary, rhs, free = masked_problem
    n = NY * NX
    solver = CapacitanceSolver(
        _periodic_poisson_pinv, boundary, n, null_vector=jnp.ones(n)
    )
    x = solver(rhs)
    assert float(jnp.abs(x[boundary]).max()) < 1e-12
    assert float(jnp.abs(_residual(x, rhs)[free]).max()) < 1e-10


def test_nonsymmetric_singular_base_uses_left_null_vector():
    """Bordered system with distinct right/left null vectors (pinv base)."""
    n = 9
    k1, k2 = jr.split(jr.key(1))
    V = jnp.eye(n) + 0.3 * jr.normal(k1, (n, n)) / jnp.sqrt(n)
    lam = jnp.concatenate([jnp.zeros(1), jr.uniform(k2, (n - 1,), minval=1, maxval=3)])
    B = (V * lam) @ jnp.linalg.inv(V)  # non-symmetric, rank n − 1
    r = V[:, 0]  # B r = 0
    ell = jnp.linalg.inv(V)[0]  # ellᵀ B = 0
    B_pinv = jnp.linalg.pinv(B)
    boundary = jnp.array([1, 5])
    rhs = jr.normal(jr.key(2), (n,))

    solver = CapacitanceSolver(
        lambda f: B_pinv @ f, boundary, n, null_vector=r, left_null_vector=ell
    )
    x = solver(rhs)
    free = jnp.ones(n, dtype=bool).at[boundary].set(False)
    assert float(jnp.abs(x[boundary]).max()) < 1e-10
    assert float(jnp.abs((B @ x - rhs)[free]).max()) < 1e-10


def test_nonsingular_base_matches_dense_masked_solve():
    """gh-259 regression: result equals a dense solve of the masked system.

    Base: dense periodic Laplacian shifted by −1 (non-singular). Reference:
    replace the constrained rows by identity rows and solve densely.
    """
    n = NY * NX
    eye = jnp.eye(n)
    A = jax.vmap(lambda e: _periodic_laplacian(e.reshape(NY, NX)).reshape(-1))(eye).T
    A = A - eye
    A_inv = jnp.linalg.inv(A)
    boundary = _ring_indices()
    rhs = jr.normal(jr.key(3), (n,))

    x = CapacitanceSolver(lambda f: A_inv @ f, boundary, n)(rhs)

    A_ref = A.at[boundary].set(eye[boundary])
    b_ref = rhs.at[boundary].set(0.0)
    assert jnp.allclose(x, jnp.linalg.solve(A_ref, b_ref), atol=1e-12)


def test_solver_stores_no_green_table(masked_problem):
    """gh-259: no (N_b, n) array is stored — only the N_b² factorization."""
    boundary, _, _ = masked_problem
    n = NY * NX
    solver = CapacitanceSolver(
        _periodic_poisson_pinv, boundary, n, null_vector=jnp.ones(n)
    )
    n_b = boundary.shape[0]
    for leaf in jax.tree_util.tree_leaves(solver):
        assert leaf.size <= max(n, (n_b + 1) ** 2)


def test_solver_works_under_plain_jit(masked_problem):
    """gh-259: ``base_solve`` is static, so the solver can be passed straight
    through plain ``jax.jit`` (all its pytree leaves are arrays)."""
    boundary, rhs, _ = masked_problem
    n = NY * NX
    solver = CapacitanceSolver(
        _periodic_poisson_pinv, boundary, n, null_vector=jnp.ones(n)
    )
    expected = solver(rhs)
    assert jnp.allclose(jax.jit(lambda s, r: s(r))(solver, rhs), expected, atol=1e-12)
    assert jnp.allclose(
        eqx.filter_jit(lambda s, r: s(r))(solver, rhs), expected, atol=1e-12
    )
