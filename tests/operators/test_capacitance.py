"""Tests for the generic capacitance-matrix solver."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

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
