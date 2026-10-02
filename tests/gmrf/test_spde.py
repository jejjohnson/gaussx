"""Tests for spde_precision, spde_precision_grid and matern_spde_params."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gaussx
from gaussx._einx import rearrange


def right_triangle_mesh(n0: int, n1: int, h: float):
    """Regular mesh of an ``n0 × n1`` grid, every square split the same way.

    Vertex ``(i0, i1)`` sits at ``(x, y) = (i1 h, i0 h)`` with index
    ``i0 n1 + i1``, the row-major layout of `spde_precision_grid`.
    """
    i0, i1 = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    vertices = np.c_[i1.ravel() * h, i0.ravel() * h]
    triangles = []
    for a in range(n0 - 1):
        for b in range(n1 - 1):
            p, q, r, s = (
                a * n1 + b,
                a * n1 + b + 1,
                (a + 1) * n1 + b + 1,
                (a + 1) * n1 + b,
            )
            triangles += [[p, q, r], [p, r, s]]
    return vertices, np.array(triangles)


def boundary_distance(n0: int, n1: int) -> np.ndarray:
    i0, i1 = np.meshgrid(np.arange(n0), np.arange(n1), indexing="ij")
    return np.minimum(np.minimum(i0, n0 - 1 - i0), np.minimum(i1, n1 - 1 - i1)).ravel()


class TestSpdePrecision:
    @pytest.mark.parametrize("alpha", [1, 2, 3])
    def test_matches_dense_recursion(self, alpha):
        vertices, triangles = right_triangle_mesh(4, 5, 0.3)
        C, G = gaussx.fem_matrices(vertices, triangles)
        kappa, tau = 1.7, 0.6
        Q = gaussx.spde_precision(C, G, kappa, tau, alpha)
        c = jnp.diag(C.as_matrix())
        K = kappa**2 * jnp.diag(c) + G.as_matrix()
        expected = tau**2 * K
        for _ in range(alpha - 1):
            expected = K @ jnp.diag(1 / c) @ expected
        assert jnp.allclose(Q.as_matrix(), expected, rtol=1e-10, atol=1e-10)
        assert Q.pattern.symmetric

    def test_pattern_is_independent_of_parameters(self):
        vertices, triangles = right_triangle_mesh(4, 4, 1.0)
        C, G = gaussx.fem_matrices(vertices, triangles)
        Q1 = gaussx.spde_precision(C, G, 0.5, 1.0, 2)
        Q2 = gaussx.spde_precision(C, G, 3.0, 2.0, 2)
        assert Q1.pattern == Q2.pattern

    def test_jit_and_grad(self):
        vertices, triangles = right_triangle_mesh(3, 4, 1.0)
        C, G = gaussx.fem_matrices(vertices, triangles)

        def logdet(log_kappa):
            return gaussx.logdet(
                gaussx.spde_precision(C, G, jnp.exp(log_kappa), 1.0, 2)
            )

        value, grad = jax.jit(jax.value_and_grad(logdet))(0.1)
        dense = lambda lk: jnp.linalg.slogdet(
            gaussx.spde_precision(C, G, jnp.exp(lk), 1.0, 2).as_matrix()
        )[1]
        assert jnp.allclose(value, dense(0.1))
        assert jnp.allclose(grad, jax.grad(dense)(0.1))

    def test_bad_alpha_raises(self):
        vertices, triangles = right_triangle_mesh(3, 3, 1.0)
        C, G = gaussx.fem_matrices(vertices, triangles)
        with pytest.raises(ValueError, match="alpha"):
            gaussx.spde_precision(C, G, 1.0, 1.0, 0)


class TestSpdePrecisionGrid:
    @pytest.mark.parametrize("alpha", [1, 2, 3])
    def test_equals_fem_on_right_triangle_mesh_in_the_interior(self, alpha):
        """Grid SPDE = FEM SPDE with lumped mass, away from the boundary.

        Interior rows at distance ≥ α from the boundary only touch interior
        vertices (lumped mass h², full-weight edges); nearer the boundary the
        mesh has smaller lumped masses and half-weight boundary edges.
        """
        n0, n1, h = 7, 8, 0.5
        kappa, tau = 1.3, 0.7
        vertices, triangles = right_triangle_mesh(n0, n1, h)
        C, G = gaussx.fem_matrices(vertices, triangles)
        fem = gaussx.spde_precision(C, G, kappa, tau, alpha).as_matrix()
        grid = gaussx.spde_precision_grid((n0, n1), kappa, tau, alpha, spacing=h)
        interior = boundary_distance(n0, n1) >= alpha
        assert jnp.allclose(grid.as_matrix()[interior], fem[interior], atol=1e-9)

    def test_symbol_on_a_raster(self):
        # Q = τ² h² (κ² I + h⁻² (L_H ⊕ L_W))^α on a 2-D raster
        shape, kappa, tau, alpha, h = (3, 4), 0.8, 1.5, 2, 0.25
        Q = gaussx.spde_precision_grid(shape, kappa, tau, alpha, spacing=h)
        L = gaussx.KroneckerSum(
            gaussx.rw1_structure(3), gaussx.rw1_structure(4)
        ).as_matrix()
        base = kappa**2 * jnp.eye(12) + L / h**2
        expected = tau**2 * h**2 * jnp.linalg.matrix_power(base, alpha)
        assert jnp.allclose(Q.as_matrix(), expected)

    def test_periodic_axis(self):
        Q = gaussx.spde_precision_grid((3, 5), 0.5, 1.0, 1, periodic=(False, True))
        L_path = gaussx.rw1_structure(3).as_matrix()
        L_cycle = gaussx.rw1_structure(5, cyclic=True).as_matrix()
        L = jnp.kron(L_path, jnp.eye(5)) + jnp.kron(jnp.eye(3), L_cycle)
        assert jnp.allclose(Q.as_matrix(), 0.25 * jnp.eye(15) + L)

    def test_exact_operations(self):
        Q = gaussx.spde_precision_grid((4, 3, 2), 0.9, 1.2, 2, periodic=True)
        dense = Q.as_matrix()
        b = jnp.arange(24.0)
        assert jnp.allclose(gaussx.solve(Q, b), jnp.linalg.solve(dense, b))
        assert jnp.allclose(gaussx.logdet(Q), jnp.linalg.slogdet(dense)[1])
        assert jnp.allclose(gaussx.diag_inv(Q), jnp.diag(jnp.linalg.inv(dense)))

    def test_grad_through_kappa(self):
        def logdet(kappa):
            return gaussx.logdet(gaussx.spde_precision_grid((4, 5), kappa, 1.0, 2))

        def dense(kappa):
            Q = gaussx.spde_precision_grid((4, 5), kappa, 1.0, 2)
            return jnp.linalg.slogdet(Q.as_matrix())[1]

        assert jnp.allclose(jax.jit(jax.grad(logdet))(0.7), jax.grad(dense)(0.7))

    def test_periodic_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="periodic"):
            gaussx.spde_precision_grid((3, 4), 1.0, 1.0, 2, periodic=(True,))


class TestMaternSpdeParams:
    def test_formulas(self):
        kappa, tau, alpha = gaussx.matern_spde_params(50.0, 2.0, 1.0, 2)
        assert alpha == 2
        assert jnp.allclose(kappa, math.sqrt(8.0) / 50.0)
        # σ² = Γ(nu) / (Γ(α) (4π)^{d/2} κ^{2nu} τ²) = 1 / (4π κ² τ²) for nu = 1, d = 2
        assert jnp.allclose(1.0 / (4 * math.pi * kappa**2 * tau**2), 4.0)

    def test_one_dimension(self):
        kappa, tau, alpha = gaussx.matern_spde_params(10.0, 1.0, 0.5, 1)
        assert alpha == 1
        # Γ(1/2) / (Γ(1) (4π)^{1/2} κ τ²) = 1 / (2 κ τ²)
        assert jnp.allclose(1.0 / (2 * kappa * tau**2), 1.0)

    def test_non_integer_alpha_raises(self):
        with pytest.raises(ValueError, match="integer"):
            gaussx.matern_spde_params(1.0, 1.0, 1.5, 2)

    @pytest.mark.slow
    def test_marginal_variance_away_from_the_boundary(self):
        """Variance ≈ σ² in the middle of a raster, range 20 cells.

        The FEM / grid discretisation error is O((κh)²) ≈ 2 % at κh = 0.14;
        the centre is 2.5 ranges from every edge, so boundary inflation is
        negligible there, while an edge shows about 2σ².
        """
        sigma = 2.0
        kappa, tau, alpha = gaussx.matern_spde_params(20.0, sigma, 1.0, 2)
        Q = gaussx.spde_precision_grid((101, 101), kappa, tau, alpha)
        variance = rearrange(gaussx.diag_inv(Q), "(h w) -> h w", h=101, w=101)
        assert jnp.allclose(variance[50, 50], sigma**2, rtol=0.03)
        assert jnp.allclose(variance[0, 50], 2 * sigma**2, rtol=0.05)
