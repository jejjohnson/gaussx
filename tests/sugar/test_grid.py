"""Tests for grid construction and cubic interpolation weights."""

import jax
import jax.numpy as jnp

from gaussx import create_grid, cubic_interpolation_weights, grid_data


class TestCreateGrid:
    def test_shapes(self):
        """Grid arrays have correct shapes."""
        grid = create_grid([5, 10], [(-1.0, 1.0), (0.0, 2.0)])
        assert len(grid) == 2
        assert grid[0].shape == (5,)
        assert grid[1].shape == (10,)

    def test_bounds(self):
        """Grid arrays span the specified bounds."""
        grid = create_grid([20], [(-3.0, 3.0)])
        assert jnp.allclose(grid[0][0], -3.0)
        assert jnp.allclose(grid[0][-1], 3.0)


class TestGridData:
    def test_shape(self):
        """Cartesian product has correct shape."""
        grid = create_grid([5, 10], [(-1.0, 1.0), (0.0, 2.0)])
        data = grid_data(grid)
        assert data.shape == (50, 2)

    def test_1d(self):
        """1-D grid data is just the grid points reshaped."""
        grid = create_grid([7], [(0.0, 1.0)])
        data = grid_data(grid)
        assert data.shape == (7, 1)
        assert jnp.allclose(data[:, 0], grid[0])


class TestCubicInterpolationWeights:
    def test_weights_sum_to_one(self):
        """Interpolation weights sum to ~1 for interior points."""
        grid = create_grid([20], [(-1.0, 1.0)])
        x = jnp.array([[0.0], [0.3], [-0.5]])
        _indices, weights = cubic_interpolation_weights(x, grid)
        sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_exact_at_grid_points(self):
        """Interpolation at grid points recovers exact values."""
        grid = create_grid([11], [(-1.0, 1.0)])
        gd = grid_data(grid)  # (11, 1)
        # Use interior points (avoid boundary clipping)
        x = gd[2:-2]
        indices, weights = cubic_interpolation_weights(x, grid)

        # Interpolate a known function: f(x) = x^2
        f_grid = gd[:, 0] ** 2  # (11,)
        f_interp = jnp.sum(weights * f_grid[indices], axis=1)
        f_exact = x[:, 0] ** 2
        assert jnp.allclose(f_interp, f_exact, atol=1e-5)

    def test_1d_accuracy(self):
        """1-D cubic interpolation is accurate for smooth functions."""
        grid = create_grid([50], [(-2.0, 2.0)])
        gd = grid_data(grid)

        # Test points between grid points
        x = jnp.linspace(-1.5, 1.5, 30)[:, None]
        indices, weights = cubic_interpolation_weights(x, grid)

        # Interpolate sin(x)
        f_grid = jnp.sin(gd[:, 0])
        f_interp = jnp.sum(weights * f_grid[indices], axis=1)
        f_exact = jnp.sin(x[:, 0])
        assert jnp.allclose(f_interp, f_exact, atol=1e-3)

    def test_2d_shapes(self):
        """2-D interpolation produces correct shapes."""
        grid = create_grid([10, 10], [(-1.0, 1.0), (-1.0, 1.0)])
        B = 5
        x = jax.random.normal(jax.random.PRNGKey(0), (B, 2)) * 0.5
        indices, weights = cubic_interpolation_weights(x, grid)
        assert indices.shape == (B, 16)  # 4^2
        assert weights.shape == (B, 16)

    def test_2d_weights_sum_to_one(self):
        """2-D weights sum to ~1."""
        grid = create_grid([10, 10], [(-1.0, 1.0), (-1.0, 1.0)])
        x = jnp.array([[0.0, 0.0], [0.3, -0.2]])
        _indices, weights = cubic_interpolation_weights(x, grid)
        sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_jit_compatible(self):
        """Works under jax.jit."""
        grid = create_grid([20], [(-1.0, 1.0)])
        x = jnp.array([[0.0], [0.5]])

        @jax.jit
        def interp(x):
            return cubic_interpolation_weights(x, grid)

        _indices, weights = interp(x)
        assert jnp.all(jnp.isfinite(weights))
