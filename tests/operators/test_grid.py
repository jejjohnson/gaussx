"""Tests for grid construction and the interpolation operator."""

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import create_grid, grid_data, interpolation_operator
from gaussx._einx import einsum
from gaussx._testing import default_tolerances


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


def _axis_1():
    return jnp.array([0.0, 0.5, 2.0, 2.5, 4.0], dtype=jnp.result_type(float))


def _axis_2():
    return jnp.array([-1.0, 0.0, 3.0, 3.5], dtype=jnp.result_type(float))


def _axis_3():
    return jnp.array([0.0, 1.0, 1.25], dtype=jnp.result_type(float))


def _grids():
    return [
        pytest.param([_axis_1()], id="1d"),
        pytest.param([_axis_1(), _axis_2()], id="2d"),
        pytest.param([_axis_1(), _axis_2(), _axis_3()], id="3d"),
    ]


def _field(grid):
    """A random field on the grid, one value per node."""
    shape = tuple(axis.shape[0] for axis in grid)
    return jr.normal(jr.key(0), shape, grid[0].dtype)


def _targets(grid, key=1, n=20):
    lows = jnp.array([axis[0] for axis in grid])
    highs = jnp.array([axis[-1] for axis in grid])
    unit = jr.uniform(jr.key(key), (n, len(grid)), grid[0].dtype)
    return lows + unit * (highs - lows)


def _linear_stencil(points, grid):
    """Clamped multilinear ``(indices, weights)`` rows on a non-uniform grid.

    The stencil layout `interpolation_operator` takes, as functax's
    ``Linear().plan(...).to_gaussx()`` returns it with ``outside="clamp"``
    (gh-813); gaussx does not import functax, so the tests build it here.
    """
    indices = jnp.zeros((points.shape[0], 1), dtype=jnp.int32)
    weights = jnp.ones((points.shape[0], 1), dtype=points.dtype)
    stride = 1
    for d in reversed(range(len(grid))):
        axis = grid[d]
        left = jnp.clip(jnp.searchsorted(axis, points[:, d]) - 1, 0, axis.shape[0] - 2)
        offset = (points[:, d] - axis[left]) / (axis[left + 1] - axis[left])
        offset = jnp.clip(offset, 0.0, 1.0)
        corners = jnp.column_stack([left, left + 1]) * stride
        indices = einx.add("b p, b k -> b (p k)", corners, indices)
        weights = einx.multiply(
            "b p, b k -> b (p k)", jnp.column_stack([1.0 - offset, offset]), weights
        )
        stride *= axis.shape[0]
    return indices, weights


class TestInterpolationOperator:
    @pytest.mark.parametrize("grid", _grids())
    def test_transpose_matches_the_dense_transpose(self, grid):
        field = _field(grid)
        indices, weights = _linear_stencil(_targets(grid), grid)
        W = interpolation_operator(indices, weights, field.size)
        dense = W.as_matrix()
        assert dense.shape == (20, field.size)
        vector = jr.normal(jr.key(2), (20,), weights.dtype)
        expected = einsum(dense, vector, "b g, b -> g")
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(W.T.mv(vector), expected, rtol=rtol, atol=10 * atol)

    def test_jit_and_vmap_over_target_batches(self):
        grid = [_axis_1(), _axis_2()]
        field = jnp.ravel(_field(grid))

        def sample(points):
            indices, weights = _linear_stencil(points, grid)
            return interpolation_operator(indices, weights, field.size).mv(field)

        batches = jnp.stack([_targets(grid, key=3), _targets(grid, key=4)])
        batched = jax.jit(jax.vmap(sample))(batches)
        loop = jnp.stack([sample(batch) for batch in batches])
        rtol, atol = default_tolerances(loop)
        assert jnp.allclose(batched, loop, rtol=rtol, atol=atol)

    def test_gradient_in_the_field_and_the_coordinates(self):
        """On one cell of a 1-D axis, ∂/∂f is the weights and ∂/∂p the slope."""
        axis = _axis_1()
        field = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0], dtype=axis.dtype)
        point = jnp.array([[1.0]], dtype=axis.dtype)  # in the cell [0.5, 2]

        def sample(values, points):
            indices, weights = _linear_stencil(points, [axis])
            return interpolation_operator(indices, weights, 5).mv(values)[0]

        by_field = jax.grad(sample, argnums=0)(field, point)
        expected = jnp.array([0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0], dtype=axis.dtype)
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(by_field, expected, rtol=10 * rtol, atol=10 * atol)
        by_point = jax.grad(sample, argnums=1)(field, point)
        slope = jnp.array([[2.0]], dtype=axis.dtype)  # (4 − 1) / (2 − 0.5)
        assert jnp.allclose(by_point, slope, rtol=10 * rtol, atol=10 * atol)
