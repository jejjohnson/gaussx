"""Tests for grid construction and cubic interpolation weights."""

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from gaussx import (
    create_grid,
    cubic_interpolation_weights,
    grid_data,
    interpolation_operator,
    linear_interpolation_weights,
)
from gaussx._deprecation import GaussxDeprecationWarning
from gaussx._einx import einsum, reduce
from gaussx._testing import default_tolerances


# The two weight builders are deprecated in favour of functax stencils (gh-813);
# their behaviour is tested here until gaussx 0.7.0 removes them.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:gaussx.cubic_interpolation_weights is deprecated"
    ),
    pytest.mark.filterwarnings(
        "ignore:gaussx.linear_interpolation_weights is deprecated"
    ),
]


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

    def test_rejects_small_grid(self):
        """Raises ValueError when grid has fewer than 4 points."""
        grid = create_grid([3], [(-1.0, 1.0)])
        x = jnp.array([[0.0]])
        with pytest.raises(ValueError, match="at least 4"):
            cubic_interpolation_weights(x, grid)


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


def _scipy(grid, field, points):
    interpolator = RegularGridInterpolator(
        [np.asarray(axis, dtype=np.float64) for axis in grid],
        np.asarray(field, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interpolator(np.asarray(points, dtype=np.float64))


class TestLinearInterpolationWeights:
    @pytest.mark.parametrize("grid", _grids())
    def test_matches_scipy_on_non_uniform_grids(self, grid):
        field = _field(grid)
        points = _targets(grid)
        indices, weights = linear_interpolation_weights(points, grid)
        assert indices.shape == (20, 2 ** len(grid))
        W = interpolation_operator(indices, weights, field.size)
        actual = W.mv(jnp.ravel(field))
        rtol, atol = default_tolerances(actual)
        expected = jnp.asarray(_scipy(grid, field, points), actual.dtype)
        assert jnp.allclose(actual, expected, rtol=100 * rtol, atol=100 * atol)

    @pytest.mark.parametrize("grid", _grids())
    def test_weights_sum_to_one_and_are_non_negative(self, grid):
        _, weights = linear_interpolation_weights(_targets(grid), grid)
        rtol, atol = default_tolerances(weights)
        total = reduce(weights, "b k -> b", "sum")
        assert jnp.allclose(total, jnp.ones_like(total), rtol=rtol, atol=10 * atol)
        assert jnp.all(weights >= 0.0)

    @pytest.mark.parametrize("grid", _grids())
    def test_nodes_are_reproduced(self, grid):
        field = _field(grid)
        nodes = grid_data(grid)
        indices, weights = linear_interpolation_weights(nodes, grid)
        W = interpolation_operator(indices, weights, field.size)
        actual = W.mv(jnp.ravel(field))
        rtol, atol = default_tolerances(actual)
        assert jnp.allclose(actual, jnp.ravel(field), rtol=rtol, atol=10 * atol)

    def test_clamp_matches_a_bracket_helper_at_the_edges(self):
        """Clamped points take the edge value, as vardax's ``_bracket`` does."""
        axis = _axis_1()
        points = jnp.array([[-3.0], [0.0], [4.0], [9.0]], dtype=axis.dtype)
        indices, weights = linear_interpolation_weights(points, [axis])
        left = jnp.clip(jnp.searchsorted(axis, points[:, 0]) - 1, 0, axis.shape[0] - 2)
        span = axis[left + 1] - axis[left]
        fraction = jnp.clip((points[:, 0] - axis[left]) / span, 0.0, 1.0)
        assert jnp.array_equal(indices[:, 0], left)
        assert jnp.array_equal(weights[:, 1], fraction)
        field = jnp.arange(5.0, dtype=axis.dtype)
        W = interpolation_operator(indices, weights, 5)
        expected = jnp.array([0.0, 0.0, 4.0, 4.0], dtype=axis.dtype)
        assert jnp.allclose(W.mv(field), expected)

    @pytest.mark.parametrize("grid", _grids())
    def test_no_clamp_extrapolates_like_scipy(self, grid):
        field = _field(grid)
        points = _targets(grid) + 10.0
        indices, weights = linear_interpolation_weights(points, grid, clamp=False)
        W = interpolation_operator(indices, weights, field.size)
        actual = W.mv(jnp.ravel(field))
        rtol, atol = default_tolerances(actual)
        expected = jnp.asarray(_scipy(grid, field, points), actual.dtype)
        assert jnp.allclose(actual, expected, rtol=1000 * rtol, atol=1000 * atol)

    def test_a_short_axis_raises(self):
        with pytest.raises(ValueError, match="at least 2 grid points"):
            linear_interpolation_weights(jnp.zeros((1, 1)), [jnp.zeros(1)])


class TestInterpolationOperator:
    @pytest.mark.parametrize("grid", _grids())
    def test_transpose_matches_the_dense_transpose(self, grid):
        field = _field(grid)
        indices, weights = linear_interpolation_weights(_targets(grid), grid)
        W = interpolation_operator(indices, weights, field.size)
        dense = W.as_matrix()
        assert dense.shape == (20, field.size)
        vector = jr.normal(jr.key(2), (20,), weights.dtype)
        expected = einsum(dense, vector, "b g, b -> g")
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(W.T.mv(vector), expected, rtol=rtol, atol=10 * atol)

    def test_takes_cubic_weights_too(self):
        grid = create_grid([6], [(0.0, 1.0)])
        x = jnp.array([[0.31], [0.75]], dtype=grid[0].dtype)
        indices, weights = cubic_interpolation_weights(x, grid)
        W = interpolation_operator(indices, weights, 6)
        # the Keys cubic reproduces a linear field
        actual = W.mv(grid[0])
        rtol, atol = default_tolerances(actual)
        assert jnp.allclose(actual, x[:, 0], rtol=10 * rtol, atol=10 * atol)

    def test_jit_and_vmap_over_target_batches(self):
        grid = [_axis_1(), _axis_2()]
        field = jnp.ravel(_field(grid))

        def sample(points):
            indices, weights = linear_interpolation_weights(points, grid)
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
            indices, weights = linear_interpolation_weights(points, [axis])
            return interpolation_operator(indices, weights, 5).mv(values)[0]

        by_field = jax.grad(sample, argnums=0)(field, point)
        expected = jnp.array([0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0], dtype=axis.dtype)
        rtol, atol = default_tolerances(expected)
        assert jnp.allclose(by_field, expected, rtol=10 * rtol, atol=10 * atol)
        by_point = jax.grad(sample, argnums=1)(field, point)
        slope = jnp.array([[2.0]], dtype=axis.dtype)  # (4 − 1) / (2 − 0.5)
        assert jnp.allclose(by_point, slope, rtol=10 * rtol, atol=10 * atol)


class TestDeprecation:
    @pytest.mark.parametrize(
        ("function", "replacement"),
        [
            (cubic_interpolation_weights, "functax.interp.Cubic()"),
            (linear_interpolation_weights, "functax.interp.Linear()"),
        ],
    )
    def test_warns_once_per_call_site_and_names_functax(self, function, replacement):
        grid = create_grid([6], [(0.0, 1.0)])
        x = jnp.array([[0.31]], dtype=grid[0].dtype)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default", GaussxDeprecationWarning)
            for _ in range(3):
                function(x, grid)
        assert len(caught) == 1
        assert caught[0].category is GaussxDeprecationWarning
        assert caught[0].filename == __file__
        message = str(caught[0].message)
        assert replacement in message
        assert "to_gaussx()" in message
        assert "gaussx 0.7.0" in message
