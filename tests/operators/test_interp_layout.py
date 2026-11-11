"""The interpolation weights at 1adbf07, frozen, and the layout they share (gh-813).

``data/interp_weights.json`` holds the output of `cubic_interpolation_weights`
and `linear_interpolation_weights` on fixed grids and targets, in float64.
These tests keep the deprecated functions returning exactly that until they are
removed, and pin the ``(indices, weights)`` layout `InterpolatedOperator` and
`interpolation_operator` take: row m holds the S flat row-major grid indices of
target m and their weights, repeated indices add, and a row of zero weights (how
functax's ``Plan.to_gaussx()`` marks a target off the domain) is a zero row of
W. gaussx does not import functax, so the fixture is the contract; functax
checks its ``Cubic`` and ``Linear`` plans against the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._einx import einsum, repeat
from gaussx._testing import default_tolerances


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:gaussx.cubic_interpolation_weights is deprecated"
    ),
    pytest.mark.filterwarnings(
        "ignore:gaussx.linear_interpolation_weights is deprecated"
    ),
]

_FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "interp_weights.json").read_text()
)
_CASES = {case["name"]: case for case in _FIXTURE["cases"]}
_FUNCTIONS = {
    "cubic_interpolation_weights": gaussx.cubic_interpolation_weights,
    "linear_interpolation_weights": gaussx.linear_interpolation_weights,
}


def _dtype():
    return jnp.result_type(float)


def _inputs(case):
    axes = [jnp.asarray(axis, _dtype()) for axis in case["axes"]]
    return axes, jnp.asarray(case["targets"], _dtype())


def _grid_size(case) -> int:
    size = 1
    for axis in case["axes"]:
        size *= len(axis)
    return size


def _dense_interpolation(indices, weights, grid_size):
    """W with [W]_m,j = ∑_s w_m,s [i_m,s = j], built by scatter-add."""
    rows = repeat(jnp.arange(indices.shape[0]), "row -> row tap", tap=indices.shape[1])
    W = jnp.zeros((indices.shape[0], grid_size), weights.dtype)
    return W.at[rows, indices].add(weights)


@pytest.mark.parametrize("name", list(_CASES))
def test_weights_match_the_frozen_output(name):
    case = _CASES[name]
    axes, targets = _inputs(case)
    function = _FUNCTIONS[case["function"]]
    indices, weights = function(targets, axes, **case["kwargs"])
    expected = jnp.asarray(case["weights"], weights.dtype)
    assert indices.tolist() == case["indices"]
    if weights.dtype == jnp.float64:
        assert jnp.array_equal(weights, expected)
    else:
        rtol, atol = default_tolerances(weights)
        assert jnp.allclose(weights, expected, rtol=10 * rtol, atol=10 * atol)


@pytest.mark.parametrize("name", list(_CASES))
def test_interpolated_operator_is_w_k_wt_for_the_frozen_arrays(name):
    case = _CASES[name]
    size = _grid_size(case)
    indices = jnp.asarray(case["indices"])
    weights = jnp.asarray(case["weights"], _dtype())
    factor = jr.normal(jr.key(0), (size, size), _dtype())
    K = lx.MatrixLinearOperator(factor @ factor.T, lx.positive_semidefinite_tag)
    W = _dense_interpolation(indices, weights, size)
    expected = einsum(W, K.as_matrix(), W, "m i, i j, n j -> m n")
    actual = gaussx.InterpolatedOperator(K, indices, weights).as_matrix()
    rtol, atol = default_tolerances(expected)
    assert jnp.allclose(actual, expected, rtol=10 * rtol, atol=100 * atol)
    gathered = gaussx.interpolation_operator(indices, weights, size).as_matrix()
    assert jnp.allclose(gathered, W, rtol=rtol, atol=atol)


def test_a_zeroed_row_is_a_zero_row_of_w_and_repeated_indices_add():
    """The two cases ``Plan.to_gaussx()`` emits beyond the gaussx builders."""
    dtype = _dtype()
    # target 0 reads grid point 2 twice; target 1 is invalid (index 0, weight 0)
    indices = jnp.array([[2, 2, 3], [0, 0, 0]])
    weights = jnp.array([[0.25, 0.25, 0.5], [0.0, 0.0, 0.0]], dtype)
    K = lx.MatrixLinearOperator(
        jnp.eye(4, dtype=dtype) + 1.0, lx.positive_semidefinite_tag
    )
    W = gaussx.interpolation_operator(indices, weights, 4).as_matrix()
    assert W.tolist() == [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]
    A = gaussx.InterpolatedOperator(K, indices, weights).as_matrix()
    expected = einsum(W, K.as_matrix(), W, "m i, i j, n j -> m n")
    rtol, atol = default_tolerances(expected)
    assert jnp.allclose(A, expected, rtol=rtol, atol=atol)
    assert A[1].tolist() == [0.0, 0.0]
