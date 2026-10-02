"""Tests for ``hadamard_transform``, moved from kernellib's FastFood (G11).

kernellib K7 re-exports this function unchanged, so it must stay
bit-identical to kernellib's original, reproduced verbatim below.

Speed tiers mirror kernellib's FastFood tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.linalg

import gaussx as gx
from gaussx._einx import rearrange


def _kernellib_hadamard_transform(x):
    """kernellib 0.0.12 ``_operators/_fastfood.py::hadamard_transform``, verbatim."""
    d = x.shape[-1]
    h = 1
    while h < d:
        y = rearrange(x, "... (m two h) -> ... m two h", two=2, h=h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = rearrange(
            jnp.stack([a + b, a - b], axis=-2), "... m two h -> ... (m two h)"
        )
        h *= 2
    return x


@pytest.mark.slow
@pytest.mark.parametrize("d", [1, 2, 4, 8, 64])
def test_matches_dense_sylvester_matrix(d):
    x = jr.normal(jr.key(0), (3, d))
    H = jnp.asarray(scipy.linalg.hadamard(d), dtype=x.dtype)
    expected = gx._einx.einsum(H, x, "i j, b j -> b i")
    assert jnp.allclose(gx.hadamard_transform(x), expected, atol=1e-10)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bit_identical_to_kernellib(dtype):
    x = jr.normal(jr.key(3), (4, 2, 32), dtype=dtype)
    out = gx.hadamard_transform(x)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, _kernellib_hadamard_transform(x))


@pytest.mark.slow
def test_is_its_own_inverse_up_to_d():
    x = jr.normal(jr.key(1), (5, 16))
    assert jnp.allclose(gx.hadamard_transform(gx.hadamard_transform(x)), 16 * x)


@pytest.mark.slow
def test_jits_and_batches():
    x = jr.normal(jr.key(2), (2, 3, 8))
    expected = gx.hadamard_transform(x)
    assert jnp.allclose(jax.jit(gx.hadamard_transform)(x), expected)


@pytest.mark.parametrize("d", [3, 6, 12])
def test_rejects_non_power_of_two(d):
    with pytest.raises(ValueError, match="power-of-two"):
        gx.hadamard_transform(jnp.ones(d))
