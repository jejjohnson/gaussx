"""Tests for mixed-precision stable squared distances."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gaussx._linalg._mixed_precision import stable_squared_distances
from gaussx._testing import tree_allclose


# float64 does not exist in the no-x64 lane (GAUSSX_TEST_X64=0).
FLOAT64 = pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case"))


# ---------------------------------------------------------------------------
# stable_squared_distances
# ---------------------------------------------------------------------------


class TestStableSquaredDistances:
    @pytest.mark.slow
    def test_matches_naive_float64(self, getkey):
        """Should match direct ||x - z||^2 in float64."""
        X = jr.normal(getkey(), (10, 5)).astype(jnp.float64)
        Z = jr.normal(getkey(), (8, 5)).astype(jnp.float64)
        expected = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        result = stable_squared_distances(
            X,
            Z,
            compute_dtype=jnp.float64,
            accumulate_dtype=jnp.float64,
        )
        assert tree_allclose(result, expected, rtol=1e-10)

    def test_non_negative_high_dim(self, getkey):
        """All distances should be >= 0 even in float32 with D=1000."""
        D = 1000
        X = jr.normal(getkey(), (50, D)).astype(jnp.float32)
        Z = jr.normal(getkey(), (50, D)).astype(jnp.float32)
        dist_sq = stable_squared_distances(X, Z)
        assert jnp.all(dist_sq >= 0.0)

    def test_self_distance_zero(self, getkey):
        """Distance of a point to itself should be ~0."""
        X = jr.normal(getkey(), (5, 10)).astype(jnp.float32)
        dist_sq = stable_squared_distances(X, X)
        diag = jnp.diag(dist_sq)
        assert jnp.allclose(diag, 0.0, atol=1e-5)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (7, 3))
        Z = jr.normal(getkey(), (4, 3))
        assert stable_squared_distances(X, Z).shape == (7, 4)

    @pytest.mark.slow
    def test_symmetric(self, getkey):
        """D(X, Z) should equal D(Z, X)^T."""
        X = jr.normal(getkey(), (6, 4))
        Z = jr.normal(getkey(), (8, 4))
        d1 = stable_squared_distances(X, Z)
        d2 = stable_squared_distances(Z, X)
        assert tree_allclose(d1, d2.T, rtol=1e-5)


def _near_duplicates():
    """Points far from the origin with near-duplicate partners (gh-414).

    Norms are ~6.4e7 and the paired distances ~6.4e-3, so the raw
    expansion cancels ~10 orders of magnitude.
    """
    rng = np.random.default_rng(0)
    X = (1000.0 + rng.standard_normal((4, 64))).astype(np.float32)
    Z = X + 1e-2 * rng.standard_normal((4, 64)).astype(np.float32)
    exact = ((X.astype(np.float64)[:, None] - Z.astype(np.float64)[None]) ** 2).sum(-1)
    return X, Z, exact


@pytest.mark.parametrize(
    "accumulate_dtype",
    [
        None,
        jnp.float32,
        FLOAT64,
    ],
)
def test_near_duplicates_far_from_origin(accumulate_dtype):
    """gh-414: centring removes the cancellation, in any accumulate dtype.

    Before, the error on the near-duplicate pairs was 1.2e3 relative. With
    float32 accumulation the centred expansion leaves ~1e-3 (a 1e4 ratio
    between the centred norms and the distances, times float32 eps).
    """
    X, Z, exact = _near_duplicates()
    d = stable_squared_distances(
        jnp.asarray(X), jnp.asarray(Z), accumulate_dtype=accumulate_dtype
    )
    rel = np.abs(np.diag(np.asarray(d, np.float64)) - np.diag(exact)) / np.diag(exact)
    assert d.dtype == jnp.float32
    assert rel.max() <= 1e-2


def test_default_arguments_are_stable_without_x64():
    """gh-414: with x64 off the default call is stable and does not warn, and
    an explicit float64 accumulate dtype raises instead of truncating.

    Runs in a fresh interpreter because conftest.py enables x64 globally.
    """
    script = textwrap.dedent(
        """
        import warnings
        import jax
        import jax.numpy as jnp
        import numpy as np
        import gaussx

        assert not jax.config.jax_enable_x64
        rng = np.random.default_rng(0)
        X = (1000.0 + rng.standard_normal((4, 64))).astype(np.float32)
        Z = X + 1e-2 * rng.standard_normal((4, 64)).astype(np.float32)
        X64, Z64 = X.astype(np.float64), Z.astype(np.float64)
        exact = ((X64[:, None] - Z64[None]) ** 2).sum(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = gaussx.stable_squared_distances(jnp.asarray(X), jnp.asarray(Z))
        diag = np.diag(np.asarray(d, np.float64))
        rel = np.abs(diag - np.diag(exact)) / np.diag(exact)
        assert rel.max() <= 1e-2, rel.max()
        try:
            gaussx.stable_squared_distances(
                jnp.asarray(X), jnp.asarray(Z), accumulate_dtype=jnp.float64
            )
        except ValueError as e:
            assert "not available" in str(e)
        else:
            raise AssertionError("float64 accumulate_dtype without x64 must raise")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
