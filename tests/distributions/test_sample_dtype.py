"""gh-381: every Gaussian sampler follows the parameter dtype.

conftest.py enables x64, so a float32 model here is exactly the case that
used to return float64 draws (or crash, for a precision-form sampler).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import GaussianMRF, sample_mvn


# float64 does not exist in the no-x64 lane (GAUSSX_TEST_X64=0).
FLOAT64 = pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case"))


D = 2


def _sample(name: str, dtype):
    eye = jnp.eye(D, dtype=dtype)
    operator = lx.MatrixLinearOperator(2 * eye, lx.positive_semidefinite_tag)
    mu = jnp.zeros(D, dtype)
    if name == "sample_mvn":
        # x ~ 𝒩(μ, Σ), Σ = 2 I
        return sample_mvn(mu, operator, key=jr.key(0), num_samples=3)
    # x ~ 𝒩(μ, Λ⁻¹), Λ = 2 I
    return GaussianMRF(mu, operator).sample(jr.key(0), (3,))


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        FLOAT64,
    ],
)
@pytest.mark.parametrize("name", ["sample_mvn", "GaussianMRF"])
def test_sample_follows_parameter_dtype(name, dtype):
    draws = _sample(name, dtype)
    assert draws.dtype == dtype
    assert jnp.all(jnp.isfinite(draws))
