"""gh-381: every distribution's ``.sample`` follows the parameter dtype.

conftest.py enables x64, so a float32 model here is exactly the case that
used to return float64 draws (or crash, for MultivariateNormalPrecision).
"""

from __future__ import annotations

import pytest


pytest.importorskip("numpyro")

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import (
    LGSSM,
    MarkovGaussian,
    MaskedLGSSM,
    MultivariateNormal,
    MultivariateNormalPrecision,
    sample_mvn,
)


# float64 does not exist in the no-x64 lane (GAUSSX_TEST_X64=0).
FLOAT64 = pytest.param(jnp.float64, marks=pytest.mark.x64_only(reason="float64 case"))


D, T = 2, 4


def _build(name: str, dtype):
    eye = jnp.eye(D, dtype=dtype)
    cov = lx.MatrixLinearOperator(2 * eye, lx.positive_semidefinite_tag)
    mu = jnp.zeros(D, dtype)
    if name == "MultivariateNormal":
        return MultivariateNormal(mu, cov)
    if name == "MultivariateNormalPrecision":
        return MultivariateNormalPrecision(mu, cov)
    if name == "MarkovGaussian":
        return MarkovGaussian(
            jnp.stack([0.9 * eye] * (T - 1)), jnp.stack([0.1 * eye] * (T - 1)), mu, eye
        )
    if name == "LGSSM":
        return LGSSM(0.9 * eye, eye, 0.1 * eye, 0.1 * eye, mu, eye, n_steps=T)
    return MaskedLGSSM(
        0.9 * eye,
        eye,
        0.1 * eye,
        0.1 * eye,
        mu,
        eye,
        n_steps=T,
        obs_mask=jnp.ones((T, D), dtype=bool),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        FLOAT64,
    ],
)
@pytest.mark.parametrize(
    "name",
    [
        "MultivariateNormal",
        "MultivariateNormalPrecision",
        "MarkovGaussian",
        "LGSSM",
        "MaskedLGSSM",
    ],
)
def test_sample_follows_parameter_dtype(name, dtype):
    draws = _build(name, dtype).sample(jr.key(0), (3,))
    assert draws.dtype == dtype
    assert jnp.all(jnp.isfinite(draws))


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        FLOAT64,
    ],
)
def test_class_and_functional_samplers_agree_on_dtype(dtype):
    """The class API matches sample_mvn, which already followed the dtype."""
    d = _build("MultivariateNormal", dtype)
    functional = sample_mvn(d.loc, d.cov_operator, key=jr.key(0), num_samples=3)
    assert d.sample(jr.key(0), (3,)).dtype == functional.dtype == dtype
