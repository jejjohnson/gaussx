"""Tests for structure-aware multivariate-normal sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._primitives._cholesky as cholesky_module
from gaussx._operators import low_rank_plus_diag
from gaussx._testing import assert_sample_moments, random_pd_operator


# 4096 draws per test; each bound comes from assert_sample_moments, which
# scales with the estimator's own standard error rather than a fixed atol.
_NUM_SAMPLES = 4096


def _pd(seed: int, n: int) -> lx.MatrixLinearOperator:
    return random_pd_operator(jr.key(seed), n)


def _structured_covariances() -> dict[str, lx.AbstractLinearOperator]:
    """Small instances of every structure `sample_mvn` dispatches on."""
    lags = jnp.arange(6.0)
    kronecker = gaussx.Kronecker(_pd(1, 2), _pd(2, 3))
    return {
        "dense": _pd(0, 5),
        "diagonal": lx.DiagonalLinearOperator(jnp.array([0.5, 1.0, 2.0, 4.0])),
        "kronecker": kronecker,
        "block_diag": gaussx.BlockDiag(_pd(3, 2), _pd(4, 3)),
        # Diagonally dominant, so positive definite: 3 blocks of 2x2.
        "block_tridiag": gaussx.BlockTriDiag(
            jnp.stack([4.0 * jnp.eye(2)] * 3),
            jnp.stack([jnp.array([[0.5, 0.2], [0.1, 0.4]])] * 2),
        ),
        "toeplitz": gaussx.Toeplitz(jnp.exp(-0.5 * lags**2)),
        "kronecker_sum": gaussx.KroneckerSum(_pd(5, 2), _pd(6, 3)),
        "sum_of_kroneckers": gaussx.SumOfKroneckers(
            gaussx.Kronecker(_pd(7, 2), _pd(8, 3)),
            gaussx.Kronecker(_pd(9, 2), _pd(10, 3)),
        ),
        "low_rank": low_rank_plus_diag(
            jnp.linspace(0.5, 1.5, 5), jr.normal(jr.key(11), (5, 2))
        ),
        "tagged": lx.TaggedLinearOperator(kronecker, lx.positive_semidefinite_tag),
        "scaled": 2.5 * kronecker,
    }


_COVARIANCES = _structured_covariances()


@pytest.mark.parametrize("name", list(_COVARIANCES))
def test_samples_have_the_requested_moments(name: str) -> None:
    covariance = _COVARIANCES[name]
    n = covariance.in_size()
    mean = jnp.linspace(-1.0, 1.0, n)

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(12), num_samples=_NUM_SAMPLES
    )

    assert samples.shape == (_NUM_SAMPLES, n)
    assert_sample_moments(samples, mean, covariance.as_matrix())


@pytest.mark.parametrize(
    "name",
    [
        "diagonal",
        "kronecker",
        "block_diag",
        "block_tridiag",
        "toeplitz",
        "kronecker_sum",
        "sum_of_kroneckers",
        "low_rank",
        "tagged",
        "scaled",
    ],
)
def test_structured_covariances_never_take_the_dense_cholesky(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Small dense factors (a Kronecker factor, one block) are expected to be
    # factorised densely; the full covariance never is.
    covariance = _COVARIANCES[name]
    dense_cholesky = cholesky_module._cholesky_dense

    def refuse_full_size(operator):
        if operator.in_size() >= covariance.in_size():
            raise AssertionError(f"dense Cholesky of the full {name} covariance")
        return dense_cholesky(operator)

    monkeypatch.setattr(cholesky_module, "_cholesky_dense", refuse_full_size)

    gaussx.sample_mvn(jnp.zeros(covariance.in_size()), covariance, key=jr.key(13))


def test_batched_mean_gets_independent_noise() -> None:
    covariance = lx.DiagonalLinearOperator(jnp.array([1.0, 4.0]))
    mean = jnp.array([[0.0, 0.0], [10.0, -10.0], [5.0, 5.0]])

    samples = gaussx.sample_mvn(mean, covariance, key=jr.key(14), num_samples=4)

    assert samples.shape == (4, 3, 2)
    # Same covariance, different noise: no two batch elements share a draw.
    centred = samples - mean[None]
    assert not jnp.allclose(centred[:, 0], centred[:, 1])
    assert not jnp.allclose(centred[:, 1], centred[:, 2])


def test_batched_draws_have_the_requested_moments() -> None:
    covariance = gaussx.Kronecker(_pd(15, 2), _pd(16, 2))
    mean = jnp.stack([jnp.zeros(4), jnp.ones(4)])

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(17), num_samples=_NUM_SAMPLES
    )

    for index in range(2):
        assert_sample_moments(samples[:, index], mean[index], covariance.as_matrix())


def test_negative_low_rank_weights_fall_back_to_dense() -> None:
    # A Woodbury downdate D - uuᵀ is still positive definite here, but it has
    # no U√D factor; the draw must come from the dense Cholesky instead.
    u = jnp.array([[0.5], [0.3], [0.2]])
    covariance = gaussx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.array([2.0, 2.0, 2.0])), u, jnp.array([-1.0])
    )
    mean = jnp.zeros(3)

    samples = gaussx.sample_mvn(
        mean, covariance, key=jr.key(18), num_samples=_NUM_SAMPLES
    )

    assert_sample_moments(samples, mean, covariance.as_matrix())


def test_traced_negative_low_rank_weights_raise() -> None:
    u = jnp.array([[0.5], [0.3], [0.2]])
    base = lx.DiagonalLinearOperator(jnp.array([2.0, 2.0, 2.0]))

    @jax.jit
    def draw(weights):
        covariance = gaussx.LowRankUpdate(base, u, weights)
        return gaussx.sample_mvn(jnp.zeros(3), covariance, key=jr.key(19))

    assert jnp.all(jnp.isfinite(draw(jnp.array([1.0]))))
    with pytest.raises(Exception, match="non-negative"):
        draw(jnp.array([-1.0]))


def test_jit_matches_eager() -> None:
    covariance = _COVARIANCES["kronecker"]
    mean = jnp.ones(covariance.in_size())

    eager = gaussx.sample_mvn(mean, covariance, key=jr.key(20), num_samples=3)
    jitted = jax.jit(
        lambda m, key: gaussx.sample_mvn(m, covariance, key=key, num_samples=3)
    )(mean, jr.key(20))

    assert jnp.allclose(eager, jitted)


def test_vmap_over_keys() -> None:
    covariance = _COVARIANCES["diagonal"]
    keys = jr.split(jr.key(21), 5)

    samples = jax.vmap(
        lambda key: gaussx.sample_mvn(jnp.zeros(4), covariance, key=key)
    )(keys)

    assert samples.shape == (5, 1, 4)
    assert not jnp.allclose(samples[0], samples[1])


def test_rejects_rectangular_covariance() -> None:
    with pytest.raises(ValueError, match="square"):
        gaussx.sample_mvn(
            jnp.zeros(3), lx.MatrixLinearOperator(jnp.ones((3, 2))), key=jr.key(0)
        )


def test_rejects_mismatched_mean() -> None:
    with pytest.raises(ValueError, match="mean must have shape"):
        gaussx.sample_mvn(jnp.zeros(4), _COVARIANCES["dense"], key=jr.key(0))


def test_rejects_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="num_samples"):
        gaussx.sample_mvn(
            jnp.zeros(5), _COVARIANCES["dense"], key=jr.key(0), num_samples=0
        )
