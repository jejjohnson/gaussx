"""Tests for partitioned joint / conditional Gaussian sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._primitives._cholesky as cholesky_module
from gaussx._testing import assert_sample_moments, random_pd_operator


# Moment tests draw 8192 samples; assert_sample_moments bounds each estimator
# by its own standard error (7 sigma), so the tolerance is not a fixed atol.
_NUM_SAMPLES = 8192
_N, _M = 4, 3


def _partition(seed: int = 0):
    """A dense (N + M) joint covariance, its blocks and a mean."""
    joint = random_pd_operator(jr.key(seed), _N + _M).as_matrix()
    blocks = {
        "aa": lx.MatrixLinearOperator(joint[:_N, :_N], lx.positive_semidefinite_tag),
        "ab": lx.MatrixLinearOperator(joint[:_N, _N:]),
        "bb": lx.MatrixLinearOperator(joint[_N:, _N:], lx.positive_semidefinite_tag),
    }
    mean = jnp.linspace(-1.0, 2.0, _N + _M)
    return joint, blocks, (mean[:_N], mean[_N:])


def _dense_conditional(joint, mean, observed_index, value):
    """Reference mean and covariance of the unobserved block."""
    mean_a, mean_b = mean
    k_aa, k_ab, k_bb = joint[:_N, :_N], joint[:_N, _N:], joint[_N:, _N:]
    if observed_index == 1:
        gain = k_ab @ jnp.linalg.inv(k_bb)
        return mean_a + gain @ (value - mean_b), k_aa - gain @ k_ab.T
    gain = k_ab.T @ jnp.linalg.inv(k_aa)
    return mean_b + gain @ (value - mean_a), k_bb - gain @ k_ab


@pytest.mark.parametrize("observed_index", [0, 1])
def test_joint_draws_have_the_joint_moments(observed_index: int) -> None:
    joint, blocks, mean = _partition()

    draws = gaussx.sample_joint_conditional(
        mean,
        blocks,
        key=jr.key(1),
        observed_index=observed_index,
        num_samples=_NUM_SAMPLES,
    )

    samples_a, samples_b = draws["joint"]
    assert samples_a.shape == (_NUM_SAMPLES, _N)
    assert samples_b.shape == (_NUM_SAMPLES, _M)
    stacked = jnp.concatenate([samples_a, samples_b], axis=1)
    assert_sample_moments(stacked, jnp.concatenate(mean), joint)


@pytest.mark.parametrize("observed_index", [0, 1])
def test_conditional_draws_have_the_schur_moments(observed_index: int) -> None:
    joint, blocks, mean = _partition()
    size = _M if observed_index == 1 else _N
    value = jnp.linspace(0.5, -0.5, size)

    draws = gaussx.sample_joint_conditional(
        mean,
        blocks,
        key=jr.key(2),
        observed_index=observed_index,
        observed_value=value,
        num_samples=_NUM_SAMPLES,
    )

    expected_mean, expected_cov = _dense_conditional(joint, mean, observed_index, value)
    assert_sample_moments(draws["conditional"], expected_mean, expected_cov)


def test_conditional_draws_are_matheron_updates_of_the_joint_draws() -> None:
    _, blocks, mean = _partition()
    value = jnp.array([0.3, -0.2, 0.1])

    draws = gaussx.sample_joint_conditional(
        mean, blocks, key=jr.key(3), observed_value=value, num_samples=16
    )
    samples_a, samples_b = draws["joint"]

    updated = gaussx.matheron_update(
        samples_a, samples_b, value, blocks["ab"], blocks["bb"]
    )
    assert jnp.allclose(draws["conditional"], updated, atol=1e-10)


def test_conditional_draws_are_matheron_updates_with_an_iterative_solver() -> None:
    # A truncated CG solve is not additive across right-hand sides, so the
    # conditional draws must come from the joint draws themselves. A 40-dim
    # observed block and a loose tolerance keep the solve far from exact.
    size_t, size_o = 3, 40
    joint = random_pd_operator(jr.key(14), size_t + size_o).as_matrix()
    blocks = {
        "aa": lx.MatrixLinearOperator(joint[:size_t, :size_t]),
        "ab": lx.MatrixLinearOperator(joint[:size_t, size_t:]),
        "bb": lx.MatrixLinearOperator(
            joint[size_t:, size_t:], lx.positive_semidefinite_tag
        ),
    }
    value = jnp.linspace(-1.0, 1.0, size_o)
    solver = gaussx.CGSolver(rtol=1e-2, atol=1e-2)

    draws = gaussx.sample_joint_conditional(
        (jnp.zeros(size_t), jnp.zeros(size_o)),
        blocks,
        key=jr.key(15),
        observed_value=value,
        num_samples=8,
        solver=solver,
    )
    samples_a, samples_b = draws["joint"]

    updated = gaussx.matheron_update(
        samples_a, samples_b, value, blocks["ab"], blocks["bb"], solver=solver
    )
    assert jnp.allclose(draws["conditional"], updated, atol=1e-12)


def test_a_singular_observed_block_does_not_poison_the_draws() -> None:
    # K_oo = diag(0, 1, 2) is only semi-definite; a structured diagonal solve
    # divides 0 / 0. The target block is correlated with the non-degenerate
    # observations only, so the joint covariance is valid.
    k_oo = jnp.diag(jnp.array([0.0, 1.0, 2.0]))
    cross = jnp.array([[0.0, 0.3, 0.1], [0.0, -0.2, 0.4]])
    k_tt = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    joint = jnp.block([[k_tt, cross], [cross.T, k_oo]])
    blocks = {
        "aa": lx.MatrixLinearOperator(k_tt),
        "ab": lx.MatrixLinearOperator(cross),
        "bb": lx.DiagonalLinearOperator(jnp.diag(k_oo)),
    }

    draws = gaussx.sample_joint_conditional(
        (jnp.zeros(2), jnp.zeros(3)),
        blocks,
        key=jr.key(15),
        observed_value=jnp.array([0.0, 0.5, -0.5]),
        num_samples=_NUM_SAMPLES,
    )

    samples_a, samples_b = draws["joint"]
    stacked = jnp.concatenate([samples_a, samples_b], axis=1)
    assert jnp.all(jnp.isfinite(stacked))
    assert jnp.all(jnp.isfinite(draws["conditional"]))
    assert_sample_moments(stacked, jnp.zeros(5), joint)


def test_structured_observed_block_is_not_densified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GP-prediction shape: diagonal test covariance, Kronecker training one.
    k_bb = gaussx.Kronecker(
        random_pd_operator(jr.key(4), 2), random_pd_operator(jr.key(5), 3)
    )
    cross = 0.1 * jr.normal(jr.key(6), (_N, 6))
    blocks = {
        "aa": lx.DiagonalLinearOperator(jnp.full((_N,), 2.0)),
        "ab": lx.MatrixLinearOperator(cross),
        "bb": k_bb,
    }
    dense_cholesky = cholesky_module._cholesky_dense

    def refuse_full_size(operator):
        if operator.in_size() >= k_bb.in_size():
            raise AssertionError("dense Cholesky of the full observed covariance")
        return dense_cholesky(operator)

    monkeypatch.setattr(cholesky_module, "_cholesky_dense", refuse_full_size)

    draws = gaussx.sample_joint_conditional(
        (jnp.zeros(_N), jnp.zeros(6)), blocks, key=jr.key(7), num_samples=4
    )

    assert draws["joint"][1].shape == (4, 6)


def test_targets_that_coincide_with_observations_are_pinned() -> None:
    # Target 0 duplicates observation 0, so the Schur complement is singular;
    # a Cholesky would return NaNs, the clipped eigh root must not.
    points = jnp.array([0.0, 0.7, 1.9, 0.0, 1.3])
    kernel = jnp.exp(-0.5 * (points[:, None] - points[None, :]) ** 2)
    blocks = {
        "aa": lx.MatrixLinearOperator(kernel[:3, :3]),
        "ab": lx.MatrixLinearOperator(kernel[:3, 3:]),
        "bb": lx.MatrixLinearOperator(
            kernel[3:, 3:] + 1e-10 * jnp.eye(2), lx.positive_semidefinite_tag
        ),
    }
    value = jnp.array([1.5, -0.5])

    draws = gaussx.sample_joint_conditional(
        (jnp.zeros(3), jnp.zeros(2)),
        blocks,
        key=jr.key(8),
        observed_value=value,
        num_samples=64,
    )

    assert jnp.all(jnp.isfinite(draws["conditional"]))
    assert jnp.allclose(draws["conditional"][:, 0], value[0], atol=1e-4)


def test_gradients_stay_finite_for_an_isotropic_schur_complement() -> None:
    # Uncorrelated blocks leave S = sigma^2 I: every eigenvalue repeated, where
    # differentiating through eigh's eigenvectors returns NaN.
    k_bb = random_pd_operator(jr.key(12), _M)

    def loss(sigma):
        blocks = {
            "aa": lx.MatrixLinearOperator(sigma**2 * jnp.eye(_N)),
            "ab": lx.MatrixLinearOperator(jnp.zeros((_N, _M))),
            "bb": k_bb,
        }
        draws = gaussx.sample_joint_conditional(
            (jnp.zeros(_N), jnp.zeros(_M)),
            blocks,
            key=jr.key(13),
            observed_value=jnp.zeros(_M),
            num_samples=8,
        )
        return jnp.sum(draws["conditional"])

    sigma = 1.5
    gradient = jax.grad(loss)(sigma)

    # Draws are sigma * z, so the gradient is sum(z) = loss / sigma.
    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, loss(sigma) / sigma)


def test_conditional_key_is_absent_without_an_observed_value() -> None:
    _, blocks, mean = _partition()

    draws = gaussx.sample_joint_conditional(mean, blocks, key=jr.key(9))

    assert set(draws) == {"joint"}


def test_solver_strategy_matches_structural_dispatch() -> None:
    _, blocks, mean = _partition()
    value = jnp.array([0.3, -0.2, 0.1])
    kwargs = dict(key=jr.key(10), observed_value=value, num_samples=8)

    direct = gaussx.sample_joint_conditional(mean, blocks, **kwargs)
    iterative = gaussx.sample_joint_conditional(
        mean, blocks, solver=gaussx.CGSolver(rtol=1e-12, atol=1e-12), **kwargs
    )

    assert jnp.allclose(direct["conditional"], iterative["conditional"], atol=1e-8)


def test_jit_matches_eager() -> None:
    _, blocks, mean = _partition()
    value = jnp.array([0.3, -0.2, 0.1])

    eager = gaussx.sample_joint_conditional(
        mean, blocks, key=jr.key(11), observed_value=value, num_samples=3
    )
    jitted = jax.jit(
        lambda key, value: gaussx.sample_joint_conditional(
            mean, blocks, key=key, observed_value=value, num_samples=3
        )
    )(jr.key(11), value)

    assert jnp.allclose(eager["conditional"], jitted["conditional"])
    assert jnp.allclose(eager["joint"][0], jitted["joint"][0])


def test_rejects_a_missing_block() -> None:
    _, blocks, mean = _partition()
    del blocks["ab"]
    with pytest.raises(ValueError, match="missing"):
        gaussx.sample_joint_conditional(mean, blocks, key=jr.key(0))


def test_rejects_a_mismatched_block() -> None:
    _, blocks, mean = _partition()
    blocks["ab"] = lx.MatrixLinearOperator(jnp.ones((_N, _M + 1)))
    with pytest.raises(ValueError, match=r'joint_covariance\["ab"\]'):
        gaussx.sample_joint_conditional(mean, blocks, key=jr.key(0))


def test_rejects_an_invalid_observed_index() -> None:
    _, blocks, mean = _partition()
    with pytest.raises(ValueError, match="observed_index"):
        gaussx.sample_joint_conditional(mean, blocks, key=jr.key(0), observed_index=2)


def test_rejects_a_mismatched_observed_value() -> None:
    _, blocks, mean = _partition()
    with pytest.raises(ValueError, match="observed_value"):
        gaussx.sample_joint_conditional(
            mean, blocks, key=jr.key(0), observed_value=jnp.zeros(_N)
        )


def test_rejects_non_positive_sample_count() -> None:
    _, blocks, mean = _partition()
    with pytest.raises(ValueError, match="num_samples"):
        gaussx.sample_joint_conditional(mean, blocks, key=jr.key(0), num_samples=0)
