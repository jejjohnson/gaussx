"""Tests for `rts_sample`, joint draws of the state path by backward sampling.

The contracts:

* on a small model, the draws of the whole path x_1:T have the mean and the
  joint covariance of the dense Gaussian conditional p(x_1:T | y_1:T), and
  each step's draws have `rts_smoother`'s moments, within bounds from the
  estimators' own sampling distributions (`assert_sample_moments`);
* a masked step's observation never reaches the draws;
* time-invariant, time-varying and operator transitions give the same draws;
* the same key gives the same draws and different keys different ones;
* the draws run under ``jit`` and ``vmap``, and are differentiable in the
  model's parameters for a fixed key (the reparameterised path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import kalman_filter, rts_sample, rts_smoother
from gaussx._einx import rearrange
from gaussx._ssm._kalman import _empty_filter_state
from gaussx._testing import assert_sample_moments


T, N = 5, 2
A = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
H = jnp.array([[1.0, 0.5]])
Q = jnp.array([[0.2, 0.05], [0.05, 0.1]])
R = jnp.array([[0.3]])
M0 = jnp.array([0.3, -0.2])
P0 = jnp.array([[1.0, 0.2], [0.2, 0.5]])
Y = jnp.array([[0.4], [-0.1], [0.7], [0.2], [-0.3]])


def _model():
    """The arrays in the active default float, so both lanes build them."""
    dtype = jnp.zeros(()).dtype
    return tuple(jnp.asarray(x, dtype=dtype) for x in (A, H, Q, R, Y, M0, P0))


def _dense_conditional(transition, obs, process, noise, y, mean0, cov0):
    """Mean ``(T N,)`` and covariance ``(T N, T N)`` of x_1:T given y_1:T.

    Unrolls x_t = F x_t−1 + η_t from x_0 (the filter predicts before each
    update, so x_1 = F x_0 + η_1), builds the joint Gaussian of the stacked
    states and observations, and conditions on the observations.
    """
    steps = y.shape[0]
    blocks = [[None] * steps for _ in range(steps)]
    means, covs = [], []
    mean, cov = mean0, cov0
    for _ in range(steps):
        mean = transition @ mean
        cov = transition @ cov @ transition.T + process
        means.append(mean)
        covs.append(cov)
    for i in range(steps):
        for j in range(steps):
            lag = jnp.linalg.matrix_power(transition, abs(i - j))
            blocks[i][j] = lag @ covs[j] if i >= j else (lag @ covs[i]).T
    state_mean = jnp.concatenate(means)
    state_cov = jnp.block(blocks)
    obs_matrix = jax.scipy.linalg.block_diag(*([obs] * steps))
    obs_cov = obs_matrix @ state_cov @ obs_matrix.T + jax.scipy.linalg.block_diag(
        *([noise] * steps)
    )
    gain = jnp.linalg.solve(obs_cov, obs_matrix @ state_cov).T
    residual = rearrange(y, "t m -> (t m)") - obs_matrix @ state_mean
    return state_mean + gain @ residual, state_cov - gain @ obs_matrix @ state_cov


def test_draws_have_the_dense_conditional_joint_moments():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0)
    draws = rts_sample(state, transition, jr.key(0), num_samples=20_000)
    assert draws.shape == (20_000, T, N)
    assert draws.dtype == y.dtype
    mean, cov = _dense_conditional(transition, obs, process, noise, y, mean0, cov0)
    # Every entry of the sample mean and of the (T N) × (T N) sample covariance
    # lies within 7 of its own estimator's standard deviations (gh-220).
    assert_sample_moments(rearrange(draws, "s t n -> s (t n)"), mean, cov)


def test_each_step_has_the_smoothers_moments():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    mask = jnp.array([True, False, True, True, False])
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0, mask=mask)
    draws = rts_sample(state, transition, jr.key(1), num_samples=20_000)
    means, covs = rts_smoother(state, transition)
    for t in range(T):
        assert_sample_moments(draws[:, t], means[t], covs[t])


def test_a_masked_observation_never_reaches_the_draws():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    mask = jnp.array([True, True, False, True, True])
    other = y.at[2, 0].set(100.0)
    draws = rts_sample(
        kalman_filter(transition, obs, process, noise, y, mean0, cov0, mask=mask),
        transition,
        jr.key(2),
        num_samples=4,
    )
    draws_other = rts_sample(
        kalman_filter(transition, obs, process, noise, other, mean0, cov0, mask=mask),
        transition,
        jr.key(2),
        num_samples=4,
    )
    assert jnp.array_equal(draws, draws_other)


def test_time_varying_and_operator_transitions_give_the_same_draws():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0)
    dense = rts_sample(state, transition, jr.key(3), num_samples=6)
    stacked = rts_sample(
        state, jnp.broadcast_to(transition, (T, N, N)), jr.key(3), num_samples=6
    )
    operator = rts_sample(
        state, lx.MatrixLinearOperator(transition), jr.key(3), num_samples=6
    )
    tol = 10 * jnp.finfo(y.dtype).eps
    assert jnp.allclose(stacked, dense, rtol=tol, atol=tol)
    assert jnp.allclose(operator, dense, rtol=tol, atol=tol)


def test_the_same_key_gives_the_same_draws():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0)
    first = rts_sample(state, transition, jr.key(4), num_samples=3)
    again = rts_sample(state, transition, jr.key(4), num_samples=3)
    other = rts_sample(state, transition, jr.key(5), num_samples=3)
    assert jnp.array_equal(first, again)
    assert not jnp.allclose(first, other)
    # The draws within one call differ from one another.
    assert not jnp.allclose(first[0], first[1])


def test_jit_vmap_and_reparameterised_gradients():
    transition, obs, process, noise, y, mean0, cov0 = _model()

    def mean_path_end(scale):
        state = kalman_filter(transition, obs, scale * process, noise, y, mean0, cov0)
        draws = rts_sample(state, transition, jr.key(6), num_samples=16)
        return jnp.mean(draws[:, -1, 0])

    value = jax.jit(mean_path_end)(jnp.asarray(1.0, dtype=y.dtype))
    assert jnp.isfinite(value)
    grad = jax.jit(jax.grad(mean_path_end))(jnp.asarray(1.0, dtype=y.dtype))
    step = jnp.asarray(1e-3, dtype=y.dtype)
    central = (mean_path_end(1.0 + step) - mean_path_end(1.0 - step)) / (2 * step)
    # A fixed key makes the draws a smooth function of the scale; the central
    # difference carries O(step²) truncation and, in float32, ~1e-4 round-off.
    assert jnp.allclose(grad, central, rtol=5e-2, atol=5e-3)

    keys = jr.split(jr.key(7), 3)
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0)
    batched = jax.vmap(lambda k: rts_sample(state, transition, k, num_samples=2))(keys)
    assert batched.shape == (3, 2, T, N)
    for i in range(3):
        assert jnp.allclose(batched[i], rts_sample(state, transition, keys[i], 2))


def test_one_step_and_empty_records():
    transition, obs, process, noise, y, mean0, cov0 = _model()
    one = kalman_filter(transition, obs, process, noise, y[:1], mean0, cov0)
    draws = rts_sample(one, transition, jr.key(8), num_samples=5)
    assert draws.shape == (5, 1, N)
    empty = _empty_filter_state(mean0, cov0)
    assert rts_sample(empty, transition, jr.key(9), num_samples=5).shape == (5, 0, N)


@pytest.mark.parametrize("ndim", [1, 4])
def test_a_transition_of_the_wrong_rank_raises(ndim):
    transition, obs, process, noise, y, mean0, cov0 = _model()
    state = kalman_filter(transition, obs, process, noise, y, mean0, cov0)
    bad = jnp.ones((N,) * ndim, dtype=y.dtype)
    with pytest.raises(ValueError, match="ndim 2 or 3"):
        rts_sample(state, bad, jr.key(10))
