"""The consolidated Kalman-family surface (gh-364).

The shims that kept the pre-gh-364 forms working were removed in gaussx 0.7.0;
these tests pin the one remaining form of each. Keys are pinned: any
well-conditioned model will do.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import (
    DenseSolver,
    FilterState,
    infinite_horizon_filter,
    infinite_horizon_smoother,
    naturals_to_ssm,
    parallel_kalman_filter,
    ssm_to_naturals,
)


def _model():
    A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
    H = jnp.array([[1.0, 0.0]])
    Q = 0.1 * jnp.eye(2)
    R = jnp.array([[0.5]])
    y = jr.normal(jr.key(0), (6, 1))
    return A, H, Q, R, y, jnp.zeros(2), jnp.eye(2)


def _chain(T=5, d=2):
    k1, k2 = jr.split(jr.key(0))
    A = jnp.tile((0.9 * jnp.eye(d) + 0.05 * jr.normal(k1, (d, d)))[None], (T - 1, 1, 1))
    L = jr.normal(k2, (d, d))
    Q = jnp.tile((L @ L.T + 0.1 * jnp.eye(d))[None], (T - 1, 1, 1))
    P0 = 2.0 * jnp.eye(d)
    return A, Q, jnp.ones(d), P0


# --- one state type ---------------------------------------------------------


@pytest.mark.slow
def test_infinite_horizon_filter_returns_filter_state():
    A, H, Q, R, y, *_ = _model()
    assert isinstance(infinite_horizon_filter(A, H, Q, R, y), FilterState)


# --- smoother argument order ------------------------------------------------


def test_infinite_horizon_smoother_requires_dare_result():
    A, H, Q, R, y, *_ = _model()
    state = infinite_horizon_filter(A, H, Q, R, y)
    with pytest.raises(TypeError, match="dare_result"):
        infinite_horizon_smoother(state, A, Q)


# --- solver where it is unused ----------------------------------------------


@pytest.mark.slow
def test_parallel_filter_solver_raises_where_unused():
    A, H, Q, R, y, m0, P0 = _model()
    with pytest.raises(ValueError, match="woodbury_innovation"):
        parallel_kalman_filter(A, H, Q, R, y, m0, P0, solver=DenseSolver())
    # With woodbury_innovation the solver is used (delegated to kalman_filter).
    parallel_kalman_filter(
        A, H, Q, R, y, m0, P0, solver=DenseSolver(), woodbury_innovation=True
    )


# --- one Q layout -----------------------------------------------------------


@pytest.mark.parametrize(
    "blocks",
    [lambda Q, P0: Q[1:], lambda Q, P0: jnp.concatenate([P0[None], Q])],
    ids=["too_few", "stacked_p0"],
)
def test_ssm_to_naturals_rejects_a_wrong_number_of_blocks(blocks):
    A, Q, mu_0, P0 = _chain()
    with pytest.raises(ValueError, match="transition-noise blocks"):
        ssm_to_naturals(A, blocks(Q, P0), mu_0, P0)


def test_naturals_to_ssm_returns_the_transition_noise():
    A, Q, mu_0, P0 = _chain()
    theta = ssm_to_naturals(A, Q, mu_0, P0)
    A_new, Q_new, mu_new, P0_new = naturals_to_ssm(*theta)
    assert Q_new.shape == Q.shape
    assert jnp.allclose(Q_new, Q, atol=1e-10)
    assert jnp.allclose(A_new, A, atol=1e-10)
    assert jnp.allclose(P0_new, P0, atol=1e-10)
    assert jnp.allclose(mu_new, mu_0, atol=1e-10)
