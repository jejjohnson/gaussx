"""The consolidated Kalman-family surface and its deprecation shims (gh-364).

Every deprecated form must still give the identical result, with a
``DeprecationWarning``; the new forms must not warn. Keys are pinned: the
checks are identities, so any well-conditioned model will do.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import jax.random as jr
import pytest

import gaussx
from gaussx import (
    DenseSolver,
    FilterState,
    dare,
    infinite_horizon_filter,
    infinite_horizon_smoother,
    kalman_filter,
    meanfield_rts_smoother,
    naturals_to_ssm,
    nonlinear_kalman_filter,
    nonlinear_rts_smoother,
    parallel_kalman_filter,
    parallel_rts_smoother,
    rts_smoother,
    ssm_to_naturals,
    udl_from_ssm_params,
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


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return fn(*args, **kwargs)


# --- one state type ---------------------------------------------------------


@pytest.mark.slow
def test_infinite_horizon_filter_returns_filter_state():
    A, H, Q, R, y, *_ = _model()
    assert isinstance(infinite_horizon_filter(A, H, Q, R, y), FilterState)


def test_infinite_horizon_state_is_a_deprecated_alias():
    with pytest.warns(DeprecationWarning, match="InfiniteHorizonState"):
        alias = gaussx.InfiniteHorizonState
    assert alias is FilterState


# --- smoother argument order ------------------------------------------------


def test_infinite_horizon_smoother_new_and_old_order_agree():
    A, H, Q, R, y, *_ = _model()
    d = dare(A, H, Q, R)
    state = infinite_horizon_filter(A, H, Q, R, y, dare_result=d)
    new = _no_warnings(infinite_horizon_smoother, state, A, Q, dare_result=d)
    with pytest.warns(DeprecationWarning, match="dare_result as a keyword"):
        old = infinite_horizon_smoother(state, A, d, Q)
    for a, b in zip(new, old, strict=True):
        assert jnp.array_equal(a, b)


def test_infinite_horizon_smoother_requires_dare_result():
    A, H, Q, R, y, *_ = _model()
    state = infinite_horizon_filter(A, H, Q, R, y)
    with pytest.raises(TypeError, match="dare_result"):
        infinite_horizon_smoother(state, A, Q)


# --- dead process_noise / solver --------------------------------------------


@pytest.mark.parametrize(
    "smoother",
    [
        rts_smoother,
        parallel_rts_smoother,
        lambda s, A, *q: meanfield_rts_smoother(s, A, *q, block_size=1),
    ],
    ids=["rts", "parallel_rts", "meanfield_rts"],
)
def test_process_noise_is_deprecated_and_ignored(smoother):
    A, H, Q, R, y, m0, P0 = _model()
    state = kalman_filter(A, H, Q, R, y, m0, P0)
    new = _no_warnings(smoother, state, A)
    with pytest.warns(DeprecationWarning, match="process_noise"):
        old = smoother(state, A, Q)
    for a, b in zip(new, old, strict=True):
        assert jnp.array_equal(a, b)


def test_nonlinear_smoother_without_process_noise_does_not_warn():
    A, H, Q, R, y, m0, P0 = _model()
    state = nonlinear_kalman_filter(lambda x: A @ x, lambda x: H @ x, Q, R, y, m0, P0)
    _no_warnings(nonlinear_rts_smoother, state, lambda x: A @ x)


@pytest.mark.slow
def test_parallel_filter_solver_only_warns_where_unused():
    A, H, Q, R, y, m0, P0 = _model()
    with pytest.warns(DeprecationWarning, match="woodbury_innovation"):
        parallel_kalman_filter(A, H, Q, R, y, m0, P0, solver=DenseSolver())
    # With woodbury_innovation the solver is used (delegated to kalman_filter).
    _no_warnings(
        parallel_kalman_filter,
        A,
        H,
        Q,
        R,
        y,
        m0,
        P0,
        solver=DenseSolver(),
        woodbury_innovation=True,
    )


# --- one Q layout -----------------------------------------------------------


@pytest.mark.slow
def test_ssm_to_naturals_accepts_the_markov_gaussian_layout():
    A, Q, mu_0, P0 = _chain()
    new = _no_warnings(ssm_to_naturals, A, Q, mu_0, P0)
    stacked = jnp.concatenate([P0[None], Q])
    with pytest.warns(DeprecationWarning, match="Q\\[0\\] == P_0"):
        old = ssm_to_naturals(A, stacked, mu_0, P0)
    assert jnp.array_equal(new[0], old[0])
    assert jnp.array_equal(new[1].as_matrix(), old[1].as_matrix())


def test_ssm_to_naturals_rejects_a_wrong_number_of_blocks():
    A, Q, mu_0, P0 = _chain()
    with pytest.raises(ValueError, match="transition-noise blocks"):
        ssm_to_naturals(A, Q[1:], mu_0, P0)


def test_naturals_to_ssm_layouts():
    A, Q, mu_0, P0 = _chain()
    theta = ssm_to_naturals(A, Q, mu_0, P0)
    A_new, Q_new, mu_new, P0_new = _no_warnings(
        naturals_to_ssm, *theta, initial_in_q=False
    )
    assert Q_new.shape == Q.shape
    assert jnp.allclose(Q_new, Q, atol=1e-10)
    assert jnp.allclose(A_new, A, atol=1e-10)
    assert jnp.allclose(P0_new, P0, atol=1e-10)
    assert jnp.allclose(mu_new, mu_0, atol=1e-10)
    _, Q_old, _, _ = _no_warnings(naturals_to_ssm, *theta, initial_in_q=True)
    assert jnp.array_equal(Q_old, jnp.concatenate([P0_new[None], Q_new]))
    with pytest.warns(DeprecationWarning, match="initial_in_q"):
        _, Q_default, _, _ = naturals_to_ssm(*theta)
    assert jnp.array_equal(Q_default, Q_old)
    with pytest.warns(DeprecationWarning, match="solver"):
        naturals_to_ssm(*theta, solver=DenseSolver(), initial_in_q=False)


def test_udl_from_ssm_params_layouts():
    A, Q, _, P0 = _chain()
    new = _no_warnings(udl_from_ssm_params, A, Q, P0)
    with pytest.warns(DeprecationWarning, match="P0 separately"):
        old = udl_from_ssm_params(A, jnp.concatenate([P0[None], Q]))
    assert jnp.array_equal(new.D_diag, old.D_diag)
    assert jnp.array_equal(new.U_sub, old.U_sub)


def test_markov_gaussian_no_longer_translates_layouts():
    pytest.importorskip("numpyro")
    A, Q, mu_0, P0 = _chain()
    chain = gaussx.MarkovGaussian(A, Q, mu_0, P0)
    assert not hasattr(chain, "_Q_with_initial")
    precision = _no_warnings(lambda: chain.precision)
    expected = -2.0 * ssm_to_naturals(A, Q, mu_0, P0)[1].as_matrix()
    assert jnp.allclose(precision.as_matrix(), expected, atol=1e-10)
