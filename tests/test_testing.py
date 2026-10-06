"""Tests for gaussx._testing helper utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import BlockDiag, Kronecker, LowRankUpdate
from gaussx._testing import (
    default_tolerances,
    random_block_diag_pd,
    random_kronecker_pd,
    random_low_rank_update,
    random_pd_matrix,
    random_pd_operator,
    tree_allclose,
)


def test_tree_allclose_true():
    x = jnp.array([1.0, 2.0])
    assert tree_allclose(x, x)


def test_tree_allclose_false():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([1.0, 3.0])
    assert not tree_allclose(x, y)


def test_random_pd_matrix_shape_and_pd(getkey):
    mat = random_pd_matrix(getkey(), 5)
    assert mat.shape == (5, 5)
    # The active default float: float64 with x64, float32 without (gh-416).
    assert mat.dtype == jnp.result_type(float)
    # Positive definite: all eigenvalues > 0
    eigs = jnp.linalg.eigvalsh(mat)
    assert jnp.all(eigs > 0)


def test_random_pd_operator(getkey):
    op = random_pd_operator(getkey(), 4)
    assert isinstance(op, lx.MatrixLinearOperator)
    assert op.in_size() == 4
    assert lx.is_positive_semidefinite(op)


def test_random_kronecker_pd(getkey):
    K = random_kronecker_pd(getkey(), (3, 4))
    assert isinstance(K, Kronecker)
    assert K.in_size() == 12
    # All factors should be PSD
    for op in K.operators:
        assert lx.is_positive_semidefinite(op)


def test_random_block_diag_pd(getkey):
    BD = random_block_diag_pd(getkey(), (2, 3, 4))
    assert isinstance(BD, BlockDiag)
    assert BD.in_size() == 9
    for op in BD.operators:
        assert lx.is_positive_semidefinite(op)


def test_random_low_rank_update(getkey):
    lr = random_low_rank_update(getkey(), 10, 3)
    assert isinstance(lr, LowRankUpdate)
    assert lr.in_size() == 10
    assert lr.rank == 3


def test_default_tolerances_follow_the_lowest_precision_leaf():
    """gh-416: one assertion holds in the x64 and the no-x64 lanes."""
    f32 = jnp.ones(2, jnp.float32)
    assert default_tolerances(f32) == (1e-4, 1e-5)
    assert default_tolerances(jnp.ones(2, jnp.complex64)) == (1e-4, 1e-5)
    assert default_tolerances(jnp.arange(3)) == (1e-5, 1e-8)
    if jax.config.jax_enable_x64:
        f64 = jnp.ones(2, jnp.float64)
        assert default_tolerances(f64) == (1e-5, 1e-8)
        assert default_tolerances((f64, f32)) == (1e-4, 1e-5)


def test_tree_allclose_float32_tolerates_round_off_near_zero():
    """A float32 result 1e-7 off a near-zero reference is round-off."""
    ref = jnp.array([1e-3, 2.0], jnp.float32)
    assert tree_allclose(ref + jnp.float32(1e-7), ref)


def test_random_spd_block_tridiag_is_spd_for_every_key():
    """gh-316: SPD by construction, not by diagonal dominance."""
    from gaussx._testing import random_spd_block_tridiag

    def min_eig(key):
        dense = random_spd_block_tridiag(key, 4, 3).as_matrix()
        return jnp.linalg.eigvalsh(dense).min(), jnp.abs(dense - dense.T).max()

    eigs, asym = jax.vmap(min_eig)(jr.split(jr.key(0), 100))
    assert jnp.all(eigs > 0)
    assert jnp.all(asym == 0)


def test_random_sum_of_kroneckers_pd_is_symmetric_pd():
    from gaussx._testing import random_sum_of_kroneckers_pd

    dense = random_sum_of_kroneckers_pd(jr.key(0), (2, 3)).as_matrix()
    assert jnp.allclose(dense, dense.T)
    assert jnp.linalg.eigvalsh(dense).min() > 0


def test_empirical_moments_matches_mean_and_cov():
    from gaussx._testing import empirical_moments

    samples = jr.normal(jr.key(0), (50, 3))
    mean, cov = empirical_moments(samples)
    assert jnp.allclose(mean, jnp.mean(samples, axis=0))
    assert jnp.allclose(cov, jnp.cov(samples.T, bias=False))


def test_random_pd_matrix_jitter():
    """jitter is the shift on the diagonal; the default stays 0.1."""
    from gaussx._testing import random_pd_matrix

    base = random_pd_matrix(jr.key(0), 4)
    shifted = random_pd_matrix(jr.key(0), 4, jitter=4.0)
    assert jnp.allclose(shifted - base, 3.9 * jnp.eye(4))
