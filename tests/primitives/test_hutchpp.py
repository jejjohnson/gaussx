"""Tests for ``trace(..., algorithm="hutchpp")`` (G17, gh-486).

Hutch++ is unbiased, so its mean over many pinned keys is bounded by its
own standard error. The comparison with Hutchinson is Meyer et al.'s
(2021) headline experiment (equal matvec budgets on a spectrum
lambda_i = i^-c), with the gap measured over 200 keys stated where it is
used.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

from gaussx import trace
from gaussx._testing import default_tolerances


def _spectral_operator(n: int, eigenvalues):
    """``U diag(eigenvalues) Uᵀ`` with a random orthogonal ``U``."""
    U, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    U_lam = einx.multiply("i k, k -> i k", U, eigenvalues)
    return lx.MatrixLinearOperator(einx.dot("i k, j k -> i j", U_lam, U))


def _estimates(op, algorithm, num_probes, num_keys, **kwargs):
    keys = jr.split(jr.key(1), num_keys)
    return jax.jit(
        jax.vmap(
            lambda k: trace(
                op,
                stochastic=True,
                num_probes=num_probes,
                key=k,
                algorithm=algorithm,
                **kwargs,
            )
        )
    )(keys)


def test_unbiased_on_an_indefinite_operator():
    """E[Hutch++] = tr(A) for any A, not only PSD."""
    n = 60
    eigenvalues = jnp.linspace(-1.0, 2.0, n)
    op = _spectral_operator(n, eigenvalues)
    est = np.asarray(_estimates(op, "hutchpp", num_probes=12, num_keys=400))
    sem = est.std() / np.sqrt(est.size)
    # The mean of 400 i.i.d. estimates is within 5 SEM of the truth with
    # probability > 1 - 1e-6 (CLT).
    assert abs(est.mean() - float(jnp.sum(eigenvalues))) <= 5 * sem


@pytest.mark.slow
def test_beats_hutchinson_on_a_decaying_spectrum():
    """Meyer et al. (2021): lambda_i = i^-2, equal budgets of 60 matvecs."""
    n = 200
    eigenvalues = 1.0 / jnp.arange(1.0, n + 1.0) ** 2
    op = _spectral_operator(n, eigenvalues)
    exact = float(jnp.sum(eigenvalues))

    def mean_rel_err(algorithm, num_probes):
        est = np.asarray(_estimates(op, algorithm, num_probes, num_keys=50))
        return float(np.mean(np.abs(est - exact)) / exact)

    hutchinson = mean_rel_err("hutchinson", 60)
    # XTrace spends 2 matvecs per probe.
    for algorithm, num_probes in (("hutchpp", 60), ("xtrace", 30)):
        # Over 200 keys at n = 300: Hutchinson 0.094, Hutch++ 0.0015, XTrace
        # 0.0008 mean relative error; a factor 10 leaves a wide margin.
        assert mean_rel_err(algorithm, num_probes) < 0.1 * hutchinson, algorithm


@pytest.mark.parametrize("sampler", ["signs", "normal"])
def test_exact_on_a_low_rank_operator(sampler):
    """rank(A) <= k: the range sketch captures A and the remainder is zero."""
    n, rank = 40, 4
    eigenvalues = jnp.zeros(n).at[:rank].set(jnp.arange(1.0, rank + 1.0))
    op = _spectral_operator(n, eigenvalues)
    est = jax.jit(
        lambda op: trace(
            op, stochastic=True, num_probes=15, algorithm="hutchpp", sampler=sampler
        )
    )(op)
    rtol, atol = default_tolerances(eigenvalues)
    assert jnp.allclose(est, jnp.sum(eigenvalues), rtol=10 * rtol, atol=10 * atol)


def test_key_none_means_prngkey_zero():
    op = _spectral_operator(10, jnp.arange(1.0, 11.0))
    a = trace(op, stochastic=True, num_probes=6, algorithm="hutchpp")
    b = trace(
        op,
        stochastic=True,
        num_probes=6,
        algorithm="hutchpp",
        key=jax.random.PRNGKey(0),
    )
    assert jnp.array_equal(a, b)


def test_rejects_fewer_than_three_matvecs():
    op = lx.MatrixLinearOperator(jnp.eye(5))
    with pytest.raises(ValueError, match="num_probes >= 3"):
        trace(op, stochastic=True, num_probes=2, algorithm="hutchpp")
