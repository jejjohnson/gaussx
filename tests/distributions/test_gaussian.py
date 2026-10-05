"""Tests for Gaussian sugar: log-prob, entropy, KL, quadratic form, jitter."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import (
    add_jitter,
    gaussian_entropy,
    gaussian_log_prob,
    kl_standard_normal,
    quadratic_form,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def test_quadratic_form_diagonal(getkey):
    d = jnp.array([2.0, 3.0, 4.0])
    x = jnp.array([1.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(d)
    result = quadratic_form(op, x)
    expected = jnp.sum(x**2 / d)
    assert tree_allclose(result, expected)


def test_quadratic_form_dense(getkey):
    mat = random_pd_matrix(getkey(), 4)
    x = jr.normal(getkey(), (4,))
    op = lx.MatrixLinearOperator(mat)
    result = quadratic_form(op, x)
    expected = x @ jnp.linalg.solve(mat, x)
    assert tree_allclose(result, expected, rtol=1e-5)


def test_gaussian_log_prob_known(getkey):
    """Log-prob of N(0, I) at x=0 should be -N/2 log(2pi)."""
    N = 3
    mu = jnp.zeros(N)
    Sigma = lx.DiagonalLinearOperator(jnp.ones(N))
    x = jnp.zeros(N)
    result = gaussian_log_prob(mu, Sigma, x)
    expected = -0.5 * N * jnp.log(2.0 * jnp.pi)
    assert tree_allclose(result, expected, rtol=1e-5)


def test_gaussian_log_prob_matches_scipy(getkey):
    N = 4
    mat = random_pd_matrix(getkey(), N)
    mu = jr.normal(getkey(), (N,))
    x = jr.normal(getkey(), (N,))
    op = lx.MatrixLinearOperator(mat)

    result = gaussian_log_prob(mu, op, x)

    # Manual computation
    r = x - mu
    _, ld = jnp.linalg.slogdet(mat)
    quad = r @ jnp.linalg.solve(mat, r)
    expected = -0.5 * (N * jnp.log(2.0 * jnp.pi) + ld + quad)

    assert tree_allclose(result, expected, rtol=1e-4)


def test_gaussian_entropy_isotropic(getkey):
    """Entropy of N(0, sigma^2 I)."""
    N = 3
    sigma2 = 2.0
    Sigma = lx.DiagonalLinearOperator(jnp.full(N, sigma2))
    result = gaussian_entropy(Sigma)
    expected = 0.5 * (N * (1.0 + jnp.log(2.0 * jnp.pi)) + N * jnp.log(sigma2))
    assert tree_allclose(result, expected, rtol=1e-5)


def test_kl_standard_normal_zero_for_identity(getkey):
    """KL(N(0, I) || N(0, I)) = 0."""
    N = 4
    m = jnp.zeros(N)
    S = lx.DiagonalLinearOperator(jnp.ones(N))
    assert tree_allclose(kl_standard_normal(m, S), jnp.array(0.0), atol=1e-6)


def test_kl_standard_normal_positive(getkey):
    """KL should be non-negative."""
    N = 3
    m = jr.normal(getkey(), (N,))
    S = lx.MatrixLinearOperator(random_pd_matrix(getkey(), N))
    kl = kl_standard_normal(m, S)
    assert kl >= -1e-6


def test_kl_standard_normal_known(getkey):
    """KL to standard normal for isotropic case."""
    N = 3
    sigma2 = 2.0
    m = jnp.array([1.0, 2.0, 3.0])
    S = lx.DiagonalLinearOperator(jnp.full(N, sigma2))
    result = kl_standard_normal(m, S)
    expected = 0.5 * (N * sigma2 + m @ m - N - N * jnp.log(sigma2))
    assert tree_allclose(result, expected, rtol=1e-5)


def test_add_jitter(getkey):
    N = 4
    mat = random_pd_matrix(getkey(), N)
    op = lx.MatrixLinearOperator(mat)
    jittered = add_jitter(op, jitter=1e-3)
    expected = mat + 1e-3 * jnp.eye(N)
    assert tree_allclose(jittered.as_matrix(), expected, rtol=1e-6)


def test_add_jitter_default(getkey):
    d = jnp.array([1.0, 2.0, 3.0])
    op = lx.DiagonalLinearOperator(d)
    jittered = add_jitter(op)
    expected = jnp.diag(d) + 1e-6 * jnp.eye(3)
    assert tree_allclose(jittered.as_matrix(), expected, atol=1e-10)


def test_log_prob_is_exact_when_x64_is_enabled_after_import():
    """gh-369: the log(2 pi) constant must not freeze the import-time dtype.

    conftest.py enables x64 before gaussx is imported, which hides the bug,
    so the import-then-enable order runs in a fresh interpreter.
    """
    script = textwrap.dedent(
        """
        import jax
        import gaussx
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp, lineax as lx, numpy as np
        from scipy.stats import multivariate_normal

        n = 2000
        x = jnp.linspace(-1.0, 1.0, n)
        cov = lx.DiagonalLinearOperator(jnp.ones(n))
        ref = multivariate_normal(np.zeros(n), np.eye(n)).logpdf(np.asarray(x))
        lp = gaussx.gaussian_log_prob(jnp.zeros(n), cov, x)
        lp_prec = gaussx.MultivariateNormalPrecision(jnp.zeros(n), cov).log_prob(x)
        assert lp.dtype == jnp.float64, lp.dtype
        # The bug was an offset of 3.1e-8 per dimension (8.8e-6 here).
        assert abs(float(lp) - ref) < 1e-9, float(lp) - ref
        assert abs(float(lp_prec) - ref) < 1e-9, float(lp_prec) - ref
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_log_prob_keeps_float32_under_x64():
    """gh-369: the Python-float constant stays weakly typed."""
    x = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    cov = lx.DiagonalLinearOperator(jnp.ones(5, jnp.float32))
    assert gaussian_log_prob(jnp.zeros(5, jnp.float32), cov, x).dtype == jnp.float32
