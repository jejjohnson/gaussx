"""Tests for conditional Gaussian distribution."""

from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum
from gaussx._testing import (
    default_tolerances,
    random_kronecker_pd,
    random_pd_matrix,
    random_pd_operator,
)


class TestConditional:
    @pytest.mark.slow
    def test_conditional_mean(self):
        """Conditional mean matches closed-form for a known case."""
        # Joint: N([0,0], [[1, 0.5], [0.5, 1]])
        mu = jnp.zeros(2)
        Sigma = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        # Observe x_1 = 1.0
        cond_mean, cond_cov = gaussx.conditional(
            mu, cov, jnp.array([1]), jnp.array([1.0])
        )
        # mu_0|1 = 0 + 0.5 * 1.0 / 1.0 * (1.0 - 0) = 0.5
        assert jnp.allclose(cond_mean, jnp.array([0.5]), atol=1e-5)
        # Sigma_0|1 = 1 - 0.5^2 / 1 = 0.75
        assert jnp.allclose(cond_cov.as_matrix(), jnp.array([[0.75]]), atol=1e-5)

    @pytest.mark.slow
    def test_conditional_recovers_marginal(self):
        """Conditioning on nothing should return something close to prior."""
        n = 5
        key = jax.random.PRNGKey(0)
        mu = jax.random.normal(key, (n,))
        Sigma = random_pd_matrix(jax.random.PRNGKey(1), n, jitter=1.0)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        # Observe 2 variables
        obs_idx = jnp.array([1, 3])
        obs_values = mu[obs_idx]  # Observe at the mean

        cond_mean, _cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)
        # Conditional mean should equal prior mean at free indices
        free_idx = jnp.array([0, 2, 4])
        assert jnp.allclose(cond_mean, mu[free_idx], atol=1e-4)

    @pytest.mark.slow
    def test_conditional_cov_psd(self):
        """Conditional covariance should be PSD."""
        n = 6
        key = jax.random.PRNGKey(42)
        mu = jnp.zeros(n)
        Sigma = random_pd_matrix(key, n, jitter=1.0)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([0, 2, 4])
        obs_values = jnp.array([1.0, -0.5, 0.3])
        _, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        eigs = jnp.linalg.eigvalsh(cond_cov.as_matrix())
        assert jnp.all(eigs > -1e-6)

    @pytest.mark.slow
    def test_conditional_reduces_variance(self):
        """Conditional variance <= marginal variance."""
        n = 4
        key = jax.random.PRNGKey(7)
        mu = jnp.zeros(n)
        Sigma = random_pd_matrix(key, n, jitter=1.0)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([1])
        obs_values = jnp.array([0.5])
        _, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        free_idx = jnp.array([0, 2, 3])
        marginal_var = jnp.diag(Sigma)[free_idx]
        cond_var = jnp.diag(cond_cov.as_matrix())
        assert jnp.all(cond_var <= marginal_var + 1e-6)

    def test_conditional_against_dense(self):
        """Test against manual computation."""
        n = 5
        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        mu = jax.random.normal(k1, (n,))
        Sigma = random_pd_matrix(k2, n, jitter=1.0)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([0, 3])
        free_idx = jnp.array([1, 2, 4])
        obs_values = jnp.array([1.0, -1.0])

        cond_mean, cond_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)

        # Manual computation
        Sigma_AA = Sigma[jnp.ix_(free_idx, free_idx)]
        Sigma_AB = Sigma[jnp.ix_(free_idx, obs_idx)]
        Sigma_BB = Sigma[jnp.ix_(obs_idx, obs_idx)]
        mu_A = mu[free_idx]
        mu_B = mu[obs_idx]

        expected_mean = mu_A + Sigma_AB @ jnp.linalg.solve(Sigma_BB, obs_values - mu_B)
        expected_cov = Sigma_AA - Sigma_AB @ jnp.linalg.solve(Sigma_BB, Sigma_AB.T)

        assert jnp.allclose(cond_mean, expected_mean, atol=1e-4)
        assert jnp.allclose(cond_cov.as_matrix(), expected_cov, atol=1e-4)

    def test_conditional_with_dense_solver_matches_default(self):
        """Passing solver=DenseSolver() must match default structural dispatch."""
        n = 5
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        mu = jax.random.normal(k1, (n,))
        Sigma = random_pd_matrix(k2, n, jitter=1.0)
        cov = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)

        obs_idx = jnp.array([0, 3])
        obs_values = jnp.array([1.0, -1.0])

        default_mean, default_cov = gaussx.conditional(mu, cov, obs_idx, obs_values)
        dense_mean, dense_cov = gaussx.conditional(
            mu, cov, obs_idx, obs_values, solver=gaussx.DenseSolver()
        )

        assert jnp.allclose(dense_mean, default_mean, atol=1e-5)
        assert jnp.allclose(dense_cov.as_matrix(), default_cov.as_matrix(), atol=1e-5)

    def test_duplicate_obs_idx_raises(self):
        """Duplicate observed indices should raise a clear validation error."""
        mu = jnp.zeros(4)
        cov = lx.MatrixLinearOperator(jnp.eye(4), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="duplicates"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([1, 1, 2]),
                jnp.array([0.0, 0.0, 0.0]),
            )

    def test_obs_idx_out_of_bounds_raises(self):
        """Observed indices must lie within the state dimension."""
        mu = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="bounds"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([0, 3]),
                jnp.array([0.0, 1.0]),
            )

    def test_obs_values_shape_mismatch_raises(self):
        """Observed values must align one-to-one with observed indices."""
        mu = jnp.zeros(3)
        cov = lx.MatrixLinearOperator(jnp.eye(3), lx.positive_semidefinite_tag)

        with pytest.raises(ValueError, match="same shape"):
            gaussx.conditional(
                mu,
                cov,
                jnp.array([0, 2]),
                jnp.array([1.0]),
            )


class TestConditionalUnderJit:
    """gh-351: index validation used ``bool()`` on (traced) arrays."""

    @pytest.fixture
    def joint(self):
        S = random_pd_matrix(jax.random.key(0), 4, jitter=1.0)
        return jnp.zeros(4), lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)

    @pytest.mark.parametrize(
        "make_idx",
        [lambda: jnp.array([1, 3]), lambda: np.array([1, 3])],
        ids=["jnp-constant", "numpy-constant"],
    )
    def test_constant_indices(self, joint, make_idx):
        mu, cov = joint
        vals = jnp.array([0.5, -1.0])
        eager_mean, eager_cov = gaussx.conditional(mu, cov, make_idx(), vals)

        @jax.jit
        def f(v):
            mean, c = gaussx.conditional(mu, cov, make_idx(), v)
            return mean, c.as_matrix()

        mean, cov_mat = f(vals)
        assert jnp.allclose(mean, eager_mean, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(cov_mat, eager_cov.as_matrix(), rtol=1e-12, atol=1e-12)

    def test_traced_indices(self, joint):
        mu, cov = joint
        idx, vals = jnp.array([1, 3]), jnp.array([0.5, -1.0])
        eager_mean, eager_cov = gaussx.conditional(mu, cov, idx, vals)

        @jax.jit
        def f(i, v):
            mean, c = gaussx.conditional(mu, cov, i, v)
            return mean, c.as_matrix()

        mean, cov_mat = f(idx, vals)
        assert jnp.allclose(mean, eager_mean, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(cov_mat, eager_cov.as_matrix(), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "idx", [jnp.array([1, 4]), jnp.array([-1, 2]), jnp.array([1, 1])]
    )
    def test_traced_invalid_indices_raise_at_run_time(self, joint, idx):
        mu, cov = joint
        f = jax.jit(lambda i, v: gaussx.conditional(mu, cov, i, v)[0])
        with pytest.raises(Exception, match="obs_idx must be within bounds"):
            f(idx, jnp.array([0.5, -1.0])).block_until_ready()

    def test_traced_invalid_indices_raise_when_only_covariance_is_used(self, joint):
        # The check must not be dead-code-eliminated with the mean.
        mu, cov = joint
        f = jax.jit(lambda i, v: gaussx.conditional(mu, cov, i, v)[1].as_matrix())
        with pytest.raises(Exception, match="obs_idx must be within bounds"):
            f(jnp.array([1, 1]), jnp.array([0.5, -1.0])).block_until_ready()

    def test_concrete_non_integral_indices_are_validated_after_casting(self, joint):
        mu, cov = joint
        with pytest.raises(ValueError, match="duplicates"):
            gaussx.conditional(mu, cov, np.array([1.1, 1.2]), jnp.array([0.5, -1.0]))


# ---------------------------------------------------------------------------
# linear_gaussian_update: x_a = x_b + B Hᵀ S⁻¹ d, P_a = B − B Hᵀ S⁻¹ H B
# ---------------------------------------------------------------------------


_N_STATE, _N_OBS = 6, 4


def _float():
    return jnp.result_type(float)


def _prior_cases():
    return [
        pytest.param(lambda: random_pd_operator(jr.key(0), _N_STATE), id="dense"),
        pytest.param(
            lambda: lx.DiagonalLinearOperator(jnp.arange(1.0, 7.0, dtype=_float())),
            id="diagonal",
        ),
        pytest.param(lambda: random_kronecker_pd(jr.key(1), (2, 3)), id="kronecker"),
    ]


def _obs_matrix():
    return jr.normal(jr.key(2), (_N_OBS, _N_STATE), _float())


def _obs_cases():
    def jacobian():
        matrix = _obs_matrix()
        return lx.JacobianLinearOperator(
            lambda state, _: matrix @ state, jnp.zeros(_N_STATE, _float())
        )

    return [
        pytest.param(lambda: lx.MatrixLinearOperator(_obs_matrix()), id="matrix"),
        pytest.param(jacobian, id="jacobian"),
    ]


def _dense_update(prior_mean, B, H, R, obs):
    """The dense formulas, as the reference."""
    gain_factor = einsum(B, H, "i k, j k -> i j")
    innovation_cov = einsum(H, gain_factor, "i k, k j -> i j") + R
    gain = einsum(gain_factor, jnp.linalg.inv(innovation_cov), "i k, k j -> i j")
    mean = prior_mean + gain @ (obs - H @ prior_mean)
    return mean, B - einsum(gain, H @ B, "i k, k j -> i j")


class TestLinearGaussianUpdate:
    @pytest.mark.parametrize("build_prior", _prior_cases())
    @pytest.mark.parametrize("build_obs", _obs_cases())
    def test_matches_the_dense_formulas(self, build_prior, build_obs):
        B, H = build_prior(), build_obs()
        R = random_pd_operator(jr.key(3), _N_OBS)
        prior_mean = jr.normal(jr.key(4), (_N_STATE,), _float())
        obs = jr.normal(jr.key(5), (_N_OBS,), _float())
        posterior = gaussx.linear_gaussian_update(prior_mean, B, H, R, obs)
        mean, cov = _dense_update(
            prior_mean, B.as_matrix(), _obs_matrix(), R.as_matrix(), obs
        )
        rtol, atol = default_tolerances(cov)
        assert jnp.allclose(posterior.mean, mean, rtol=100 * rtol, atol=100 * atol)
        assert jnp.allclose(
            posterior.cov.as_matrix(), cov, rtol=100 * rtol, atol=100 * atol
        )
        assert lx.is_positive_semidefinite(posterior.cov)

    def test_posterior_is_no_wider_than_the_prior(self):
        B = random_pd_operator(jr.key(0), _N_STATE)
        R = random_pd_operator(jr.key(3), _N_OBS)
        posterior = gaussx.linear_gaussian_update(
            jnp.zeros(_N_STATE, _float()),
            B,
            lx.MatrixLinearOperator(_obs_matrix()),
            R,
            jnp.ones(_N_OBS, _float()),
        )
        reduction = B.as_matrix() - posterior.cov.as_matrix()
        eigenvalues = jnp.linalg.eigvalsh(gaussx.symmetrize(reduction))
        # B − P_a is PSD to rounding
        rounding = 100 * jnp.finfo(eigenvalues.dtype).eps * jnp.max(eigenvalues)
        assert jnp.all(eigenvalues >= -rounding)

    def test_infinite_noise_returns_the_prior(self):
        B = random_pd_operator(jr.key(0), _N_STATE)
        prior_mean = jr.normal(jr.key(4), (_N_STATE,), _float())
        posterior = gaussx.linear_gaussian_update(
            prior_mean,
            B,
            lx.MatrixLinearOperator(_obs_matrix()),
            lx.DiagonalLinearOperator(jnp.full(_N_OBS, 1e8, _float())),
            jnp.ones(_N_OBS, _float()),
        )
        assert jnp.allclose(posterior.mean, prior_mean, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(posterior.cov.as_matrix(), B.as_matrix(), atol=1e-2)

    def test_shaped_state_and_observation(self):
        """A (T, N) state observed through a 2-D mask, against the flat update."""
        shape = (2, 3)
        dtype = _float()
        variances = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)
        weights = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype)
        noise = jnp.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], dtype)
        prior_mean = jr.normal(jr.key(4), shape, dtype)
        obs = jr.normal(jr.key(5), shape, dtype)
        posterior = gaussx.linear_gaussian_update(
            prior_mean,
            lx.DiagonalLinearOperator(variances),
            lx.DiagonalLinearOperator(weights),
            lx.DiagonalLinearOperator(noise),
            obs,
            solver=gaussx.CGSolver(rtol=1e-5, atol=1e-5),
        )
        assert posterior.mean.shape == shape
        assert posterior.cov.in_structure().shape == shape
        # entrywise: gain = b w / (w² b + r), variance = b − gain w b
        gain = variances * weights / (weights**2 * variances + noise)
        mean = prior_mean + gain * (obs - weights * prior_mean)
        applied = posterior.cov.mv(jnp.ones(shape, dtype))
        assert jnp.allclose(posterior.mean, mean, rtol=1e-3, atol=1e-3)
        expected = variances - gain * weights * variances
        assert jnp.allclose(applied, expected, rtol=1e-3, atol=1e-3)

    def test_cg_stays_matrix_free(self):
        B = random_kronecker_pd(jr.key(1), (2, 3))
        R = random_pd_operator(jr.key(3), _N_OBS)
        prior_mean = jnp.zeros(_N_STATE, _float())
        obs = jnp.ones(_N_OBS, _float())
        H = lx.MatrixLinearOperator(_obs_matrix())
        reference = _dense_update(
            prior_mean, B.as_matrix(), _obs_matrix(), R.as_matrix(), obs
        )
        direction = jnp.ones(_N_STATE, _float())
        with mock.patch.object(
            gaussx.Kronecker, "as_matrix", side_effect=AssertionError("densified")
        ):
            posterior = gaussx.linear_gaussian_update(
                prior_mean,
                B,
                H,
                R,
                obs,
                solver=gaussx.CGSolver(rtol=1e-5, atol=1e-5),
            )
            applied = posterior.cov.mv(direction)
        assert jnp.allclose(posterior.mean, reference[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(applied, reference[1] @ direction, rtol=1e-3, atol=1e-3)

    def test_jit_vmap_and_grad_in_the_observation(self):
        B = random_pd_operator(jr.key(0), _N_STATE)
        H = lx.MatrixLinearOperator(_obs_matrix())
        R = random_pd_operator(jr.key(3), _N_OBS)
        prior_mean = jnp.zeros(_N_STATE, _float())

        def mean_of(obs):
            return gaussx.linear_gaussian_update(prior_mean, B, H, R, obs).mean

        obs = jr.normal(jr.key(5), (2, _N_OBS), _float())
        batched = eqx.filter_jit(jax.vmap(mean_of))(obs)
        rtol, atol = default_tolerances(batched)
        loop = jnp.stack([mean_of(row) for row in obs])
        assert jnp.allclose(batched, loop, rtol=100 * rtol, atol=100 * atol)
        # x_a is linear in y, so ∂ sum(x_a) / ∂y is 1ᵀ B Hᵀ S⁻¹
        gradient = jax.grad(lambda y: jnp.sum(mean_of(y)))(obs[0])
        matrix = _obs_matrix()
        prior = B.as_matrix()
        projected = einsum(matrix, prior, "i k, k j -> i j")
        innovation_cov = einsum(projected, matrix, "i k, j k -> i j") + R.as_matrix()
        expected = jnp.linalg.solve(innovation_cov, projected @ jnp.ones(_N_STATE))
        assert jnp.allclose(gradient, expected, rtol=100 * rtol, atol=100 * atol)
