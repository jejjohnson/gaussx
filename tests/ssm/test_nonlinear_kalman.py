"""Tests for the moment-matched nonlinear Kalman filter and smoother."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx import (
    AbstractIntegrator,
    CubatureIntegrator,
    FifthOrderCubatureIntegrator,
    GaussHermiteIntegrator,
    GaussianState,
    MonteCarloIntegrator,
    PropagationResult,
    TaylorIntegrator,
    UnscentedIntegrator,
    kalman_filter,
    nonlinear_kalman_filter,
    nonlinear_rts_smoother,
    rts_smoother,
)
from gaussx._testing import tree_allclose


_N, _M, _T = 3, 2, 12

_A = jnp.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.7]])
_H = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])
_Q = 0.05 * jnp.eye(_N)
_R = jnp.diag(jnp.array([0.2, 0.3]))
_M0 = jnp.array([0.5, -0.2, 0.1])
_P0 = 0.4 * jnp.eye(_N)
_YS = jax.random.normal(jax.random.key(0), (_T, _M))


def _linear_dynamics(x):
    return _A @ x


def _linear_obs(x):
    return _H @ x


# The scaled unscented transform with the default alpha=1e-3 places its sigma
# points ~1e-3 away from the mean and recovers the moments by cancellation,
# which costs roughly seven digits. That is a property of the rule, not of the
# filter wiring -- alpha=1.0 reaches machine precision on the same problem.
_INTEGRATORS = [
    pytest.param(TaylorIntegrator(), 1e-12, id="taylor-ekf"),
    pytest.param(UnscentedIntegrator(alpha=1.0), 1e-12, id="unscented-alpha1"),
    pytest.param(UnscentedIntegrator(), 1e-8, id="unscented-default"),
    pytest.param(CubatureIntegrator(), 1e-12, id="cubature-ckf"),
    pytest.param(FifthOrderCubatureIntegrator(), 1e-12, id="cubature-degree5"),
    pytest.param(GaussHermiteIntegrator(order=4), 1e-12, id="gauss-hermite-ghkf"),
    pytest.param(
        MonteCarloIntegrator(n_samples=20_000, key=jax.random.key(0)), 5e-2, id="mc"
    ),
]


@pytest.mark.parametrize(("integrator", "atol"), _INTEGRATORS)
@pytest.mark.parametrize("joseph", [True, False])
def test_linear_reduction_matches_kalman_filter(integrator, atol, joseph):
    """With affine maps the filter must reproduce ``kalman_filter`` exactly.

    Every moment-matching rule here is exact for affine maps, so any
    discrepancy beyond the rule's own conditioning is a bug in the wiring,
    not an approximation. This is the load-bearing test.
    """
    reference = kalman_filter(_A, _H, _Q, _R, _YS, _M0, _P0)
    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
        joseph=joseph,
    )

    assert tree_allclose(result.filtered_means, reference.filtered_means, atol=atol)
    assert tree_allclose(result.filtered_covs, reference.filtered_covs, atol=atol)
    assert tree_allclose(result.predicted_means, reference.predicted_means, atol=atol)
    assert tree_allclose(result.predicted_covs, reference.predicted_covs, atol=atol)
    assert tree_allclose(result.log_likelihood, reference.log_likelihood, atol=atol)


@pytest.mark.parametrize(("integrator", "atol"), _INTEGRATORS)
def test_cross_covariance_contract(integrator, atol):
    """Every supported integrator supplies the cross-covariance the gain needs."""
    del atol
    state = GaussianState(
        mean=_M0, cov=lx.MatrixLinearOperator(_P0, lx.positive_semidefinite_tag)
    )
    result = integrator.integrate(_linear_obs, state)

    assert result.cross_cov is not None
    assert result.cross_cov.shape == (_N, _M)


def test_integrator_without_cross_covariance_is_rejected():
    """A missing cross-covariance is a loud failure, not a silent zero gain."""

    class _NoCrossCov(AbstractIntegrator):
        def integrate(self, fn, state):
            values = jax.vmap(fn)(state.mean[None, :])
            cov = lx.MatrixLinearOperator(
                jnp.eye(values.shape[-1]), lx.positive_semidefinite_tag
            )
            return PropagationResult(
                state=GaussianState(mean=values[0], cov=cov), cross_cov=None
            )

    with pytest.raises(TypeError, match="cross_cov=None"):
        nonlinear_kalman_filter(
            _linear_dynamics,
            _linear_obs,
            _Q,
            _R,
            _YS,
            _M0,
            _P0,
            integrator=_NoCrossCov(),
        )


def test_default_integrator_is_unscented():
    """Omitting the integrator gives the derivative-free default.

    The default is ``alpha=1.0``, not `UnscentedIntegrator`'s own
    ``alpha=1e-3``: see ``test_default_is_float32_safe``.
    """
    explicit = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=UnscentedIntegrator(alpha=1.0),
    )
    default = nonlinear_kalman_filter(
        _linear_dynamics, _linear_obs, _Q, _R, _YS, _M0, _P0
    )

    assert tree_allclose(default.filtered_means, explicit.filtered_means, atol=1e-14)


def test_time_varying_noise():
    """``(T, N, N)`` noise stacks are accepted, as in ``kalman_filter``."""
    Q_seq = (
        jnp.broadcast_to(_Q, (_T, _N, _N)) * jnp.linspace(0.5, 2.0, _T)[:, None, None]
    )
    R_seq = jnp.broadcast_to(_R, (_T, _M, _M))

    reference = kalman_filter(_A, _H, Q_seq, R_seq, _YS, _M0, _P0)
    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        Q_seq,
        R_seq,
        _YS,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
    )

    assert tree_allclose(result.filtered_means, reference.filtered_means, atol=1e-10)
    assert tree_allclose(result.log_likelihood, reference.log_likelihood, atol=1e-10)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        jnp.array([True] * 6 + [False] * 6),
        jnp.stack([jnp.array([True, False])] * _T),
        jnp.stack([jnp.array([True, True])] * _T),
    ],
    ids=["step-gate", "channel-gate", "channel-all-true"],
)
def test_masks_match_the_linear_filter(mask):
    """Both mask forms behave exactly as ``kalman_filter``'s."""
    reference = kalman_filter(_A, _H, _Q, _R, _YS, _M0, _P0, mask=mask)
    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
        mask=mask,
    )

    assert tree_allclose(result.filtered_means, reference.filtered_means, atol=1e-10)
    assert tree_allclose(result.filtered_covs, reference.filtered_covs, atol=1e-10)
    assert tree_allclose(result.log_likelihood, reference.log_likelihood, atol=1e-10)


def test_fully_masked_step_leaves_state_at_prediction():
    """A gated-off step contributes nothing and does not move the state."""
    mask = jnp.ones((_T,), dtype=bool).at[5].set(False)
    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
        mask=mask,
    )

    assert tree_allclose(result.filtered_means[5], result.predicted_means[5], atol=0.0)
    assert tree_allclose(result.filtered_covs[5], result.predicted_covs[5], atol=0.0)


def test_masked_channels_may_be_nan():
    """Masked entries are never read, so they may carry the usual NaN sentinel."""
    observations = _YS.at[:, 1].set(jnp.nan)
    mask = jnp.stack([jnp.array([True, False])] * _T)

    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        observations,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
        mask=mask,
    )
    reference = kalman_filter(_A, _H, _Q, _R, observations, _M0, _P0, mask=mask)

    assert bool(jnp.all(jnp.isfinite(result.filtered_means)))
    assert tree_allclose(result.log_likelihood, reference.log_likelihood, atol=1e-10)


@pytest.mark.parametrize(
    "bad", [jnp.ones((_T + 1,), bool), jnp.ones((_T, _M + 1), bool)]
)
def test_bad_mask_shape_is_rejected(bad):
    """A misshapen mask fails before the scan, with a clear message."""
    with pytest.raises(ValueError, match="mask must be"):
        nonlinear_kalman_filter(
            _linear_dynamics, _linear_obs, _Q, _R, _YS, _M0, _P0, mask=bad
        )


# --------------------------------------------------------------------------
# Joseph form
# --------------------------------------------------------------------------


def _nonlinear_dynamics(x):
    return 0.9 * x + 0.2 * jnp.sin(x)


def _nonlinear_obs(x):
    return jnp.array([jnp.tanh(2 * x[0] + x[2]), jnp.sin(1.5 * x[1])])


def test_joseph_coincides_with_standard_form_for_affine_maps():
    """For affine maps the statistical-linearisation residual vanishes.

    The two covariance updates differ by ``K Omega K^T``; ``Omega`` is zero
    for an affine ``obs_fn``, so the Joseph default cannot change the
    linear-reduction result.
    """
    joseph = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
        joseph=True,
    )
    standard = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=FifthOrderCubatureIntegrator(),
        joseph=False,
    )

    assert tree_allclose(joseph.filtered_covs, standard.filtered_covs, atol=1e-12)


def test_joseph_agrees_with_the_standard_form_on_a_nonlinear_step():
    """Joseph reproduces the matched-joint posterior, it does not shrink it.

    Joseph form linearises ``obs_fn`` as ``H_eff x + b + eps``, so its
    effective noise is ``R + Omega``, not ``R``. Passing ``R`` alone would
    return ``standard - K Omega K^T`` -- systematically overconfident on
    nonlinear maps. Checked on a single step so the two runs cannot
    diverge through the carry.
    """
    from gaussx import statistical_linear_regression

    integrator = FifthOrderCubatureIntegrator()
    observations = jnp.array([[0.3, -0.4]])
    kwargs = {
        "process_noise": _Q,
        "obs_noise": _R,
        "observations": observations,
        "init_mean": _M0,
        "init_cov": _P0,
        "integrator": integrator,
    }
    joseph = nonlinear_kalman_filter(
        _nonlinear_dynamics, _nonlinear_obs, joseph=True, **kwargs
    )
    standard = nonlinear_kalman_filter(
        _nonlinear_dynamics, _nonlinear_obs, joseph=False, **kwargs
    )
    # The two covariance updates now agree analytically.
    assert tree_allclose(joseph.filtered_covs[0], standard.filtered_covs[0], atol=1e-12)

    # And the residual they used to differ by is genuinely nonzero here, so
    # the agreement above is a real check rather than a vacuous one.
    predicted = GaussianState(
        mean=joseph.predicted_means[0],
        cov=lx.MatrixLinearOperator(joseph.predicted_covs[0], lx.symmetric_tag),
    )
    slr = statistical_linear_regression(
        _nonlinear_obs, lambda f: _R, predicted, integrator
    )
    residual_cov = slr.omega - _R
    assert float(jnp.linalg.eigvalsh(residual_cov).min()) > 1e-6


def test_joseph_keeps_covariances_psd_on_an_ill_conditioned_problem():
    """Joseph form stays PSD where the gain is only approximately optimal."""
    steps = 30
    process_noise = 1e-6 * jnp.eye(2)
    obs_noise = jnp.array([[1e-3]])
    init_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])  # near-singular

    def dynamics(x):
        return jnp.array([x[0] + 0.1 * jnp.sin(3 * x[1]), 0.98 * x[1]])

    def obs_fn(x):
        return jnp.array([jnp.tanh(4.0 * x[0]) * jnp.exp(-0.5 * x[1] ** 2)])

    observations = 0.5 * jnp.sin(jnp.arange(steps, dtype=float))[:, None]

    result = nonlinear_kalman_filter(
        dynamics,
        obs_fn,
        process_noise,
        obs_noise,
        observations,
        jnp.array([1.0, 0.3]),
        init_cov,
        integrator=UnscentedIntegrator(alpha=1.0),
        joseph=True,
    )

    eigenvalues = jnp.linalg.eigvalsh(result.filtered_covs)
    assert float(eigenvalues.min()) > 0.0
    assert bool(jnp.all(jnp.isfinite(result.log_likelihood)))


# --------------------------------------------------------------------------
# Smoother
# --------------------------------------------------------------------------


def test_smoother_linear_reduction():
    """With affine dynamics the smoother reproduces ``rts_smoother``."""
    integrator = FifthOrderCubatureIntegrator()
    reference_filter = kalman_filter(_A, _H, _Q, _R, _YS, _M0, _P0)
    ref_means, ref_covs = rts_smoother(reference_filter, _A, _Q)

    filtered = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
    )
    means, covs = nonlinear_rts_smoother(
        filtered, _linear_dynamics, integrator=integrator
    )

    assert tree_allclose(means, ref_means, atol=1e-10)
    assert tree_allclose(covs, ref_covs, atol=1e-10)


def test_smoothed_variances_do_not_exceed_filtered():
    """Conditioning on the future cannot increase marginal variance."""
    integrator = FifthOrderCubatureIntegrator()
    filtered = nonlinear_kalman_filter(
        _nonlinear_dynamics,
        _nonlinear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
    )
    _, covs = nonlinear_rts_smoother(
        filtered, _nonlinear_dynamics, integrator=integrator
    )

    smoothed_var = jnp.diagonal(covs, axis1=1, axis2=2)
    filtered_var = jnp.diagonal(filtered.filtered_covs, axis1=1, axis2=2)
    assert bool(jnp.all(smoothed_var <= filtered_var + 1e-10))


def test_smoother_accepts_process_noise_for_symmetry():
    """``process_noise`` is accepted and ignored, as in ``rts_smoother``."""
    integrator = FifthOrderCubatureIntegrator()
    filtered = nonlinear_kalman_filter(
        _linear_dynamics, _linear_obs, _Q, _R, _YS, _M0, _P0, integrator=integrator
    )

    without = nonlinear_rts_smoother(filtered, _linear_dynamics, integrator=integrator)
    with_noise = nonlinear_rts_smoother(
        filtered, _linear_dynamics, _Q, integrator=integrator
    )

    assert tree_allclose(without[0], with_noise[0], atol=0.0)


# --------------------------------------------------------------------------
# Nonlinear behaviour, tracing, gradients
# --------------------------------------------------------------------------


def test_rules_are_ordered_by_polynomial_exactness_on_one_step():
    """On a nonlinear update the rules order by their degree of exactness.

    Tested on a *single* step: over a trajectory the filters diverge
    through the carry, so a whole-run comparison measures compounding
    rather than the quality of the moment transform. A high-order
    Gauss-Hermite rule is the converged reference (order 11 and 15 agree
    to 1e-7 here), and the degree-5 cubature rule must land closer to it
    than the degree-3 unscented rule.
    """
    single_step = {
        "process_noise": 0.02 * jnp.eye(_N),
        "obs_noise": jnp.diag(jnp.array([0.1, 0.15])),
        "observations": jnp.array([[0.3, -0.25]]),
        "init_mean": _M0,
        "init_cov": 0.1 * jnp.eye(_N),
    }

    def run(integrator):
        return nonlinear_kalman_filter(
            _nonlinear_dynamics, _nonlinear_obs, integrator=integrator, **single_step
        ).filtered_means

    reference = run(GaussHermiteIntegrator(order=15))
    coarse_reference = run(GaussHermiteIntegrator(order=11))
    cubature = run(FifthOrderCubatureIntegrator())
    unscented = run(UnscentedIntegrator(alpha=1.0))

    # The reference is converged, so the comparisons below are meaningful.
    assert tree_allclose(coarse_reference, reference, atol=1e-5)

    cubature_error = jnp.abs(cubature - reference).max()
    unscented_error = jnp.abs(unscented - reference).max()

    assert float(cubature_error) < 1e-2
    assert float(cubature_error) < float(unscented_error)


def test_filter_beats_the_prior_on_a_nonlinear_tracking_problem():
    """The filter actually uses the observations."""
    steps = 40

    def dynamics(x):
        return jnp.array([0.95 * x[0] + 0.1 * jnp.sin(x[1]), 0.9 * x[1]])

    def obs_fn(x):
        return jnp.array([jnp.tanh(x[0])])

    key = jax.random.key(7)
    truth = [jnp.array([1.5, 0.8])]
    for _ in range(steps - 1):
        truth.append(dynamics(truth[-1]))
    states = jnp.stack(truth)
    observations = jax.vmap(obs_fn)(states) + 0.02 * jax.random.normal(key, (steps, 1))

    result = nonlinear_kalman_filter(
        dynamics,
        obs_fn,
        1e-4 * jnp.eye(2),
        jnp.array([[4e-4]]),
        observations,
        jnp.array([0.0, 0.8]),  # wrong initial x0
        jnp.eye(2),
        integrator=UnscentedIntegrator(alpha=1.0),
    )

    filtered_error = jnp.abs(result.filtered_means[-1, 0] - states[-1, 0])
    prior_error = jnp.abs(result.predicted_means[0, 0] - states[0, 0])
    assert float(filtered_error) < float(prior_error)
    assert bool(jnp.isfinite(result.log_likelihood))


@pytest.mark.parametrize(
    "integrator",
    [TaylorIntegrator(), UnscentedIntegrator(alpha=1.0)],
    ids=["taylor", "unscented"],
)
def test_gradient_through_closed_over_parameters(integrator):
    """``jax.grad`` of the log-likelihood w.r.t. dynamics parameters is finite."""

    def negative_log_likelihood(decay):
        def dynamics(x):
            return decay * x + 0.2 * jnp.sin(x)

        result = nonlinear_kalman_filter(
            dynamics,
            _nonlinear_obs,
            _Q,
            _R,
            _YS,
            _M0,
            _P0,
            integrator=integrator,
        )
        return -result.log_likelihood

    grad = jax.grad(negative_log_likelihood)(jnp.asarray(0.9))
    assert bool(jnp.isfinite(grad))
    assert float(jnp.abs(grad)) > 0.0


def test_jit():
    """The filter traces under ``eqx.filter_jit`` with the integrator static."""
    integrator = FifthOrderCubatureIntegrator()

    @eqx.filter_jit
    def run(observations):
        return nonlinear_kalman_filter(
            _nonlinear_dynamics,
            _nonlinear_obs,
            _Q,
            _R,
            observations,
            _M0,
            _P0,
            integrator=integrator,
        )

    jitted = run(_YS)
    eager = nonlinear_kalman_filter(
        _nonlinear_dynamics,
        _nonlinear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
    )

    assert tree_allclose(jitted.filtered_means, eager.filtered_means, atol=1e-12)
    assert bool(jnp.all(jnp.isfinite(jitted.log_likelihood)))


# --------------------------------------------------------------------------
# The functional per-step API
# --------------------------------------------------------------------------


def test_manual_loop_reproduces_the_filter():
    """The wrapper is exactly a scan over the public per-step functions.

    This is the contract that makes the step functions usable on their
    own: someone driving their own loop must get bit-for-bit what the
    wrapper produces.
    """
    from gaussx import nonlinear_kalman_predict, nonlinear_kalman_update

    integrator = FifthOrderCubatureIntegrator()
    wrapped = nonlinear_kalman_filter(
        _nonlinear_dynamics,
        _nonlinear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
    )

    mean, cov = _M0, _P0
    total_ll = jnp.zeros(())
    means, covs = [], []
    for t in range(_T):
        mean_pred, cov_pred = nonlinear_kalman_predict(
            _nonlinear_dynamics, mean, cov, _Q, integrator=integrator
        )
        mean, cov, ll_inc = nonlinear_kalman_update(
            _nonlinear_obs,
            mean_pred,
            cov_pred,
            _YS[t],
            _R,
            integrator=integrator,
        )
        total_ll = total_ll + ll_inc
        means.append(mean)
        covs.append(cov)

    assert tree_allclose(jnp.stack(means), wrapped.filtered_means, atol=0.0)
    assert tree_allclose(jnp.stack(covs), wrapped.filtered_covs, atol=0.0)
    assert tree_allclose(total_ll, wrapped.log_likelihood, atol=1e-12)


def test_manual_backward_loop_reproduces_the_smoother():
    """Likewise for ``nonlinear_rts_step``."""
    from gaussx import nonlinear_rts_step

    integrator = FifthOrderCubatureIntegrator()
    filtered = nonlinear_kalman_filter(
        _nonlinear_dynamics,
        _nonlinear_obs,
        _Q,
        _R,
        _YS,
        _M0,
        _P0,
        integrator=integrator,
    )
    ref_means, ref_covs = nonlinear_rts_smoother(
        filtered, _nonlinear_dynamics, integrator=integrator
    )

    mean = filtered.filtered_means[-1]
    cov = filtered.filtered_covs[-1]
    means = [mean]
    covs = [cov]
    for t in range(_T - 2, -1, -1):
        mean, cov = nonlinear_rts_step(
            _nonlinear_dynamics,
            filtered.filtered_means[t],
            filtered.filtered_covs[t],
            filtered.predicted_means[t + 1],
            filtered.predicted_covs[t + 1],
            mean,
            cov,
            integrator=integrator,
        )
        means.insert(0, mean)
        covs.insert(0, cov)

    assert tree_allclose(jnp.stack(means), ref_means, atol=0.0)
    assert tree_allclose(jnp.stack(covs), ref_covs, atol=0.0)


def test_step_functions_default_to_the_unscented_rule():
    """Each step function stands alone, integrator included."""
    from gaussx import nonlinear_kalman_predict

    explicit = nonlinear_kalman_predict(
        _nonlinear_dynamics, _M0, _P0, _Q, integrator=UnscentedIntegrator(alpha=1.0)
    )
    default = nonlinear_kalman_predict(_nonlinear_dynamics, _M0, _P0, _Q)

    assert tree_allclose(default[0], explicit[0], atol=0.0)


def test_update_step_honours_a_channel_mask():
    """The standalone update marginalises masked channels the same way."""
    from gaussx import nonlinear_kalman_update

    integrator = FifthOrderCubatureIntegrator()
    mask = jnp.array([True, False])

    masked = nonlinear_kalman_update(
        _linear_obs,
        _M0,
        _P0,
        jnp.array([0.3, jnp.nan]),
        _R,
        integrator=integrator,
        mask=mask,
    )

    # Equivalent to dropping the second channel outright.
    dropped = nonlinear_kalman_update(
        lambda x: _linear_obs(x)[:1],
        _M0,
        _P0,
        jnp.array([0.3]),
        _R[:1, :1],
        integrator=integrator,
    )

    assert tree_allclose(masked[0], dropped[0], atol=1e-10)
    assert tree_allclose(masked[1], dropped[1], atol=1e-10)
    assert tree_allclose(masked[2], dropped[2], atol=1e-10)


def test_masked_moment_inputs_makes_channels_inert():
    """The public mask substitution zeroes exactly the right blocks."""
    from gaussx import masked_moment_inputs

    obs_cov = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    cross_cov = jnp.arange(6.0).reshape(3, 2)
    obs_noise = jnp.array([[0.4, 0.1], [0.1, 0.6]])
    y = jnp.array([1.0, jnp.nan])
    y_hat = jnp.array([0.25, 7.0])
    mask = jnp.array([True, False])

    cov_e, cross_e, noise_e, residual, n_missing = masked_moment_inputs(
        obs_cov, cross_cov, obs_noise, y, y_hat, mask
    )

    # Masked row and column of the matched covariance are cleared, and the
    # noise picks up a unit block there instead.
    assert tree_allclose(cov_e, jnp.array([[2.0, 0.0], [0.0, 0.0]]), atol=0.0)
    assert tree_allclose(noise_e, jnp.array([[0.4, 0.0], [0.0, 1.0]]), atol=0.0)

    # Masked column of the cross-covariance is cleared, so the gain cannot
    # move the state through that channel.
    assert tree_allclose(cross_e[:, 0], cross_cov[:, 0], atol=0.0)
    assert tree_allclose(cross_e[:, 1], jnp.zeros(3), atol=0.0)

    # The NaN never reaches the residual.
    assert tree_allclose(residual, jnp.array([0.75, 0.0]), atol=0.0)
    assert float(n_missing) == 1.0


def test_masked_moment_inputs_is_the_identity_when_nothing_is_masked():
    """An all-True mask must not perturb the moments."""
    from gaussx import masked_moment_inputs

    obs_cov = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    cross_cov = jnp.arange(6.0).reshape(3, 2)
    obs_noise = jnp.array([[0.4, 0.1], [0.1, 0.6]])
    y = jnp.array([1.0, -0.5])
    y_hat = jnp.array([0.25, 0.5])

    cov_e, cross_e, noise_e, residual, n_missing = masked_moment_inputs(
        obs_cov, cross_cov, obs_noise, y, y_hat, jnp.array([True, True])
    )

    assert tree_allclose(cov_e, obs_cov, atol=0.0)
    assert tree_allclose(cross_e, cross_cov, atol=0.0)
    assert tree_allclose(noise_e, obs_noise, atol=0.0)
    assert tree_allclose(residual, y - y_hat, atol=0.0)
    assert float(n_missing) == 0.0


def test_masked_moment_inputs_gradient_is_nan_free():
    """A NaN in a masked observation must not poison the gradient."""
    from gaussx import masked_moment_inputs

    obs_cov = jnp.eye(2)
    cross_cov = jnp.ones((3, 2))
    obs_noise = jnp.eye(2)
    mask = jnp.array([True, False])

    def loss(y_hat):
        y = jnp.array([1.0, jnp.nan])
        _, _, _, residual, _ = masked_moment_inputs(
            obs_cov, cross_cov, obs_noise, y, y_hat, mask
        )
        return jnp.sum(residual**2)

    grad = jax.grad(loss)(jnp.array([0.25, 7.0]))
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize(
    ("noise", "name"),
    [
        (jnp.ones((1, 1)), "process_noise"),
        (jnp.ones((_N, 1)), "process_noise"),
        (jnp.ones((_N + 1, _N + 1)), "process_noise"),
    ],
)
def test_misshapen_process_noise_is_rejected(noise, name):
    """A misshapen covariance must not broadcast across the whole matrix.

    A ``(1, 1)`` process noise with ``N = 3`` would otherwise be added to
    every entry of ``Cov[f(x)]``, corrupting the filter silently.
    """
    with pytest.raises(ValueError, match=name):
        nonlinear_kalman_filter(_linear_dynamics, _linear_obs, noise, _R, _YS, _M0, _P0)


@pytest.mark.parametrize("noise", [jnp.ones((1, 1)), jnp.ones((_M + 1, _M + 1))])
def test_misshapen_obs_noise_is_rejected(noise):
    """Likewise for the observation noise."""
    with pytest.raises(ValueError, match="obs_noise"):
        nonlinear_kalman_filter(_linear_dynamics, _linear_obs, _Q, noise, _YS, _M0, _P0)


def test_indefinite_innovation_is_rejected():
    """An indefinite innovation must be rejected, not quietly used.

    ``Cov[h(x)]`` comes from a quadrature rule; a negative-weight rule can
    return it indefinite, and ``R`` may be too small to repair it. Neither
    the gain nor the likelihood is defined then -- the quadratic form can
    go negative and the log-determinant becomes ``log|det S|`` -- so the
    filter must say so rather than emit a plausible-looking number.
    """

    class _IndefiniteIntegrator(AbstractIntegrator):
        """Returns a deliberately indefinite output covariance."""

        def integrate(self, fn, state):
            mean = fn(state.mean)
            dim = mean.shape[-1]
            indefinite = jnp.diag(
                jnp.array([1.0, -0.5][:dim] + [1.0] * max(0, dim - 2))
            )
            return PropagationResult(
                state=GaussianState(
                    mean=mean,
                    cov=lx.MatrixLinearOperator(indefinite, lx.symmetric_tag),
                ),
                cross_cov=jnp.ones((state.mean.shape[-1], dim)) * 0.1,
            )

    # Exercised through the update step directly: driven through the whole
    # filter, the predict guard would catch this stub's dynamics covariance
    # first, and it is the innovation check under test here.
    from gaussx import nonlinear_kalman_update

    with pytest.raises(Exception, match="not positive definite"):
        jax.block_until_ready(
            nonlinear_kalman_update(
                _linear_obs,
                _M0,
                _P0,
                _YS[0],
                1e-3 * jnp.eye(_M),  # too small to rescue the indefinite block
                integrator=_IndefiniteIntegrator(),
                joseph=True,
            )
        )


def test_default_is_the_float32_safe_unscented_rule():
    """The default must be ``alpha=1.0``, not the classic ``alpha=1e-3``.

    `UnscentedIntegrator`'s own default places the sigma points ~1e-3 from
    the mean and recovers the moments by cancellation. Under x64 that shows
    up only as the looser tolerance this file gives ``unscented-default``;
    in float32 -- JAX's default, and the configuration this suite cannot
    reach because ``conftest`` enables x64 globally -- it misplaces the
    log-likelihood of a *linear* problem by over one nat, where
    ``alpha=1.0`` stays at 4e-6.

    Pinning the default here is what keeps that from silently regressing.
    """
    common = (_linear_dynamics, _linear_obs, _Q, _R, _YS, _M0, _P0)

    default = nonlinear_kalman_filter(*common)
    safe = nonlinear_kalman_filter(*common, integrator=UnscentedIntegrator(alpha=1.0))
    classic = nonlinear_kalman_filter(*common, integrator=UnscentedIntegrator())

    assert tree_allclose(default.filtered_means, safe.filtered_means, atol=0.0)
    assert tree_allclose(default.log_likelihood, safe.log_likelihood, atol=0.0)

    # And the classic rule really is a different, worse-conditioned answer.
    # rtol=0: on a log-likelihood of order -40, allclose's default
    # rtol=1e-5 would tolerate 4e-4 and hide the difference entirely.
    assert not bool(
        jnp.allclose(
            default.log_likelihood, classic.log_likelihood, atol=1e-12, rtol=0.0
        )
    )

    reference = kalman_filter(_A, _H, _Q, _R, _YS, _M0, _P0)
    default_error = jnp.abs(default.log_likelihood - reference.log_likelihood)
    classic_error = jnp.abs(classic.log_likelihood - reference.log_likelihood)
    assert float(default_error) < float(classic_error)


@pytest.mark.parametrize(
    ("init_cov", "process_noise", "label"),
    [
        (jnp.zeros((_N, _N)), jnp.zeros((_N, _N)), "deterministic-init"),
        (jnp.diag(jnp.array([1.0, 1.0, 0.0])), jnp.zeros((_N, _N)), "rank-deficient"),
    ],
)
def test_singular_predicted_covariance_is_handled(init_cov, process_noise, label):
    """A singular ``P^-`` is a valid belief, not a failure.

    A deterministic initial state or zero process noise leaves ``P^-``
    singular while the update stays well defined, because ``R`` keeps the
    innovation invertible. Joseph form needs ``C^T (P^-)^-1``, so it must
    use a least-squares solve rather than a well-posed one, which would
    return NaN here — and Joseph is the default.
    """
    del label
    result = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        process_noise,
        _R,
        _YS,
        _M0,
        init_cov,
        integrator=CubatureIntegrator(),
        joseph=True,
    )
    standard = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        process_noise,
        _R,
        _YS,
        _M0,
        init_cov,
        integrator=CubatureIntegrator(),
        joseph=False,
    )

    assert bool(jnp.all(jnp.isfinite(result.filtered_means)))
    assert bool(jnp.all(jnp.isfinite(result.filtered_covs)))
    assert bool(jnp.isfinite(result.log_likelihood))
    # The two covariance forms agree analytically, singular P^- included.
    assert tree_allclose(result.filtered_covs, standard.filtered_covs, atol=1e-10)


def test_smoother_handles_a_singular_predicted_covariance():
    """A deterministic process must not fail once it reaches the smoother.

    The RTS gain is ``Cov[x, f(x)] (P^-)^-1``; with zero process noise
    ``P^-`` is singular while the correction stays well defined on the
    supported subspace, so the gain needs a least-squares solve too. A
    filter run that survives a singular covariance would otherwise fail at
    the backward pass.
    """
    integrator = CubatureIntegrator()
    zero_noise = jnp.zeros((_N, _N))

    filtered = nonlinear_kalman_filter(
        _linear_dynamics,
        _linear_obs,
        zero_noise,
        _R,
        _YS,
        _M0,
        jnp.diag(jnp.array([1.0, 1.0, 0.0])),
        integrator=integrator,
    )
    means, covs = nonlinear_rts_smoother(
        filtered, _linear_dynamics, integrator=integrator
    )

    assert bool(jnp.all(jnp.isfinite(means)))
    assert bool(jnp.all(jnp.isfinite(covs)))


def test_linearisation_keeps_small_covariance_modes():
    """The rank policy must not discard representable low-variance modes.

    ``lstsq``'s default cutoff scales with the dtype epsilon, so in float32
    a mode at 1e-8 relative to 1 is thrown away -- and the Joseph update
    then *grows* that variance instead of shrinking it. ``rcond=0.0``
    discards only exactly-zero singular values.

    Checked directly on the solve, because the suite runs in x64 where the
    default cutoff would not bite.
    """
    cov = jnp.diag(jnp.array([1.0, 1e-8], dtype=jnp.float32))
    cross = jnp.array([[0.0], [1e-8]], dtype=jnp.float32)

    kept = jnp.linalg.lstsq(cov, cross, rcond=0.0)[0].T
    discarded = jnp.linalg.lstsq(cov, cross)[0].T

    # C^T P^-1 = [0, 1] here.
    assert tree_allclose(kept, jnp.array([[0.0, 1.0]], dtype=jnp.float32), atol=1e-5)
    # The default policy loses it entirely -- this is what we avoid.
    assert float(jnp.abs(discarded).max()) == 0.0

    # And an exactly singular covariance still gives the min-norm answer.
    singular = jnp.linalg.lstsq(jnp.zeros((2, 2)), jnp.zeros((2, 1)), rcond=0.0)[0].T
    assert tree_allclose(singular, jnp.zeros((1, 2)), atol=0.0)


def test_inconsistent_matched_joint_is_rejected():
    """A positive-definite S is not enough; the joint must be consistent.

    With ``P = 1``, ``S_yy = 1``, ``C = 2`` the residual
    ``Omega = S_yy - C^T P^-1 C`` is negative while ``S = S_yy + R`` stays
    positive definite, so the innovation check passes and the posterior
    variance still goes negative.
    """

    class _InconsistentIntegrator(AbstractIntegrator):
        def integrate(self, fn, state):
            dim_in = state.mean.shape[-1]
            mean = fn(state.mean)
            dim_out = mean.shape[-1]
            return PropagationResult(
                state=GaussianState(
                    mean=mean,
                    cov=lx.MatrixLinearOperator(jnp.eye(dim_out), lx.symmetric_tag),
                ),
                # Far too large to be a valid cross-covariance for this pair.
                cross_cov=2.0 * jnp.ones((dim_in, dim_out)),
            )

    with pytest.raises(Exception, match="not positive semi-definite"):
        jax.block_until_ready(
            nonlinear_kalman_filter(
                lambda x: x,
                lambda x: x[:1],
                0.1 * jnp.eye(1),
                jnp.array([[0.1]]),
                jnp.zeros((3, 1)),
                jnp.zeros(1),
                jnp.eye(1),
                integrator=_InconsistentIntegrator(),
            )
        )


def test_indefinite_posterior_with_positive_variances_is_rejected():
    """The PSD guard checks the spectrum, not just the variances.

    An indefinite covariance can have entirely positive diagonal entries --
    ``[[0.36, -0.64], [-0.64, 0.36]]`` has smallest eigenvalue ``-0.28`` --
    so a diagonal-only test would pass it and hand the next predict step a
    covariance it treats as PSD.
    """

    class _CorrelatedInconsistent(AbstractIntegrator):
        """Cross-covariance too correlated to be consistent with P and S."""

        def integrate(self, fn, state):
            mean = fn(state.mean)
            dim_out = mean.shape[-1]
            return PropagationResult(
                state=GaussianState(
                    mean=mean,
                    cov=lx.MatrixLinearOperator(jnp.eye(dim_out), lx.symmetric_tag),
                ),
                cross_cov=jnp.array([[0.8, 0.0], [0.8, 0.0]]),
            )

    with pytest.raises(Exception, match="not positive semi-definite"):
        jax.block_until_ready(
            nonlinear_kalman_filter(
                lambda x: x,
                lambda x: x,
                jnp.zeros((2, 2)),
                jnp.zeros((2, 2)),
                jnp.zeros((2, 2)),
                jnp.zeros(2),
                jnp.eye(2),
                integrator=_CorrelatedInconsistent(),
            )
        )


@pytest.mark.parametrize("bad", [jnp.ones((1, 1)), jnp.ones((_N, 1))])
def test_predict_step_rejects_misshapen_process_noise(bad):
    """The public step functions validate shapes, not just the wrapper.

    These are advertised for custom loops, so they cannot rely on
    ``nonlinear_kalman_filter`` having checked first: a ``(1, 1)`` noise
    would broadcast across the whole covariance and corrupt the prediction
    silently.
    """
    from gaussx import nonlinear_kalman_predict

    with pytest.raises(ValueError, match="process_noise must have shape"):
        nonlinear_kalman_predict(_linear_dynamics, _M0, _P0, bad)


@pytest.mark.parametrize("bad", [jnp.ones((1, 1)), jnp.ones((_M + 1, _M + 1))])
def test_update_step_rejects_misshapen_obs_noise(bad):
    """Likewise for the observation noise on the standalone update."""
    from gaussx import nonlinear_kalman_update

    with pytest.raises(ValueError, match="obs_noise must have shape"):
        nonlinear_kalman_update(_linear_obs, _M0, _P0, _YS[0], bad)


def test_predicted_covariance_is_validated():
    """An indefinite prediction is rejected, not passed on.

    A step-level ``False`` mask would expose it directly as the filtered
    covariance, and nothing downstream would object: the next moment
    transform tags it PSD and the dense square-root path clips negative
    eigenvalues to zero, quietly altering the belief.
    """
    from gaussx import nonlinear_kalman_predict

    class _IndefiniteDynamics(AbstractIntegrator):
        def integrate(self, fn, state):
            dim = state.mean.shape[-1]
            return PropagationResult(
                state=GaussianState(
                    mean=fn(state.mean),
                    cov=lx.MatrixLinearOperator(
                        jnp.diag(jnp.array([1.0, 1.0, -0.5])), lx.symmetric_tag
                    ),
                ),
                cross_cov=jnp.zeros((dim, dim)),
            )

    with pytest.raises(Exception, match="not positive semi-definite"):
        jax.block_until_ready(
            nonlinear_kalman_predict(
                _linear_dynamics,
                _M0,
                _P0,
                jnp.zeros((_N, _N)),
                integrator=_IndefiniteDynamics(),
            )
        )


@pytest.mark.parametrize("bad", [jnp.array([False]), jnp.ones((_M + 1,), bool)])
def test_update_step_rejects_a_misshapen_mask(bad):
    """A per-step mask must be exactly ``(M,)``.

    A ``(1,)`` mask broadcasts across every channel: ``False`` suppresses
    them all and returns an identity update, ``True`` enables them all —
    either way silently, which is precisely what a custom loop cannot
    afford.
    """
    from gaussx import nonlinear_kalman_update

    with pytest.raises(ValueError, match="mask must have shape"):
        nonlinear_kalman_update(_linear_obs, _M0, _P0, _YS[0], _R, mask=bad)


def test_smoothed_covariance_is_validated():
    """An indefinite smoothed covariance is rejected, as in predict/update.

    The filtered and predicted covariances can each be PSD while an
    inconsistent cross-covariance still drives the correction indefinite.
    """
    from gaussx import nonlinear_rts_step

    class _InconsistentDynamics(AbstractIntegrator):
        def integrate(self, fn, state):
            dim = state.mean.shape[-1]
            return PropagationResult(
                state=GaussianState(
                    mean=fn(state.mean),
                    cov=lx.MatrixLinearOperator(jnp.eye(dim), lx.symmetric_tag),
                ),
                cross_cov=2.0 * jnp.eye(dim),
            )

    with pytest.raises(Exception, match="not positive semi-definite"):
        jax.block_until_ready(
            nonlinear_rts_step(
                lambda x: x,
                jnp.zeros(1),
                jnp.eye(1),
                jnp.zeros(1),
                jnp.eye(1),
                jnp.zeros(1),
                0.1 * jnp.eye(1),
                integrator=_InconsistentDynamics(),
            )
        )
