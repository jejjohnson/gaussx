"""Tests for matrix-fraction discretisation of linear SDEs."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
import scipy.linalg

from gaussx import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    discretise_mfd,
    discretise_mfd_sequence,
)
from gaussx._testing import tree_allclose


# An undamped oscillator: eigenvalues +/- i*omega, so lambda + conj(lambda) = 0
# and the Lyapunov equation is singular. This is the case the function exists
# for -- CosineSDE has exactly this drift.
_OMEGA = 1.4
_OSCILLATOR = jnp.array([[0.0, -_OMEGA], [_OMEGA, 0.0]])

# Unstable but non-degenerate: eigenvalues sum to 0.4, 0.6, 0.2 -- none zero.
_UNSTABLE = jnp.array([[0.3, 0.0], [0.0, 0.1]])


def _quadrature_Q(F, Q_c, dt, n_points=40_000):
    """Ground truth: trapezoid quadrature of the defining integral."""
    grid = jnp.linspace(0.0, dt, n_points)

    def integrand(s):
        expm_s = jsl.expm(F * s)
        return expm_s @ Q_c @ expm_s.T

    return jnp.trapezoid(jax.vmap(integrand)(grid), grid, axis=0)


def _lyapunov_Q(F, Q_c, dt):
    """The stationary route, via a Sylvester solve for ``P_inf``."""
    P = scipy.linalg.solve_sylvester(np.asarray(F), np.asarray(F).T, -np.asarray(Q_c))
    A = np.asarray(jsl.expm(F * dt))
    residual = np.abs(F @ P + P @ np.asarray(F).T + np.asarray(Q_c)).max()
    return jnp.asarray(P - A @ P @ A.T), float(residual)


@pytest.mark.parametrize("dt", [0.01, 0.5, 3.0])
def test_agrees_with_stationary_route_on_matern(dt):
    """Where both routes are valid they must agree to near machine precision."""
    kernel = MaternSDE(variance=jnp.asarray(1.3), lengthscale=jnp.asarray(0.8), order=1)
    params = kernel.sde_params()

    A_stat, Q_stat = kernel.discretise(jnp.asarray(dt))
    A_mfd, Q_mfd = discretise_mfd(
        params.F, params.L @ params.Q_c @ params.L.T, jnp.asarray(dt)
    )

    assert tree_allclose(A_mfd, A_stat, atol=1e-13)
    assert tree_allclose(Q_mfd, Q_stat, atol=1e-13)


def test_matches_quadrature_on_unstable_drift():
    """The defining integral is reproduced even when ``F`` is unstable.

    Instability is not the obstruction: the stationary identity holds here
    too, despite the Lyapunov solution not being a valid covariance.
    """
    dt = jnp.asarray(0.5)
    Q_c = jnp.eye(2)

    _, Q_mfd = discretise_mfd(_UNSTABLE, Q_c, dt)

    assert tree_allclose(Q_mfd, _quadrature_Q(_UNSTABLE, Q_c, dt), atol=1e-9)


def test_undamped_oscillator_where_lyapunov_fails():
    """The load-bearing case: MFD is exact where the Lyapunov route is not.

    For ``F = [[0, -w], [w, 0]]`` the eigenvalues are ``+/- i w``, so
    ``lambda_i + lambda_j = 0`` and the Sylvester system is singular. The
    assertion on the Lyapunov side is deliberate -- it documents why this
    function exists, and will fail loudly if that ever stops being true.
    """
    dt = jnp.asarray(0.5)
    Q_c = jnp.eye(2)
    truth = _quadrature_Q(_OSCILLATOR, Q_c, dt)

    _, Q_mfd = discretise_mfd(_OSCILLATOR, Q_c, dt)
    assert tree_allclose(Q_mfd, truth, atol=1e-9)

    # The stationary route is not merely less accurate here -- it is wrong.
    Q_lyap, residual = _lyapunov_Q(_OSCILLATOR, Q_c, dt)
    assert residual > 1e-3, "Sylvester system was expected to be singular"
    assert float(jnp.abs(Q_lyap - truth).max()) > 1e-2


# A *correlated* diffusion. The choice matters: with a diagonal ``F`` the
# singular Sylvester system splits elementwise as
# ``(lambda_i + lambda_j) P_ij = -Q_c[i, j]``, so the degenerate off-diagonal
# equation reads ``0 * P_01 = -Q_c[0, 1]``. That is inconsistent -- and so
# actually unsolvable -- only when ``Q_c[0, 1] != 0``. With ``Q_c = I`` the
# system is singular but consistent, the free off-diagonal never affects the
# answer, and the Lyapunov route returns the right result by luck.
_CORRELATED_Q_C = jnp.array([[1.0, 0.4], [0.4, 1.0]])


@pytest.mark.parametrize("lambda_2", [-0.29, -0.2999, -0.3])
def test_degeneracy_sweep(lambda_2):
    """MFD is unaffected as the eigenvalue pair approaches degeneracy.

    With ``lambda_1 = 0.3``, the Lyapunov route stays accurate while
    ``lambda_1 + lambda_2`` is merely small, and breaks at exactly
    ``-0.3``. MFD is indifferent throughout.
    """
    dt = jnp.asarray(0.5)
    Q_c = _CORRELATED_Q_C
    F = jnp.array([[0.3, 0.0], [0.0, lambda_2]])
    truth = _quadrature_Q(F, Q_c, dt)

    _, Q_mfd = discretise_mfd(F, Q_c, dt)
    assert tree_allclose(Q_mfd, truth, atol=1e-9)

    Q_lyap, residual = _lyapunov_Q(F, Q_c, dt)
    is_degenerate = lambda_2 == -0.3
    if is_degenerate:
        assert residual > 1e-3
        assert float(jnp.abs(Q_lyap - truth).max()) > 1e-2
    else:
        assert residual < 1e-6
        assert tree_allclose(Q_lyap, truth, atol=1e-9)


def test_diagonal_degeneracy_is_benign_for_uncorrelated_diffusion():
    """Singular does not always mean wrong -- and that is a trap.

    For ``F = diag(0.3, -0.3)`` with ``Q_c = I`` the Sylvester system is
    singular, yet consistent, so a solver may return a valid particular
    solution and the stationary route looks fine. Degeneracy of the
    eigenvalue pair is therefore not on its own a usable runtime check for
    whether the stationary route can be trusted, which is the argument for
    reaching for MFD whenever ``F`` is fitted rather than only when a
    diagnostic fires.
    """
    dt = jnp.asarray(0.5)
    F = jnp.array([[0.3, 0.0], [0.0, -0.3]])
    truth = _quadrature_Q(F, jnp.eye(2), dt)

    _, Q_mfd = discretise_mfd(F, jnp.eye(2), dt)
    Q_lyap, residual = _lyapunov_Q(F, jnp.eye(2), dt)

    assert tree_allclose(Q_mfd, truth, atol=1e-9)
    assert residual < 1e-9  # consistent despite being singular
    assert tree_allclose(Q_lyap, truth, atol=1e-9)


@pytest.mark.parametrize(
    "F", [_OSCILLATOR, _UNSTABLE, jnp.array([[-1.0, 0.5], [0.0, -2.0]])]
)
def test_process_noise_is_psd_and_symmetric(F):
    """``Q`` is a covariance: exactly symmetric and non-negative definite."""
    _, Q = discretise_mfd(F, jnp.eye(2), jnp.asarray(0.7))

    # Exactly, not approximately -- the symmetrise call is not optional,
    # since ``C @ A.T`` is asymmetric to floating point.
    assert jnp.array_equal(Q, Q.T)
    assert float(jnp.linalg.eigvalsh(Q).min()) >= -1e-12


def test_zero_timestep_is_the_identity():
    """A step of zero transitions nothing and accumulates no noise."""
    A, Q = discretise_mfd(_OSCILLATOR, jnp.eye(2), jnp.asarray(0.0))

    assert tree_allclose(A, jnp.eye(2), atol=1e-14)
    assert tree_allclose(Q, jnp.zeros((2, 2)), atol=1e-14)


def test_negative_timestep_is_rejected():
    """A negative step is an error, not a silently reversed exponential."""
    with pytest.raises(Exception, match="dt >= 0"):
        jax.block_until_ready(
            discretise_mfd(_OSCILLATOR, jnp.eye(2), jnp.asarray(-0.5))
        )


def test_sequence_matches_vmap_of_scalar():
    """``discretise_mfd_sequence`` is the vectorised scalar function."""
    steps = jnp.array([0.0, 0.1, 0.5, 2.0])
    Q_c = jnp.eye(2)

    A_seq, Q_seq = discretise_mfd_sequence(_OSCILLATOR, Q_c, steps)

    for i, step in enumerate(steps):
        A_i, Q_i = discretise_mfd(_OSCILLATOR, Q_c, step)
        assert tree_allclose(A_seq[i], A_i, atol=1e-14)
        assert tree_allclose(Q_seq[i], Q_i, atol=1e-14)


@pytest.mark.parametrize(
    "kernel",
    [
        MaternSDE(variance=jnp.asarray(1.3), lengthscale=jnp.asarray(0.8), order=1),
        CosineSDE(variance=jnp.asarray(1.0), frequency=jnp.asarray(_OMEGA)),
        PeriodicSDE(
            variance=jnp.asarray(1.0),
            lengthscale=jnp.asarray(1.0),
            period=jnp.asarray(2.0),
        ),
        ConstantSDE(variance=jnp.asarray(2.0)),
    ],
)
def test_existing_kernels_keep_their_analytic_paths(kernel):
    """Kernels that supply ``P_inf`` must not be captured by the fallback.

    Each of these has a closed form that is more precise than MFD; the
    fallback is dead code for them and must stay that way.
    """
    dt = jnp.asarray(0.3)
    params = kernel.sde_params()
    assert params.P_inf is not None

    A, Q = kernel.discretise(dt)
    assert bool(jnp.all(jnp.isfinite(A)))
    assert bool(jnp.all(jnp.isfinite(Q)))


def test_fallback_engages_when_p_inf_is_none():
    """A kernel with a learned drift and no ``P_inf`` routes through MFD."""
    import equinox as eqx

    from gaussx import SDEKernel, SDEParams

    class LearnedSDE(SDEKernel):
        """Drift is a free parameter, so no stationary covariance exists."""

        F: jax.Array

        @property
        def state_dim(self) -> int:
            return 2

        def sde_params(self) -> SDEParams:
            return SDEParams(
                F=self.F,
                L=jnp.eye(2),
                H=jnp.array([[1.0, 0.0]]),
                Q_c=jnp.eye(2),
            )

    kernel = LearnedSDE(F=_OSCILLATOR)
    assert kernel.sde_params().P_inf is None

    dt = jnp.asarray(0.5)
    A, Q = kernel.discretise(dt)
    A_ref, Q_ref = discretise_mfd(_OSCILLATOR, jnp.eye(2), dt)

    assert tree_allclose(A, A_ref, atol=1e-14)
    assert tree_allclose(Q, Q_ref, atol=1e-14)

    # And it is differentiable w.r.t. the learned drift.
    def loss(F):
        _, Q_f = LearnedSDE(F=F).discretise(dt)
        return jnp.trace(Q_f)

    grad = eqx.filter_grad(loss)(_OSCILLATOR)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_grad_through_drift_is_finite():
    """``jax.grad`` w.r.t. ``F`` returns finite values."""

    def loss(F):
        _, Q = discretise_mfd(F, jnp.eye(2), jnp.asarray(0.4))
        return jnp.sum(Q**2)

    grad = jax.grad(loss)(_UNSTABLE)
    assert grad.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_jit():
    """The function is traceable."""
    dt = jnp.asarray(0.5)
    jitted = jax.jit(discretise_mfd)(_OSCILLATOR, jnp.eye(2), dt)
    eager = discretise_mfd(_OSCILLATOR, jnp.eye(2), dt)

    assert tree_allclose(jitted[0], eager[0], atol=1e-14)
    assert tree_allclose(jitted[1], eager[1], atol=1e-14)


def _learned_sde_class():
    """A kernel whose drift is a free parameter, so it has no ``P_inf``."""
    from gaussx import SDEKernel, SDEParams

    class LearnedSDE(SDEKernel):
        F: jax.Array

        @property
        def state_dim(self) -> int:
            return 2

        def sde_params(self) -> SDEParams:
            return SDEParams(
                F=self.F,
                L=jnp.eye(2),
                H=jnp.array([[1.0, 0.0]]),
                Q_c=jnp.eye(2),
            )

    return LearnedSDE


def test_sum_composition_propagates_missing_p_inf():
    """A sum containing a ``P_inf``-less component has no ``P_inf`` either."""
    from gaussx import SumSDE

    learned = _learned_sde_class()(F=_OSCILLATOR)
    matern = MaternSDE(variance=jnp.asarray(1.0), lengthscale=jnp.asarray(1.0), order=1)

    assert SumSDE(kernels=(matern, matern)).sde_params().P_inf is not None

    composed = SumSDE(kernels=(matern, learned))
    assert composed.sde_params().P_inf is None

    # And it still discretises, via the MFD fallback on the composed F.
    A, Q = composed.discretise(jnp.asarray(0.3))
    assert A.shape == (4, 4)
    assert bool(jnp.all(jnp.isfinite(Q)))
    assert tree_allclose(Q, Q.T, atol=1e-14)


def test_product_composition_propagates_missing_p_inf():
    """Likewise for the Kronecker product, whose fast path needs both."""
    from gaussx import ProductSDE

    learned = _learned_sde_class()(F=_OSCILLATOR)
    matern = MaternSDE(variance=jnp.asarray(1.0), lengthscale=jnp.asarray(1.0), order=1)

    composed = ProductSDE(kernel1=matern, kernel2=learned)
    assert composed.sde_params().P_inf is None

    # ProductSDE.discretise overrides with a factorised congruence that
    # needs both P_inf; it must defer rather than fail.
    A, Q = composed.discretise(jnp.asarray(0.3))
    assert A.shape == (4, 4)
    assert bool(jnp.all(jnp.isfinite(Q)))

    # The factorised path is still taken when both factors supply P_inf.
    both = ProductSDE(kernel1=matern, kernel2=matern)
    assert both.sde_params().P_inf is not None
    A_both, Q_both = both.discretise(jnp.asarray(0.3))
    assert bool(jnp.all(jnp.isfinite(A_both)))
    assert bool(jnp.all(jnp.isfinite(Q_both)))


def test_autocovariance_rejects_missing_p_inf():
    """The stationary autocovariance is undefined without ``P_inf``."""
    from gaussx import sde_autocovariance

    learned = _learned_sde_class()(F=_OSCILLATOR)

    with pytest.raises(ValueError, match="no stationary covariance"):
        sde_autocovariance(learned, jnp.array([0.0, 0.5]))
