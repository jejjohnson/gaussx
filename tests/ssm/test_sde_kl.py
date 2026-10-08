"""Tests for `linearize_sde` and `sde_kl_divergence`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gaussx
from gaussx._einx import rearrange
from gaussx._testing import key_sequence, psd_operator, random_pd_matrix


def _gauss_moments(m: float, s: float, kmax: int) -> list[float]:
    """Raw moments E[x^k] of N(m, s): E[x^k] = m E[x^{k-1}] + (k-1) s E[x^{k-2}]."""
    mom = [1.0, m]
    for k in range(2, kmax + 1):
        mom.append(m * mom[k - 1] + (k - 1) * s * mom[k - 2])
    return mom


def _double_well_oracle(m: float, s: float) -> tuple[float, float, float]:
    """Exact A*, b* and E[(f - A* x - b*)^2] for f(x) = x - x^3, x ~ N(m, s)."""
    mo = _gauss_moments(m, s, 6)
    E_f = mo[1] - mo[3]
    cov_xf = (mo[2] - mo[4]) - m * E_f
    A = cov_xf / s
    b = E_f - A * m
    var_f = (mo[2] - 2 * mo[4] + mo[6]) - E_f**2
    # Residual of the linear regression of f on x: Var(f) - Cov(x, f)^2 / s.
    return A, b, var_f - cov_xf**2 / s


def _random_path(nextkey, T: int, d: int):
    means = jax.random.normal(nextkey(), (T, d))
    covs = jnp.stack([random_pd_matrix(nextkey(), d, jitter=0.2) for _ in range(T)])
    return means, 0.1 * covs


class TestLinearizeSDE:
    def test_linear_drift_is_recovered_exactly(self):
        """f(x) = M x + c gives A* = M, b* = c and a zero KL."""
        nextkey = key_sequence(0)
        T, d = 4, 3
        M = jax.random.normal(nextkey(), (d, d))
        c = jax.random.normal(nextkey(), (d,))
        means, covs = _random_path(nextkey, T, d)

        def drift(x):
            return M @ x + c

        lin = gaussx.linearize_sde(drift, jnp.eye(d), means, covs)
        np.testing.assert_allclose(lin.A, jnp.broadcast_to(M, (T, d, d)), atol=1e-10)
        np.testing.assert_allclose(lin.b, jnp.broadcast_to(c, (T, d)), atol=1e-10)
        kl = gaussx.sde_kl_divergence(drift, lin, means, covs, dt=0.1)
        assert abs(float(kl)) < 1e-18

    def test_double_well_matches_closed_form(self):
        """A*, b* and the KL match the Gaussian-moment closed form.

        Gauss-Hermite with 4 points is exact to degree 7, and the KL
        integrand of f(x) = x - x^3 has degree 6, so the match is exact.
        """
        ms = [0.0, 0.5, -1.2]
        ss = [0.05, 0.3, 0.8]
        means = rearrange(jnp.array(ms), "t -> t 1")
        covs = rearrange(jnp.array(ss), "t -> t 1 1")
        sigma2, dt = 0.4, 0.01
        integrator = gaussx.GaussHermiteIntegrator(order=4)

        def drift(x):
            return x - x**3

        lin = gaussx.linearize_sde(
            drift, sigma2 * jnp.eye(1), means, covs, integrator=integrator
        )
        oracle = [_double_well_oracle(m, s) for m, s in zip(ms, ss, strict=True)]
        np.testing.assert_allclose(lin.A[:, 0, 0], [o[0] for o in oracle], rtol=1e-10)
        np.testing.assert_allclose(lin.b[:, 0], [o[1] for o in oracle], atol=1e-10)
        np.testing.assert_allclose(lin.Q, jnp.full((3, 1, 1), sigma2))

        kl = gaussx.sde_kl_divergence(
            drift, lin, means, covs, dt=dt, integrator=integrator
        )
        expected = 0.5 * dt * sum(o[2] for o in oracle) / sigma2
        np.testing.assert_allclose(kl, expected, rtol=1e-10)

    def test_matches_expected_jacobian(self):
        """Stein's form equals E[df/dx] computed from the Jacobian itself."""
        nextkey = key_sequence(1)
        means, covs = _random_path(nextkey, 3, 2)
        integrator = gaussx.GaussHermiteIntegrator(order=20)

        def drift(x):
            return jnp.array([jnp.sin(x[0]) * x[1], jnp.tanh(x[0] - x[1])])

        lin = gaussx.linearize_sde(
            drift, jnp.eye(2), means, covs, integrator=integrator
        )

        def expected_jac(m, S):
            state = gaussx.GaussianState(mean=m, cov=psd_operator(S))
            flat = gaussx.mean_expectation(
                lambda x: rearrange(jax.jacfwd(drift)(x), "i j -> (i j)"),
                state,
                integrator,
            )
            return rearrange(flat, "(i j) -> i j", i=2)

        ref = jax.vmap(expected_jac)(means, covs)
        # Stein's lemma is exact under q; the residual is the quadrature
        # error of a 20-point-per-axis rule on smooth integrands.
        np.testing.assert_allclose(lin.A, ref, atol=1e-6)

    def test_integrators_agree_on_polynomial_drift(self):
        """Rules exact to degree 5 agree on a cubic drift."""
        nextkey = key_sequence(2)
        means, covs = _random_path(nextkey, 3, 2)

        def drift(x):
            return jnp.array([x[0] - x[0] ** 3 + x[1], -x[1] + x[0] * x[1]])

        a = gaussx.linearize_sde(
            drift, jnp.eye(2), means, covs, gaussx.FifthOrderCubatureIntegrator()
        )
        b = gaussx.linearize_sde(
            drift, jnp.eye(2), means, covs, gaussx.GaussHermiteIntegrator(order=3)
        )
        np.testing.assert_allclose(a.A, b.A, atol=1e-10)
        np.testing.assert_allclose(a.b, b.b, atol=1e-10)

    def test_time_varying_diffusion_and_dt(self):
        """Per-step diffusion and dt weight each step's contribution."""
        means = jnp.array([[0.2], [0.4]])
        covs = jnp.full((2, 1, 1), 0.3)
        Q = jnp.array([[[0.5]], [[2.0]]])
        dt = jnp.array([0.1, 0.3])
        integrator = gaussx.GaussHermiteIntegrator(order=4)

        def drift(x):
            return x - x**3

        lin = gaussx.linearize_sde(drift, Q, means, covs, integrator=integrator)
        np.testing.assert_allclose(lin.Q, Q)
        kl = gaussx.sde_kl_divergence(drift, lin, means, covs, dt, integrator)
        resid = [_double_well_oracle(0.2, 0.3)[2], _double_well_oracle(0.4, 0.3)[2]]
        expected = 0.5 * (0.1 * resid[0] / 0.5 + 0.3 * resid[1] / 2.0)
        np.testing.assert_allclose(kl, expected, rtol=1e-10)

    def test_kl_is_nonnegative_and_minimised_by_linearisation(self):
        """The optimal drift has a smaller KL than a perturbed one."""
        nextkey = key_sequence(3)
        means, covs = _random_path(nextkey, 4, 2)

        def drift(x):
            return x - x**3

        lin = gaussx.linearize_sde(drift, 0.5 * jnp.eye(2), means, covs)
        kl = gaussx.sde_kl_divergence(drift, lin, means, covs, dt=0.05)
        worse = gaussx.LinearizedSDE(A=lin.A + 0.1, b=lin.b, Q=lin.Q)
        kl_worse = gaussx.sde_kl_divergence(drift, worse, means, covs, dt=0.05)
        assert kl > 0
        assert kl_worse > kl

    def test_jit_and_grad(self):
        """jit matches eager and the KL is differentiable in the path mean."""
        means = jnp.array([[0.1], [0.3], [-0.2]])
        covs = jnp.full((3, 1, 1), 0.2)
        integrator = gaussx.GaussHermiteIntegrator(order=4)

        def drift(x):
            return x - x**3

        def loss(mu):
            lin = gaussx.linearize_sde(drift, jnp.eye(1), mu, covs, integrator)
            return gaussx.sde_kl_divergence(drift, lin, mu, covs, 0.1, integrator)

        np.testing.assert_allclose(jax.jit(loss)(means), loss(means), rtol=1e-12)
        g = jax.grad(loss)(means)
        assert g.shape == means.shape
        assert jnp.all(jnp.isfinite(g))
        # Central differences on the first mean; float64 makes 1e-5 safe.
        eps = 1e-5
        e0 = jnp.zeros_like(means).at[0, 0].set(eps)
        fd = (loss(means + e0) - loss(means - e0)) / (2 * eps)
        np.testing.assert_allclose(g[0, 0], fd, rtol=1e-6)

    def test_vmap_over_paths(self):
        """vmap over a batch of paths matches a Python loop."""
        nextkey = key_sequence(4)
        batch = [_random_path(nextkey, 3, 2) for _ in range(2)]
        means = jnp.stack([b[0] for b in batch])
        covs = jnp.stack([b[1] for b in batch])

        def drift(x):
            return jnp.sin(x)

        out = jax.vmap(lambda m, S: gaussx.linearize_sde(drift, jnp.eye(2), m, S).A)(
            means, covs
        )
        for i, (m, S) in enumerate(batch):
            np.testing.assert_allclose(
                out[i], gaussx.linearize_sde(drift, jnp.eye(2), m, S).A, rtol=1e-12
            )

    @pytest.mark.parametrize(
        ("means_shape", "covs_shape", "diff_shape"),
        [
            ((3,), (3, 1, 1), (1, 1)),
            ((3, 2), (3, 2, 1), (2, 2)),
            ((3, 2), (3, 2, 2), (4, 2, 2)),
        ],
    )
    def test_shape_validation(self, means_shape, covs_shape, diff_shape):
        with pytest.raises(ValueError, match="must have shape"):
            gaussx.linearize_sde(
                jnp.sin,
                jnp.ones(diff_shape),
                jnp.zeros(means_shape),
                jnp.ones(covs_shape),
            )

    def test_kl_rejects_mismatched_linear_drift(self):
        lin = gaussx.linearize_sde(
            jnp.sin, jnp.eye(1), jnp.zeros((3, 1)), jnp.ones((3, 1, 1))
        )
        with pytest.raises(ValueError, match="linear_drift must have"):
            gaussx.sde_kl_divergence(
                jnp.sin, lin, jnp.zeros((4, 1)), jnp.ones((4, 1, 1)), dt=0.1
            )


@pytest.mark.parametrize("order", [1, 2])
def test_kl_rejects_taylor(order):
    """Order 1 sees the residual only at the mean; order 2 can go negative."""
    m, S = jnp.zeros((2, 1)), jnp.full((2, 1, 1), 0.5)

    def drift(x):
        return x - x**3

    lin = gaussx.linearize_sde(drift, jnp.eye(1), m, S)
    with pytest.raises(ValueError, match="TaylorIntegrator"):
        gaussx.sde_kl_divergence(
            drift, lin, m, S, dt=0.1, integrator=gaussx.TaylorIntegrator(order=order)
        )


def test_kl_rejects_negative_dt():
    m, S = jnp.zeros((2, 1)), jnp.full((2, 1, 1), 0.5)
    lin = gaussx.linearize_sde(jnp.sin, jnp.eye(1), m, S)
    with pytest.raises(Exception, match="dt must be non-negative"):
        gaussx.sde_kl_divergence(jnp.sin, lin, m, S, dt=jnp.array([0.1, -0.1]))
