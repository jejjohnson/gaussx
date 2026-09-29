"""Tests for SDE autocovariance utility."""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl
import pytest

from gaussx import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SumSDE,
    sde_autocovariance,
)


class TestSDEAutocovariance:
    def test_zero_lag_equals_variance(self):
        kern = MaternSDE(variance=jnp.array(2.5), lengthscale=jnp.array(1.0), order=1)
        k0 = sde_autocovariance(kern, jnp.array(0.0))
        assert jnp.allclose(k0, 2.5, atol=1e-6)

    def test_symmetry(self):
        kern = MaternSDE(variance=jnp.array(1.0), lengthscale=jnp.array(1.0), order=1)
        k_pos = sde_autocovariance(kern, jnp.array(0.5))
        k_neg = sde_autocovariance(kern, jnp.array(-0.5))
        assert jnp.allclose(k_pos, k_neg, atol=1e-6)

    def test_matern12_analytical(self):
        sigma2 = jnp.array(2.0)
        ell = jnp.array(1.5)
        kern = MaternSDE(variance=sigma2, lengthscale=ell, order=0)
        taus = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.exp(-jnp.abs(taus) / ell)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_cosine_kernel(self):
        sigma2 = jnp.array(1.5)
        w = jnp.array(3.0)
        kern = CosineSDE(variance=sigma2, frequency=w)
        taus = jnp.array([0.0, 0.1, 0.5, 1.0])
        k_vals = sde_autocovariance(kern, taus)
        expected = sigma2 * jnp.cos(w * taus)
        assert jnp.allclose(k_vals, expected, atol=1e-5)

    def test_constant_kernel(self):
        sigma2 = jnp.array(3.0)
        kern = ConstantSDE(variance=sigma2)
        taus = jnp.array([0.0, 1.0, 10.0, 100.0])
        k_vals = sde_autocovariance(kern, taus)
        assert jnp.allclose(k_vals, sigma2, atol=1e-5)

    def test_differentiable(self):
        def loss(variance, lengthscale):
            kern = MaternSDE(variance=variance, lengthscale=lengthscale, order=1)
            return sde_autocovariance(kern, jnp.array(0.5))

        grad_fn = jax.grad(loss, argnums=(0, 1))
        g_var, g_ell = grad_fn(jnp.array(1.0), jnp.array(1.0))
        assert jnp.isfinite(g_var)
        assert jnp.isfinite(g_ell)


class TestPeriodicClosedForm:
    """gh-289: the j = 0 (constant) harmonic was missing, so k(0) != σ²."""

    @staticmethod
    def _periodic(variance, ell, n_harmonics=10):
        return PeriodicSDE(
            variance=jnp.array(variance),
            lengthscale=jnp.array(ell),
            period=jnp.array(1.0),
            n_harmonics=n_harmonics,
        )

    @pytest.mark.parametrize("ell", [0.5, 1.0, 2.0])
    def test_matches_mackay_kernel(self, ell):
        # The truncation tail 2σ² Σ_{j>10} I_j(x) e^{-x} is 3e-6·σ² at ell = 0.5.
        variance = 2.0
        taus = jnp.linspace(0.0, 1.0, 21)
        k_sde = sde_autocovariance(self._periodic(variance, ell), taus)
        expected = variance * jnp.exp(-2.0 * jnp.sin(jnp.pi * taus) ** 2 / ell**2)
        assert jnp.allclose(k_sde, expected, rtol=0.0, atol=1e-5 * variance)
        assert jnp.allclose(k_sde[0], variance, rtol=1e-5)

    def test_quasi_periodic_zero_lag_is_product_of_variances(self):
        matern = MaternSDE(
            variance=jnp.array(1.5), lengthscale=jnp.array(10.0), order=0
        )
        kern = QuasiPeriodicSDE(kernel1=matern, kernel2=self._periodic(2.0, 1.0))
        assert jnp.allclose(sde_autocovariance(kern, jnp.array([0.0])), 3.0, rtol=1e-6)

    def test_sde_params_and_discretise_agree(self):
        kern = self._periodic(2.0, 0.7, n_harmonics=4)
        params = kern.sde_params()
        A, Q = kern.discretise(jnp.array(0.13))
        assert jnp.allclose(A, jsl.expm(params.F * 0.13), atol=1e-10)
        assert jnp.allclose(A @ params.P_inf @ A.T + Q, params.P_inf, atol=1e-12)


# ---------------------------------------------------------------------------
# gh-412: every stationary kernel against its closed form
# ---------------------------------------------------------------------------

_S2, _ELL, _W = 1.3, 0.7, 2.0


def _matern(order):
    return MaternSDE(variance=jnp.array(_S2), lengthscale=jnp.array(_ELL), order=order)


def _k_matern(order, tau):
    tau = jnp.abs(tau)
    if order == 0:
        return _S2 * jnp.exp(-tau / _ELL)
    if order == 1:
        r = jnp.sqrt(3.0) * tau / _ELL
        return _S2 * (1.0 + r) * jnp.exp(-r)
    r = jnp.sqrt(5.0) * tau / _ELL
    return _S2 * (1.0 + r + r**2 / 3.0) * jnp.exp(-r)


def _cosine():
    return CosineSDE(variance=jnp.array(0.5), frequency=jnp.array(_W))


def _k_cosine(tau):
    return 0.5 * jnp.cos(_W * tau)


def _periodic():
    # ell = 1 with 10 harmonics: the truncation tail is ~1e-11.
    return PeriodicSDE(
        variance=jnp.array(0.8),
        lengthscale=jnp.array(1.0),
        period=jnp.array(1.5),
        n_harmonics=10,
    )


def _k_periodic(tau):
    return 0.8 * jnp.exp(-2.0 * jnp.sin(jnp.pi * tau / 1.5) ** 2)


# name -> (kernel factory, closed form k(tau), tolerance)
_STATIONARY = {
    "matern12": (lambda: _matern(0), lambda t: _k_matern(0, t), 1e-12),
    "matern32": (lambda: _matern(1), lambda t: _k_matern(1, t), 1e-12),
    "matern52": (lambda: _matern(2), lambda t: _k_matern(2, t), 1e-12),
    "cosine": (_cosine, _k_cosine, 1e-12),
    "constant": (
        lambda: ConstantSDE(variance=jnp.array(_S2)),
        lambda t: _S2 + 0 * t,
        1e-12,
    ),
    "periodic": (_periodic, _k_periodic, 1e-9),
    "sum": (
        lambda: SumSDE(kernels=(_matern(1), _cosine())),
        lambda t: _k_matern(1, t) + _k_cosine(t),
        1e-12,
    ),
    "product": (
        lambda: ProductSDE(kernel1=_matern(1), kernel2=_cosine()),
        lambda t: _k_matern(1, t) * _k_cosine(t),
        1e-12,
    ),
    "quasi_periodic": (
        lambda: QuasiPeriodicSDE(kernel1=_matern(0), kernel2=_periodic()),
        lambda t: _k_matern(0, t) * _k_periodic(t),
        1e-9,
    ),
}


@pytest.mark.parametrize("name", list(_STATIONARY))
def test_autocovariance_matches_closed_form(name):
    make, closed_form, tol = _STATIONARY[name]
    taus = jnp.linspace(0.0, 2.0, 9)
    got = sde_autocovariance(make(), taus)
    assert jnp.allclose(got, closed_form(taus), rtol=0.0, atol=tol)


@pytest.mark.parametrize("name", list(_STATIONARY))
def test_chain_gram_matches_kernel_on_irregular_grid(name):
    # The covariance implied by the discretised chain -- what a Markov GP
    # actually uses -- equals the kernel's Gram matrix. The grid is pinned;
    # any sorted irregular grid would do.
    make, closed_form, tol = _STATIONARY[name]
    kern = make()
    params = kern.sde_params()
    t = jnp.sort(jr.uniform(jr.key(1), (8,), maxval=3.0, dtype=jnp.float64))
    A_seq, _ = kern.discretise_sequence(jnp.diff(t))

    # Cov(f(t_j), f(t_i)) = H A_{j-1} ... A_i P∞ Hᵀ for j >= i: carry every
    # column u_i = Φ(j, i) P∞ Hᵀ forward, starting column i at step i.
    def step(U, inputs):
        j, A_prev = inputs
        U = jnp.where(jnp.arange(U.shape[0])[:, None] < j, U @ A_prev.T, U)
        return U, U @ params.H[0]

    n = t.shape[0]
    U0 = jnp.tile((params.P_inf @ params.H[0])[None], (n, 1))
    A_prev = jnp.concatenate([jnp.eye(kern.state_dim)[None], A_seq])
    _, rows = jax.lax.scan(step, U0, (jnp.arange(n), A_prev))
    lower = jnp.tril(rows)  # rows[j, i] = Cov(f(t_j), f(t_i)) for i <= j
    gram = lower + jnp.tril(lower, -1).T
    expected = closed_form(t[:, None] - t[None, :])
    assert jnp.allclose(gram, expected, rtol=0.0, atol=tol)
