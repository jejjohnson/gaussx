"""Tests for `mixture_quantile`, `Chandrupatla` and the Gaussian fast path."""

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
import scipy.optimize
import scipy.stats
from jax.scipy.special import betainc
from jax.scipy.stats import norm

import gaussx
from gaussx._einx import reduce


LEVELS = jnp.array([0.025, 0.1, 0.5, 0.9, 0.975])


def _eps() -> float:
    return float(jnp.finfo(jnp.result_type(float)).eps)


def _two_normals(x):
    """Equal-weight mixture of N(-1, 1) and N(1, 1)."""
    return 0.5 * (norm.cdf(x + 1.0) + norm.cdf(x - 1.0))


class TestMixtureQuantile:
    def test_single_gaussian_matches_ppf(self):
        """A pure Gaussian CDF inverts to norm.ppf at the requested rtol."""
        x = gaussx.mixture_quantile(norm.cdf, LEVELS, -10.0, 10.0)
        assert x.shape == LEVELS.shape
        assert x.dtype == jnp.result_type(float)
        # Default rtol=1e-4, atol=1e-6 bound the error in x; the IQI steps
        # land far inside it on a smooth CDF.
        np.testing.assert_allclose(x, norm.ppf(LEVELS), rtol=1e-4, atol=1e-6)

    def test_tight_tolerance_reaches_machine_precision(self):
        x = gaussx.mixture_quantile(
            norm.cdf, LEVELS, -10.0, 10.0, rtol=_eps(), atol=0.0
        )
        # The root of norm.cdf(x) - q is only defined to the CDF's own
        # rounding, eps / pdf(x) ~ 20 eps in x at the 2.5% tails.
        np.testing.assert_allclose(
            x, norm.ppf(LEVELS), rtol=50 * _eps(), atol=50 * _eps()
        )

    def test_two_component_mixture(self):
        """Median 0 by symmetry; the tails match scipy's brentq."""
        x = gaussx.mixture_quantile(
            _two_normals, LEVELS, -10.0, 10.0, rtol=_eps(), atol=0.0
        )
        assert abs(float(x[2])) < 50 * _eps()

        def cdf_np(v):
            return 0.5 * (scipy.stats.norm.cdf(v + 1) + scipy.stats.norm.cdf(v - 1))

        ref = [
            scipy.optimize.brentq(lambda v, q=q: cdf_np(v) - q, -10, 10, xtol=1e-14)
            for q in np.asarray(LEVELS)
        ]
        np.testing.assert_allclose(x, ref, rtol=100 * _eps(), atol=100 * _eps())

    def test_batched_brackets_give_batch_by_levels(self):
        """(B, 1) brackets and (Q,) levels give a (B, Q) result."""
        means = jnp.array([[-2.0], [0.0], [3.0]])

        def cdf(x):
            return norm.cdf(x - means)

        x = gaussx.mixture_quantile(cdf, LEVELS, means - 10.0, means + 10.0)
        assert x.shape == (3, 5)
        expected = einx.add("b 1, q -> b q", means, norm.ppf(LEVELS))
        np.testing.assert_allclose(x, expected, rtol=1e-4, atol=1e-5)

    def test_ensemble_mixture(self):
        """Per-point ensemble CDFs: each row matches a scalar solve."""
        key_m, key_s = jax.random.split(jax.random.key(0))
        means = jax.random.normal(key_m, (4, 6))
        stds = 0.5 + jax.random.uniform(key_s, (4, 6))

        def cdf(x):  # x: (4, Q) -> (4, Q)
            z = einx.divide(
                "b q e, b e -> b q e",
                einx.subtract("b q, b e -> b q e", x, means),
                stds,
            )
            return reduce(norm.cdf(z), "b q e -> b q", "mean")

        lo = reduce(means - 8 * stds, "b e -> b 1", "min")
        hi = reduce(means + 8 * stds, "b e -> b 1", "max")
        x = gaussx.mixture_quantile(cdf, LEVELS, lo, hi)
        assert x.shape == (4, 5)
        # The CDF at the returned point is q to within |F'| * tolerance.
        np.testing.assert_allclose(cdf(x), jnp.broadcast_to(LEVELS, (4, 5)), atol=1e-4)

    def test_vmap_matches_loop(self):
        """vmap over a batch of CDFs equals a Python loop."""
        shifts = jnp.array([-1.0, 2.0])

        @jax.jit
        def one(shift):
            return gaussx.mixture_quantile(
                lambda x: _two_normals(x - shift), LEVELS, -12.0, 12.0
            )

        batched = jax.vmap(one)(shifts)
        for i, s in enumerate(shifts):
            np.testing.assert_allclose(
                batched[i], one(s), rtol=10 * _eps(), atol=10 * _eps()
            )

    def test_jit_does_not_retrace_for_new_levels(self):
        traces = []

        @jax.jit
        def f(q):
            traces.append(None)
            return gaussx.mixture_quantile(norm.cdf, q, -10.0, 10.0)

        a = f(LEVELS)
        b = f(LEVELS[::-1])
        assert len(traces) == 1
        np.testing.assert_allclose(a[::-1], b)

    def test_shift_equivariance_gradient(self):
        """d x_q / d shift = 1 exactly for F(x - shift)."""
        g = jax.grad(
            lambda m: gaussx.mixture_quantile(
                lambda x: _two_normals(x - m), 0.95, -10.0, 10.0
            )
        )(0.3)
        np.testing.assert_allclose(g, 1.0, rtol=10 * _eps())

    def test_gradient_wrt_level_is_inverse_density(self):
        x = gaussx.mixture_quantile(norm.cdf, 0.9, -10.0, 10.0, rtol=_eps(), atol=0.0)
        g = jax.grad(
            lambda q: gaussx.mixture_quantile(
                norm.cdf, q, -10.0, 10.0, rtol=_eps(), atol=0.0
            )
        )(0.9)
        np.testing.assert_allclose(g, 1.0 / norm.pdf(x), rtol=1e-5)

    @pytest.mark.x64_only(reason="central finite differences need float64")
    def test_gradient_matches_finite_differences(self):
        """Gradient of the 95th percentile w.r.t. a mixture scale."""

        def q95(scale):
            def cdf(x):
                return 0.5 * (norm.cdf(x / scale + 1.0) + norm.cdf((x - 1.0) / 2.0))

            return gaussx.mixture_quantile(cdf, 0.95, -50.0, 50.0, rtol=1e-14, atol=0.0)

        s, h = 1.3, 1e-6
        fd = (q95(s + h) - q95(s - h)) / (2 * h)
        # Central-difference error: h^2 f''' is negligible, but each root
        # is only known to ~ulp(x) ~ 4e-16, giving ~4e-10 / |g| ~ 5e-8
        # relative on this small derivative (|g| ~ 0.009).
        np.testing.assert_allclose(jax.grad(q95)(s), fd, rtol=1e-6)

    def test_unbracketed_level_returns_closest_endpoint(self):
        x = gaussx.mixture_quantile(norm.cdf, jnp.array([0.5, 0.999]), -1.0, 1.0)
        assert abs(float(x[0])) < 1e-6
        assert float(x[1]) == 1.0

    def test_unbracketed_level_raises_when_throw(self):
        with pytest.raises(Exception, match="not bracketed"):
            gaussx.mixture_quantile(norm.cdf, 0.999, -1.0, 1.0, throw=True)

    def test_negative_binomial_continuous_relaxation(self):
        """NB quantile by root-finding the continuous CDF, then ceil.

        The NB(r, p) CDF at integer k is I_p(r, k + 1); extending k to the
        reals gives a continuous monotone CDF whose root, rounded up, is
        the integer quantile (bayesnf's ``_get_nb_quantiles_root``).
        """
        r, p = 3.0, 0.4

        def cdf(k):
            return betainc(r, k + 1.0, p)

        x = gaussx.mixture_quantile(cdf, LEVELS, -1.0 + 1e-6, 200.0)
        np.testing.assert_array_equal(
            jnp.ceil(x), scipy.stats.nbinom.ppf(np.asarray(LEVELS), r, p)
        )

    def test_custom_solver(self):
        """Any optimistix root finder that takes lower/upper options works."""
        x = gaussx.mixture_quantile(
            norm.cdf, 0.8, -10.0, 10.0, solver=optx.Bisection(rtol=1e-6, atol=1e-6)
        )
        np.testing.assert_allclose(x, norm.ppf(0.8), atol=1e-5)


class TestChandrupatla:
    def test_root_find_contract(self):
        """Composes with optx.root_find: value, result and step count."""
        sol = optx.root_find(
            lambda x, _: jnp.cos(x) - x,
            gaussx.Chandrupatla(rtol=_eps(), atol=0.0),
            jnp.array(0.5),
            options=dict(lower=0.0, upper=1.0),
        )
        assert sol.result == optx.RESULTS.successful
        np.testing.assert_allclose(sol.value, 0.7390851332151607, rtol=10 * _eps())
        # Superlinear: far fewer steps than bisection's ~log2(1 / eps).
        assert int(sol.stats["num_steps"]) < 15

    def test_elementwise_brackets_converge_independently(self):
        targets = jnp.array([0.1, 1.0, 5.0, 30.0])
        sol = optx.root_find(
            lambda x, t: x**3 - t,
            gaussx.Chandrupatla(rtol=_eps(), atol=0.0),
            jnp.ones(4),
            args=targets,
            options=dict(lower=0.0, upper=jnp.array([1.0, 2.0, 2.0, 4.0])),
        )
        np.testing.assert_allclose(sol.value, jnp.cbrt(targets), rtol=10 * _eps())

    def test_rejects_non_elementwise_function(self):
        with pytest.raises(ValueError, match="elementwise"):
            optx.root_find(
                lambda x, _: jnp.sum(x),
                gaussx.Chandrupatla(rtol=1e-6, atol=1e-6),
                jnp.ones(2),
                options=dict(lower=-1.0, upper=1.0),
            )

    def test_max_steps_reported(self):
        sol = optx.root_find(
            lambda x, _: jnp.cos(x) - x,
            gaussx.Chandrupatla(rtol=0.0, atol=0.0),
            jnp.array(0.5),
            options=dict(lower=0.0, upper=1.0),
            max_steps=2,
            throw=False,
        )
        assert sol.result == optx.RESULTS.nonlinear_max_steps_reached


class TestGaussianApprox:
    def test_single_component_is_exact(self):
        means, stds = jnp.array([[1.5], [-0.5]]), jnp.array([[0.7], [2.0]])
        x = gaussx.mixture_quantile_gaussian_approx(means, stds, LEVELS)
        expected = einx.add(
            "b 1, b q -> b q",
            means,
            einx.multiply("b 1, q -> b q", stds, norm.ppf(LEVELS)),
        )
        assert x.shape == (2, 5)
        np.testing.assert_allclose(x, expected, rtol=10 * _eps(), atol=10 * _eps())

    def test_moments_match_mixture(self):
        """Mean and variance are the mixture's first two moments."""
        means = jnp.array([[-1.0, 1.0, 3.0]])
        stds = jnp.array([[0.5, 1.0, 2.0]])
        mu = float(jnp.mean(means))
        var = float(jnp.mean(stds**2 + means**2)) - mu**2
        x = gaussx.mixture_quantile_gaussian_approx(
            means, stds, jnp.array([0.5, norm.cdf(1.0)])
        )
        np.testing.assert_allclose(x[0], [mu, mu + np.sqrt(var)], rtol=1e-5)


class TestReviewEdgeCases:
    def test_aux_belongs_to_the_returned_root(self):
        sol = optx.root_find(
            lambda x, _: (x**3 - 2.0, x),
            gaussx.Chandrupatla(rtol=_eps(), atol=0.0),
            jnp.array(1.0),
            options=dict(lower=0.0, upper=2.0),
            has_aux=True,
        )
        assert sol.aux == sol.value

    @pytest.mark.parametrize(
        ("lower", "upper"), [(10.0, -10.0), (jnp.nan, 10.0)], ids=["reversed", "nan"]
    )
    def test_throw_rejects_reversed_or_nan_brackets(self, lower, upper):
        with pytest.raises(Exception, match="not bracketed"):
            gaussx.mixture_quantile(norm.cdf, 0.5, lower, upper, throw=True)

    def test_narrow_bracket_honours_atol(self):
        """A bracket narrower than 2 atol used to stop at an endpoint."""

        def cdf(x):
            return jnp.clip(jnp.where(x < 0, 0.5 + 0.01 * x, 0.5 + x), 0.0, 1.0)

        x = gaussx.mixture_quantile(cdf, 0.5, -0.14, 0.01, rtol=0.0, atol=0.1)
        assert abs(float(x)) <= 0.1

    @pytest.mark.x64_only(reason="needs a float64 iterate with a float32 residual")
    def test_bracket_keeps_the_iterate_dtype(self):
        sol = optx.root_find(
            lambda x, _: (x - 1.0 / 3.0).astype(jnp.float32),
            gaussx.Chandrupatla(rtol=0.0, atol=1e-12),
            jnp.array(0.5, jnp.float64),
            options=dict(lower=0.0, upper=1.0),
        )
        assert sol.value.dtype == jnp.float64
        # float32 rounding of the bracket would cap the error at ~3e-8.
        assert abs(float(sol.value) - 1.0 / 3.0) < 1e-7

    @pytest.mark.x64_only(reason="1e308 is a float64 bracket")
    def test_wide_finite_bracket_does_not_overflow(self):
        def cdf(x):
            # A Cauchy CDF with scale 1e300 (x / 1e308 would compile to a
            # multiply by a subnormal reciprocal that XLA flushes to zero).
            return 0.5 + jnp.arctan(x / 1e300) / jnp.pi

        x = gaussx.mixture_quantile(cdf, 0.5, -1e308, 1e308, atol=1.0)
        assert jnp.isfinite(x) and abs(float(x)) <= 1.0

    def test_gaussian_approx_promotes_low_precision_inputs(self):
        x = gaussx.mixture_quantile_gaussian_approx(
            jnp.array([[300.0]], jnp.float16),
            jnp.array([[300.0]], jnp.float16),
            jnp.array([0.5], jnp.float32),
        )
        assert jnp.isfinite(x).all()
        np.testing.assert_allclose(x, [[300.0]], rtol=1e-3)


def test_nan_endpoint_residual_prefers_the_finite_endpoint():
    # gh-121 review: with throw=False and a NaN CDF at lower only, the
    # unbracketed level returns the finite endpoint, not the NaN one.
    def cdf(x):
        return jnp.where(x < -5.0, jnp.nan, norm.cdf(x))

    x = gaussx.mixture_quantile(cdf, jnp.array([0.5]), -10.0, -6.0, throw=False)
    assert jnp.all(x == -6.0)


@pytest.mark.parametrize(("lower", "upper"), [(-jnp.inf, 10.0), (-10.0, jnp.inf)])
def test_infinite_brackets_are_rejected(lower, upper):
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        gaussx.mixture_quantile(norm.cdf, jnp.array([0.5]), lower, upper, throw=True)
    x = gaussx.mixture_quantile(norm.cdf, jnp.array([0.5]), lower, upper, throw=False)
    assert jnp.all((x == lower) | (x == upper))


def test_gaussian_approx_scale_does_not_overflow():
    # gh-121 review: stds**2 overflows float32 at 1e30; the rescaled norm
    # keeps the single-component quantiles exact.
    x = gaussx.mixture_quantile_gaussian_approx(
        jnp.array([[0.0]], dtype=jnp.float32),
        jnp.array([[1e30]], dtype=jnp.float32),
        jnp.array([0.5, 0.8413447], dtype=jnp.float32),
    )
    assert jnp.all(jnp.isfinite(x))
    assert jnp.allclose(x[0], jnp.array([0.0, 1e30]), rtol=1e-5)
