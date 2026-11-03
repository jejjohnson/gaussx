"""Quantiles by CDF inversion, and the moment-matched Gaussian fast path.

F is the CDF of a scalar y, τ ∈ [0, 1] a quantile level and F⁻¹(τ) the
quantile, the root of F(y) − τ. A mixture of E members with CDFs F_e,
e = 1, …, E, has the CDF F_mix(y) = (1/E) ∑_e F_e(y); a Gaussian member is
𝒩(μ_e, σ_e²). Φ is the standard normal CDF.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special
import lineax as lx
import optimistix as optx
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import reduce
from gaussx._primitives._chandrupatla import Chandrupatla


def mixture_quantile(
    cdf_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    q: Float[ArrayLike, ...],
    lower: Float[ArrayLike, ...],
    upper: Float[ArrayLike, ...],
    *,
    solver: optx.AbstractRootFinder | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 64,
    throw: bool = False,
) -> Float[Array, ...]:
    """Invert a CDF at one or more quantile levels by bracketed root finding.

        F⁻¹(τ) = y   such that   F(y) − τ = 0,   y ∈ [y_lo, y_hi]

    F is monotone, so F(y) − τ changes sign once on any bracket with
    F(y_lo) ≤ τ ≤ F(y_hi), and a bracketing method needs neither derivatives
    nor a good initial guess. A mixture or ensemble CDF
    F_mix(y) = (1/E) ∑_e F_e(y) has no closed-form inverse even when every
    F_e does; this is what the function is for.

    **Shapes.** ``q``, ``lower`` and ``upper`` broadcast together (NumPy
    rules) to the output shape, and ``cdf_fn`` is called on arrays of that
    shape. It must act **elementwise**: entry ``i`` of its output may
    depend only on entry ``i`` of its input, e.g. a scalar CDF, or a
    per-test-point mixture CDF with a trailing quantile axis. With ``Q``
    levels ``q`` of shape ``(Q,)`` and a batch of brackets, give the
    brackets a trailing axis, ``(*batch, 1)``, to get ``(*batch, Q)``.

    **Gradients** go through `optimistix.ImplicitAdjoint`. For a root y_τ
    of F(y; θ) − τ, with density p = ∂F/∂y,

        ∂y_τ/∂θ = −∂_θ F(y_τ; θ) / p(y_τ; θ),   ∂y_τ/∂τ = 1 / p(y_τ; θ)

    evaluated with one Jacobian-vector product (the Jacobian is tagged
    diagonal). Parameters θ closed over by ``cdf_fn`` are differentiated
    too. The brackets are not.

    Algorithm:
        1. Promote τ and the bracket to one floating dtype and broadcast
           them to the output shape.
        2. With ``throw``, check F(y_lo) − τ ≤ 0 ≤ F(y_hi) − τ and a finite
           bracket for every entry.
        3. Find the root of y ↦ F(y) − τ with the solver (`Chandrupatla` by
           default) from the bracket midpoint, through
           `optimistix.root_find` with the implicit adjoint and a diagonal
           Jacobian tag.

    Args:
        cdf_fn: Elementwise CDF F, mapping an array to an array of the same
            shape. It must be strictly increasing where it crosses τ: on a
            flat segment at level τ the solver returns some point of the
            segment, not the generalized inverse inf{y : F(y) ≥ τ}.
            Gaussian-mixture CDFs always qualify.
        q: Quantile levels τ in [0, 1].
        lower: Finite lower bracket y_lo, with F(y_lo) ≤ τ.
        upper: Finite upper bracket y_hi, with F(y_hi) ≥ τ. Infinite
            brackets are rejected; for a Gaussian mixture, means ± 40
            standard deviations bracket every level.
        solver: An optimistix root finder that takes
            ``options=dict(lower=..., upper=...)``. Defaults to
            ``Chandrupatla(rtol=rtol, atol=atol)``. `optimistix.Bisection`
            also fits, for scalar problems only.
        rtol: Relative tolerance on the quantile (default solver only).
        atol: Absolute tolerance on the quantile (default solver only).
        max_steps: Maximum number of solver iterations.
        throw: If ``True``, raise when a level is not bracketed
            (F(y_lo) ≤ τ ≤ F(y_hi) fails, including a NaN endpoint value, an
            infinite endpoint or a reversed bracket) or the solver does not
            converge. If ``False``, an unbracketed level returns the bracket
            endpoint whose CDF is closer to τ (its gradient is then
            meaningless), and a non-converged one the best estimate so far.

    Returns:
        The quantiles F⁻¹(τ), with the broadcast shape of ``q``, ``lower``
        and ``upper``.

    References:
        Chandrupatla, T. R. (1997). A new hybrid quadratic/bisection
        algorithm for finding the zero of a nonlinear function without
        using derivatives. *Advances in Engineering Software* 28(3),
        145-149.

    Examples:
        The median and 97.5% quantile of a two-member Gaussian mixture:

        >>> import jax.numpy as jnp
        >>> from jax.scipy.stats import norm
        >>> import gaussx
        >>> cdf = lambda y: 0.5 * (norm.cdf(y + 1.0) + norm.cdf(y - 1.0))
        >>> y = gaussx.mixture_quantile(cdf, jnp.array([0.5, 0.975]), -10.0, 10.0)
        >>> [round(float(v), 3) for v in y]
        [0.0, 2.646]

        A batch of brackets with a trailing axis gives one row per bracket:

        >>> lower = jnp.full((3, 1), -10.0)
        >>> gaussx.mixture_quantile(cdf, jnp.array([0.1, 0.9]), lower, 10.0).shape
        (3, 2)

        An unbracketed level returns the endpoint closer to τ:

        >>> float(gaussx.mixture_quantile(cdf, 0.999, -10.0, 1.0))
        1.0
    """
    q = jnp.asarray(q)
    dtype = jnp.result_type(q, lower, upper, float)
    q = q.astype(dtype)
    shape = jnp.broadcast_shapes(q.shape, jnp.shape(lower), jnp.shape(upper))
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype), shape)
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype), shape)
    if solver is None:
        solver = Chandrupatla(rtol=rtol, atol=atol)

    def residual_fn(
        y: Float[Array, ...], level: Float[Array, ...]
    ) -> Float[Array, ...]:
        # F(y) − τ
        return cdf_fn(y) - level

    if throw:
        # F(y_lo) − τ
        lower_residual = residual_fn(lower, q)
        # F(y_hi) − τ
        upper_residual = residual_fn(upper, q)
        # Directed checks: they also reject NaN endpoint values and a
        # reversed bracket, which a sign-product test would let through.
        # F(y_lo) − τ ≤ 0 ≤ F(y_hi) − τ, with y_lo and y_hi finite
        bracketed = (
            (lower_residual <= 0)
            & (upper_residual >= 0)
            & jnp.isfinite(lower)
            & jnp.isfinite(upper)
        )
        lower = eqx.error_if(
            lower,
            ~jnp.all(bracketed),
            "mixture_quantile: some levels q are not bracketed by finite "
            "endpoints, i.e. cdf_fn(lower) <= q <= cdf_fn(upper) fails or a "
            "bracket is infinite.",
        )
    # (y_lo + y_hi) / 2, halved first: no overflow for wide finite brackets
    midpoint = 0.5 * lower + 0.5 * upper
    solution = optx.root_find(
        residual_fn,
        solver,
        midpoint,
        args=jnp.broadcast_to(q, shape),
        options=dict(lower=lower, upper=upper),
        max_steps=max_steps,
        adjoint=optx.ImplicitAdjoint(),
        throw=throw,
        tags=frozenset({lx.diagonal_tag}),
    )
    return solution.value


def mixture_quantile_gaussian_approx(
    means: Float[Array, "*batch E"],
    stds: Float[Array, "*batch E"],
    q: Float[ArrayLike, " Q"],
) -> Float[Array, "*batch Q"]:
    """Quantiles of the Gaussian matched to an ensemble's first two moments.

        F_mix⁻¹(τ) ≈ μ_mix + σ_mix Φ⁻¹(τ)
        μ_mix = (1/E) ∑_e μ_e,   σ_mix² = (1/E) ∑_e (σ_e² + (μ_e − μ_mix)²)

    μ_mix and σ_mix² are the exact mean and variance of the equal-weight
    mixture of 𝒩(μ_e, σ_e²); its quantiles are approximated by those of
    𝒩(μ_mix, σ_mix²), the deep-ensemble predictive of Lakshminarayanan,
    Pritzel & Blundell (2017). It is exact for a single member and cheap
    otherwise: a diagnostic, or a seed for the brackets of
    `mixture_quantile`.

    Algorithm:
        1. Average the member means into μ_mix and take the deviations
           μ_e − μ_mix, which avoid the cancellation of the raw-moment form
           (1/E) ∑_e (σ_e² + μ_e²) − μ_mix².
        2. Rescale σ_e and the deviations by their largest magnitude s, so
           squaring cannot overflow, and take
           σ_mix = s ((1/E) ∑_e ((σ_e/s)² + ((μ_e − μ_mix)/s)²))^½.
        3. Return μ_mix + σ_mix Φ⁻¹(τ) for every level.

    Args:
        means: Member means μ_e, shape ``(*batch, E)``.
        stds: Member standard deviations σ_e, shape ``(*batch, E)``.
        q: Quantile levels τ, shape ``(Q,)``.

    Returns:
        The approximate quantiles μ_mix + σ_mix Φ⁻¹(τ), shape
        ``(*batch, Q)``.

    References:
        Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and
        scalable predictive uncertainty estimation using deep ensembles.
        *NeurIPS 30*.

    Examples:
        A single member is its own Gaussian, so the quantiles are exact:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> y = gaussx.mixture_quantile_gaussian_approx(
        ...     jnp.array([[0.0]]), jnp.array([[2.0]]), jnp.array([0.5, 0.8413447])
        ... )
        >>> [round(float(v), 4) for v in y[0]]
        [0.0, 2.0]

        Two members at ±1 with unit spread have σ_mix² = 2:

        >>> means = jnp.array([[-1.0, 1.0]])
        >>> stds = jnp.array([[1.0, 1.0]])
        >>> y = gaussx.mixture_quantile_gaussian_approx(means, stds, 0.8413447)
        >>> y.shape, round(float(y[0, 0]) ** 2, 3)
        ((1, 1), 2.0)
    """
    dtype = jnp.result_type(means, stds, q, float)
    means = jnp.asarray(means, dtype)
    stds = jnp.asarray(stds, dtype)
    q = jnp.atleast_1d(jnp.asarray(q, dtype))
    # μ_mix = (1/E) ∑_e μ_e
    mixture_mean = reduce(means, "... e -> ...", "mean")
    # μ_e − μ_mix
    deviations = einx.subtract("... e, ... -> ... e", means, mixture_mean)
    # Rescaled norm: σ_e² would overflow for σ_e near sqrt(max float).
    # |σ_e|
    abs_stds = jnp.abs(stds)
    # |μ_e − μ_mix|
    abs_deviations = jnp.abs(deviations)
    # max_e |σ_e|
    max_std = reduce(abs_stds, "... e -> ...", "max")
    # max_e |μ_e − μ_mix|
    max_deviation = reduce(abs_deviations, "... e -> ...", "max")
    # s = max(max_e |σ_e|, max_e |μ_e − μ_mix|)
    scale = jnp.maximum(max_std, max_deviation)
    # s ← 1 where s = 0, so an all-zero ensemble divides safely
    scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    # σ_e / s
    scaled_stds = einx.divide("... e, ... -> ... e", stds, scale)
    # (μ_e − μ_mix) / s
    scaled_deviations = einx.divide("... e, ... -> ... e", deviations, scale)
    # (σ_e/s)² + ((μ_e − μ_mix)/s)²
    scaled_second_moments = scaled_stds**2 + scaled_deviations**2
    # σ_mix² / s² = (1/E) ∑_e ((σ_e/s)² + ((μ_e − μ_mix)/s)²)
    scaled_variance = reduce(scaled_second_moments, "... e -> ...", "mean")
    # σ_mix = s (σ_mix² / s²)^½
    mixture_std = scale * jnp.sqrt(scaled_variance)
    # Φ⁻¹(τ)
    standard_quantiles = jax.scipy.special.ndtri(q)
    # σ_mix Φ⁻¹(τ)
    spread = einx.multiply("..., q -> ... q", mixture_std, standard_quantiles)
    # μ_mix + σ_mix Φ⁻¹(τ)
    return einx.add("..., ... q -> ... q", mixture_mean, spread)
