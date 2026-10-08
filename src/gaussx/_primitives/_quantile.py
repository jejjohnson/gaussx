r"""Quantiles by CDF inversion, and the moment-matched Gaussian fast path."""

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
    r"""Invert a CDF at one or more quantile levels by bracketed root finding.

    The $q$-quantile of a continuous CDF $F$ is the root of
    $g_q(x) = F(x) - q$. $F$ is monotone, so $g_q$ changes sign exactly
    once on any bracket $[\ell, u]$ with $F(\ell) \le q \le F(u)$, and a
    bracketing method needs neither derivatives nor a good initial guess.
    A mixture or ensemble CDF

    $$
    F_{\text{mix}}(x) = \frac{1}{E}\sum_{e=1}^{E} F_e(x)
    $$

    has no closed-form inverse even when every $F_e$ does; this is what
    the function is for. The default solver is `Chandrupatla`, run through
    `optimistix.root_find`.

    **Shapes.** ``q``, ``lower`` and ``upper`` broadcast together (NumPy
    rules) to the output shape, and ``cdf_fn`` is called on arrays of that
    shape. It must act **elementwise** — entry ``i`` of its output may
    depend only on entry ``i`` of its input — e.g. a scalar CDF, or a
    per-test-point mixture CDF with a trailing quantile axis. With ``Q``
    levels ``q`` of shape ``(Q,)`` and a batch of brackets, give the
    brackets a trailing axis, ``(*batch, 1)``, to get ``(*batch, Q)``.

    **Gradients** go through `optimistix.ImplicitAdjoint`: for a root
    $x_q$ of $F(x; \theta) - q$,

    $$
    \frac{\partial x_q}{\partial \theta}
        = -\frac{\partial_\theta F(x_q; \theta)}{p(x_q; \theta)},
    \qquad \frac{\partial x_q}{\partial q} = \frac{1}{p(x_q; \theta)},
    $$

    with $p = \partial_x F$ the density, evaluated with one Jacobian-vector
    product (the Jacobian is tagged diagonal). Parameters closed over by
    ``cdf_fn`` are differentiated too. The brackets are not.

    Pseudocode:

        F_lo, F_hi = cdf(lower) − q, cdf(upper) − q
        if throw: error unless F_lo · F_hi ≤ 0 everywhere
        x = root_find(x ↦ cdf(x) − q, Chandrupatla, bracket [lower, upper],
                      adjoint = implicit, Jacobian tag = diagonal)

    Args:
        cdf_fn: Elementwise CDF, mapping an array to an array of the same
            shape. It must be strictly increasing where it crosses ``q``: on
            a flat segment at level ``q`` the solver returns some point of
            the segment, not the generalized inverse
            ``inf{x : F(x) >= q}``. Gaussian-mixture CDFs always qualify.
        q: Quantile levels in $[0, 1]$.
        lower: Finite lower bracket, with ``cdf_fn(lower) <= q``.
        upper: Finite upper bracket, with ``cdf_fn(upper) >= q``. Infinite
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
            (``cdf_fn(lower) <= q <= cdf_fn(upper)`` fails, including a NaN
            endpoint value, an infinite endpoint or a reversed bracket) or
            the solver does not
            converge. If ``False``, an unbracketed level
            returns the bracket endpoint whose CDF is closer to $q$ (its
            gradient is then meaningless), and a non-converged one the
            best estimate so far.

    Returns:
        Quantiles, with the broadcast shape of ``q``, ``lower`` and
        ``upper``.

    References:
        Chandrupatla, T. R. (1997). A new hybrid quadratic/bisection
        algorithm for finding the zero of a nonlinear function without
        using derivatives. *Advances in Engineering Software* 28(3),
        145-149.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jax.scipy.stats import norm
        >>> import gaussx
        >>> cdf = lambda x: 0.5 * (norm.cdf(x + 1.0) + norm.cdf(x - 1.0))
        >>> x = gaussx.mixture_quantile(cdf, jnp.array([0.5, 0.975]), -10.0, 10.0)
        >>> [round(float(v), 3) for v in x]
        [0.0, 2.646]
    """
    q = jnp.asarray(q)
    dtype = jnp.result_type(q, lower, upper, float)
    q = q.astype(dtype)
    shape = jnp.broadcast_shapes(q.shape, jnp.shape(lower), jnp.shape(upper))
    lower = jnp.broadcast_to(jnp.asarray(lower, dtype), shape)
    upper = jnp.broadcast_to(jnp.asarray(upper, dtype), shape)
    if solver is None:
        solver = Chandrupatla(rtol=rtol, atol=atol)

    def fn(x: Float[Array, ...], level: Float[Array, ...]) -> Float[Array, ...]:
        return cdf_fn(x) - level

    if throw:
        # Directed checks: they also reject NaN endpoint values and a
        # reversed bracket, which a sign-product test would let through.
        g_lo, g_hi = fn(lower, q), fn(upper, q)
        lower = eqx.error_if(
            lower,
            ~jnp.all(
                (g_lo <= 0) & (g_hi >= 0) & jnp.isfinite(lower) & jnp.isfinite(upper)
            ),
            "mixture_quantile: some levels q are not bracketed by finite "
            "endpoints, i.e. cdf_fn(lower) <= q <= cdf_fn(upper) fails or a "
            "bracket is infinite.",
        )
    sol = optx.root_find(
        fn,
        solver,
        0.5 * lower + 0.5 * upper,  # no overflow for wide finite brackets
        args=jnp.broadcast_to(q, shape),
        options=dict(lower=lower, upper=upper),
        max_steps=max_steps,
        adjoint=optx.ImplicitAdjoint(),
        throw=throw,
        tags=frozenset({lx.diagonal_tag}),
    )
    return sol.value


def mixture_quantile_gaussian_approx(
    means: Float[Array, "*batch E"],
    stds: Float[Array, "*batch E"],
    q: Float[ArrayLike, " Q"],
) -> Float[Array, "*batch Q"]:
    r"""Quantiles of the Gaussian matched to an ensemble's first two moments.

    For an equal-weight ensemble of Gaussians $\mathcal{N}(\mu_e,
    \sigma_e^2)$ the mixture has mean and variance

    $$
    \hat\mu = \frac{1}{E}\sum_e \mu_e, \qquad
    \hat\sigma^2 = \frac{1}{E}\sum_e \big(\sigma_e^2 + (\mu_e - \hat\mu)^2\big),
    $$

    and its quantiles are approximated by those of
    $\mathcal{N}(\hat\mu, \hat\sigma^2)$,
    $\hat x_q = \hat\mu + \hat\sigma\,\Phi^{-1}(q)$ — the deep-ensemble
    predictive of Lakshminarayanan, Pritzel & Blundell (2017). It is exact
    for a single component and cheap otherwise: a diagnostic, or a seed
    for the brackets of `mixture_quantile`. The variance is accumulated
    from deviations about $\hat\mu$, which equals the raw-moment form
    $\frac{1}{E}\sum_e(\sigma_e^2 + \mu_e^2) - \hat\mu^2$ without its
    cancellation.

    Args:
        means: Component means, shape ``(*batch, E)``.
        stds: Component standard deviations, shape ``(*batch, E)``.
        q: Quantile levels, shape ``(Q,)``.

    Returns:
        Approximate quantiles, shape ``(*batch, Q)``.

    References:
        Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and
        scalable predictive uncertainty estimation using deep ensembles.
        *NeurIPS 30*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> x = gaussx.mixture_quantile_gaussian_approx(
        ...     jnp.array([[0.0]]), jnp.array([[2.0]]), jnp.array([0.5, 0.8413447])
        ... )
        >>> [round(float(v), 4) for v in x[0]]
        [0.0, 2.0]
    """
    dtype = jnp.result_type(means, stds, q, float)
    means = jnp.asarray(means, dtype)
    stds = jnp.asarray(stds, dtype)
    q = jnp.atleast_1d(jnp.asarray(q, dtype))
    mu = reduce(means, "... e -> ...", "mean")
    dev = einx.subtract("... e, ... -> ... e", means, mu)
    # Rescaled norm: stds**2 would overflow for stds near sqrt(max float).
    big = jnp.maximum(
        reduce(jnp.abs(stds), "... e -> ...", "max"),
        reduce(jnp.abs(dev), "... e -> ...", "max"),
    )
    big = jnp.where(big > 0, big, jnp.ones_like(big))
    s_r = einx.divide("... e, ... -> ... e", stds, big)
    d_r = einx.divide("... e, ... -> ... e", dev, big)
    sigma = big * jnp.sqrt(reduce(s_r**2 + d_r**2, "... e -> ...", "mean"))
    z = jax.scipy.special.ndtri(q)
    return einx.add(
        "..., ... q -> ... q", mu, einx.multiply("..., q -> ... q", sigma, z)
    )
