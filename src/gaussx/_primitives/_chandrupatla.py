r"""Chandrupatla's bracketing root finder as an optimistix solver."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool, Float, PyTree, Scalar


class _ChandrupatlaState(eqx.Module):
    a: Float[Array, ...]
    b: Float[Array, ...]
    c: Float[Array, ...]
    fa: Float[Array, ...]
    fb: Float[Array, ...]
    fc: Float[Array, ...]
    t: Float[Array, ...]
    x_best: Float[Array, ...]
    done: Bool[Array, ...]


class Chandrupatla(optx.AbstractRootFinder):
    r"""Chandrupatla's hybrid inverse-quadratic / bisection root finder.

    A bracketing method for $f(x) = 0$ that keeps the robustness of
    bisection and the superlinear convergence of inverse quadratic
    interpolation (Chandrupatla, 1997). The bracket $[a, b]$ with
    $f(a) f(b) \le 0$ and the previous point $c$ are updated each step
    with the new point $x_t = a + t (b - a)$. The next $t$ is the inverse
    quadratic interpolant through $(a, b, c)$,

    $$
    t = \frac{f_a}{f_b - f_a}\frac{f_c}{f_b - f_c}
      + \frac{c - a}{b - a}\frac{f_a}{f_c - f_a}\frac{f_b}{f_c - f_b},
    $$

    accepted only when $\xi = (a - b)/(c - b)$ and
    $\Phi = (f_a - f_b)/(f_c - f_b)$ satisfy $\Phi^2 < \xi$ and
    $(1 - \Phi)^2 < 1 - \xi$, i.e. when the interpolant is monotone on the
    bracket; otherwise $t = 1/2$ (bisection). $t$ is clipped to
    $[t_\ell, 1 - t_\ell]$ with $t_\ell = \tau / |b - c|$ and
    $\tau = 2\,\mathrm{rtol}\,|x_m| + \mathrm{atol}$, where $x_m$ is
    whichever of $a, b$ has the smaller $|f|$. The solve stops when
    $t_\ell > 1/2$ (the bracket is narrower than $2\tau$) or
    $f(x_m) = 0$, and returns $x_m$.

    The solver is **elementwise**: ``y`` may be an array, ``fn`` must act
    on each entry independently (as a CDF evaluated pointwise does), and
    every entry carries its own bracket and stops on its own. The loop
    ends when all entries have converged. Pass ``tags=frozenset(
    {lineax.diagonal_tag})`` to `optimistix.root_find` so the implicit
    adjoint uses the diagonal Jacobian.

    Requires ``options=dict(lower=..., upper=...)`` (broadcastable to
    ``y``), as `optimistix.Bisection` does. An entry whose bracket has no
    sign change is not iterated: it returns the endpoint with the smaller
    $|f|$, so the caller decides whether that is an error.

    Pseudocode (per entry, Scherer 2010, §6.1):

        b, a, c = lower, upper, upper;  t = 1/2
        repeat:
            x_t = a + t (b − a);  f_t = f(x_t)
            if sign f_t = sign f_a:  c ← a          else:  c ← b; b ← a
            a ← x_t
            x_m = argmin over a, b of |f|;  t_lim = (2 rtol |x_m| + atol) / |b − c|
            stop if f(x_m) = 0 or t_lim > 1/2
            t = IQI(a, b, c) if Φ² < ξ and (1 − Φ)² < 1 − ξ else 1/2
            t = clip(t, t_lim, 1 − t_lim)

    Attributes:
        rtol: Relative tolerance on the root.
        atol: Absolute tolerance on the root.

    References:
        Chandrupatla, T. R. (1997). A new hybrid quadratic/bisection
        algorithm for finding the zero of a nonlinear function without
        using derivatives. *Advances in Engineering Software* 28(3),
        145-149.

        Scherer, P. O. J. (2010). *Computational Physics: Simulation of
        Classical and Quantum Systems*. Springer, §6.1.

    Examples:
        >>> import jax.numpy as jnp
        >>> import optimistix as optx
        >>> import gaussx
        >>> sol = optx.root_find(
        ...     lambda x, _: x**3 - 2.0,
        ...     gaussx.Chandrupatla(rtol=1e-10, atol=1e-12),
        ...     jnp.array(1.0),
        ...     options=dict(lower=0.0, upper=2.0),
        ... )
        >>> round(float(sol.value), 8)
        1.25992105
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def init(
        self,
        fn: Callable[[Array, Any], tuple[Array, Any]],
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        f_struct: Any,
        aux_struct: Any,
        tags: frozenset[object],
    ) -> _ChandrupatlaState:
        del aux_struct, tags
        if not isinstance(
            f_struct, jax.ShapeDtypeStruct
        ) or f_struct.shape != jnp.shape(y):
            raise ValueError(
                "Chandrupatla needs an elementwise function: the output must be "
                "a single array with the same shape as y."
            )
        shape, dtype = f_struct.shape, f_struct.dtype
        lower = jnp.broadcast_to(jnp.asarray(options["lower"], dtype), shape)
        upper = jnp.broadcast_to(jnp.asarray(options["upper"], dtype), shape)
        f_lower, _ = fn(lower, args)
        f_upper, _ = fn(upper, args)
        # Scherer's initialisation: b = lower, a = c = upper.
        a, b, c = upper, lower, upper
        fa, fb, fc = f_upper, f_lower, f_upper
        x_best = jnp.where(jnp.abs(fa) < jnp.abs(fb), a, b)
        f_best = jnp.where(jnp.abs(fa) < jnp.abs(fb), fa, fb)
        bracketed = jnp.sign(fa) * jnp.sign(fb) <= 0
        tol = 2 * self.rtol * jnp.abs(x_best) + self.atol
        done = (~bracketed) | (f_best == 0) | (jnp.abs(b - a) < 2 * tol)
        return _ChandrupatlaState(
            a=a,
            b=b,
            c=c,
            fa=fa,
            fb=fb,
            fc=fc,
            t=jnp.full(shape, 0.5, dtype),
            x_best=x_best,
            done=done,
        )

    def step(
        self,
        fn: Callable[[Array, Any], tuple[Array, Any]],
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
    ) -> tuple[Array, _ChandrupatlaState, Any]:
        del y, options, tags
        s = state
        x_t = s.a + s.t * (s.b - s.a)
        f_t, aux = fn(x_t, args)

        same = jnp.sign(f_t) == jnp.sign(s.fa)
        c = jnp.where(same, s.a, s.b)
        fc = jnp.where(same, s.fa, s.fb)
        b = jnp.where(same, s.b, s.a)
        fb = jnp.where(same, s.fb, s.fa)
        a, fa = x_t, f_t

        a_better = jnp.abs(fa) < jnp.abs(fb)
        x_best = jnp.where(a_better, a, b)
        f_best = jnp.where(a_better, fa, fb)
        tol = 2 * self.rtol * jnp.abs(x_best) + self.atol
        t_lim = tol / jnp.abs(b - c)
        converged = (f_best == 0) | (t_lim > 0.5)

        # The divisions below may produce inf/nan on degenerate entries;
        # those entries either fail the IQI test (nan comparisons are
        # False) and bisect, or have already converged.
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        use_iqi = (phi**2 < xi) & ((1 - phi) ** 2 < 1 - xi)
        t_iqi = fa / (fb - fa) * fc / (fb - fc) + (c - a) / (b - a) * fa / (
            fc - fa
        ) * fb / (fc - fb)
        t = jnp.clip(jnp.where(use_iqi, t_iqi, 0.5), t_lim, 1 - t_lim)

        new = _ChandrupatlaState(
            a=a,
            b=b,
            c=c,
            fa=fa,
            fb=fb,
            fc=fc,
            t=t,
            x_best=x_best,
            done=converged,
        )
        # Freeze entries that had already converged.
        new = jax.tree.map(lambda old, nw: jnp.where(s.done, old, nw), s, new)
        return new.x_best, new, aux

    def terminate(
        self,
        fn: Callable[[Array, Any], tuple[Array, Any]],
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        del fn, y, args, options, tags
        return jnp.all(state.done), optx.RESULTS.successful

    def postprocess(
        self,
        fn: Callable[[Array, Any], tuple[Array, Any]],
        y: Array,
        aux: Any,
        args: PyTree,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[Array, Any, dict[str, Any]]:
        del fn, y, args, options, tags, result
        return state.x_best, aux, {}
