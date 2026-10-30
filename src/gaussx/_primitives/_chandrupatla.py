"""Chandrupatla's bracketing root finder as an optimistix solver.

The solver finds a root of a scalar function f, entry by entry, keeping the
letters of Chandrupatla (1997) and Scherer (2010): a is the newest point, b
the bracket's other end, c the point a replaced, and f_a, f_b, f_c their
function values. A step tries x_t = (1 − t) a + t b for a fraction
t ∈ (0, 1); ξ and Φ decide whether the inverse quadratic step is safe, and
the tolerance is tol = rtol |x_m| + atol around the best point x_m.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Bool, Float, PyTree, Scalar


class _ChandrupatlaState(eqx.Module):
    """Per-entry iteration state of `Chandrupatla`.

    Attributes:
        a: Newest point a.
        b: Bracket end b, with f_a f_b ≤ 0.
        c: Previous point c.
        fa: Function value f_a = f(a).
        fb: Function value f_b = f(b).
        fc: Function value f_c = f(c).
        t: Fraction t of the next step x_t = (1 − t) a + t b.
        x_best: Best point x_m, whichever of a, b has the smaller |f|.
        done: Whether the entry has converged and is frozen.
    """

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
    """Chandrupatla's hybrid inverse-quadratic / bisection root finder.

        x_t = (1 − t) a + t b
        t_IQI = f_a/(f_b − f_a) · f_c/(f_b − f_c)
                + (c − a)/(b − a) · f_a/(f_c − f_a) · f_b/(f_c − f_b)

    A bracketing method for f(x) = 0 that keeps the robustness of bisection
    and the superlinear convergence of inverse quadratic interpolation
    (Chandrupatla, 1997). The bracket [a, b] with f_a f_b ≤ 0 and the
    previous point c are updated each step with the new point x_t. The next
    t is the inverse quadratic interpolant t_IQI through (a, b, c), accepted
    only when ξ = (a − b)/(c − b) and Φ = (f_a − f_b)/(f_c − f_b) satisfy
    Φ² < ξ and (1 − Φ)² < 1 − ξ, i.e. when the interpolant is monotone on
    the bracket; otherwise t = ½ (bisection). t is clipped to
    [t_lim, 1 − t_lim] with t_lim = ½ tol / |b − c| and
    tol = rtol |x_m| + atol, where x_m is whichever of a, b has the smaller
    |f|. The solve stops when t_lim > ½, i.e. |b − c| < tol, which bounds
    the bracket |b − a| ≤ |b − c| and so puts x_m within tol of the root; or
    when f(x_m) = 0; or when the bracket has shrunk to adjacent floats and
    x_t can no longer move. It returns x_m. (Scherer's formulation stops at
    2 tol; halving keeps the requested tolerance.)

    The solver is **elementwise**: ``y`` may be an array, ``fn`` must act
    on each entry independently (as a CDF evaluated pointwise does), and
    every entry carries its own bracket and stops on its own. The loop
    ends when all entries have converged. Pass ``tags=frozenset(
    {lineax.diagonal_tag})`` to `optimistix.root_find` so the implicit
    adjoint uses the diagonal Jacobian.

    Requires ``options=dict(lower=..., upper=...)`` (broadcastable to
    ``y``), as `optimistix.Bisection` does. The bracket and every iterate
    keep ``y``'s dtype; only the function values take ``fn``'s. An entry
    whose bracket has no sign change (or a NaN endpoint value) is not
    iterated: it returns the endpoint with the smaller |f|, so the caller
    decides whether that is an error. With ``has_aux=True`` the auxiliary
    output is that of ``fn`` at the returned root (one extra evaluation).

    Algorithm:
        Per entry, following Scherer (2010), §6.1:

        1. Start from b = lower, a = c = upper and t = ½; an entry whose
           bracket has no sign change, or is infinite, is done at once.
        2. Evaluate f at x_t = (1 − t) a + t b. If f(x_t) has the sign of
           f_a, set c ← a; otherwise set c ← b and b ← a. Then a ← x_t.
        3. Take x_m, whichever of a, b has the smaller |f|, and
           t_lim = (rtol |x_m| + atol) / (2 |b − c|). Stop if f(x_m) = 0,
           t_lim > ½, or x_t equalled a or b before the update.
        4. Take t = t_IQI if Φ² < ξ and (1 − Φ)² < 1 − ξ, else t = ½, and
           clip it to [t_lim, 1 − t_lim]. Repeat from step 2.

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
        The cube root of 2 on the bracket [0, 2]:

        >>> import jax.numpy as jnp
        >>> import optimistix as optx
        >>> import gaussx
        >>> sol = optx.root_find(
        ...     lambda x, _: x**3 - 2.0,
        ...     gaussx.Chandrupatla(rtol=1e-10, atol=1e-12),
        ...     jnp.array(1.0),
        ...     options=dict(lower=0.0, upper=2.0),
        ... )
        >>> round(float(sol.value), 6)
        1.259921

        Every entry carries its own bracket and target:

        >>> sol = optx.root_find(
        ...     lambda x, target: x**2 - target,
        ...     gaussx.Chandrupatla(rtol=1e-6, atol=1e-8),
        ...     jnp.ones(2),
        ...     args=jnp.array([4.0, 9.0]),
        ...     options=dict(lower=0.0, upper=jnp.array([5.0, 10.0])),
        ... )
        >>> [round(float(v), 4) for v in sol.value]
        [2.0, 3.0]
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
        """Set up the bracket b = lower, a = c = upper and t = ½.

        The arguments are those of ``optimistix.AbstractRootFinder.init``.

        Raises:
            ValueError: If ``fn``'s output is not one array of ``y``'s shape.
        """
        del aux_struct, tags
        if not isinstance(
            f_struct, jax.ShapeDtypeStruct
        ) or f_struct.shape != jnp.shape(y):
            raise ValueError(
                "Chandrupatla needs an elementwise function: the output must be "
                "a single array with the same shape as y."
            )
        shape, dtype = f_struct.shape, jnp.result_type(y)
        lower = jnp.broadcast_to(jnp.asarray(options["lower"], dtype), shape)
        upper = jnp.broadcast_to(jnp.asarray(options["upper"], dtype), shape)
        f_lower, _ = fn(lower, args)
        f_upper, _ = fn(upper, args)
        # Scherer's initialisation: b = lower, a = c = upper.
        a, b, c = upper, lower, upper
        fa, fb, fc = f_upper, f_lower, f_upper
        # NaN-aware: a finite residual always beats a NaN one.
        # |f_a| < |f_b|, or f_b is NaN
        a_better = (jnp.abs(fa) < jnp.abs(fb)) | jnp.isnan(fb)
        # x_m = argmin over a, b of |f|
        x_best = jnp.where(a_better, a, b)
        # f(x_m)
        f_best = jnp.where(a_better, fa, fb)
        # An infinite endpoint makes the convex-combination step NaN, so it
        # counts as unbracketed (finite brackets only).
        # f_a f_b ≤ 0, with a and b finite
        bracketed = (
            (jnp.sign(fa) * jnp.sign(fb) <= 0) & jnp.isfinite(a) & jnp.isfinite(b)
        )
        # tol = rtol |x_m| + atol
        tol = self.rtol * jnp.abs(x_best) + self.atol
        # done if unbracketed, f(x_m) = 0 or |b − a| < tol
        done = (~bracketed) | (f_best == 0) | (jnp.abs(b - a) < tol)
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
        """One Chandrupatla step on every entry that has not converged.

            x_t = (1 − t) a + t b

        The arguments are those of ``optimistix.AbstractRootFinder.step``;
        it returns the best point x_m, the new state and ``fn``'s aux at
        x_t.
        """
        del y, options, tags
        previous = state
        # The convex combination cannot overflow for finite a, b, unlike
        # a + t (b − a) on a bracket wider than the largest float.
        # x_t = (1 − t) a + t b
        x_t = (1 - previous.t) * previous.a + previous.t * previous.b
        # x_t ∈ {a, b}: the bracket is down to adjacent floats
        stalled = (x_t == previous.a) | (x_t == previous.b)
        f_t, aux = fn(x_t, args)

        # --- update the bracket ---
        # sign f(x_t) = sign f_a
        same_sign = jnp.sign(f_t) == jnp.sign(previous.fa)
        # c ← a if the signs agree, else c ← b
        c = jnp.where(same_sign, previous.a, previous.b)
        fc = jnp.where(same_sign, previous.fa, previous.fb)
        # b ← a if the signs differ
        b = jnp.where(same_sign, previous.b, previous.a)
        fb = jnp.where(same_sign, previous.fb, previous.fa)
        # a ← x_t
        a, fa = x_t, f_t

        # --- convergence ---
        # |f_a| < |f_b|, or f_b is NaN
        a_better = (jnp.abs(fa) < jnp.abs(fb)) | jnp.isnan(fb)
        # x_m = argmin over a, b of |f|
        x_best = jnp.where(a_better, a, b)
        # f(x_m)
        f_best = jnp.where(a_better, fa, fb)
        # tol = rtol |x_m| + atol
        tol = self.rtol * jnp.abs(x_best) + self.atol
        # t_lim = ½ tol / |b − c|
        t_lim = 0.5 * tol / jnp.abs(b - c)
        converged = (f_best == 0) | (t_lim > 0.5) | stalled

        # --- next fraction t ---
        # The divisions below may produce inf/nan on degenerate entries;
        # those entries either fail the IQI test (nan comparisons are
        # False) and bisect, or have already converged.
        # ξ = (a − b)/(c − b)
        xi = (a - b) / (c - b)
        # Φ = (f_a − f_b)/(f_c − f_b)
        phi = (fa - fb) / (fc - fb)
        # Φ² < ξ and (1 − Φ)² < 1 − ξ
        use_iqi = (phi**2 < xi) & ((1 - phi) ** 2 < 1 - xi)
        # f_a/(f_b − f_a) · f_c/(f_b − f_c)
        iqi_first = fa / (fb - fa) * fc / (fb - fc)
        # (c − a)/(b − a) · f_a/(f_c − f_a) · f_b/(f_c − f_b)
        iqi_second = (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
        # t_IQI
        t_iqi = iqi_first + iqi_second
        # t = t_IQI if accepted, else ½
        t = jnp.where(use_iqi, t_iqi, 0.5)
        # t ← clip(t, t_lim, 1 − t_lim)
        t = jnp.clip(t, t_lim, 1 - t_lim)

        updated = _ChandrupatlaState(
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
        updated = jax.tree.map(
            lambda old, new: jnp.where(previous.done, old, new), previous, updated
        )
        return updated.x_best, updated, aux

    def terminate(
        self,
        fn: Callable[[Array, Any], tuple[Array, Any]],
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        state: _ChandrupatlaState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], optx.RESULTS]:
        """Stop once every entry has converged.

        The arguments are those of ``optimistix.AbstractRootFinder.terminate``.
        """
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
        """Return x_m with ``fn``'s aux evaluated there.

        The arguments are those of
        ``optimistix.AbstractRootFinder.postprocess``.
        """
        del y, aux, options, tags, result
        # Re-evaluate so ``aux`` belongs to the returned point; the step's
        # aux is that of the last trial point, which may be the other end.
        _, aux = fn(state.x_best, args)
        return state.x_best, aux, {}
