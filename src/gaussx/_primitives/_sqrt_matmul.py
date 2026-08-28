r"""Matrix-free $A^{\pm 1/2} b$ via contour-integral quadrature.

Implements the Hale-Higham-Trefethen (2008) elliptic-integral quadrature for
the square root of a symmetric positive-definite operator. Only the *action*
of $A^{\pm 1/2}$ on a right-hand side is formed, as a weighted sum of shifted
solves $(A + \sigma_j I)^{-1} b$ dispatched through `gaussx.solve`, so the
cost is $J$ structural solves rather than an $O(N^3)$ factorisation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._primitives._eig import eigvals
from gaussx._primitives._solve import solve


# The arithmetic-geometric mean converges quadratically, so 12 steps take a
# float64 pair to machine precision even for a modulus within 1e-10 of 1.
_AGM_ITERATIONS = 12

_DEFAULT_NUM_QUADRATURE = 15
_DEFAULT_MAX_LANCZOS_ITER = 20

# Widening applied to a partial-Lanczos eigenvalue bracket; see
# ``estimate_spectral_bounds``.
_DEFAULT_SAFETY = 10.0

# Largest condition number the contour is allowed to be parameterised for.
# The elliptic modulus is ``1 - lam_min / lam_max``; pinning it strictly below
# one keeps ``ellipk`` finite (it diverges logarithmically at one).
_MIN_SPECTRAL_RATIO = 1e-14


def estimate_spectral_bounds(
    operator: lx.AbstractLinearOperator,
    *,
    max_lanczos_iter: int = _DEFAULT_MAX_LANCZOS_ITER,
    safety: float = _DEFAULT_SAFETY,
    key: jax.Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Estimate $(\lambda_{\min}, \lambda_{\max})$ of a symmetric operator.

    Routes through `gaussx.eigvals`, so structured operators (diagonal,
    Kronecker, block-diagonal, Kronecker-sum) return their exact spectra and
    everything else runs a partial Lanczos decomposition.

    Ritz values interlace the true spectrum, so a partial Lanczos run brackets
    it from the *inside* — and the smallest eigenvalue is the slowest one to
    converge, so the gap can be a factor of several. The estimate is therefore
    widened by ``safety`` in each direction whenever it came from a partial
    run. Widening is close to free for the contour quadrature that consumes
    these bounds: its node count grows only with $\log \kappa$.

    Args:
        operator: A square symmetric positive-definite operator.
        max_lanczos_iter: Number of Lanczos iterations (clamped to the
            operator size).
        safety: Factor by which a partial-Lanczos bracket is widened in each
            direction. Ignored when the full spectrum is available.
        key: PRNG key for the Lanczos start vector. Defaults to
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(lam_min, lam_max)`` of scalar eigenvalue estimates, floored
        at a positive value so downstream square roots stay finite.
    """
    if operator.in_size() != operator.out_size():
        raise ValueError("spectral bounds require a square operator")
    if safety < 1.0:
        raise ValueError("safety must be at least 1")
    n = operator.in_size()
    order = min(max_lanczos_iter, n)
    values = jnp.real(eigvals(operator, rank=order, key=key))
    # A full-length spectrum is exact -- either structurally or because
    # Lanczos ran for as many steps as the operator has dimensions.
    widen = 1.0 if values.shape[0] >= n else safety

    floor = jnp.finfo(values.dtype).tiny
    lam_min = jnp.maximum(jnp.min(values) / widen, floor)
    lam_max = jnp.maximum(jnp.max(values) * widen, lam_min)
    return lam_min, lam_max


def sqrt_inv_matmul(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    *,
    num_quadrature: int = _DEFAULT_NUM_QUADRATURE,
    max_lanczos_iter: int = _DEFAULT_MAX_LANCZOS_ITER,
    spectral_bounds: tuple[float, float] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, "N C"]:
    r"""Compute $A^{-1/2} b$ via contour-integral quadrature.

    Uses the integral representation

    $$A^{-1/2} = \frac{2}{\pi} \int_0^{\infty} (A + t^2 I)^{-1} \, dt$$

    discretised by the Hale-Higham-Trefethen elliptic-integral rule, which
    turns it into $J$ shifted solves

    $$A^{-1/2} b \approx \sum_{j=1}^{J} w_j \, (A + \sigma_j I)^{-1} b,
    \qquad \sigma_j \ge 0 .$$

    Each shifted operator is positive definite, so every solve goes through
    the ordinary `gaussx.solve` dispatch. Accuracy improves geometrically in
    $J$ with a rate that depends only *logarithmically* on the condition
    number $\kappa = \lambda_{\max} / \lambda_{\min}$: $J = 15$ reaches
    ``1e-13`` at $\kappa = 10^3$ and ``1e-7`` at $\kappa = 10^6$.

    The quadrature nodes and weights are treated as constants by
    `jax.grad` — they parameterise the contour, not the function value, and
    the rule is designed to be insensitive to them.

    Args:
        operator: Square symmetric positive-definite operator $A$.
        rhs: Right-hand side of shape ``(N, C)``.
        num_quadrature: Number of contour-quadrature nodes $J$.
        max_lanczos_iter: Lanczos iterations used to estimate the spectral
            bounds when ``spectral_bounds`` is not given.
        spectral_bounds: Optional ``(lam_min, lam_max)`` bracketing the
            spectrum. Supplying known bounds skips the Lanczos estimate.
        solver: Optional lineax solver used for the shifted solves. Pass
            ``lineax.CG(...)`` for large matrix-free operators.

    Returns:
        The array $A^{-1/2} b$ of shape ``(N, C)``.
    """
    shifts, weights = _contour_rule(
        operator,
        num_quadrature=num_quadrature,
        max_lanczos_iter=max_lanczos_iter,
        spectral_bounds=spectral_bounds,
        rhs=rhs,
    )
    solves = jax.vmap(lambda shift: _shifted_solve(operator, shift, rhs, solver))(
        shifts
    )
    return einsum(weights, solves, "j, j n c -> n c")


def sqrt_matmul(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    *,
    num_quadrature: int = _DEFAULT_NUM_QUADRATURE,
    max_lanczos_iter: int = _DEFAULT_MAX_LANCZOS_ITER,
    spectral_bounds: tuple[float, float] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, "N C"]:
    r"""Compute $A^{1/2} b$ via contour-integral quadrature.

    Shares the quadrature of `sqrt_inv_matmul` through the identity
    $A^{1/2} b = A \, (A^{-1/2} b)$, which costs one extra matvec per column.

    Args:
        operator: Square symmetric positive-definite operator $A$.
        rhs: Right-hand side of shape ``(N, C)``.
        num_quadrature: Number of contour-quadrature nodes $J$.
        max_lanczos_iter: Lanczos iterations used to estimate the spectral
            bounds when ``spectral_bounds`` is not given.
        spectral_bounds: Optional ``(lam_min, lam_max)`` bracketing the
            spectrum.
        solver: Optional lineax solver used for the shifted solves.

    Returns:
        The array $A^{1/2} b$ of shape ``(N, C)``.
    """
    whitened = sqrt_inv_matmul(
        operator,
        rhs,
        num_quadrature=num_quadrature,
        max_lanczos_iter=max_lanczos_iter,
        spectral_bounds=spectral_bounds,
        solver=solver,
    )
    return jax.vmap(operator.mv, in_axes=1, out_axes=1)(whitened)


def _contour_rule(
    operator: lx.AbstractLinearOperator,
    *,
    num_quadrature: int,
    max_lanczos_iter: int,
    spectral_bounds: tuple[float, float] | None,
    rhs: Float[Array, "N C"],
) -> tuple[Float[Array, " J"], Float[Array, " J"]]:
    """Validate the call and build the shifts/weights of the contour rule."""
    if operator.in_size() != operator.out_size():
        raise ValueError("contour-integral square roots require a square operator")
    if jnp.ndim(rhs) != 2:
        raise ValueError(
            f"rhs must have shape (N, C); got array with {jnp.ndim(rhs)} axes"
        )
    if rhs.shape[0] != operator.in_size():
        raise ValueError(
            f"rhs has {rhs.shape[0]} rows but operator has size {operator.in_size()}"
        )
    if num_quadrature < 1:
        raise ValueError("num_quadrature must be at least 1")

    if spectral_bounds is None:
        lam_min, lam_max = estimate_spectral_bounds(
            operator, max_lanczos_iter=max_lanczos_iter
        )
    else:
        lam_min, lam_max = (jnp.asarray(b, dtype=rhs.dtype) for b in spectral_bounds)
    return _quadrature_nodes(lam_min, lam_max, num_quadrature)


def _quadrature_nodes(
    lam_min: Float[Array, ""],
    lam_max: Float[Array, ""],
    num_quadrature: int,
) -> tuple[Float[Array, " J"], Float[Array, " J"]]:
    r"""Shifts and weights of the Hale-Higham-Trefethen rule.

    With $t = \sqrt{\lambda_{\min}}\,\mathrm{sn}(iu \mid k)$, $k^2 =
    \lambda_{\min} / \lambda_{\max}$, the midpoint rule on $u \in [0, K']$
    maps the $t$-integral onto nodes that cluster around the spectrum, which
    is what makes the convergence rate depend on $\kappa$ only through
    $\log \kappa$. Applying Jacobi's imaginary transformation turns the
    complex-argument elliptic functions into real ones and leaves purely real,
    non-negative shifts.
    """
    # The contour is a quadrature parameter, not part of the value: the rule is
    # constructed to be insensitive to it, and its Lanczos estimate is far too
    # noisy to differentiate through.
    lam_min = jax.lax.stop_gradient(lam_min)
    lam_max = jax.lax.stop_gradient(lam_max)

    ratio = jnp.clip(lam_min / lam_max, _MIN_SPECTRAL_RATIO, 1.0)
    modulus = 1.0 - ratio
    quarter_period = _ellipk(modulus)

    index = jnp.arange(1, num_quadrature + 1, dtype=lam_min.dtype)
    nodes = (index - 0.5) * quarter_period / num_quadrature
    sn, cn, dn = _ellipj(nodes, modulus)

    shifts = lam_min * (sn / cn) ** 2
    scale = 2.0 * quarter_period * jnp.sqrt(lam_min) / (jnp.pi * num_quadrature)
    weights = scale * dn / cn**2
    return shifts, weights


def _shifted_solve(
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, ""],
    rhs: Float[Array, "N C"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, "N C"]:
    """Solve ``(A + shift I) X = rhs`` column-by-column."""
    shifted = _shift_operator(operator, shift)
    return jax.vmap(
        lambda column: solve(shifted, column, solver=solver),
        in_axes=1,
        out_axes=1,
    )(rhs)


def _shift_operator(
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, ""],
) -> lx.AbstractLinearOperator:
    """Build ``A + shift I``, keeping diagonal structure where it exists.

    ``lineax`` does not propagate the positive-semidefinite tag across
    `lineax.AddLinearOperator`, so the sum is re-tagged: every shift is
    non-negative and ``A`` is assumed positive definite, which is what lets
    the fallback solver pick a Cholesky factorisation.
    """
    if isinstance(operator, lx.DiagonalLinearOperator | lx.IdentityLinearOperator):
        return lx.DiagonalLinearOperator(lx.diagonal(operator) + shift)
    identity = lx.IdentityLinearOperator(operator.in_structure())
    return lx.TaggedLinearOperator(
        operator + shift * identity, lx.positive_semidefinite_tag
    )


def _ellipk(modulus: Float[Array, ""]) -> Float[Array, ""]:
    r"""Complete elliptic integral of the first kind $K(m)$.

    Uses the arithmetic-geometric mean, $K(m) = \pi / (2\,\mathrm{agm}(1,
    \sqrt{1-m}))$, so the whole evaluation is a fixed-length JAX-traceable
    loop. ``modulus`` is the *parameter* $m = k^2$, matching the
    ``scipy.special.ellipk`` convention.
    """
    a = jnp.ones_like(modulus)
    b = jnp.sqrt(1.0 - modulus)
    for _ in range(_AGM_ITERATIONS):
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
    return 0.5 * jnp.pi / a


def _ellipj(
    argument: Float[Array, " J"],
    modulus: Float[Array, ""],
) -> tuple[Float[Array, " J"], Float[Array, " J"], Float[Array, " J"]]:
    r"""Jacobi elliptic functions $(\mathrm{sn}, \mathrm{cn}, \mathrm{dn})$.

    Descending Landen transformation (Abramowitz and Stegun 16.4): run the
    arithmetic-geometric mean forward to collapse the modulus to zero, where
    the functions reduce to $\sin$ and $\cos$, then descend the recorded
    sequence back to the requested modulus. ``modulus`` is the parameter
    $m = k^2$, matching ``scipy.special.ellipj``.
    """
    a = jnp.ones_like(modulus)
    b = jnp.sqrt(1.0 - modulus)
    c = jnp.sqrt(modulus)
    means, complements = [a], [c]
    for _ in range(_AGM_ITERATIONS):
        a, b, c = 0.5 * (a + b), jnp.sqrt(a * b), 0.5 * (a - b)
        means.append(a)
        complements.append(c)

    phase = (2.0**_AGM_ITERATIONS) * means[-1] * argument
    for step in range(_AGM_ITERATIONS, 0, -1):
        ratio = complements[step] / means[step]
        # ``ratio`` is below one in exact arithmetic; the clip only guards the
        # rounding boundary, where arcsin would otherwise return NaN.
        phase = 0.5 * (phase + jnp.arcsin(jnp.clip(ratio * jnp.sin(phase), -1.0, 1.0)))

    sn = jnp.sin(phase)
    cn = jnp.cos(phase)
    dn = jnp.sqrt(jnp.clip(1.0 - modulus * sn**2, 0.0, None))
    return sn, cn, dn
