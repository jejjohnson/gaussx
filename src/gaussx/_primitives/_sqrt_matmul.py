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
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
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

    Structured operators (diagonal, identity, Kronecker, block-diagonal,
    Kronecker-sum) return their exact spectra, including when wrapped in
    `lineax.TaggedLinearOperator` or scaled by a scalar -- `gaussx.eigvals`
    only recognises the bare classes. Everything else runs a partial Lanczos
    decomposition with full reorthogonalisation, which stops at the first
    breakdown: an operator such as ``cI + UUᵀ`` exhausts its Krylov space
    after ``rank(U) + 1`` steps, and continuing past that point only
    normalises rounding noise into spurious Ritz values.

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
    lam_min, lam_max = _spectral_bracket(operator, max_lanczos_iter, safety, key)
    floor = jnp.finfo(jnp.result_type(lam_min)).tiny
    lam_min = jnp.maximum(lam_min, floor)
    return lam_min, jnp.maximum(lam_max, lam_min)


def _spectral_bracket(
    operator: lx.AbstractLinearOperator,
    max_lanczos_iter: int,
    safety: float,
    key: jax.Array | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Unfloored ``(lam_min, lam_max)``; see `estimate_spectral_bounds`."""
    diagonal = _diagonal_of(operator)
    if diagonal is not None:
        return jnp.min(diagonal), jnp.max(diagonal)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _spectral_bracket(operator.operator, max_lanczos_iter, safety, key)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        low, high = _spectral_bracket(operator.operator, max_lanczos_iter, safety, key)
        factor = operator.scalar
        if isinstance(operator, lx.DivLinearOperator):
            factor = 1.0 / factor
        return (
            jnp.minimum(low * factor, high * factor),
            jnp.maximum(low * factor, high * factor),
        )
    if isinstance(operator, BlockDiag | Kronecker | KroneckerSum):
        values = jnp.real(eigvals(operator))
        return jnp.min(values), jnp.max(values)

    n = operator.in_size()
    order = min(max_lanczos_iter, n)
    low, high = _lanczos_extremes(operator, order, key)
    # Lanczos run for as many steps as the operator has dimensions is exact.
    widen = 1.0 if order >= n else safety
    return low / widen, high * widen


def _lanczos_extremes(
    operator: lx.AbstractLinearOperator,
    order: int,
    key: jax.Array | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Extreme Ritz values of ``order`` Lanczos steps, stopping at breakdown.

    Steps after a breakdown are masked out of the tridiagonal: their diagonal
    is set to the first Rayleigh quotient, which lies inside the spectrum and
    so cannot widen the bracket, and their couplings are zeroed.
    """
    n = operator.in_size()
    dtype = operator.in_structure().dtype
    if key is None:
        key = jr.PRNGKey(0)
    start = jr.normal(key, (n,), dtype=dtype)
    start = start / jnp.linalg.norm(start)
    eps = jnp.finfo(dtype).eps

    def step(carry, index):
        basis, vector, previous, beta_previous, alive = carry
        residual = operator.mv(vector) - beta_previous * previous
        alpha = jnp.dot(vector, residual)
        residual = residual - alpha * vector
        # Full reorthogonalisation; rows not yet written are zero.
        residual = residual - basis.T @ (basis @ residual)
        beta = jnp.linalg.norm(residual)
        # An exhausted Krylov space leaves nothing but rounding in the residual.
        tolerance = jnp.sqrt(eps) * (jnp.abs(alpha) + beta_previous)
        alive_next = alive & (beta > tolerance)
        following = jnp.where(
            alive_next, residual / jnp.where(alive_next, beta, 1.0), 0.0
        )
        basis = basis.at[index + 1].set(following, mode="drop")
        carry = (basis, following, vector, beta, alive_next)
        return carry, (alpha, beta, alive, alive_next)

    basis = jnp.zeros((order, n), dtype=dtype).at[0].set(start)
    init = (
        basis,
        start,
        jnp.zeros_like(start),
        jnp.zeros((), dtype=dtype),
        jnp.asarray(True),
    )
    _, (alphas, betas, alive, alive_next) = jax.lax.scan(step, init, jnp.arange(order))

    diagonal = jnp.where(alive, alphas, alphas[0])
    off_diagonal = jnp.where(alive_next[:-1], betas[:-1], 0.0)
    tridiagonal = (
        jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    )
    values = jnp.linalg.eigvalsh(tridiagonal)
    return values[0], values[-1]


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

    # ``ratio`` is the complementary parameter ``1 - modulus``. It is passed
    # alongside the modulus rather than recovered as ``1 - modulus``, which
    # rounds to zero in float32 once the condition number passes ~1e7.
    ratio = jnp.clip(lam_min / lam_max, _MIN_SPECTRAL_RATIO, 1.0)
    modulus = 1.0 - ratio
    quarter_period = _ellipk(modulus, complement=ratio)

    index = jnp.arange(1, num_quadrature + 1, dtype=lam_min.dtype)
    nodes = (index - 0.5) * quarter_period / num_quadrature
    sn, cn, dn = _ellipj(nodes, modulus, complement=ratio)

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
    """Build ``A + shift I``, keeping structure where it exists.

    Diagonals and identities -- tagged, scaled or bare -- stay diagonal; tags
    and scalars are peeled off (``cA + sI = c(A + (s/c)I)``) so the structure
    underneath is shifted instead; a `gaussx.BlockDiag` shifts each block; and
    a `gaussx.LowRankUpdate` shifts its base so the Woodbury solve still
    applies. Anything else would fall through to a dense factorisation per
    quadrature node.

    ``lineax`` does not propagate the positive-semidefinite tag across
    `lineax.AddLinearOperator`, so the sum is re-tagged: every shift is
    non-negative and ``A`` is assumed positive definite, which is what lets
    the fallback solver pick a Cholesky factorisation.
    """
    diagonal = _diagonal_of(operator)
    if diagonal is not None:
        # A diagonal needs no tags to take the structural solve path.
        return lx.DiagonalLinearOperator(diagonal + shift)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _shift_operator(operator.operator, shift)
    # ``A`` is positive definite, so the scalar is positive and the inner shift
    # stays non-negative.
    if isinstance(operator, lx.MulLinearOperator):
        inner = _shift_operator(operator.operator, shift / operator.scalar)
        return inner * operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        inner = _shift_operator(operator.operator, shift * operator.scalar)
        return inner / operator.scalar
    if isinstance(operator, BlockDiag):
        return BlockDiag(
            *(_shift_operator(block, shift) for block in operator.operators),
            tags=operator.tags,
        )
    if isinstance(operator, LowRankUpdate):
        return LowRankUpdate(
            _shift_operator(operator.base, shift),
            operator.U,
            operator.d,
            operator.V,
            tags=operator.tags,
            orthonormal=operator.orthonormal,
        )
    identity = lx.IdentityLinearOperator(operator.in_structure())
    return lx.TaggedLinearOperator(
        operator + shift * identity, lx.positive_semidefinite_tag
    )


def _diagonal_of(operator: lx.AbstractLinearOperator) -> Array | None:
    """The diagonal of a (tagged, scaled) diagonal or identity, else ``None``.

    Covers e.g. the PSD-tagged base of `gaussx.low_rank_plus_diag` and
    expressions such as ``2.0 * lineax.DiagonalLinearOperator(d)``.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, lx.IdentityLinearOperator):
        structure = operator.in_structure()
        return jnp.ones(structure.shape, dtype=structure.dtype)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _diagonal_of(operator.operator)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        inner = _diagonal_of(operator.operator)
        if inner is None:
            return None
        if isinstance(operator, lx.MulLinearOperator):
            return inner * operator.scalar
        return inner / operator.scalar
    return None


def _ellipk(
    modulus: Float[Array, ""],
    *,
    complement: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    r"""Complete elliptic integral of the first kind $K(m)$.

    Uses the arithmetic-geometric mean, $K(m) = \pi / (2\,\mathrm{agm}(1,
    \sqrt{1-m}))$, so the whole evaluation is a fixed-length JAX-traceable
    loop. ``modulus`` is the *parameter* $m = k^2$, matching the
    ``scipy.special.ellipk`` convention. Pass ``complement`` $= 1 - m$ when it
    is known more accurately than ``1 - modulus`` can be rounded.
    """
    if complement is None:
        complement = 1.0 - modulus
    a = jnp.ones_like(modulus)
    b = jnp.sqrt(complement)
    for _ in range(_AGM_ITERATIONS):
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
    return 0.5 * jnp.pi / a


def _ellipj(
    argument: Float[Array, " J"],
    modulus: Float[Array, ""],
    *,
    complement: Float[Array, ""] | None = None,
) -> tuple[Float[Array, " J"], Float[Array, " J"], Float[Array, " J"]]:
    r"""Jacobi elliptic functions $(\mathrm{sn}, \mathrm{cn}, \mathrm{dn})$.

    Descending Landen transformation (Abramowitz and Stegun 16.4): run the
    arithmetic-geometric mean forward to collapse the modulus to zero, where
    the functions reduce to $\sin$ and $\cos$, then descend the recorded
    sequence back to the requested modulus. ``modulus`` is the parameter
    $m = k^2$, matching ``scipy.special.ellipj``; ``complement`` is $1 - m$,
    as for `_ellipk`.
    """
    if complement is None:
        complement = 1.0 - modulus
    a = jnp.ones_like(modulus)
    b = jnp.sqrt(complement)
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
    # 1 - m sn^2 = cn^2 + (1 - m) sn^2, which avoids cancelling near m = 1.
    dn = jnp.sqrt(cn**2 + complement * sn**2)
    return sn, cn, dn
