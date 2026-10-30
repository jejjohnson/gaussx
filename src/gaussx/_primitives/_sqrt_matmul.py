"""Matrix-free A^½ B and A^−½ B by contour-integral quadrature.

A is a symmetric positive-definite N × N operator with eigenvalues in
[γ_min, γ_max] and condition number κ = γ_max / γ_min; B is an N × C
right-hand side. This implements the Hale-Higham-Trefethen (2008)
elliptic-integral quadrature with J nodes: only the *action* of A^±½ on B is
formed, as a weighted sum ∑_j w_j (A + σ_j I)⁻¹ B of shifted solves
dispatched through `gaussx.solve`, so the cost is J structural solves rather
than an O(N³) factorisation. The contour is built from Jacobi's elliptic
functions sn, cn and dn of parameter m and the complete elliptic integral
K(m), in the ``scipy.special`` convention m = k².
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
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
# The elliptic parameter is m = 1 − γ_min / γ_max; pinning it strictly below
# one keeps K(m) finite (it diverges logarithmically at one).
_MIN_SPECTRAL_RATIO = 1e-14


def estimate_spectral_bounds(
    operator: lx.AbstractLinearOperator,
    *,
    max_lanczos_iter: int = _DEFAULT_MAX_LANCZOS_ITER,
    safety: float = _DEFAULT_SAFETY,
    key: jax.Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Estimate the extreme eigenvalues (γ_min, γ_max) of a symmetric operator.

        γ_min ≤ γ_i(A) ≤ γ_max

    Structured operators (diagonal, identity, Kronecker, block-diagonal,
    Kronecker-sum) return their exact spectra, including when wrapped in
    `lineax.TaggedLinearOperator` or scaled by a scalar (`gaussx.eigvals`
    only recognises the bare classes). Everything else runs a partial
    Lanczos decomposition with full reorthogonalisation, which stops at the
    first breakdown: an operator such as c I + U Uᵀ exhausts its Krylov
    space after rank(U) + 1 steps, and continuing past that point only
    normalises rounding noise into spurious Ritz values.

    Ritz values interlace the true spectrum, so a partial Lanczos run
    brackets it from the *inside*, and the smallest eigenvalue is the
    slowest one to converge, so the gap can be a factor of several. The
    estimate is therefore widened by ``safety`` in each direction whenever
    it came from a partial run. Widening is close to free for the contour
    quadrature that consumes these bounds: its node count grows only with
    log κ.

    Algorithm:
        1. Read the exact extremes off a (tagged, scaled) diagonal or
           identity, or off the eigenvalues of a block-diagonal, Kronecker
           or Kronecker-sum operator.
        2. Otherwise run up to ``max_lanczos_iter`` Lanczos steps, stopping
           at breakdown, and take the extreme Ritz values.
        3. Widen a partial-Lanczos bracket to (γ_min / safety,
           γ_max · safety).
        4. Floor γ_min at the smallest positive normal number and γ_max at
           γ_min.

    Args:
        operator: The square symmetric positive-definite operator A.
        max_lanczos_iter: Number of Lanczos iterations (clamped to the
            operator size N).
        safety: Factor by which a partial-Lanczos bracket is widened in each
            direction. Ignored when the full spectrum is available.
        key: PRNG key for the Lanczos start vector. Defaults to
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(lam_min, lam_max)`` holding the scalar estimates γ_min and
        γ_max, floored at a positive value so downstream square roots stay
        finite.

    Raises:
        ValueError: If A is not square or ``safety`` is below one.

    Examples:
        A diagonal operator gives its exact extremes:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([3.0, 1.0, 4.0]))
        >>> [round(float(g), 4) for g in gaussx.estimate_spectral_bounds(A)]
        [1.0, 4.0]

        A dense operator no larger than the Lanczos budget is exact too:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> [round(float(g), 4) for g in gaussx.estimate_spectral_bounds(A)]
        [1.0, 3.0]
    """
    if operator.in_size() != operator.out_size():
        raise ValueError("spectral bounds require a square operator")
    if safety < 1.0:
        raise ValueError("safety must be at least 1")
    lam_min, lam_max = _spectral_bracket(operator, max_lanczos_iter, safety, key)
    floor = jnp.finfo(jnp.result_type(lam_min)).tiny
    # γ_min ← max(γ_min, floor)
    lam_min = jnp.maximum(lam_min, floor)
    # γ_max ← max(γ_max, γ_min)
    return lam_min, jnp.maximum(lam_max, lam_min)


def _spectral_bracket(
    operator: lx.AbstractLinearOperator,
    max_lanczos_iter: int,
    safety: float,
    key: jax.Array | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Unfloored (γ_min, γ_max); see `estimate_spectral_bounds`.

    Args:
        operator: The square symmetric operator A.
        max_lanczos_iter: Maximum number of Lanczos iterations.
        safety: Widening factor for a partial-Lanczos bracket.
        key: PRNG key for the Lanczos start vector, or ``None``.

    Returns:
        Tuple ``(lam_min, lam_max)`` holding γ_min and γ_max.
    """
    diagonal = _diagonal_of(operator)
    if diagonal is not None:
        # (min_i a_i, max_i a_i) for A = diag(a)
        return jnp.min(diagonal), jnp.max(diagonal)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _spectral_bracket(operator.operator, max_lanczos_iter, safety, key)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        low, high = _spectral_bracket(operator.operator, max_lanczos_iter, safety, key)
        factor = operator.scalar
        if isinstance(operator, lx.DivLinearOperator):
            # A / c = (1 / c) A
            factor = 1.0 / factor
        # c γ_min and c γ_max, in either order for a negative c
        return (
            jnp.minimum(low * factor, high * factor),
            jnp.maximum(low * factor, high * factor),
        )
    if isinstance(operator, BlockDiag | Kronecker | KroneckerSum):
        # Re γ, exact from the blocks or factors
        values = jnp.real(eigvals(operator))
        return jnp.min(values), jnp.max(values)

    size = operator.in_size()
    order = min(max_lanczos_iter, size)
    low, high = _lanczos_extremes(operator, order, key)
    # Lanczos run for as many steps as the operator has dimensions is exact.
    widen = 1.0 if order >= size else safety
    # (γ_min / safety, γ_max · safety)
    return low / widen, high * widen


def _lanczos_extremes(
    operator: lx.AbstractLinearOperator,
    order: int,
    key: jax.Array | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Extreme Ritz values of ``order`` Lanczos steps, stopping at breakdown.

        A P_m = P_m T_m + β_m p_m+1 e_mᵀ,   T_m = tridiag(β, α, β)

    Steps after a breakdown are masked out of the tridiagonal: their
    diagonal is set to the first Rayleigh quotient α_1, which lies inside
    the spectrum and so cannot widen the bracket, and their couplings are
    zeroed.

    Args:
        operator: The square symmetric operator A.
        order: Number of Lanczos steps m.
        key: PRNG key for the start vector p_1, or ``None``.

    Returns:
        Tuple ``(lam_min, lam_max)`` holding the extreme eigenvalues of T_m.
    """
    size = operator.in_size()
    dtype = operator.in_structure().dtype
    if key is None:
        key = jr.PRNGKey(0)
    start = jr.normal(key, (size,), dtype=dtype)
    # p_1 = z / ‖z‖₂
    start = start / jnp.linalg.norm(start)
    eps = jnp.finfo(dtype).eps

    def step(carry, index):
        basis, vector, previous, beta_previous, alive = carry
        # A p_i
        applied = operator.mv(vector)
        # r = A p_i − β_i−1 p_i−1
        residual = applied - beta_previous * previous
        # α_i = p_iᵀ r
        alpha = jnp.dot(vector, residual)
        # r ← r − α_i p_i
        residual = residual - alpha * vector
        # Full reorthogonalisation; rows not yet written are zero.
        # P_mᵀ r, one coefficient per basis row
        coefficients = basis @ residual
        # P_m P_mᵀ r
        projection = einsum(basis, coefficients, "step row, step -> row")
        # r ← (I − P_m P_mᵀ) r
        residual = residual - projection
        # β_i = ‖r‖₂
        beta = jnp.linalg.norm(residual)
        # An exhausted Krylov space leaves nothing but rounding in the residual.
        # √ε (∣α_i∣ + β_i−1)
        tolerance = jnp.sqrt(eps) * (jnp.abs(alpha) + beta_previous)
        alive_next = alive & (beta > tolerance)
        # p_i+1 = r / β_i, or 0 after a breakdown
        safe_beta = jnp.where(alive_next, beta, 1.0)
        following = jnp.where(alive_next, residual / safe_beta, 0.0)
        basis = basis.at[index + 1].set(following, mode="drop")
        carry = (basis, following, vector, beta, alive_next)
        return carry, (alpha, beta, alive, alive_next)

    basis = jnp.zeros((order, size), dtype=dtype).at[0].set(start)
    init = (
        basis,
        start,
        jnp.zeros_like(start),
        jnp.zeros((), dtype=dtype),
        jnp.asarray(True),
    )
    _, (alphas, betas, alive, alive_next) = jax.lax.scan(step, init, jnp.arange(order))

    # α_i on live steps, α_1 after a breakdown
    diagonal = jnp.where(alive, alphas, alphas[0])
    # β_i on live couplings, 0 after a breakdown
    off_diagonal = jnp.where(alive_next[:-1], betas[:-1], 0.0)
    # T_m = tridiag(β, α, β)
    tridiagonal = (
        jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    )
    # γ(T_m), ascending
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
    """Inverse square root times a matrix, A^−½ B, by contour quadrature.

        A^−½ = (2 / π) ∫₀^∞ (A + t² I)⁻¹ dt
        A^−½ B ≈ ∑_j w_j (A + σ_j I)⁻¹ B,   σ_j ≥ 0,  j = 1, …, J

    The integral is discretised by the Hale-Higham-Trefethen
    elliptic-integral rule, which turns it into J shifted solves. Each
    shifted operator is positive definite, so every solve goes through the
    ordinary `gaussx.solve` dispatch. Accuracy improves geometrically in J
    with a rate that depends only *logarithmically* on the condition number
    κ = γ_max / γ_min: J = 15 reaches ``1e-13`` at κ = 10³ and ``1e-7`` at
    κ = 10⁶.

    The quadrature nodes and weights are treated as constants by
    `jax.grad`: they parameterise the contour, not the function value, and
    the rule is designed to be insensitive to them.

    Algorithm:
        1. Validate the shapes and bracket the spectrum, from
           ``spectral_bounds`` or `estimate_spectral_bounds`.
        2. Build the J shifts σ_j and weights w_j of the elliptic rule.
        3. Solve (A + σ_j I) X_j = B for every node, keeping the structure
           of A in the shifted operator where it can.
        4. Sum the weighted solutions, ∑_j w_j X_j.

    Args:
        operator: The square symmetric positive-definite operator A.
        rhs: The right-hand side B, N × C.
        num_quadrature: Number of contour-quadrature nodes J.
        max_lanczos_iter: Lanczos iterations used to estimate the spectral
            bounds when ``spectral_bounds`` is not given.
        spectral_bounds: Optional ``(lam_min, lam_max)``, a bracket
            (γ_min, γ_max) of the spectrum. Supplying known bounds skips the
            Lanczos estimate.
        solver: Optional lineax solver used for the shifted solves. Pass
            ``lineax.CG(...)`` for large matrix-free operators.

    Returns:
        The product A^−½ B, N × C.

    Raises:
        ValueError: If A is not square, B is not N × C, or
            ``num_quadrature`` is below one.

    Examples:
        On a diagonal operator, A^−½ divides by the roots of its entries:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([4.0, 9.0]))
        >>> X = gaussx.sqrt_inv_matmul(A, jnp.ones((2, 1)))
        >>> [round(float(x), 4) for x in X[:, 0]]
        [0.5, 0.3333]

        It whitens a dense covariance, ‖A^−½ b‖₂² = bᵀ A⁻¹ b:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([[1.0], [2.0]])
        >>> whitened = gaussx.sqrt_inv_matmul(A, b)
        >>> quadratic = b[:, 0] @ jnp.linalg.solve(matrix, b[:, 0])
        >>> bool(jnp.allclose(jnp.sum(whitened**2), quadratic, atol=1e-4))
        True
    """
    shifts, weights = _contour_rule(
        operator,
        num_quadrature=num_quadrature,
        max_lanczos_iter=max_lanczos_iter,
        spectral_bounds=spectral_bounds,
        rhs=rhs,
    )
    # X_j = (A + σ_j I)⁻¹ B, one per node
    solves = jax.vmap(lambda shift: _shifted_solve(operator, shift, rhs, solver))(
        shifts
    )
    # A^−½ B ≈ ∑_j w_j X_j
    return einsum(weights, solves, "node, node row col -> row col")


def sqrt_matmul(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    *,
    num_quadrature: int = _DEFAULT_NUM_QUADRATURE,
    max_lanczos_iter: int = _DEFAULT_MAX_LANCZOS_ITER,
    spectral_bounds: tuple[float, float] | None = None,
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, "N C"]:
    """Square root times a matrix, A^½ B, by contour quadrature.

        A^½ B = A (A^−½ B)

    Shares the quadrature of `sqrt_inv_matmul` through the identity above,
    which costs one extra matvec per column.

    Algorithm:
        1. Whiten: A^−½ B by `sqrt_inv_matmul`.
        2. Apply A to each column.

    Args:
        operator: The square symmetric positive-definite operator A.
        rhs: The right-hand side B, N × C.
        num_quadrature: Number of contour-quadrature nodes J.
        max_lanczos_iter: Lanczos iterations used to estimate the spectral
            bounds when ``spectral_bounds`` is not given.
        spectral_bounds: Optional ``(lam_min, lam_max)``, a bracket
            (γ_min, γ_max) of the spectrum.
        solver: Optional lineax solver used for the shifted solves.

    Returns:
        The product A^½ B, N × C.

    Raises:
        ValueError: If A is not square, B is not N × C, or
            ``num_quadrature`` is below one.

    Examples:
        On a diagonal operator, A^½ multiplies by the roots of its entries:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([4.0, 9.0]))
        >>> Y = gaussx.sqrt_matmul(A, jnp.ones((2, 1)))
        >>> [round(float(y), 4) for y in Y[:, 0]]
        [2.0, 3.0]

        It agrees with the dense symmetric root:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> B = jnp.eye(2)
        >>> dense_root = gaussx.sqrt(A).as_matrix()
        >>> bool(jnp.allclose(gaussx.sqrt_matmul(A, B), dense_root, atol=1e-4))
        True
    """
    # A^−½ B
    whitened = sqrt_inv_matmul(
        operator,
        rhs,
        num_quadrature=num_quadrature,
        max_lanczos_iter=max_lanczos_iter,
        spectral_bounds=spectral_bounds,
        solver=solver,
    )
    # A^½ B = A (A^−½ B), column by column
    return jax.vmap(operator.mv, in_axes=1, out_axes=1)(whitened)


def _contour_rule(
    operator: lx.AbstractLinearOperator,
    *,
    num_quadrature: int,
    max_lanczos_iter: int,
    spectral_bounds: tuple[float, float] | None,
    rhs: Float[Array, "N C"],
) -> tuple[Float[Array, " J"], Float[Array, " J"]]:
    """Validate the call and build the shifts σ_j and weights w_j of the rule.

    Args:
        operator: The square symmetric positive-definite operator A.
        num_quadrature: Number of nodes J.
        max_lanczos_iter: Lanczos iterations for the spectral estimate.
        spectral_bounds: Optional bracket (γ_min, γ_max).
        rhs: The right-hand side B, which fixes the bounds' dtype.

    Returns:
        Tuple ``(shifts, weights)`` holding σ_j and w_j, each of length J.

    Raises:
        ValueError: If A is not square, B is not N × C, or J < 1.
    """
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
        lam_min, lam_max = (
            jnp.asarray(bound, dtype=rhs.dtype) for bound in spectral_bounds
        )
    return _quadrature_nodes(lam_min, lam_max, num_quadrature)


def _quadrature_nodes(
    lam_min: Float[Array, ""],
    lam_max: Float[Array, ""],
    num_quadrature: int,
) -> tuple[Float[Array, " J"], Float[Array, " J"]]:
    """Shifts σ_j and weights w_j of the Hale-Higham-Trefethen rule.

        u_j = (j − ½) K(m) / J,   m = 1 − γ_min / γ_max
        σ_j = γ_min (sn(u_j) / cn(u_j))²
        w_j = 2 K(m) √γ_min dn(u_j) / (π J cn(u_j)²)

    With t = √γ_min sn(iu ∣ k), k² = γ_min / γ_max, the midpoint rule on
    u ∈ [0, K′] maps the t-integral onto nodes that cluster around the
    spectrum, which is what makes the convergence rate depend on κ only
    through log κ. Applying Jacobi's imaginary transformation turns the
    complex-argument elliptic functions into real ones of the complementary
    parameter m = 1 − k² and leaves purely real, non-negative shifts.

    Args:
        lam_min: Lower spectral bound γ_min.
        lam_max: Upper spectral bound γ_max.
        num_quadrature: Number of nodes J.

    Returns:
        Tuple ``(shifts, weights)`` holding σ_j and w_j, each of length J.
    """
    # The contour is a quadrature parameter, not part of the value: the rule is
    # constructed to be insensitive to it, and its Lanczos estimate is far too
    # noisy to differentiate through.
    lam_min = jax.lax.stop_gradient(lam_min)
    lam_max = jax.lax.stop_gradient(lam_max)

    # ``ratio`` is the complementary parameter 1 − m. It is passed alongside
    # the parameter rather than recovered as 1 − m, which rounds to zero in
    # float32 once the condition number passes ~1e7.
    # 1 − m = γ_min / γ_max, clipped to [1e-14, 1]
    ratio = jnp.clip(lam_min / lam_max, _MIN_SPECTRAL_RATIO, 1.0)
    # m = 1 − γ_min / γ_max
    modulus = 1.0 - ratio
    # K(m)
    quarter_period = _ellipk(modulus, complement=ratio)

    index = jnp.arange(1, num_quadrature + 1, dtype=lam_min.dtype)
    # (j − ½) K(m)
    scaled_index = (index - 0.5) * quarter_period
    # u_j = (j − ½) K(m) / J
    nodes = scaled_index / num_quadrature
    sn, cn, dn = _ellipj(nodes, modulus, complement=ratio)

    # sn / cn
    sn_over_cn = sn / cn
    # σ_j = γ_min (sn / cn)²
    shifts = lam_min * sn_over_cn**2
    # 2 K(m)
    twice_period = 2.0 * quarter_period
    # 2 K(m) √γ_min
    scale_numerator = twice_period * jnp.sqrt(lam_min)
    # 2 K(m) √γ_min / (π J)
    scale = scale_numerator / (jnp.pi * num_quadrature)
    # w_j = 2 K(m) √γ_min dn / (π J cn²)
    weights = scale * dn / cn**2
    return shifts, weights


def _shifted_solve(
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, ""],
    rhs: Float[Array, "N C"],
    solver: lx.AbstractLinearSolver | None,
) -> Float[Array, "N C"]:
    """Solve (A + σ I) X = B column by column.

    Args:
        operator: The square operator A.
        shift: The scalar shift σ.
        rhs: The right-hand side B, N × C.
        solver: Optional lineax solver for each solve.

    Returns:
        The solution X, N × C.
    """
    # A + σ I
    shifted = _shift_operator(operator, shift)
    # X = (A + σ I)⁻¹ B, one column at a time
    return jax.vmap(
        lambda column: solve(shifted, column, solver=solver),
        in_axes=1,
        out_axes=1,
    )(rhs)


def _shift_operator(
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, ""],
) -> lx.AbstractLinearOperator:
    """Build A + σ I, keeping structure where it exists.

    Diagonals and identities (tagged, scaled or bare) stay diagonal; tags
    and scalars are peeled off (c A + σ I = c (A + (σ / c) I)) so the
    structure underneath is shifted instead; a `gaussx.BlockDiag` shifts
    each block; a `gaussx.LowRankUpdate` shifts its base so the Woodbury
    solve still applies; a `gaussx.KroneckerSum` shifts one factor
    ((A_1 ⊕ A_2) + σ I = (A_1 + σ I) ⊕ A_2); and a `gaussx.BlockTriDiag`
    shifts its diagonal blocks. Anything else, including a multi-factor
    `gaussx.Kronecker`, which has no cheap shift, falls through to a dense
    factorisation per quadrature node.

    ``lineax`` does not propagate the positive-semidefinite tag across
    `lineax.AddLinearOperator`, so the sum is re-tagged: every shift is
    non-negative and A is assumed positive definite, which is what lets
    the fallback solver pick a Cholesky factorisation.

    Args:
        operator: The square symmetric positive-definite operator A.
        shift: The non-negative scalar shift σ.

    Returns:
        The operator A + σ I, structured where A is.
    """
    diagonal = _diagonal_of(operator)
    if diagonal is not None:
        # A diagonal needs no tags to take the structural solve path.
        # diag(a + σ)
        return lx.DiagonalLinearOperator(diagonal + shift)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _shift_operator(operator.operator, shift)
    # A is positive definite, so the scalar is positive and the inner shift
    # stays non-negative.
    if isinstance(operator, lx.MulLinearOperator):
        # c A + σ I = c (A + (σ / c) I)
        inner = _shift_operator(operator.operator, shift / operator.scalar)
        return inner * operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        # A / c + σ I = (A + σ c I) / c
        inner = _shift_operator(operator.operator, shift * operator.scalar)
        return inner / operator.scalar
    if isinstance(operator, BlockDiag):
        # blockdiag(A_d + σ I), block by block
        return BlockDiag(
            *(_shift_operator(block, shift) for block in operator.operators),
            tags=operator.tags,
        )
    if isinstance(operator, LowRankUpdate):
        # (A₀ + σ I) + U D Vᵀ
        return LowRankUpdate(
            _shift_operator(operator.base, shift),
            operator.U,
            operator.d,
            operator.V,
            tags=operator.tags,
            orthonormal=operator.orthonormal,
        )
    if isinstance(operator, KroneckerSum):
        # (A_1 ⊕ A_2) + σ I = (A_1 + σ I) ⊕ A_2, since I = I_1 ⊗ I_2 (gh-337).
        return KroneckerSum(
            _shift_operator(operator.A, shift), operator.B, tags=operator.tags
        )
    if isinstance(operator, BlockTriDiag):
        # Only the diagonal blocks change (gh-337).
        eye = jnp.eye(operator._block_size, dtype=operator.diagonal.dtype)
        # σ I
        shifted_eye = shift * eye
        # A_tt + σ I, for every diagonal block
        shifted_diagonal = einx.add(
            "block row col, row col -> block row col",
            operator.diagonal,
            shifted_eye,
        )
        return BlockTriDiag(
            shifted_diagonal,
            operator.sub_diagonal,
            symmetric=operator.symmetric,
            tags=operator.tags,
        )
    identity = lx.IdentityLinearOperator(operator.in_structure())
    # A + σ I, re-tagged symmetric PSD
    return lx.TaggedLinearOperator(
        operator + shift * identity,
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


def _diagonal_of(operator: lx.AbstractLinearOperator) -> Array | None:
    """The diagonal a of a (tagged, scaled) diagonal or identity, else ``None``.

    Covers e.g. the PSD-tagged base of `gaussx.low_rank_plus_diag` and
    expressions such as ``2.0 * lineax.DiagonalLinearOperator(d)``.

    Args:
        operator: The square operator A.

    Returns:
        The diagonal a with A = diag(a), or ``None`` when A is not a
        (wrapped) diagonal or identity.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, lx.IdentityLinearOperator):
        structure = operator.in_structure()
        # I = diag(1)
        return jnp.ones(structure.shape, dtype=structure.dtype)
    if isinstance(operator, lx.TaggedLinearOperator):
        return _diagonal_of(operator.operator)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        inner = _diagonal_of(operator.operator)
        if inner is None:
            return None
        if isinstance(operator, lx.MulLinearOperator):
            # c a
            return inner * operator.scalar
        # a / c
        return inner / operator.scalar
    return None


def _ellipk(
    modulus: Float[Array, ""],
    *,
    complement: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    """Complete elliptic integral of the first kind K(m).

        K(m) = π / (2 agm(1, √(1 − m)))

    The arithmetic-geometric mean makes the whole evaluation a fixed-length
    JAX-traceable loop. ``modulus`` is the *parameter* m = k², matching the
    ``scipy.special.ellipk`` convention. Pass ``complement`` = 1 − m when it
    is known more accurately than ``1 - modulus`` can be rounded.

    Args:
        modulus: The parameter m.
        complement: The complementary parameter 1 − m, or ``None`` to form
            it from m.

    Returns:
        The scalar K(m).
    """
    if complement is None:
        # 1 − m
        complement = 1.0 - modulus
    # a₀ = 1
    arithmetic_mean = jnp.ones_like(modulus)
    # b₀ = √(1 − m)
    geometric_mean = jnp.sqrt(complement)
    for _ in range(_AGM_ITERATIONS):
        # a_n+1 = (a_n + b_n) / 2
        next_arithmetic = 0.5 * (arithmetic_mean + geometric_mean)
        # b_n+1 = √(a_n b_n)
        next_geometric = jnp.sqrt(arithmetic_mean * geometric_mean)
        arithmetic_mean, geometric_mean = next_arithmetic, next_geometric
    # K(m) = π / (2 agm(1, √(1 − m)))
    return 0.5 * jnp.pi / arithmetic_mean


def _ellipj(
    argument: Float[Array, " J"],
    modulus: Float[Array, ""],
    *,
    complement: Float[Array, ""] | None = None,
) -> tuple[Float[Array, " J"], Float[Array, " J"], Float[Array, " J"]]:
    """Jacobi elliptic functions (sn, cn, dn) of u ∣ m.

        sn(u) = sin φ₀,   cn(u) = cos φ₀,   dn(u) = √(1 − m sn(u)²)

    Descending Landen transformation (Abramowitz and Stegun 16.4): run the
    arithmetic-geometric mean forward to collapse the parameter to zero,
    where the functions reduce to sin and cos, then descend the recorded
    sequence back to the requested parameter. ``modulus`` is the parameter
    m = k², matching ``scipy.special.ellipj``; ``complement`` is 1 − m, as
    for `_ellipk`.

    Args:
        argument: The arguments u, one per node.
        modulus: The parameter m.
        complement: The complementary parameter 1 − m, or ``None`` to form
            it from m.

    Returns:
        Tuple ``(sn, cn, dn)``, each the length of ``argument``.
    """
    if complement is None:
        # 1 − m
        complement = 1.0 - modulus
    # a₀ = 1
    arithmetic_mean = jnp.ones_like(modulus)
    # b₀ = √(1 − m)
    geometric_mean = jnp.sqrt(complement)
    # c₀ = √m
    half_gap = jnp.sqrt(modulus)
    means, half_gaps = [arithmetic_mean], [half_gap]
    for _ in range(_AGM_ITERATIONS):
        # a_n+1 = (a_n + b_n) / 2
        next_arithmetic = 0.5 * (arithmetic_mean + geometric_mean)
        # b_n+1 = √(a_n b_n)
        next_geometric = jnp.sqrt(arithmetic_mean * geometric_mean)
        # c_n+1 = (a_n − b_n) / 2
        half_gap = 0.5 * (arithmetic_mean - geometric_mean)
        arithmetic_mean, geometric_mean = next_arithmetic, next_geometric
        means.append(arithmetic_mean)
        half_gaps.append(half_gap)

    # φ_N = 2ᴺ a_N u
    phase = (2.0**_AGM_ITERATIONS) * means[-1] * argument
    for step in range(_AGM_ITERATIONS, 0, -1):
        # c_n / a_n
        ratio = half_gaps[step] / means[step]
        # (c_n / a_n) sin φ_n
        sine_term = ratio * jnp.sin(phase)
        # The ratio is below one in exact arithmetic; the clip only guards the
        # rounding boundary, where arcsin would otherwise return NaN.
        clipped = jnp.clip(sine_term, -1.0, 1.0)
        # φ_n−1 = (φ_n + arcsin((c_n / a_n) sin φ_n)) / 2
        phase = 0.5 * (phase + jnp.arcsin(clipped))

    # sn = sin φ₀
    sn = jnp.sin(phase)
    # cn = cos φ₀
    cn = jnp.cos(phase)
    # 1 − m sn² = cn² + (1 − m) sn², which avoids cancelling near m = 1.
    # dn = √(cn² + (1 − m) sn²)
    dn = jnp.sqrt(cn**2 + complement * sn**2)
    return sn, cn, dn
