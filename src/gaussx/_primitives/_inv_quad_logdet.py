"""Joint inverse-quadratic and log-determinant from one shared CG pass.

A is a square N × N symmetric positive-definite operator and B an N × C
right-hand side with columns b_c, c = 1, …, C. The Gaussian marginal
log-likelihood needs both ∑_c b_cᵀ A⁻¹ b_c and log det A. Computing them
separately runs the Krylov machinery twice; the modified batched conjugate
gradient (mBCG) of Gardner et al. (2018) gets both from a single pass,
because the coefficients α_j, β_j that CG produces on the way to a solution
*are* the Lanczos tridiagonal T that stochastic Lanczos quadrature (SLQ)
needs for the log-determinant. The probes are z^(i), i = 1, …, N_probe, and
P ≈ A is an optional log-determinant preconditioner.
"""

from __future__ import annotations

import functools as ft

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange, reduce, repeat
from gaussx._operators._diagonalised import as_diagonalized
from gaussx._primitives._logdet import _has_structural_logdet, logdet as _logdet
from gaussx._primitives._solve import solve as _solve
from gaussx._primitives._sqrt_matmul import sqrt_inv_matmul, sqrt_matmul
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolverStrategy
from gaussx._strategies._bbmm import BBMMSolver
from gaussx._strategies._composed import ComposedSolver
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._renamed import UNSET, default, renamed
from gaussx._strategies._slq_logdet import DenseLogdet
from gaussx._strategies._tolerances import resolve_tolerance


def inv_quad_logdet(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    *,
    strategy: AbstractSolverStrategy | None = None,
    reduce_inv_quad: bool = True,
    logdet_preconditioner: lx.AbstractLinearOperator | None = None,
    preconditioner: lx.AbstractLinearOperator | None = UNSET,
) -> tuple[Float[Array, " *C"], Float[Array, ""]]:
    """Joint inverse-quadratic and log-determinant via shared BBMM CG.

        inv_quad(A, B) = tr(Bᵀ A⁻¹ B) = ∑_c b_cᵀ A⁻¹ b_c,   logdet(A) = log det A

    Together with the −(N / 2) log 2π constant these make up the Gaussian
    log-density. For a structured operator with an exact log-determinant the
    default is the exact structural pair. Otherwise, with
    `gaussx.BBMMSolver`, both come out of one modified-batched-CG pass over
    ``[rhs | probes]``, so a marginal-likelihood step costs roughly half the
    matvecs of a separate `gaussx.solve` plus `gaussx.logdet`.

    Passing a ``logdet_preconditioner`` P ≈ A applies the variance reduction
    of Artemev et al. (2021): the same CG pass then produces the Lanczos
    tridiagonal of P⁻¹ A, so the stochastic part of

        log det A = log det P + tr(log(P⁻¹ A))

    only has to estimate a trace of a near-identity operator.

    Algorithm:
        1. If ``strategy`` is ``None`` and A has an exact structural
           log-determinant, solve each column structurally and return
           (∑_c b_cᵀ A⁻¹ b_c, `gaussx.logdet`).
        2. With `gaussx.BBMMSolver` (the default otherwise), run one mBCG pass
           over [B ∣ z^(1) … z^(N_probe)]: the B columns give A⁻¹ b_c, and the
           probe columns' CG coefficients give the Lanczos tridiagonals T^(i)
           for SLQ, log det A ≈ log det P + (1 / N_probe) ∑_i w_i e₁ᵀ log(T^(i)) e₁.
        3. With any other strategy, use its own solve per column and its own
           logdet, whitened by P when the logdet is stochastic.
        4. Sum the per-column quadratics into the trace if asked.

    Args:
        operator: Square symmetric positive-definite operator A.
        rhs: Right-hand side B, N × C.
        strategy: Solver strategy. ``None`` (the default) takes the exact
            structural path (`gaussx.solve` per column plus `gaussx.logdet`)
            when the operator has one (diagonal, `Kronecker`, `BlockDiag`,
            `LowRankUpdate`, `BlockTriDiag`, `KroneckerSum`, ... and scalar
            or tagged wrappers of them), and `gaussx.BBMMSolver` otherwise.
            `gaussx.BBMMSolver` takes the shared-work path; any other
            strategy falls back to its own `solve` per column plus its own
            `logdet`.
        reduce_inv_quad: Whether to sum the per-column quadratic forms into
            the trace. When ``False`` the first return value is the length-C
            vector of b_cᵀ A⁻¹ b_c.
        logdet_preconditioner: Optional operator P ≈ A with a cheap
            log-determinant, used for variance reduction as above. This is the
            approximation to A itself, **not** an approximate inverse like
            the ``preconditioner=`` of `gaussx.CGSolver` and
            `gaussx.linear_solve`. Ignored when the strategy's
            log-determinant is already exact (e.g. `gaussx.DenseSolver`),
            since there is no variance to reduce.
        preconditioner: Deprecated alias of ``logdet_preconditioner``
            (gh-405), renamed because everywhere else in gaussx
            ``preconditioner=`` means M⁻¹ ≈ A⁻¹.

    Returns:
        Tuple ``(inv_quad, logdet)``. ``inv_quad`` is a scalar when
        ``reduce_inv_quad`` is ``True`` and a length-C vector otherwise.

    Raises:
        ValueError: If A is not square, or B is not N × C.

    Examples:
        A diagonal operator takes the exact structural pair:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 4.0]))
        >>> B = jnp.array([[1.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
        >>> inv_quad, logdet = gaussx.inv_quad_logdet(A, B)
        >>> round(float(inv_quad), 4), round(float(logdet), 4)  # 4.5, log 8
        (4.5, 2.0794)

        Per-column quadratics instead of their sum:

        >>> per_column, _ = gaussx.inv_quad_logdet(A, B, reduce_inv_quad=False)
        >>> [round(float(v), 4) for v in per_column]
        [3.0, 1.5]

        A dense matrix with an exact strategy matches the dense formulas:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> B = jnp.array([[1.0], [2.0]])
        >>> inv_quad, logdet = gaussx.inv_quad_logdet(
        ...     A, B, strategy=gaussx.DenseSolver()
        ... )
        >>> expected = B[:, 0] @ jnp.linalg.solve(matrix, B[:, 0])
        >>> bool(jnp.allclose(inv_quad, expected, atol=1e-5))
        True
        >>> bool(jnp.allclose(logdet, jnp.log(11.0), atol=1e-5))
        True
    """
    preconditioner = default(
        renamed(
            "inv_quad_logdet",
            "preconditioner",
            preconditioner,
            "logdet_preconditioner",
            logdet_preconditioner if logdet_preconditioner is not None else UNSET,
        ),
        None,
    )
    if operator.in_size() != operator.out_size():
        raise ValueError("inv_quad_logdet requires a square operator")
    if jnp.ndim(rhs) != 2:
        raise ValueError(
            f"rhs must have shape (N, C); got array with {jnp.ndim(rhs)} axes"
        )
    if rhs.shape[0] != operator.in_size():
        raise ValueError(
            f"rhs has {rhs.shape[0]} rows but operator has size {operator.in_size()}"
        )
    if strategy is None and (
        as_diagonalized(operator) is not None or _has_structural_logdet(operator)
    ):
        # Exact and cheap: a structural solve per column plus the structural
        # log-determinant, instead of BBMM's stochastic one (gh-340).
        # A⁻¹ b_c, one structural solve per column
        solutions = jax.vmap(ft.partial(_solve, operator), 1, 1)(rhs)
        # b_cᵀ A⁻¹ b_c
        # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
        columns = jnp.sum(rhs * solutions, axis=0)
        # ∑_c b_cᵀ A⁻¹ b_c
        inv_quad = jnp.sum(columns) if reduce_inv_quad else columns
        return inv_quad, _logdet(operator)
    if strategy is None:
        strategy = BBMMSolver()

    if isinstance(strategy, BBMMSolver):
        columns, logdet_value = _shared_work((operator, rhs), strategy, preconditioner)
    else:
        columns, logdet_value = _separate_work(operator, rhs, strategy, preconditioner)

    # ∑_c b_cᵀ A⁻¹ b_c
    inv_quad = jnp.sum(columns) if reduce_inv_quad else columns
    return inv_quad, logdet_value


# `equinox.filter_custom_vjp` rather than `jax.custom_vjp`: the operator may
# carry non-array leaves -- the function of a `lineax.FunctionLinearOperator`,
# static tags -- which `jax.custom_vjp` rejects as arguments.
@eqx.filter_custom_vjp
def _shared_work(
    differentiable: tuple[lx.AbstractLinearOperator, Float[Array, "N C"]],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
) -> tuple[Float[Array, " C"], Float[Array, ""]]:
    """Per-column inverse-quadratics and the log-determinant, in one CG pass.

        (b_cᵀ A⁻¹ b_c for each c,   log det A)

    Args:
        differentiable: The pair ``(operator, rhs)``, A and B.
        strategy: The BBMM strategy (probes, seed, tolerances, budget).
        preconditioner: Log-determinant preconditioner P, or ``None``.

    Returns:
        Tuple ``(columns, logdet)``.
    """
    operator, rhs = differentiable
    outputs, _ = _shared_work_core(operator, rhs, strategy, preconditioner)
    return outputs


@_shared_work.def_fwd
def _shared_work_fwd(
    perturbed,
    differentiable: tuple[lx.AbstractLinearOperator, Float[Array, "N C"]],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
):
    """Forward pass of `_shared_work`, keeping the solves for the backward pass.

    Args:
        perturbed: Which leaves of ``differentiable`` are differentiated.
        differentiable: The pair ``(operator, rhs)``, A and B.
        strategy: The BBMM strategy.
        preconditioner: Log-determinant preconditioner P, or ``None``.

    Returns:
        The outputs of `_shared_work` and the residuals of
        `_shared_work_core`.
    """
    del perturbed
    operator, rhs = differentiable
    return _shared_work_core(operator, rhs, strategy, preconditioner)


@_shared_work.def_bwd
def _shared_work_bwd(
    residuals,
    cotangents,
    perturbed,
    differentiable: tuple[lx.AbstractLinearOperator, Float[Array, "N C"]],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
):
    """Differentiate the quantities, not the recurrence that estimated them.

        ∂(bᵀ A⁻¹ b)/∂A = −x xᵀ,   ∂(bᵀ A⁻¹ b)/∂b = 2 x,   ∂(log det A)/∂A = A⁻¹

    with x = A⁻¹ b from the same pass.

    Unrolling the mBCG scan is a poor way to get a gradient: as CG converges,
    α = rᵀz / dᵀAd becomes a ratio of two quantities that are pure rounding
    noise, and reverse mode amplifies that noise without bound. The exact
    derivatives above are simple and every term is already in hand. The
    log-determinant term reuses the probe solves as the Hutchinson estimate

        A⁻¹ ≈ (1 / N_probe) ∑_i (A⁻¹ z^(i)) (P⁻¹ z^(i))ᵀ

    which is unbiased because 𝔼[z zᵀ] = P by construction (P = I when
    unpreconditioned). It is symmetrised so that operators parameterised
    asymmetrically (a Cholesky factor, say) still see a symmetric
    cotangent.

    The preconditioner receives a zero cotangent: it is a variance-reduction
    device, and the quantities being differentiated do not depend on it.

    Args:
        residuals: ``(solutions, probe_solutions, probe_rights)`` from the
            forward pass: A⁻¹ b_c, A⁻¹ z^(i) and P⁻¹ z^(i).
        cotangents: Cotangents of ``(columns, logdet)``; ``None`` marks an
            output that is not being differentiated.
        perturbed: Which leaves of ``differentiable`` are differentiated.
        differentiable: The pair ``(operator, rhs)``, A and B.
        strategy: The BBMM strategy (unused).
        preconditioner: Log-determinant preconditioner P (unused).

    Returns:
        Cotangents for ``(operator, rhs)``, ``None`` where unperturbed.
    """
    del strategy, preconditioner
    operator, _ = differentiable
    solutions, probe_solutions, probe_rights = residuals
    # ``None`` marks an output that is not being differentiated.
    ct_columns, ct_logdet = cotangents
    if ct_columns is None:
        ct_columns = jnp.zeros(solutions.shape[1], dtype=solutions.dtype)
    if ct_logdet is None:
        ct_logdet = jnp.zeros((), dtype=solutions.dtype)
    num_probes = probe_rights.shape[1]

    # ∂/∂b_c = 2 x_c, scaled by the column cotangents
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    ct_rhs = 2.0 * ct_columns[None, :] * solutions

    # ½ (log-det cotangent) / N_probe, split over the two symmetrised halves
    probe_scale = 0.5 * ct_logdet / num_probes
    # −x_c, scaled by the column cotangents
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    quadratic_lefts = -ct_columns[None, :] * solutions
    # ∑_k ℓ_k r_kᵀ = −∑_c x_c x_cᵀ + ½ ∑_i (A⁻¹ z^(i) (P⁻¹ z^(i))ᵀ + transpose)
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    lefts = jnp.concatenate(
        [quadratic_lefts, probe_scale * probe_solutions, probe_scale * probe_rights],
        axis=1,
    )
    # [x_c ∣ P⁻¹ z^(i) ∣ A⁻¹ z^(i)]
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    rights = jnp.concatenate([solutions, probe_rights, probe_solutions], axis=1)
    ct_operator = _operator_cotangent(operator, lefts, rights)

    # Only leaves that are actually being differentiated get a cotangent.
    return jax.tree.map(
        lambda gradient, wanted: gradient if wanted else None,
        (ct_operator, ct_rhs),
        perturbed,
        is_leaf=lambda leaf: leaf is None,
    )


def _operator_cotangent(
    operator: lx.AbstractLinearOperator,
    lefts: Float[Array, "N K"],
    rights: Float[Array, "N K"],
):
    """Cotangent of ∑_k ℓ_kᵀ A r_k with respect to ``operator``.

        ∂(∑_k ℓ_kᵀ A r_k)/∂A = ∑_k ℓ_k r_kᵀ

    Pulling the rank-one terms back through the operator's own ``mv`` keeps
    this working for any parameterisation (a dense matrix, a Kronecker
    factorisation, a kernel closure) without ever materialising A.

    Args:
        operator: The operator A.
        lefts: The left vectors ℓ_k as columns.
        rights: The right vectors r_k as columns.

    Returns:
        The cotangent, a pytree shaped like ``operator``.
    """

    def apply(op: lx.AbstractLinearOperator) -> Float[Array, "N K"]:
        # A r_k, one matvec per column
        return jax.vmap(op.mv, in_axes=1, out_axes=1)(rights)

    # `equinox.filter_vjp` passes non-array leaves (a matvec closure, static
    # tags) through untouched and gives them a ``None`` cotangent. A scalar the
    # operator closes over can reach the backward pass as a Python float, so
    # promote those to arrays first; the caller drops unperturbed cotangents.
    operator = jax.tree.map(
        lambda leaf: jnp.asarray(leaf) if isinstance(leaf, float) else leaf,
        operator,
    )
    _, pullback = eqx.filter_vjp(apply, operator)
    # ∑_k ℓ_k r_kᵀ, pulled back through A's parameters
    (cotangent,) = pullback(lefts)
    return cotangent


def _shared_work_core(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
) -> tuple[
    tuple[Float[Array, " C"], Float[Array, ""]],
    tuple[Float[Array, "N C"], Float[Array, "N P"], Float[Array, "N P"]],
]:
    """Run one mBCG pass over the right-hand sides and the SLQ probes.

        log det A ≈ log det P + (1 / N_probe) ∑_i w_i e₁ᵀ log(T^(i)) e₁

    Algorithm:
        1. Draw the probes z^(i) and their quadrature weights w_i.
        2. Run mBCG on [B ∣ z^(1) … z^(N_probe)], the B columns to CG's
           tolerance and the probes to numerical breakdown.
        3. Score b_cᵀ A⁻¹ b_c; raise if a B column used the whole budget.
        4. Turn each probe's CG coefficients into its Lanczos tridiagonal
           T^(i) and average the Gauss quadratures of log.

    Args:
        operator: The operator A.
        rhs: Right-hand side B, N × C.
        strategy: The BBMM strategy (probes, seed, tolerances, budget).
        preconditioner: Log-determinant preconditioner P, or ``None``.

    Returns:
        Tuple ``((columns, logdet), cache)`` with the per-column
        b_cᵀ A⁻¹ b_c and log det A, and the cache ``(solutions,
        probe_solutions, probe_rights)`` holding A⁻¹ b_c, A⁻¹ z^(i) and
        P⁻¹ z^(i) for the backward pass.
    """
    size = operator.in_size()
    num_rhs = rhs.shape[1]
    dtype = rhs.dtype
    key = jr.PRNGKey(strategy.seed)

    # z^(i), w_i and log det P
    probes, probe_weights, base_logdet = _probe_vectors(
        preconditioner, size, strategy.num_probes, dtype, key
    )
    apply_inverse = _preconditioner_inverse(preconditioner)

    # The right-hand sides only need CG's own tolerance; the probes carry the
    # log-determinant, so they run until numerical breakdown to keep as much
    # of the Lanczos tridiagonal as the arithmetic supports.
    # The mBCG floors are relative, so they take BBMM's dtype-aware relative
    # tolerance (gh-327): 1e-4 in float64, as before, and 1e-3 in float32.
    eps = jnp.finfo(dtype).eps
    rhs_tol = resolve_tolerance(strategy.rtol, dtype, 1e-4)
    floors = jnp.concatenate(
        [
            jnp.full((num_rhs,), rhs_tol**2, dtype=dtype),
            jnp.full((strategy.num_probes,), eps**2, dtype=dtype),
        ]
    )

    # [B ∣ z^(1) … z^(N_probe)]
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    block = jnp.concatenate([rhs, probes], axis=1)
    max_iter = strategy.max_steps
    solutions, initial, alphas, betas, active = _mbcg(
        operator, block, apply_inverse, max_iter, floors
    )

    # b_cᵀ A⁻¹ b_c
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    columns = jnp.sum(rhs * solutions[:, :num_rhs], axis=0)
    if strategy.throw:
        # A right-hand side still active at the last step used the whole
        # budget (lineax's max_steps_reached), like BBMMSolver.solve.
        columns = eqx.error_if(
            columns,
            jnp.any(active[-1, :num_rhs]),
            "BBMM's mBCG did not converge within `max_steps` steps. Increase "
            "`max_steps`, loosen `rtol`, or pass `throw=False`.",
        )
    # Deliberately *not* clamped to N. mBCG does not reorthogonalise, so
    # steps past N add ghost copies of converged Ritz values, but the Gauss
    # rule splits their weight between the copies, and those extra steps are
    # what recover the accuracy lost to orthogonality (Greenbaum 1989).
    # Truncating at N costs digits on ill-conditioned operators.
    order = min(strategy.lanczos_order, max_iter)
    # T^(i), one Lanczos tridiagonal per probe
    diagonal, off_diagonal = _lanczos_coefficients(
        alphas[:, num_rhs:], betas[:, num_rhs:], active[:, num_rhs:], order
    )
    # e₁ᵀ log(T^(i)) e₁
    quadrature = _log_quadrature(diagonal, off_diagonal)
    # log det A ≈ log det P + (1 / N_probe) ∑_i w_i e₁ᵀ log(T^(i)) e₁
    logdet_value = base_logdet + jnp.mean(probe_weights * quadrature)

    # ``initial`` is P⁻¹ times the block, so its probe columns are the
    # P⁻¹ z^(i) that the backward pass pairs with the probe solves (the
    # probes themselves when there is no preconditioner).
    cache = (solutions[:, :num_rhs], solutions[:, num_rhs:], initial[:, num_rhs:])
    return (columns, logdet_value), cache


def _separate_work(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    strategy: AbstractSolverStrategy,
    preconditioner: lx.AbstractLinearOperator | None,
) -> tuple[Float[Array, " C"], Float[Array, ""]]:
    """Fall back to the strategy's own solve and logdet, without shared work.

        log det A = log det P + log det(P^(−1/2) A P^(−1/2))

    The whitened form is used only when P is given and the strategy's
    log-determinant is stochastic.

    Args:
        operator: The operator A.
        rhs: Right-hand side B, N × C.
        strategy: The solver strategy.
        preconditioner: Log-determinant preconditioner P, or ``None``.

    Returns:
        Tuple ``(columns, logdet)`` with the per-column b_cᵀ A⁻¹ b_c and
        log det A.
    """
    # A⁻¹ b_c, one strategy solve per column
    solutions = jax.vmap(
        lambda column: strategy.solve(operator, column), in_axes=1, out_axes=1
    )(rhs)
    # b_cᵀ A⁻¹ b_c
    # jnp, not einx: einx cannot resolve a zero-column right-hand side (C = 0).
    columns = jnp.sum(rhs * solutions, axis=0)
    # An exact log-determinant has no variance to reduce, and whitening would
    # only swap it for one that carries contour-quadrature error.
    if preconditioner is None or _has_exact_logdet(strategy, operator):
        return columns, strategy.logdet(operator)
    # P^(−1/2) A P^(−1/2)
    whitened = _whitened_operator(operator, preconditioner)
    # log det A = log det P + log det(P^(−1/2) A P^(−1/2))
    return columns, _logdet(preconditioner) + strategy.logdet(whitened)


def _has_exact_logdet(
    strategy: AbstractLogdetStrategy,
    operator: lx.AbstractLinearOperator,
) -> bool:
    """Whether ``strategy`` computes log det A deterministically and exactly.

    Args:
        strategy: The log-determinant strategy.
        operator: The operator A, for strategies that route on it.

    Returns:
        ``True`` for dense strategies, and for automatic or composed ones
        that resolve to a dense log-determinant.
    """
    if isinstance(strategy, DenseSolver | DenseLogdet):
        return True
    if isinstance(strategy, AutoSolver):
        return _has_exact_logdet(strategy._get_strategy(operator), operator)
    if isinstance(strategy, ComposedSolver):
        return _has_exact_logdet(strategy.logdet_strategy, operator)
    return False


def _probe_vectors(
    preconditioner: lx.AbstractLinearOperator | None,
    n: int,
    num_probes: int,
    dtype,
    key: jax.Array,
) -> tuple[Float[Array, "N P"], Float[Array, " P"], Float[Array, ""]]:
    """Probe vectors z^(i), their SLQ weights w_i and the deterministic part.

        z^(i) ∈ {−1, 1}ᴺ, w_i = N (no P);   z^(i) = P^(1/2) u^(i), w_i = ‖u^(i)‖²

    Unpreconditioned, Hutchinson wants 𝔼[z zᵀ] = I and sign probes give
    ‖z‖² = N exactly. Preconditioned, the tridiagonal that CG yields belongs
    to P^(−1/2) A P^(−1/2) started at P^(−1/2) z, so it is *that* vector which
    has to be isotropic: draw u^(i) ~ 𝒩(0, I) and use z^(i) = P^(1/2) u^(i),
    whose quadrature weight ‖u^(i)‖² is already in hand.

    Args:
        preconditioner: Log-determinant preconditioner P, or ``None``.
        n: Operator size N.
        num_probes: Number of probes N_probe.
        dtype: Floating dtype of the probes.
        key: PRNG key for the draws.

    Returns:
        Tuple ``(probes, weights, base_logdet)``: z^(i) as columns, w_i, and
        log det P (zero without a preconditioner).
    """
    if preconditioner is None:
        signs = jr.bernoulli(key, 0.5, (n, num_probes)).astype(dtype)
        # z^(i) ∈ {−1, 1}ᴺ
        probes = 2.0 * signs - 1.0
        # w_i = ‖z^(i)‖² = N
        weights = jnp.full((num_probes,), float(n), dtype=dtype)
        return probes, weights, jnp.zeros((), dtype=dtype)

    # u^(i) ~ 𝒩(0, I)
    normals = jr.normal(key, (n, num_probes), dtype=dtype)
    # z^(i) = P^(1/2) u^(i)
    probes = sqrt_matmul(preconditioner, normals)
    # w_i = ‖u^(i)‖²
    weights = reduce(normals**2, "row probe -> probe", "sum")
    return probes, weights, _logdet(preconditioner)


def _preconditioner_inverse(preconditioner: lx.AbstractLinearOperator | None):
    """Build the M(V) = P⁻¹ V step of preconditioned CG.

    Args:
        preconditioner: Preconditioner P, or ``None``.

    Returns:
        A function applying P⁻¹ column by column, or ``None``.
    """
    if preconditioner is None:
        return None

    def apply_inverse(block: Float[Array, "N T"]) -> Float[Array, "N T"]:
        # P⁻¹ V, one solve per column
        return jax.vmap(
            lambda column: _solve(preconditioner, column), in_axes=1, out_axes=1
        )(block)

    return apply_inverse


def _whitened_operator(
    operator: lx.AbstractLinearOperator,
    preconditioner: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Build the symmetric whitened operator P^(−1/2) A P^(−1/2).

        P^(−1/2) A P^(−1/2) ~ P⁻¹ A

    It is similar to P⁻¹ A, so it has the same log-determinant while staying
    symmetric enough for Lanczos.

    Only used on the non-BBMM path, where the strategy's own `logdet` has no
    way to absorb a preconditioner. Each matvec costs two contour-quadrature
    square roots, so this is worthwhile only when P is cheap to solve
    against.

    Args:
        operator: The operator A.
        preconditioner: The preconditioner P.

    Returns:
        A matrix-free PSD-tagged operator P^(−1/2) A P^(−1/2).
    """

    def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        column = rearrange(vector, "row -> row 1")
        # P^(−1/2) v
        whitened = sqrt_inv_matmul(preconditioner, column)
        # A P^(−1/2) v
        applied = operator.mv(rearrange(whitened, "row 1 -> row"))
        # P^(−1/2) A P^(−1/2) v
        result = sqrt_inv_matmul(preconditioner, rearrange(applied, "row -> row 1"))
        return rearrange(result, "row 1 -> row")

    return lx.FunctionLinearOperator(
        matvec, operator.in_structure(), lx.positive_semidefinite_tag
    )


def _mbcg(
    operator: lx.AbstractLinearOperator,
    block: Float[Array, "N T"],
    apply_inverse,
    max_iter: int,
    floors: Float[Array, " T"],
) -> tuple[
    Float[Array, "N T"],
    Float[Array, "N T"],
    Float[Array, "K T"],
    Float[Array, "K T"],
    Array,
]:
    """Modified batched conjugate gradients (Gardner et al. 2018, Alg. 2).

        α_j = r_jᵀ z_j / d_jᵀ A d_j,   β_j = r_j+1ᵀ z_j+1 / r_jᵀ z_j

    with residual r_j, preconditioned residual z_j = P⁻¹ r_j and direction
    d_j, all per column. Solves every column of ``block`` simultaneously and
    records the per-column CG coefficients, which are the Lanczos
    tridiagonal in disguise.

    A column is *frozen* once its (preconditioned) residual falls below its
    entry of ``floors`` times the initial residual, or once the curvature
    d_jᵀ A d_j stops being positive: its step size is then zeroed so the
    iterate, the residual and the recorded coefficients all stand still. The
    freeze is permanent, and the loop exits as soon as every column is frozen,
    so an easy system does not pay for ``max_iter`` batched matvecs. Steps
    that never ran are recorded as inactive.

    Args:
        operator: The PSD operator A.
        block: Right-hand sides, shape ``(N, T)``.
        apply_inverse: Callable applying P⁻¹ column-wise, or ``None``.
        max_iter: Maximum number of CG steps.
        floors: Per-column *relative* squared residual floors, shape ``(T,)``.

    Returns:
        Tuple ``(solutions, initial, alphas, betas, active)``. ``solutions``
        and ``initial`` (the preconditioned starting residual P⁻¹ B) have
        shape ``(N, T)``; the rest have shape ``(max_iter, T)``.
    """

    def matmul(columns: Float[Array, "N T"]) -> Float[Array, "N T"]:
        # A V, one matvec per column
        return jax.vmap(operator.mv, in_axes=1, out_axes=1)(columns)

    def precondition(residual: Float[Array, "N T"]) -> Float[Array, "N T"]:
        # z = P⁻¹ r
        return residual if apply_inverse is None else apply_inverse(residual)

    residual = block
    preconditioned = precondition(residual)
    # r₀ᵀ z₀
    rz = reduce(residual * preconditioned, "row col -> col", "sum")
    thresholds = floors * rz

    def step(carry):
        index, solution, residual, preconditioned, direction, rz, alive, record = carry
        # A d_j
        curvature = matmul(direction)
        # d_jᵀ A d_j
        denominator = reduce(direction * curvature, "row col -> col", "sum")
        active = alive & (rz > thresholds) & (denominator > 0.0)
        # α_j = r_jᵀ z_j / d_jᵀ A d_j, zero once frozen
        alpha = jnp.where(active, rz / jnp.where(active, denominator, 1.0), 0.0)
        # x_j+1 = x_j + α_j d_j
        solution = solution + einx.multiply("col, row col -> row col", alpha, direction)
        # r_j+1 = r_j − α_j A d_j
        residual = residual - einx.multiply("col, row col -> row col", alpha, curvature)
        # z_j+1 = P⁻¹ r_j+1
        preconditioned = precondition(residual)
        # r_j+1ᵀ z_j+1
        rz_next = reduce(residual * preconditioned, "row col -> col", "sum")
        # β_j = r_j+1ᵀ z_j+1 / r_jᵀ z_j, zero once frozen
        beta = jnp.where(active, rz_next / jnp.where(active, rz, 1.0), 0.0)
        # d_j+1 = z_j+1 + β_j d_j
        direction = preconditioned + einx.multiply(
            "col, row col -> row col", beta, direction
        )
        alphas, betas, actives = record
        record = (
            alphas.at[index].set(alpha),
            betas.at[index].set(beta),
            actives.at[index].set(active),
        )
        return (
            index + 1,
            solution,
            residual,
            preconditioned,
            direction,
            rz_next,
            active,
            record,
        )

    def unfinished(carry) -> Array:
        index, *_, alive, _ = carry
        return (index < max_iter) & jnp.any(alive)

    num_columns = block.shape[1]
    record = (
        jnp.zeros((max_iter, num_columns), dtype=block.dtype),
        jnp.zeros((max_iter, num_columns), dtype=block.dtype),
        jnp.zeros((max_iter, num_columns), dtype=bool),
    )
    init = (
        jnp.asarray(0),
        jnp.zeros_like(block),
        residual,
        preconditioned,
        preconditioned,
        rz,
        jnp.ones((num_columns,), dtype=bool),
        record,
    )
    _, solutions, *_, (alphas, betas, active) = jax.lax.while_loop(
        unfinished, step, init
    )
    return solutions, preconditioned, alphas, betas, active


def _lanczos_coefficients(
    alphas: Float[Array, "K P"],
    betas: Float[Array, "K P"],
    active: Array,
    order: int,
) -> tuple[Float[Array, "P T"], Float[Array, "P T-1"]]:
    """Turn CG step sizes into the symmetric Lanczos tridiagonal T.

        T_jj = 1 / α_j + β_j−1 / α_j−1,   T_j,j+1 = √β_j / α_j

    Once a probe freezes, every coupling into the frozen steps is zeroed,
    including β_k of the last *active* step k, which was computed before the
    freeze and would otherwise connect the valid Lanczos prefix to step
    k + 1. That splits a trailing block off the tridiagonal, so
    e₁ᵀ log(T) e₁ cannot see what that block contains. We still give it the
    *distinct* diagonal entries 1, 2, 3, … rather than a repeated value,
    because the VJP of `jax.numpy.linalg.eigh` divides by eigenvalue gaps
    and a degenerate trailing block would fill `jax.grad` with NaNs even
    though the forward value is unaffected.

    Args:
        alphas: CG step sizes α_j, steps × probes.
        betas: CG direction updates β_j, steps × probes.
        active: Whether each step ran, steps × probes.
        order: Number of Lanczos steps to keep.

    Returns:
        Tuple ``(diagonal, off_diagonal)`` of T per probe.
    """
    alphas = rearrange(alphas[:order], "step probe -> probe step")
    betas = rearrange(betas[:order], "step probe -> probe step")
    # Only the prefix before the first freeze is a Lanczos run; a column that
    # broke down can report active again once its direction resets, so make
    # the freeze sticky.
    active = rearrange(active[:order], "step probe -> probe step")
    active = jnp.cumprod(active.astype(jnp.int32), axis=1).astype(bool)

    steps = jnp.arange(1, alphas.shape[1] + 1, dtype=alphas.dtype)
    # 1 / j on the frozen steps, so the trailing block has distinct entries j
    filler = repeat(1.0 / steps, "step -> probe step", probe=alphas.shape[0])
    alphas = jnp.where(active, alphas, filler)
    # ``beta_j`` couples step ``j`` to step ``j + 1``, so it survives only if
    # both ends are live; with a sticky freeze, the later end decides.
    links = jnp.where(active[:, 1:], betas[:, :-1], 0.0)

    # 1 / α_j
    diagonal = 1.0 / alphas
    # T_jj = 1 / α_j + β_j−1 / α_j−1
    diagonal = diagonal.at[:, 1:].add(links / alphas[:, :-1])

    positive = links > 0.0
    # √β_j, zero on the cut links
    root_beta = jnp.where(positive, jnp.sqrt(jnp.where(positive, links, 1.0)), 0.0)
    # T_j,j+1 = √β_j / α_j
    off_diagonal = root_beta / alphas[:, :-1]
    return diagonal, off_diagonal


def _log_quadrature(
    diagonal: Float[Array, "P T"],
    off_diagonal: Float[Array, "P T-1"],
) -> Float[Array, " P"]:
    """Evaluate e₁ᵀ log(T) e₁ for a batch of tridiagonals T.

        e₁ᵀ log(T) e₁ = ∑_j [S]₁ⱼ² log θ_j,   T = S diag(θ) Sᵀ

    Args:
        diagonal: Diagonals of T, one row per probe.
        off_diagonal: Off-diagonals of T, one row per probe.

    Returns:
        The Gauss quadrature e₁ᵀ log(T) e₁ per probe.
    """

    def quadrature(
        diag: Float[Array, " T"], off: Float[Array, " T-1"]
    ) -> Float[Array, ""]:
        tridiagonal = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
        # T = S diag(θ) Sᵀ
        ritz_values, ritz_vectors = jnp.linalg.eigh(tridiagonal)
        floor = jnp.finfo(diag.dtype).tiny
        # ∑_j [S]₁ⱼ² log θ_j, with θ_j floored at the smallest normal
        return jnp.sum(
            ritz_vectors[0, :] ** 2 * jnp.log(jnp.maximum(ritz_values, floor))
        )

    return jax.vmap(quadrature)(diagonal, off_diagonal)
