r"""Joint inverse-quadratic and log-determinant from one shared CG pass.

The Gaussian marginal log-likelihood needs both $R^{\top} A^{-1} R$ and
$\log|A|$. Computing them separately runs the Krylov machinery twice; the
modified batched conjugate gradient of Gardner et al. (2018) gets both from a
single pass, because the coefficients that CG produces on the way to a
solution *are* the Lanczos tridiagonal that stochastic Lanczos quadrature
needs for the log-determinant.
"""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._primitives._logdet import logdet as _logdet
from gaussx._primitives._solve import solve as _solve
from gaussx._primitives._sqrt_matmul import sqrt_inv_matmul, sqrt_matmul
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._bbmm import BBMMSolver


def inv_quad_logdet(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    *,
    strategy: AbstractSolverStrategy | None = None,
    reduce_inv_quad: bool = True,
    preconditioner: lx.AbstractLinearOperator | None = None,
) -> tuple[Float[Array, " *C"], Float[Array, ""]]:
    r"""Joint inverse-quadratic and log-determinant via shared BBMM CG.

    Returns the pair

    $$
    \mathrm{inv\_quad}(A, R) = \mathrm{tr}(R^{\top} A^{-1} R)
      = \sum_{c} r_c^{\top} A^{-1} r_c,
    \qquad
    \mathrm{logdet}(A) = \log |A| ,
    $$

    which together with the $-\tfrac{N}{2}\log 2\pi$ constant make up the
    Gaussian log-density. With the default `gaussx.BBMMSolver` strategy both
    come out of one modified-batched-CG pass over ``[rhs | probes]``, so a
    marginal-likelihood step costs roughly half the matvecs of a separate
    `gaussx.solve` plus `gaussx.logdet`.

    Passing a ``preconditioner`` $P \approx A$ applies the variance reduction
    of Artemev et al. (2021): the same CG pass then produces the Lanczos
    tridiagonal of $P^{-1} A$, so the stochastic part of

    $$\log |A| = \log |P| + \mathrm{tr}\bigl[\log (P^{-1} A)\bigr]$$

    only has to estimate a trace of a near-identity operator.

    Args:
        operator: Square symmetric positive-definite operator $A$.
        rhs: Right-hand side $R$ of shape ``(N, C)``.
        strategy: Solver strategy. `gaussx.BBMMSolver` (the default when
            ``None``) takes the shared-work path; any other strategy falls
            back to its own `solve` per column plus its own `logdet`.
        reduce_inv_quad: Whether to sum the per-column quadratic forms into
            the trace. When ``False`` the first return value is the ``(C,)``
            vector of $r_c^{\top} A^{-1} r_c$.
        preconditioner: Optional operator $P \approx A$ with a cheap
            log-determinant, used for variance reduction as above. This is the
            approximation to $A$ itself, not an approximate inverse.

    Returns:
        Tuple ``(inv_quad, logdet)``. ``inv_quad`` is a scalar when
        ``reduce_inv_quad`` is ``True`` and a ``(C,)`` vector otherwise.
    """
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
    if strategy is None:
        strategy = BBMMSolver()

    if isinstance(strategy, BBMMSolver):
        columns, logdet_value = _shared_work(operator, rhs, strategy, preconditioner)
    else:
        columns, logdet_value = _separate_work(operator, rhs, strategy, preconditioner)

    inv_quad = jnp.sum(columns) if reduce_inv_quad else columns
    return inv_quad, logdet_value


@ft.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _shared_work(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
) -> tuple[Float[Array, " C"], Float[Array, ""]]:
    """Per-column inverse-quadratics and the log-determinant, in one CG pass."""
    outputs, _ = _shared_work_core(operator, rhs, strategy, preconditioner)
    return outputs


def _shared_work_fwd(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    strategy: BBMMSolver,
    preconditioner: lx.AbstractLinearOperator | None,
):
    outputs, cache = _shared_work_core(operator, rhs, strategy, preconditioner)
    return outputs, (operator, preconditioner, *cache)


def _shared_work_bwd(strategy: BBMMSolver, residuals, cotangents):
    r"""Differentiate the quantities, not the recurrence that estimated them.

    Unrolling the mBCG scan is a poor way to get a gradient: as CG converges,
    ``alpha = rᵀz / dᵀAd`` becomes a ratio of two quantities that are pure
    rounding noise, and reverse-mode amplifies that noise without bound. The
    exact derivatives are simple and every term is already in hand:

    $$
    \frac{\partial}{\partial A} r^{\top} A^{-1} r = -x x^{\top},
    \qquad
    \frac{\partial}{\partial r} r^{\top} A^{-1} r = 2 x,
    \qquad
    \frac{\partial}{\partial A} \log |A| = A^{-1},
    $$

    with $x = A^{-1} r$ from the same pass. The log-determinant term reuses the
    probe solves as the Hutchinson estimate
    $A^{-1} \approx \frac{1}{m} \sum_{p=1}^{m} (A^{-1} z_p)(P^{-1} z_p)^{\top}$
    over the $m$ probes, which is unbiased because $\mathbb{E}[z z^{\top}]
    = P$ by construction ($P = I$ when unpreconditioned). It is symmetrised
    so that operators parameterised asymmetrically -- a Cholesky factor,
    say -- still see a symmetric cotangent.

    The preconditioner receives a zero cotangent: it is a variance-reduction
    device, and the quantities being differentiated do not depend on it.
    """
    del strategy
    operator, preconditioner, solutions, probe_solutions, probe_rights = residuals
    ct_columns, ct_logdet = cotangents
    num_probes = probe_rights.shape[1]

    ct_rhs = 2.0 * ct_columns[None, :] * solutions

    probe_scale = 0.5 * ct_logdet / num_probes
    lefts = jnp.concatenate(
        [
            -ct_columns[None, :] * solutions,
            probe_scale * probe_solutions,
            probe_scale * probe_rights,
        ],
        axis=1,
    )
    rights = jnp.concatenate([solutions, probe_rights, probe_solutions], axis=1)
    ct_operator = _operator_cotangent(operator, lefts, rights)

    ct_preconditioner = jax.tree.map(jnp.zeros_like, preconditioner)
    return ct_operator, ct_rhs, ct_preconditioner


_shared_work.defvjp(_shared_work_fwd, _shared_work_bwd)


def _operator_cotangent(
    operator: lx.AbstractLinearOperator,
    lefts: Float[Array, "N K"],
    rights: Float[Array, "N K"],
):
    r"""Cotangent of $\sum_k \ell_k^{\top} A r_k$ with respect to ``operator``.

    Pulling the rank-one terms back through the operator's own ``mv`` keeps
    this working for any parameterisation -- a dense matrix, a Kronecker
    factorisation, a kernel closure -- without ever materialising $A$.
    """

    def apply(op: lx.AbstractLinearOperator) -> Float[Array, "N K"]:
        return jax.vmap(op.mv, in_axes=1, out_axes=1)(rights)

    _, pullback = jax.vjp(apply, operator)
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
    """Run one mBCG pass over the right-hand sides and the SLQ probes."""
    n = operator.in_size()
    num_rhs = rhs.shape[1]
    dtype = rhs.dtype
    key = jr.PRNGKey(strategy.seed)

    probes, probe_weights, base_logdet = _probe_vectors(
        preconditioner, n, strategy.num_probes, dtype, key
    )
    apply_inverse = _preconditioner_inverse(preconditioner)

    # The right-hand sides only need CG's own tolerance; the probes carry the
    # log-determinant, so they run until numerical breakdown to keep as much
    # of the Lanczos tridiagonal as the arithmetic supports.
    eps = jnp.finfo(dtype).eps
    floors = jnp.concatenate(
        [
            jnp.full((num_rhs,), strategy.cg_tolerance**2, dtype=dtype),
            jnp.full((strategy.num_probes,), eps**2, dtype=dtype),
        ]
    )

    block = jnp.concatenate([rhs, probes], axis=1)
    max_iter = strategy.cg_max_iter
    solutions, initial, alphas, betas, active = _mbcg(
        operator, block, apply_inverse, max_iter, floors
    )

    columns = jnp.sum(rhs * solutions[:, :num_rhs], axis=0)
    order = min(strategy.lanczos_iter, max_iter)
    diagonal, off_diagonal = _lanczos_coefficients(
        alphas[:, num_rhs:], betas[:, num_rhs:], active[:, num_rhs:], order
    )
    quadrature = _log_quadrature(diagonal, off_diagonal)
    logdet_value = base_logdet + jnp.mean(probe_weights * quadrature)

    # ``initial`` is P^{-1} times the block, so its probe columns are the
    # P^{-1} z that the backward pass pairs with the probe solves (the probes
    # themselves when there is no preconditioner).
    cache = (solutions[:, :num_rhs], solutions[:, num_rhs:], initial[:, num_rhs:])
    return (columns, logdet_value), cache


def _separate_work(
    operator: lx.AbstractLinearOperator,
    rhs: Float[Array, "N C"],
    strategy: AbstractSolverStrategy,
    preconditioner: lx.AbstractLinearOperator | None,
) -> tuple[Float[Array, " C"], Float[Array, ""]]:
    """Fall back to the strategy's own solve and logdet, without shared work."""
    solutions = jax.vmap(
        lambda column: strategy.solve(operator, column), in_axes=1, out_axes=1
    )(rhs)
    columns = jnp.sum(rhs * solutions, axis=0)
    if preconditioner is None:
        return columns, strategy.logdet(operator)
    whitened = _whitened_operator(operator, preconditioner)
    return columns, _logdet(preconditioner) + strategy.logdet(whitened)


def _probe_vectors(
    preconditioner: lx.AbstractLinearOperator | None,
    n: int,
    num_probes: int,
    dtype,
    key: jax.Array,
) -> tuple[Float[Array, "N P"], Float[Array, " P"], Float[Array, ""]]:
    r"""Probe vectors, their SLQ weights, and the deterministic logdet part.

    Unpreconditioned, Hutchinson wants $\mathbb{E}[zz^{\top}] = I$ and sign
    probes give $\|z\|^2 = N$ exactly. Preconditioned, the tridiagonal that CG
    yields belongs to $P^{-1/2} A P^{-1/2}$ started at $P^{-1/2} z$, so it is
    *that* vector which has to be isotropic: draw $u \sim N(0, I)$ and use
    $z = P^{1/2} u$, whose quadrature weight $\|u\|^2$ is already in hand.
    """
    if preconditioner is None:
        probes = 2.0 * jr.bernoulli(key, 0.5, (n, num_probes)).astype(dtype) - 1.0
        weights = jnp.full((num_probes,), float(n), dtype=dtype)
        return probes, weights, jnp.zeros((), dtype=dtype)

    normals = jr.normal(key, (n, num_probes), dtype=dtype)
    probes = sqrt_matmul(preconditioner, normals)
    weights = jnp.sum(normals**2, axis=0)
    return probes, weights, _logdet(preconditioner)


def _preconditioner_inverse(preconditioner: lx.AbstractLinearOperator | None):
    """Build the ``M(V) = P^{-1} V`` step of preconditioned CG."""
    if preconditioner is None:
        return None

    def apply_inverse(block: Float[Array, "N T"]) -> Float[Array, "N T"]:
        return jax.vmap(
            lambda column: _solve(preconditioner, column), in_axes=1, out_axes=1
        )(block)

    return apply_inverse


def _whitened_operator(
    operator: lx.AbstractLinearOperator,
    preconditioner: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    r"""Build the symmetric whitened operator $P^{-1/2} A P^{-1/2}$.

    It is similar to $P^{-1} A$, so it has the same log-determinant while
    staying symmetric enough for Lanczos.

    Only used on the non-BBMM path, where the strategy's own `logdet` has no
    way to absorb a preconditioner. Each matvec costs two contour-quadrature
    square roots, so this is worthwhile only when $P$ is cheap to solve
    against.
    """

    def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        whitened = sqrt_inv_matmul(preconditioner, vector[:, None])
        applied = operator.mv(whitened[:, 0])
        return sqrt_inv_matmul(preconditioner, applied[:, None])[:, 0]

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

    Solves every column of ``block`` simultaneously and records the per-column
    CG coefficients, which are the Lanczos tridiagonal in disguise.

    A column is *frozen* once its (preconditioned) residual falls below its
    entry of ``floors`` times the initial residual, or once the curvature
    ``dᵀAd`` stops being positive: its step size is then zeroed so the
    iterate, the residual and the recorded coefficients all stand still.

    Args:
        operator: The PSD operator ``A``.
        block: Right-hand sides, shape ``(N, T)``.
        apply_inverse: Callable applying ``P^{-1}`` column-wise, or ``None``.
        max_iter: Number of CG steps.
        floors: Per-column *relative* squared residual floors, shape ``(T,)``.

    Returns:
        Tuple ``(solutions, initial, alphas, betas, active)``. ``solutions``
        and ``initial`` (the preconditioned starting residual ``P^{-1} B``)
        have shape ``(N, T)``; the rest have shape ``(max_iter, T)``.
    """

    def matmul(columns: Float[Array, "N T"]) -> Float[Array, "N T"]:
        return jax.vmap(operator.mv, in_axes=1, out_axes=1)(columns)

    def precondition(residual: Float[Array, "N T"]) -> Float[Array, "N T"]:
        return residual if apply_inverse is None else apply_inverse(residual)

    residual = block
    preconditioned = precondition(residual)
    rz = jnp.sum(residual * preconditioned, axis=0)
    thresholds = floors * rz

    def step(carry, _):
        solution, residual, preconditioned, direction, rz = carry
        curvature = matmul(direction)
        denominator = jnp.sum(direction * curvature, axis=0)
        active = (rz > thresholds) & (denominator > 0.0)
        alpha = jnp.where(active, rz / jnp.where(active, denominator, 1.0), 0.0)
        solution = solution + alpha * direction
        residual = residual - alpha * curvature
        preconditioned = precondition(residual)
        rz_next = jnp.sum(residual * preconditioned, axis=0)
        beta = jnp.where(active, rz_next / jnp.where(active, rz, 1.0), 0.0)
        direction = preconditioned + beta * direction
        carry = (solution, residual, preconditioned, direction, rz_next)
        return carry, (alpha, beta, active)

    init = (
        jnp.zeros_like(block),
        residual,
        preconditioned,
        preconditioned,
        rz,
    )
    (solutions, *_), (alphas, betas, active) = jax.lax.scan(
        step, init, xs=None, length=max_iter
    )
    return solutions, preconditioned, alphas, betas, active


def _lanczos_coefficients(
    alphas: Float[Array, "K P"],
    betas: Float[Array, "K P"],
    active: Array,
    order: int,
) -> tuple[Float[Array, "P T"], Float[Array, "P T-1"]]:
    r"""Turn CG step sizes into the symmetric Lanczos tridiagonal.

    The standard correspondence is $T_{jj} = 1/\alpha_j +
    \beta_{j-1}/\alpha_{j-1}$ and $T_{j,j+1} = \sqrt{\beta_j}/\alpha_j$.

    Frozen steps get $\beta = 0$, which zeroes the off-diagonal and splits a
    trailing block off the tridiagonal; $e_1^{\top} \log(T) e_1$ therefore
    cannot see what that block contains. We still give it the *distinct*
    diagonal entries $1, 2, 3, \dots$ rather than a repeated value, because
    the VJP of `jax.numpy.linalg.eigh` divides by eigenvalue gaps and a
    degenerate trailing block would fill `jax.grad` with NaNs even though the
    forward value is unaffected.
    """
    alphas = rearrange(alphas[:order], "k p -> p k")
    betas = rearrange(betas[:order], "k p -> p k")
    active = rearrange(active[:order], "k p -> p k")

    steps = jnp.arange(1, alphas.shape[1] + 1, dtype=alphas.dtype)
    alphas = jnp.where(active, alphas, 1.0 / steps[None, :])
    betas = jnp.where(active, betas, 0.0)

    diagonal = 1.0 / alphas
    diagonal = diagonal.at[:, 1:].add(betas[:, :-1] / alphas[:, :-1])

    positive = betas[:, :-1] > 0.0
    root_beta = jnp.where(
        positive, jnp.sqrt(jnp.where(positive, betas[:, :-1], 1.0)), 0.0
    )
    off_diagonal = root_beta / alphas[:, :-1]
    return diagonal, off_diagonal


def _log_quadrature(
    diagonal: Float[Array, "P T"],
    off_diagonal: Float[Array, "P T-1"],
) -> Float[Array, " P"]:
    r"""Evaluate $e_1^{\top} \log(T) e_1$ for a batch of tridiagonals."""

    def quadrature(
        diag: Float[Array, " T"], off: Float[Array, " T-1"]
    ) -> Float[Array, ""]:
        tridiagonal = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
        values, vectors = jnp.linalg.eigh(tridiagonal)
        floor = jnp.finfo(diag.dtype).tiny
        return jnp.sum(vectors[0, :] ** 2 * jnp.log(jnp.maximum(values, floor)))

    return jax.vmap(quadrature)(diagonal, off_diagonal)
