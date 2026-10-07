"""Structured trace with dispatch on operator type."""

from __future__ import annotations

import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._operators._toeplitz import Toeplitz
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys
from gaussx._randomized._trace import hutchpp_trace


def trace(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName | None = None,
    algorithm: Literal["hutchinson", "hutchpp", "xtrace"] = "hutchinson",
) -> Float[Array, ""]:
    r"""Compute the trace of an operator.

    When ``stochastic=True``, uses a stochastic estimator — only requires
    matvec access, no materialization. Hutchinson averages $z^\top A z$
    over probes, with variance $O(\|A\|_F^2 / m)$ after $m$ matvecs. The
    variance-reduced estimators deflate a low-rank part first,

    $$
    \operatorname{tr}(A) = \operatorname{tr}(Q^\top A Q)
    + \operatorname{tr}\big((I - QQ^\top) A (I - QQ^\top)\big),
    $$

    with $Q$ from a randomized range finder: the first term is exact and
    only the (small) remainder is probed. For PSD $A$ with a decaying
    spectrum that reaches relative error $\varepsilon$ in
    $O(1/\varepsilon)$ matvecs instead of $O(1/\varepsilon^2)$.

    - ``"hutchpp"`` (Hutch++, Meyer, Musco, Musco & Woodruff, 2021):
      ``num_probes`` is the total matvec budget $m \ge 3$, a third each
      for the range sketch, the exact projection and Hutchinson on the
      remainder.
    - ``"xtrace"`` (Epperly, Tropp & Webber, 2024; matfree's
      ``leave_one_out_xtrace``): every probe both builds $Q$ and estimates
      the remainder, by leave-one-out; ``num_probes`` probes cost
      ``2 * num_probes`` matvecs. Usually the most accurate per matvec.

    Args:
        operator: A square linear operator.
        stochastic: If ``True``, use stochastic trace estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``). Defaults to ``"signs"`` for Hutchinson and
            ``"sphere"`` for XTrace (which requires a rotationally
            invariant distribution).
        algorithm: ``"hutchinson"`` (Monte-Carlo), ``"hutchpp"``
            (Hutch++, low-rank deflation plus Hutchinson) or ``"xtrace"``
            (leave-one-out, Epperly et al. 2024 — much lower variance
            for the same number of matvecs).

    Returns:
        Scalar trace value (exact or estimated).

    Raises:
        ValueError: For an unknown ``algorithm``, ``algorithm="xtrace"``
            with ``sampler="signs"``, or ``algorithm="hutchpp"`` with
            ``num_probes < 3``.

    References:
        Meyer, R. A., Musco, C., Musco, C. & Woodruff, D. P. (2021).
        Hutch++: optimal stochastic trace estimation. *SOSA*, 142-155.

        Epperly, E. N., Tropp, J. A. & Webber, R. J. (2024). XTrace: making
        the most of every sample in stochastic trace estimation. *SIAM J.
        Matrix Anal. Appl.*, 45(1), 1-23.

    Examples:
        >>> import einx, jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> U, _ = jnp.linalg.qr(jr.normal(jr.key(0), (200, 200)))
        >>> lam = 1.0 / jnp.arange(1.0, 201.0) ** 2  # decaying spectrum
        >>> U_lam = einx.multiply("i k, k -> i k", U, lam)
        >>> A = lx.MatrixLinearOperator(einx.dot("i k, j k -> i j", U_lam, U))
        >>> est = gx.trace(A, stochastic=True, num_probes=30, algorithm="hutchpp")
        >>> bool(jnp.abs(est - jnp.sum(lam)) / jnp.sum(lam) < 0.05)
        True
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def rec(op: lx.AbstractLinearOperator, k: jax.Array | None = key) -> Array:
        return trace(
            op,
            stochastic=stochastic,
            num_probes=num_probes,
            key=k,
            sampler=sampler,
            algorithm=algorithm,
        )

    def rec_all(ops) -> list[Array]:
        ops = tuple(ops)
        keys = split_keys(key, len(ops))
        return [rec(op, k) for op, k in zip(ops, keys, strict=True)]

    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.asarray(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return jnp.sum(lx.diagonal(operator))
    if isinstance(operator, DiagonalizedOperator):
        total = jnp.sum(operator.eigenvalues)
        return jnp.real(total) if operator.real_output else total
    if isinstance(operator, Toeplitz):
        # Constant diagonal c[0] (gh-373).
        return operator.in_size() * operator.column[0]
    if isinstance(operator, BlockDiag):
        return ft.reduce(jnp.add, rec_all(operator.operators))
    if isinstance(operator, Kronecker):
        # trace(A ⊗ B) = trace(A) · trace(B).
        return ft.reduce(jnp.multiply, rec_all(operator.operators))
    if isinstance(operator, BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag):
        return _trace_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _trace_low_rank(operator, rec(operator.base))
    if isinstance(operator, KroneckerSum):
        # trace(A ⊕ B) = n_b · trace(A) + n_a · trace(B).
        trace_a, trace_b = rec_all((operator.A, operator.B))
        return operator.B.out_size() * trace_a + operator.A.out_size() * trace_b
    if isinstance(operator, SumOfKroneckers):
        return ft.reduce(jnp.add, rec_all(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return rec(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        first, second = rec_all((operator.operator1, operator.operator2))
        return first + second
    if isinstance(operator, lx.MulLinearOperator):
        return operator.scalar * rec(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return rec(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        return -rec(operator.operator)
    if stochastic:
        return _trace_stochastic(operator, num_probes, key, sampler, algorithm)
    return jnp.trace(operator.as_matrix())


def _trace_block_tridiag(
    operator: BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, ""]:
    """trace of block-tridiagonal = sum of traces of diagonal blocks."""
    return jnp.sum(jax.vmap(jnp.trace)(operator.diagonal))


def _trace_low_rank(
    operator: LowRankUpdate, base_trace: Float[Array, ""]
) -> Float[Array, ""]:
    """trace(L + U diag(d) V^T) = trace(L) + sum_k d[k] (V[:, k] . U[:, k])."""
    update = jnp.sum(operator.U * operator.d * operator.V)
    return base_trace + update


def _trace_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName | None,
    algorithm: Literal["hutchinson", "hutchpp", "xtrace"],
) -> Float[Array, ""]:
    """Stochastic trace estimator (matfree Hutchinson / XTrace, or Hutch++)."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    if algorithm == "xtrace":
        # XTrace requires rotationally invariant probes.
        if sampler == "signs":
            raise ValueError(
                "XTrace requires a rotationally invariant sampler; "
                'use sampler="normal" or sampler="sphere".'
            )
        probe_fn = resolve_sampler(
            sampler or "sphere", n, num_probes, dtype=operator.in_structure().dtype
        )
        integrand = matfree.stochtrace.leave_one_out_xtrace()
        estimate = matfree.stochtrace.estimator_leave_one_out(integrand, probe_fn)
    elif algorithm == "hutchinson":
        probe_fn = resolve_sampler(
            sampler or "signs", n, num_probes, dtype=operator.in_structure().dtype
        )
        integrand = matfree.stochtrace.monte_carlo_trace()
        estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    elif algorithm == "hutchpp":
        return hutchpp_trace(operator, num_probes, key, sampler or "signs")
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; expected "
            '"hutchinson", "hutchpp" or "xtrace".'
        )
    return estimate(operator.mv, key)


def trace_and_diag(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> tuple[Float[Array, ""], Float[Array, " n"]]:
    """Jointly estimate the trace and diagonal from one probe pass.

    Halves the matvec budget relative to calling
    ``trace(..., stochastic=True)`` and ``diag(..., stochastic=True)``
    separately — both statistics are accumulated from the same
    ``A @ probe`` products.

    Args:
        operator: A square linear operator.
        num_probes: Number of probe vectors.
        key: PRNG key. If ``None``, uses ``jax.random.PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    Returns:
        Tuple ``(trace_estimate, diagonal_estimate)``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, n, num_probes, dtype=operator.in_structure().dtype
    )
    integrand = matfree.stochtrace.monte_carlo_trace_and_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    result = estimate(operator.mv, key)
    return result["trace"], result["diagonal"]
