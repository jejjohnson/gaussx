"""Leave-one-out cross-validation via the bordered-system identity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._diag_inv import diag_inv
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve


class LOOResult(eqx.Module):
    """Result of leave-one-out cross-validation.

    Attributes:
        loo_log_likelihood: Scalar LOO-CV log-likelihood.
        loo_means: Per-point LOO predictive means, shape ``(N,)``.
        loo_variances: Per-point LOO predictive variances, shape ``(N,)``.
    """

    loo_log_likelihood: Float[Array, ""]
    loo_means: Float[Array, " N"]
    loo_variances: Float[Array, " N"]


def leave_one_out_cv(
    operator: lx.AbstractLinearOperator,
    y: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | None = None,
    diag_inv_method: str = "solve",
    diag_inv_num_probes: int = 30,
    diag_inv_key: jax.Array | None = None,
) -> LOOResult:
    """LOO-CV via the bordered-system identity.

    Computes leave-one-out predictive means, variances, and
    log-likelihood without refitting the model N times.

    Math:

        alpha = K_y^{-1} y
        mu_LOO_i   = y_i - alpha_i / [K_y^{-1}]_{ii}
        sigma^2_LOO_i = 1 / [K_y^{-1}]_{ii}
        LOO-CV = -(1/2) sum_i [ log sigma^2_LOO_i
                                + (y_i - mu_LOO_i)^2 / sigma^2_LOO_i
                                + log 2 pi ]

    Args:
        operator: A linear operator representing the (noise-inclusive)
            kernel matrix K_y.
        y: Observation vector of shape ``(N,)``.
        solver: Optional solve strategy for computing K_y^{-1} y
            and for the ``diag_inv`` computation. When ``None``,
            falls back to structural dispatch.
        diag_inv_method: Method passed to `diag_inv`. Defaults to
            ``"solve"`` (exact, deterministic, and honours ``solver``)
            rather than `diag_inv`'s own ``"auto"``, which switches to the
            stochastic Hutchinson estimator for unstructured ``N > 2048``.
            ``"cholesky"`` is equally exact and deterministic and is
            usually faster for a dense ``K_y`` (one factorisation instead
            of ``N`` solves); ``"auto"`` is preferable for structured
            operators with an exact fast path (`Kronecker`,
            `BlockTriDiag`, ...).
        diag_inv_num_probes: Number of Hutchinson probes when
            ``diag_inv_method="hutchinson"``.
        diag_inv_key: PRNG key for probe generation when
            ``diag_inv_method="hutchinson"``.

    Returns:
        A `LOOResult` containing the LOO log-likelihood,
        predictive means, and predictive variances.
    """
    alpha = dispatch_solve(operator, y, solver)
    diag_Kinv = diag_inv(
        operator,
        method=diag_inv_method,
        num_probes=diag_inv_num_probes,
        key=diag_inv_key,
        solver=solver,
    )

    loo_means = y - alpha / diag_Kinv
    loo_variances = 1.0 / diag_Kinv

    # (y_i - mu_i)^2 / sigma^2_i simplifies to alpha_i^2 / diag_Kinv_i
    loo_ll = -0.5 * jnp.sum(
        jnp.log(loo_variances) + alpha**2 / diag_Kinv + jnp.log(2.0 * jnp.pi)
    )

    return LOOResult(
        loo_log_likelihood=loo_ll,
        loo_means=loo_means,
        loo_variances=loo_variances,
    )
