"""Leave-one-out cross-validation via the bordered-system identity."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve
from gaussx._sugar._diag_inv import diag_inv


class LOOResult(eqx.Module):
    """Result of leave-one-out cross-validation.

    Attributes:
        loo_log_likelihood: Scalar LOO-CV log-likelihood.
        loo_means: Per-point LOO predictive means, shape ``(N,)``.
        loo_variances: Per-point LOO predictive variances, shape ``(N,)``.
    """

    loo_log_likelihood: jnp.ndarray
    loo_means: jnp.ndarray
    loo_variances: jnp.ndarray


def leave_one_out_cv(
    operator: lx.AbstractLinearOperator,
    y: jnp.ndarray,
    *,
    solver: AbstractSolveStrategy | None = None,
) -> LOOResult:
    """LOO-CV via the bordered-system identity.

    Computes leave-one-out predictive means, variances, and
    log-likelihood without refitting the model N times.

    Math::

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
        solver: Optional solve strategy for computing K_y^{-1} y.
            When ``None``, falls back to structural dispatch.

    Returns:
        A :class:`LOOResult` containing the LOO log-likelihood,
        predictive means, and predictive variances.
    """
    alpha = dispatch_solve(operator, y, solver)
    diag_Kinv = diag_inv(operator)

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
