"""Control variate for the gradient of a stochastic log-determinant (gh-826).

With probes z^(i) ~ 𝒩(0, P), i = 1, …, N_probe, from a preconditioner P ≈ A
and w^(i) = P⁻¹ z^(i), the Hutchinson estimate of ∂ log det A / ∂θ is

    a^(i) = (A⁻¹ z^(i))ᵀ (∂A/∂θ) w^(i),

which keeps its full variance as P → A. The same probes give a control

    b^(i) = w^(i)ᵀ (∂P/∂θ) w^(i),   𝔼[b^(i)] = tr(P⁻¹ ∂P/∂θ) = ∂ log det P / ∂θ,

whose mean is known exactly, so (Gyger, Furrer & Sigrist, 2026, Appendix D)

    ∂ log det A / ∂θ ≈ (1 / N_probe) ∑_i (a^(i) − b^(i)) + ∂ log det P / ∂θ

is unbiased, and its variance vanishes as P → A, where a^(i) = b^(i).
The coefficient of the control is one; see `logdet_control_variate`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def logdet_control_variate(
    preconditioner_logdet: Float[Array, ""],
    approximation: lx.AbstractLinearOperator,
    probe_inverses: Float[Array, "n probe"],
) -> Float[Array, ""]:
    """A zero-valued term whose gradient is the control variate.

        h = log det P − (1 / N_probe) ∑_i w^(i)ᵀ P w^(i),   w^(i) held fixed,
        returns h − stop_gradient(h)

    Its value is exactly zero, so adding it to a log-determinant estimate
    changes no forward value, and its gradient is
    ∂ log det P / ∂θ − (1 / N_probe) ∑_i b^(i), the controlled part of the
    estimator above. P must be differentiable here (built from θ, or from
    the system A), while the pass that estimates log det A holds P fixed.

    The coefficient of b^(i) is one rather than the least-squares ĉ_opt:
    ĉ_opt is a ratio of per-probe covariances for each parameter, which
    reverse mode can only form from per-probe gradients (N_probe copies of
    the parameter cotangent), and it tends to one as P → A anyway. On the
    gh-826 prototype a ĉ_opt estimated from the same 50 probes was less
    accurate than ĉ = 1. A fixed coefficient also has no 0 / 0 when
    b^(i) does not vary (a P that does not depend on θ): the term is zero.

    Args:
        preconditioner_logdet: log det P, differentiable in θ.
        approximation: P, differentiable in θ.
        probe_inverses: The probe solves w^(i) = P⁻¹ z^(i) as columns; their
            gradient is stopped.

    Returns:
        A scalar zero whose gradient is the control variate.
    """
    probes = jax.lax.stop_gradient(probe_inverses)
    # P w^(i), one matvec per probe
    applied = jax.vmap(approximation.mv, in_axes=1, out_axes=1)(probes)
    # (1 / N_probe) ∑_i w^(i)ᵀ P w^(i), a full reduction over both axes
    quadratic = jnp.sum(probes * applied) / probes.shape[1]
    # h = log det P − (1 / N_probe) ∑_i w^(i)ᵀ P w^(i)
    controlled = preconditioner_logdet - quadratic
    # value 0, gradient ∇h
    return controlled - jax.lax.stop_gradient(controlled)
