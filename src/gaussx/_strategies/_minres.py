"""MINRES solver strategy: symmetric (possibly indefinite) iterative solve."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import IndefiniteSLQLogdet
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


def _minres_solve(
    matvec: Callable[[Array], Array],
    b: Float[Array, " N"],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_steps: int = 1000,
) -> tuple[Float[Array, " N"], Int[Array, ""], Bool[Array, ""]]:
    r"""Solve ``A x = b`` via MINRES for symmetric ``A``.

    Implements the Lanczos-based MINRES algorithm (Paige & Saunders, 1975).
    Unlike CG, only requires symmetry — not positive definiteness.

    The iteration is a ``lax.while_loop`` that stops as soon as the residual
    norm ``|η|`` falls to ``atol + rtol ||b||``, so it costs one matvec per
    iteration actually taken rather than ``max_steps`` (gh-336).

    Args:
        matvec: Function ``v -> A @ v``.
        b: Right-hand side vector, shape ``(N,)``.
        rtol: Relative tolerance on residual norm.
        atol: Absolute tolerance on residual norm.
        max_steps: Maximum number of iterations.

    Returns:
        ``(x, num_steps, converged)``.
    """
    n = b.shape[0]
    dtype = b.dtype

    beta1 = jnp.linalg.norm(b)
    # Guard against zero RHS
    safe_beta1 = jnp.where(beta1 > 0.0, beta1, 1.0)
    zeros = jnp.zeros(n, dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    tol = jnp.asarray(atol + rtol * beta1, dtype=dtype)

    # (x, v_prev, v_curr, w_prev, w_curr, c_prev, s_prev, c_curr, s_curr,
    #  eta, beta_curr, step)
    init_state = (
        zeros,
        zeros,
        b / safe_beta1,
        zeros,
        zeros,
        one,
        zero,
        one,
        zero,
        jnp.asarray(beta1, dtype=dtype),
        jnp.asarray(beta1, dtype=dtype),
        jnp.zeros((), dtype=jnp.int32),
    )

    def cond_fun(state):
        eta, step = state[9], state[11]
        return (step < max_steps) & (jnp.abs(eta) > tol)

    def body_fun(state):
        (
            x,
            v_prev,
            v_curr,
            w_prev,
            w_curr,
            c_prev,
            s_prev,
            c_curr,
            s_curr,
            eta,
            beta_curr,
            step,
        ) = state

        # Lanczos step
        Av = matvec(v_curr)
        alpha = jnp.dot(v_curr, Av)

        v_next = Av - alpha * v_curr - beta_curr * v_prev
        beta_next = jnp.linalg.norm(v_next)
        safe_beta_next = jnp.where(beta_next > 0.0, beta_next, 1.0)
        v_next = v_next / safe_beta_next

        # Apply previous Givens rotation
        delta = c_curr * alpha - c_prev * s_curr * beta_curr
        eps_val = s_prev * beta_curr
        gamma_bar = s_curr * alpha + c_prev * c_curr * beta_curr

        # Construct new Givens rotation to zero out beta_next
        gamma = jnp.sqrt(delta**2 + beta_next**2)
        safe_gamma = jnp.where(gamma > 0.0, gamma, 1.0)
        c_next = delta / safe_gamma
        s_next = beta_next / safe_gamma

        # Update w vectors
        w_next = (v_curr - eps_val * w_prev - gamma_bar * w_curr) / safe_gamma

        return (
            x + (c_next * eta) * w_next,
            v_curr,
            v_next,
            w_curr,
            w_next,
            c_curr,
            s_curr,
            c_next,
            s_next,
            -s_next * eta,
            beta_next,
            step + 1,
        )

    final = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return final[0], final[11], jnp.abs(final[9]) <= tol


class _MINRES(lx.AbstractLinearSolver):
    """MINRES as a lineax solver, so `lineax.linear_solve` differentiates it.

    Going through `lineax.linear_solve` gives MINRES lineax's implicit
    derivative rules: the backward pass of ``x = A⁻¹ b`` is one more MINRES
    solve with ``Aᵀ = A``, not a tape through every iteration (gh-336). The
    operator is assumed symmetric and nonsingular.
    """

    rtol: float
    atol: float
    max_steps: int

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError("MINRES may only be used with square operators.")
        return lx.linearise(operator)

    def compute(
        self,
        state: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
        options: dict[str, Any],
    ) -> tuple[Float[Array, " n"], lx.RESULTS, dict[str, Any]]:
        del options
        x, num_steps, converged = _minres_solve(
            state.mv,
            vector,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        result = lx.RESULTS.where(
            converged, lx.RESULTS.successful, lx.RESULTS.max_steps_reached
        )
        return x, result, {"num_steps": num_steps, "max_steps": self.max_steps}

    def transpose(self, state: lx.AbstractLinearOperator, options: dict[str, Any]):
        return state.transpose(), options

    def conj(self, state: lx.AbstractLinearOperator, options: dict[str, Any]):
        return lx.conj(state), options

    def assume_full_rank(self) -> bool:
        return True


class MINRESSolver(AbstractSolverStrategy):
    """MINRES solver for symmetric (possibly indefinite) systems.

    Uses the Lanczos-based MINRES algorithm for the linear solve
    and matfree's stochastic Lanczos quadrature (SLQ) for the
    log-determinant. Unlike CG, MINRES only requires symmetry —
    it works on indefinite and singular systems.

    Use cases: EP natural parameters, saddle-point systems,
    Laplace approximation Hessians.

    The solve stops as soon as it converges and goes through
    `lineax.linear_solve`, so gradients are implicit (one more MINRES solve)
    rather than a tape through the iterations. Running out of ``max_steps``
    raises, like `gaussx.CGSolver`, unless ``throw=False`` (gh-336).

    Attributes:
        rtol: Relative tolerance for MINRES. ``None``: ``1e-5`` in float64,
            ``1e-3`` in float32 (gh-327).
        atol: Absolute tolerance for MINRES. ``None``: ``1e-5`` in every
            dtype.
        max_steps: Maximum MINRES iterations.
        shift: Diagonal shift — solves ``(A + shift * I) x = b``.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
        throw: Raise when MINRES does not converge within ``max_steps`` (the
            default). With ``False`` the last iterate is returned unchecked.
    """

    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    shift: float = eqx.field(static=True, default=0.0)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    throw: bool = eqx.field(static=True, default=True)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve ``(A + shift I) x = b`` with MINRES.

        Args:
            operator: A symmetric (possibly indefinite) linear operator ``A``.
            vector: Right-hand side ``b``, shape ``(n,)``.

        Returns:
            Solution ``x``, shape ``(n,)``.
        """
        dtype = operator_dtype(operator, vector)
        solver = _MINRES(
            rtol=resolve_tolerance(self.rtol, dtype, 1e-5),
            atol=1e-5 if self.atol is None else self.atol,
            max_steps=self.max_steps,
        )
        if self.shift != 0.0:
            identity = lx.IdentityLinearOperator(operator.in_structure())
            shifted = operator + self.shift * identity
            if lx.is_symmetric(operator):
                shifted = lx.TaggedLinearOperator(shifted, lx.symmetric_tag)
            operator = shifted
        return lx.linear_solve(operator, vector, solver, throw=self.throw).value

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic ``log|det(A + shift I)|`` via Lanczos quadrature.

        Args:
            operator: A symmetric linear operator.
            key: PRNG key for probe vector sampling. If None,
                uses ``jax.random.PRNGKey(0)``.

        Returns:
            Scalar estimate of ``log|det(A + shift I)|``.
        """
        return IndefiniteSLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            shift=self.shift,
        ).logdet(operator, key=key)
