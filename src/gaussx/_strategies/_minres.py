"""MINRES solver strategy: symmetric (possibly indefinite) iterative solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._slq_logdet import IndefiniteSLQLogdet


def _minres_solve(
    matvec,
    b: jnp.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_steps: int = 1000,
    shift: float = 0.0,
) -> jnp.ndarray:
    r"""Solve ``(A + shift I) x = b`` via MINRES for symmetric ``A``.

    Implements the Lanczos-based MINRES algorithm (Paige & Saunders, 1975).
    Unlike CG, only requires symmetry — not positive definiteness.

    Args:
        matvec: Function ``v -> A @ v``.
        b: Right-hand side vector, shape ``(N,)``.
        rtol: Relative tolerance on residual norm.
        atol: Absolute tolerance on residual norm.
        max_steps: Maximum number of iterations.
        shift: Diagonal shift ``(A + shift * I) x = b``.

    Returns:
        Approximate solution ``x``, shape ``(N,)``.
    """
    n = b.shape[0]
    dtype = b.dtype

    def shifted_matvec(v):
        return matvec(v) + shift * v

    # Initialize
    beta1 = jnp.linalg.norm(b)
    # Guard against zero RHS
    safe_beta1 = jnp.where(beta1 > 0.0, beta1, 1.0)

    v_prev = jnp.zeros(n, dtype=dtype)
    v_curr = b / safe_beta1
    w_prev = jnp.zeros(n, dtype=dtype)
    w_curr = jnp.zeros(n, dtype=dtype)
    x = jnp.zeros(n, dtype=dtype)

    # Givens rotation state
    c_prev, s_prev = 1.0, 0.0
    c_curr, s_curr = 1.0, 0.0
    eta = beta1

    beta_curr = beta1

    # State tuple (no step counter — scan handles iteration)
    init_state = (
        x,
        v_prev,
        v_curr,
        w_prev,
        w_curr,
        jnp.array(c_prev, dtype=dtype),
        jnp.array(s_prev, dtype=dtype),
        jnp.array(c_curr, dtype=dtype),
        jnp.array(s_curr, dtype=dtype),
        jnp.array(eta, dtype=dtype),
        jnp.array(beta_curr, dtype=dtype),
    )

    tol = jnp.array(atol + rtol * beta1, dtype=dtype)

    def scan_body(state, _):
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
        ) = state

        # Check convergence — if already converged, return state unchanged
        converged = jnp.abs(eta) <= tol

        # Lanczos step
        Av = shifted_matvec(v_curr)
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

        # Update solution
        x_new = x + (c_next * eta) * w_next

        # Update eta
        eta_new = -s_next * eta

        new_state = (
            x_new,
            v_curr,
            v_next,
            w_curr,
            w_next,
            c_curr,
            s_curr,
            c_next,
            s_next,
            eta_new,
            beta_next,
        )

        # If converged, keep old state; otherwise use new state
        out_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(converged, old, new),
            state,
            new_state,
        )
        return out_state, None

    final_state, _ = jax.lax.scan(scan_body, init_state, None, length=max_steps)
    x_sol = final_state[0]
    return x_sol


class MINRESSolver(AbstractSolverStrategy):
    """MINRES solver for symmetric (possibly indefinite) systems.

    Uses the Lanczos-based MINRES algorithm for the linear solve
    and matfree's stochastic Lanczos quadrature (SLQ) for the
    log-determinant. Unlike CG, MINRES only requires symmetry —
    it works on indefinite and singular systems.

    Use cases: EP natural parameters, saddle-point systems,
    Laplace approximation Hessians.

    Args:
        rtol: Relative tolerance for MINRES.
        atol: Absolute tolerance for MINRES.
        max_steps: Maximum MINRES iterations.
        shift: Diagonal shift — solves ``(A + shift * I) x = b``.
        num_probes: Number of probe vectors for stochastic logdet.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
    """

    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1000
    shift: float = 0.0
    num_probes: int = 20
    lanczos_order: int = 30

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: jnp.ndarray,
    ) -> jnp.ndarray:
        return _minres_solve(
            operator.mv,
            vector,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            shift=self.shift,
        )

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> jnp.ndarray:
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
