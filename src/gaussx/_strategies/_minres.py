"""MINRES solver strategy: symmetric, possibly indefinite, iterative solve.

A is a square N × N symmetric operator (it need not be positive definite),
b the right-hand side and x the solution of A x = b. MINRES (Paige & Saunders,
1975) runs the Lanczos recurrence

    β₁ v₁ = b,   α_j = v_jᵀ A v_j,   β_j+1 v_j+1 = A v_j − α_j v_j − β_j v_j−1,

with orthonormal Lanczos vectors v_j and the j × j Lanczos tridiagonal T_j
(diagonal α_j, off-diagonal β_j+1), j = 1, 2, …, and picks the iterate
x_j ∈ span(v_1, …, v_j) that minimises ‖b − A x_j‖₂. Givens rotations with
cosines c_j and sines s_j reduce T_j to an upper-triangular R_j whose column j
holds ε_j, δ_j and γ_j (two above the diagonal, one above, on it; these are
Paige and Saunders' letters), and φ̄_j+1 is the signed residual norm after
j steps, ‖r_j‖₂ = |φ̄_j+1| for the residual r_j = b − A x_j. The search
directions are w_j = (v_j − ε_j w_j−2 − δ_j w_j−1) / γ_j. The
log-determinant is the indefinite stochastic Lanczos quadrature (SLQ)
estimate of `gaussx.IndefiniteSLQLogdet` over probes z^(i),
i = 1, …, N_probe.
"""

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
    vector: Float[Array, " N"],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    max_steps: int = 1000,
) -> tuple[Float[Array, " N"], Int[Array, ""], Bool[Array, ""]]:
    """Solve A x = b by MINRES for a symmetric A.

        x_j = argmin_{x ∈ span(v_1, …, v_j)} ‖b − A x‖₂

    stopping once ‖r_j‖₂ = |φ̄_j+1| ≤ tol_abs + tol_rel ‖b‖₂. Unlike CG,
    MINRES only needs A symmetric, not positive definite. The iteration is a
    ``lax.while_loop`` that stops as soon as it converges, so it costs one
    matvec per step actually taken rather than ``max_steps`` (gh-336).

    Algorithm:
        1. Start from x₀ = 0, β₁ = ‖b‖₂, v₁ = b / β₁ and φ̄₁ = β₁.
        2. Lanczos step: α_j = v_jᵀ A v_j and
           β_j+1 v_j+1 = A v_j − α_j v_j − β_j v_j−1.
        3. Apply the two previous rotations to column j of T_j:
           ε_j = s_j−1 β_j, δ_j = s_j α_j + c_j−1 c_j β_j and
           γ̄_j = c_j α_j − c_j−1 s_j β_j.
        4. Build the new rotation that zeroes β_j+1:
           γ_j = √(γ̄_j² + β_j+1²), c_j+1 = γ̄_j / γ_j, s_j+1 = β_j+1 / γ_j.
        5. Update w_j = (v_j − ε_j w_j−2 − δ_j w_j−1) / γ_j,
           x_j = x_j−1 + c_j+1 φ̄_j w_j and φ̄_j+1 = −s_j+1 φ̄_j.

    Args:
        matvec: The product v ↦ A v.
        vector: Right-hand side b, of length N.
        rtol: Relative tolerance tol_rel on the residual norm.
        atol: Absolute tolerance tol_abs on the residual norm.
        max_steps: Maximum number of steps j.

    Returns:
        Tuple ``(solution, num_steps, converged)`` holding x_j, the number
        of steps j taken and whether |φ̄_j+1| ≤ tol_abs + tol_rel ‖b‖₂.

    Examples:
        An indefinite diagonal system, solved exactly in two steps:

        >>> import jax.numpy as jnp
        >>> from gaussx._strategies._minres import _minres_solve
        >>> matrix = jnp.diag(jnp.array([2.0, -1.0]))
        >>> b = jnp.array([1.0, 1.0])
        >>> x, num_steps, converged = _minres_solve(lambda v: matrix @ v, b)
        >>> [round(float(v), 4) for v in x], int(num_steps), bool(converged)
        ([0.5, -1.0], 2, True)

        A zero right-hand side converges at once:

        >>> x, num_steps, converged = _minres_solve(lambda v: v, jnp.zeros(2))
        >>> int(num_steps), bool(converged)
        (0, True)
    """
    size = vector.shape[0]
    dtype = vector.dtype

    # β₁ = ‖b‖₂
    rhs_norm = jnp.linalg.norm(vector)
    # β₁ = 0 for a zero right-hand side; divide by 1 instead
    safe_rhs_norm = jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
    zeros = jnp.zeros(size, dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    # tol = tol_abs + tol_rel ‖b‖₂
    tolerance = jnp.asarray(atol + rtol * rhs_norm, dtype=dtype)

    # (x_j−1, v_j−1, v_j, w_j−2, w_j−1, c_j−1, s_j−1, c_j, s_j, φ̄_j, β_j, j − 1)
    init_state = (
        zeros,
        zeros,
        # v₁ = b / β₁
        vector / safe_rhs_norm,
        zeros,
        zeros,
        one,
        zero,
        one,
        zero,
        # φ̄₁ = β₁
        jnp.asarray(rhs_norm, dtype=dtype),
        jnp.asarray(rhs_norm, dtype=dtype),
        jnp.zeros((), dtype=jnp.int32),
    )

    def cond_fun(state):
        residual_norm = state[9]
        step = state[11]
        # j < max_steps and ‖r_j‖₂ = |φ̄_j+1| > tol
        return (step < max_steps) & (jnp.abs(residual_norm) > tolerance)

    def body_fun(state):
        (
            solution,
            basis_prev,
            basis,
            direction_prev2,
            direction_prev,
            cosine_prev,
            sine_prev,
            cosine,
            sine,
            residual_norm,
            beta,
            step,
        ) = state

        # --- Lanczos step ---
        # A v_j
        operator_basis = matvec(basis)
        # α_j = v_jᵀ A v_j
        # jnp, not einx: einx cannot resolve the empty axis of a 0 × 0 system
        alpha = jnp.dot(basis, operator_basis)
        # β_j+1 v_j+1 = A v_j − α_j v_j − β_j v_j−1
        basis_next = operator_basis - alpha * basis - beta * basis_prev
        # β_j+1 = ‖β_j+1 v_j+1‖₂
        beta_next = jnp.linalg.norm(basis_next)
        # β_j+1 = 0 once the Krylov space is exhausted; divide by 1 instead
        safe_beta_next = jnp.where(beta_next > 0.0, beta_next, 1.0)
        # v_j+1
        basis_next = basis_next / safe_beta_next

        # --- previous rotations on column j of T_j ---
        # γ̄_j = c_j α_j − c_j−1 s_j β_j
        unrotated_diagonal = cosine * alpha - cosine_prev * sine * beta
        # ε_j = s_j−1 β_j
        second_superdiagonal = sine_prev * beta
        # δ_j = s_j α_j + c_j−1 c_j β_j
        superdiagonal = sine * alpha + cosine_prev * cosine * beta

        # --- new rotation, zeroing β_j+1 ---
        # γ_j = √(γ̄_j² + β_j+1²)
        diagonal = jnp.sqrt(unrotated_diagonal**2 + beta_next**2)
        # γ_j = 0 at breakdown; divide by 1 instead
        safe_diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
        # c_j+1 = γ̄_j / γ_j
        cosine_next = unrotated_diagonal / safe_diagonal
        # s_j+1 = β_j+1 / γ_j
        sine_next = beta_next / safe_diagonal

        # --- update the direction, the iterate and the residual norm ---
        # w_j = (v_j − ε_j w_j−2 − δ_j w_j−1) / γ_j
        direction = (
            basis
            - second_superdiagonal * direction_prev2
            - superdiagonal * direction_prev
        ) / safe_diagonal
        # c_j+1 φ̄_j
        step_size = cosine_next * residual_norm
        # x_j = x_j−1 + c_j+1 φ̄_j w_j
        solution_next = solution + step_size * direction
        # φ̄_j+1 = −s_j+1 φ̄_j
        residual_norm_next = -sine_next * residual_norm

        return (
            solution_next,
            basis,
            basis_next,
            direction_prev,
            direction,
            cosine,
            sine,
            cosine_next,
            sine_next,
            residual_norm_next,
            beta_next,
            step + 1,
        )

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    solution = final_state[0]
    num_steps = final_state[11]
    # ‖r_j‖₂ = |φ̄_j+1| ≤ tol
    converged = jnp.abs(final_state[9]) <= tolerance
    return solution, num_steps, converged


class _MINRES(lx.AbstractLinearSolver):
    """MINRES as a lineax solver, so `lineax.linear_solve` differentiates it.

        x = A⁻¹ b,   by `_minres_solve`

    Going through `lineax.linear_solve` gives MINRES lineax's implicit
    derivative rules: the backward pass of x = A⁻¹ b is one more MINRES
    solve with Aᵀ = A, not a tape through every iteration (gh-336). The
    operator is assumed symmetric and nonsingular.

    Attributes:
        rtol: Relative tolerance tol_rel on the residual norm.
        atol: Absolute tolerance tol_abs on the residual norm.
        max_steps: Maximum number of steps j.
    """

    rtol: float
    atol: float
    max_steps: int

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]):
        """Linearise A once for every solve with it.

        Args:
            operator: Square symmetric operator A.
            options: Unused lineax options.

        Returns:
            The linearised operator, the solver state.

        Raises:
            ValueError: If A is not square.
        """
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
        """Solve A x = b by MINRES and report convergence to lineax.

            x = A⁻¹ b

        Args:
            state: The linearised operator A from `init`.
            vector: Right-hand side b, of length N.
            options: Unused lineax options.

        Returns:
            Tuple ``(solution, result, stats)``: x, lineax's success or
            ``max_steps_reached`` flag, and the steps taken and allowed.
        """
        del options
        solution, num_steps, converged = _minres_solve(
            state.mv,
            vector,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        result = lx.RESULTS.where(
            converged, lx.RESULTS.successful, lx.RESULTS.max_steps_reached
        )
        return solution, result, {"num_steps": num_steps, "max_steps": self.max_steps}

    def transpose(self, state: lx.AbstractLinearOperator, options: dict[str, Any]):
        """The state for solving with Aᵀ, used by the implicit derivative.

        Args:
            state: The linearised operator A.
            options: lineax options, passed through.

        Returns:
            Tuple ``(state, options)`` for Aᵀ.
        """
        return state.transpose(), options

    def conj(self, state: lx.AbstractLinearOperator, options: dict[str, Any]):
        """The state for solving with the complex conjugate of A.

        Args:
            state: The linearised operator A.
            options: lineax options, passed through.

        Returns:
            Tuple ``(state, options)`` for the conjugate of A.
        """
        return lx.conj(state), options

    def assume_full_rank(self) -> bool:
        """MINRES is used on nonsingular A, so lineax may assume full rank.

        Returns:
            ``True``.
        """
        return True


class MINRESSolver(AbstractSolverStrategy):
    """MINRES solver for symmetric, possibly indefinite, systems.

        x_j = argmin_{x ∈ span(v_1, …, v_j)} ‖b − (A + σ I) x‖₂

    with σ the diagonal ``shift``, stopping once
    ‖r_j‖₂ ≤ tol_abs + tol_rel ‖b‖₂. Lanczos-based MINRES runs the solve and
    matfree's stochastic Lanczos quadrature (SLQ) the log-determinant.
    Unlike CG, MINRES only needs symmetry: it works on indefinite and
    singular systems. Use cases: EP natural parameters, saddle-point
    systems, Laplace-approximation Hessians.

    The solve stops as soon as it converges and goes through
    `lineax.linear_solve`, so gradients are implicit (one more MINRES solve)
    rather than a tape through the iterations. Running out of ``max_steps``
    raises, like `gaussx.CGSolver`, unless ``throw=False`` (gh-336).

    Attributes:
        rtol: Relative tolerance tol_rel. ``None``: 1e-5 in float64, 1e-3 in
            float32 (gh-327).
        atol: Absolute tolerance tol_abs. ``None``: 1e-5 in every dtype.
        max_steps: Maximum number of MINRES steps j.
        shift: Diagonal shift σ, so the system is (A + σ I) x = b.
        num_probes: Number of SLQ probe vectors N_probe.
        lanczos_order: Order of the Lanczos decomposition for SLQ.
        seed: Seed of the SLQ probes when `logdet` gets no ``key`` (the same
            probes on every call; see `gaussx.KeyedSolver`).
        throw: Raise when MINRES does not converge within ``max_steps`` (the
            default). With ``False`` the last iterate is returned unchecked.

    Examples:
        An indefinite system, which CG cannot solve:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[2.0, 1.0], [1.0, -3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.symmetric_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.MINRESSolver().solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-3))
        True

        A diagonal shift solves (A + σ I) x = b:

        >>> x = gaussx.MINRESSolver(shift=0.5).solve(A, b)
        >>> bool(jnp.allclose((matrix + 0.5 * jnp.eye(2)) @ x, b, atol=1e-3))
        True

        Running out of steps raises:

        >>> tight = gaussx.MINRESSolver(max_steps=1, rtol=1e-12, atol=1e-12)
        >>> tight.solve(A, b)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        equinox.EquinoxRuntimeError: The maximum number of solver steps was reached.

    References:
        Paige & Saunders (1975), MINRES.
    """

    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    shift: float = eqx.field(static=True, default=0.0)
    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    throw: bool = eqx.field(static=True, default=True)

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve (A + σ I) x = b with MINRES.

            x = (A + σ I)⁻¹ b

        Algorithm:
            1. Resolve tol_rel for the dtype; tol_abs defaults to 1e-5.
            2. Shift A by σ I when σ ≠ 0, keeping its symmetric tag.
            3. Run `_minres_solve` through `lineax.linear_solve`, raising at
               ``max_steps`` when ``throw`` is set.

        Args:
            operator: Symmetric, possibly indefinite, operator A, N × N.
            vector: Right-hand side b, of length N.

        Returns:
            Solution x, of length N.

        Raises:
            equinox.EquinoxRuntimeError: If MINRES has not converged after
                ``max_steps`` steps and ``throw`` is set.
        """
        dtype = operator_dtype(operator, vector)
        rtol = resolve_tolerance(self.rtol, dtype, 1e-5)
        atol = 1e-5 if self.atol is None else self.atol
        solver = _MINRES(
            rtol=rtol,
            atol=atol,
            max_steps=self.max_steps,
        )
        if self.shift != 0.0:
            identity = lx.IdentityLinearOperator(operator.in_structure())
            # A + σ I
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
        """Estimate log |det(A + σ I)| by stochastic Lanczos quadrature.

            log |det(A + σ I)| ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log|T^(i)| e₁

        with T^(i) the Lanczos tridiagonal of A + σ I started at z^(i).

        Args:
            operator: Symmetric operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``: the same probes on every call.

        Returns:
            Scalar estimate of log |det(A + σ I)|.
        """
        return IndefiniteSLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            shift=self.shift,
            seed=self.seed,
        ).logdet(operator, key=key)
