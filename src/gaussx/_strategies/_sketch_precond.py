"""Sketch-and-precondition least squares (G15).

A is a tall M × N operator (M ≫ N), b ∈ ℝ^M the right-hand side and
x ∈ ℝ^N the solution of the damped least-squares problem
min_x ‖A x − b‖₂² + β ‖x‖₂² with β = damp² ≥ 0. A random sketch
S ∈ ℝ^(d × M) with d ≪ M rows (`gaussx.AbstractSketch`) compresses A to
S A ∈ ℝ^(d × N); the reduced QR factorisation [S A; √β I] = Q R gives a
sketch-and-solve warm start x₀ and the right preconditioner R⁻¹ for LSMR.
S is an ε-subspace embedding for range(A) when
(1 − ε) ‖A x‖₂ ≤ ‖S A x‖₂ ≤ (1 + ε) ‖A x‖₂ for every x (ε is the
distortion, the letter of the sketching literature).
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._sketching import (
    AbstractSketch,
    GaussianSketch,
    SparseSignSketch,
    SRHTSketch,
)
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._lsmr import LSMRSolver
from gaussx._strategies._tolerances import operator_dtype, resolve_tolerance


SketchName = Literal["sparse_sign", "srht", "gaussian"]

# A matrix-free operator is sketched this many rows of S at a time, so the
# peak extra memory is _ROW_BATCH · M (rows of Sᵀ) rather than d · M.
_ROW_BATCH = 32


def _sketch(
    operator: lx.AbstractLinearOperator, sketch: AbstractSketch
) -> Float[Array, "d n"]:
    """The sketch S A, without materialising Sᵀ for a matrix-free A.

        [S A]_i,: = (Aᵀ Sᵀ e_i)ᵀ,   i = 1, …, d

    `AbstractSketch.sketch_operator` vmaps over all d rows of S at once,
    which forms the M × d block Sᵀ: 4.8 GB in float32 at M = 10⁶, d = 1200.
    Row i of S A is Aᵀ (Sᵀ e_i), so this maps over the rows of S in batches
    of ``_ROW_BATCH``.

    Args:
        operator: Operator A, M × N.
        sketch: Sketch S, d × M.

    Returns:
        The sketch S A, d × N.

    Raises:
        ValueError: If the sketch's ``in_size`` is not M.
    """
    if operator.out_size() != sketch.in_size:
        raise ValueError(
            f"Cannot sketch an operator with {operator.out_size()} rows using a "
            f"sketch with in_size={sketch.in_size}."
        )
    if isinstance(operator, lx.MatrixLinearOperator):
        # S A, on the stored matrix
        return sketch.apply(operator.matrix)
    transpose = operator.transpose()
    # e_i, i = 1, …, d
    unit_rows = jnp.eye(sketch.out_size, dtype=operator.out_structure().dtype)
    # Aᵀ Sᵀ e_i, a batch of rows of S at a time
    return jax.lax.map(
        lambda unit_row: transpose.mv(sketch.apply_transpose(unit_row)),
        unit_rows,
        batch_size=min(_ROW_BATCH, sketch.out_size),
    )


def _check_full_rank(
    triangular_factor: Float[Array, "n n"], solution: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Raise (at run time) when the undamped triangular factor R is singular.

        minₙ |R_nn| ≤ N eps maxₙ |R_nn|

    Args:
        triangular_factor: Upper-triangular factor R of the sketch, N × N.
        solution: Array to thread the check through, e.g. x₀.

    Returns:
        ``solution``, unchanged.

    Raises:
        equinox.EquinoxRuntimeError: If R is numerically singular.
    """
    # |R_nn|
    magnitudes = jnp.abs(jnp.diagonal(triangular_factor))
    # N eps maxₙ |R_nn|
    threshold = (
        triangular_factor.shape[0]
        * jnp.finfo(triangular_factor.dtype).eps
        * jnp.max(magnitudes)
    )
    return eqx.error_if(
        solution,
        jnp.min(magnitudes) <= threshold,
        "The sketch SA is numerically rank-deficient: A has (numerically) "
        "dependent columns, so the least-squares solution is not unique. "
        "Pass damp > 0 for the ridge solution.",
    )


def _sketch_qr(
    operator: lx.AbstractLinearOperator, sketch: AbstractSketch, damp: float
) -> tuple[Float[Array, "d n"], Float[Array, "n n"]]:
    """Reduced QR of the (augmented) sketch.

        [S A; √β I] = Q R,   √β = ``damp``

    Args:
        operator: Operator A, M × N.
        sketch: Sketch S, d × M.
        damp: Tikhonov damping √β ≥ 0; the √β I block is stacked only when
            it is nonzero.

    Returns:
        Tuple ``(orthogonal_top, triangular_factor)``: the first d rows
        Q_1:d of Q, the block that meets S b, and R.
    """
    # S A
    sketched_operator = _sketch(operator, sketch)
    size = operator.in_size()
    stacked = sketched_operator
    if damp != 0.0:
        # [S A; √β I]
        stacked = jnp.concatenate(
            [sketched_operator, damp * jnp.eye(size, dtype=sketched_operator.dtype)]
        )
    # [S A; √β I] = Q R
    orthogonal, triangular_factor = jnp.linalg.qr(stacked)
    return orthogonal[: sketch.out_size], triangular_factor


def _warm_start(
    orthogonal_top: Float[Array, "d n"],
    triangular_factor: Float[Array, "n n"],
    sketch: AbstractSketch,
    vector: Float[Array, " m"],
) -> Float[Array, " n"]:
    """Sketch-and-solve estimate, the warm start of the preconditioned LSMR.

        x₀ = R⁻¹ Q_1:dᵀ S b

    Args:
        orthogonal_top: First d rows Q_1:d of the QR factor Q.
        triangular_factor: Upper-triangular R, N × N.
        sketch: Sketch S, d × M.
        vector: Right-hand side b, of length M.

    Returns:
        The estimate x₀, of length N.
    """
    # S b
    sketched_rhs = sketch.apply(vector)
    # Q_1:dᵀ S b
    projected_rhs = einsum(orthogonal_top, sketched_rhs, "sketch col, sketch -> col")
    # x₀ = R⁻¹ Q_1:dᵀ S b
    return jsl.solve_triangular(triangular_factor, projected_rhs, lower=False)


def sketch_and_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " m"],
    *,
    sketch: AbstractSketch,
    damp: float = 0.0,
) -> Float[Array, " n"]:
    """Approximate least squares from a sketch.

        x̂ = argmin_x ‖S (A x − b)‖₂² + β ‖x‖₂² = R⁻¹ Q_1:dᵀ S b,
        [S A; √β I] = Q R,   β = damp²

    Solves the sketched problem exactly, through the reduced QR
    factorisation of the d × N sketch S A (stacked on √β I when
    ``damp`` > 0). If S is an ε-subspace embedding for range([A b]), the
    residual is within a constant factor of the optimum x^⋆ (Woodruff,
    2014, §2):

        ‖A x̂ − b‖₂ ≤ ‖S (A x̂ − b)‖₂ / (1 − ε)
                   ≤ ‖S (A x^⋆ − b)‖₂ / (1 − ε)
                   ≤ (1 + ε) / (1 − ε) ‖A x^⋆ − b‖₂.

    The solution itself is only O(ε)-accurate, so this is a function, not
    a solver strategy; `gaussx.SketchAndPrecondLSMR` uses it as its warm
    start and then solves to tolerance.

    Algorithm:
        1. Sketch the operator, S A (d × N), matrix-free through
           transpose-matvecs when A is not stored.
        2. Factor [S A; √β I] = Q R, O(d N²).
        3. Solve x̂ = R⁻¹ Q_1:dᵀ (S b).
        4. With β = 0, raise when R is numerically singular.

    Args:
        operator: Tall operator A, M × N.
        vector: Right-hand side b, of length M.
        sketch: A sampled sketch S with ``in_size == M`` and, for a unique
            solution when ``damp == 0``, ``out_size >= N``.
        damp: Tikhonov damping √β ≥ 0.

    Returns:
        The sketch-and-solve estimate x̂, of length N.

    Raises:
        ValueError: If ``damp < 0`` or the sketch's ``in_size`` is not M.
        equinox.EquinoxRuntimeError: If ``damp == 0`` and S A is
            numerically rank-deficient (A has dependent columns).

    Examples:
        A noisy overdetermined system with true coefficients one:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jr.normal(jr.key(0), (2000, 10))
        >>> b = matrix @ jnp.ones(10) + 0.1 * jr.normal(jr.key(1), (2000,))
        >>> sketch = gaussx.GaussianSketch.sample(jr.key(2), d=200, m=2000)
        >>> A = lx.MatrixLinearOperator(matrix)
        >>> x = gaussx.sketch_and_solve(A, b, sketch=sketch)
        >>> bool(jnp.max(jnp.abs(x - 1.0)) < 0.05)
        True

        A negative damping is rejected:

        >>> gaussx.sketch_and_solve(
        ...     A, b, sketch=sketch, damp=-1.0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: sketch_and_solve needs damp >= 0, got -1.0.

    References:
        Woodruff, D. P. (2014). Sketching as a tool for numerical linear
        algebra. *Foundations and Trends in Theoretical Computer Science*,
        10(1-2), 1-157.
    """
    if damp < 0:
        raise ValueError(f"sketch_and_solve needs damp >= 0, got {damp}.")
    # [S A; √β I] = Q R
    orthogonal_top, triangular_factor = _sketch_qr(operator, sketch, damp)
    # x̂ = R⁻¹ Q_1:dᵀ S b
    solution = _warm_start(orthogonal_top, triangular_factor, sketch, vector)
    if damp != 0.0:
        return solution
    return _check_full_rank(triangular_factor, solution)


class SketchAndPrecondLSMR(AbstractSolverStrategy):
    """Sketch-and-precondition LSMR for tall least squares (Rokhlin & Tygert, 2008).

        x = argmin_x ‖A x − b‖₂² + β ‖x‖₂²,   A ∈ ℝ^(M × N),   M ≫ N

    that is, plain least squares on the augmented system A_aug = [A; √β I],
    b_aug = [b; 0] (A_aug = A when β = 0). A sketch S ∈ ℝ^(d × M),
    d = ⌈γ N⌉, gives the QR factorisation [S A; √β I] = Q R, and R⁻¹ is a
    right preconditioner: if S is an ε-embedding for range(A), then
    ‖[S A; √β I] R⁻¹ y‖₂ = ‖y‖₂ and

        ‖A_aug R⁻¹ y‖₂ ∈ [1 / (1 + ε), 1 / (1 − ε)] ‖y‖₂
            ⟹   κ(A_aug R⁻¹) ≤ (1 + ε) / (1 − ε),

    so LSMR on A_aug R⁻¹ contracts the error by about (κ − 1) / (κ + 1) per
    step, independently of M and of κ(A). The default sampling factor
    γ = 4 gives ε ≈ √(1 / γ) = 0.5 for a Gaussian sketch (and comparable
    for the sparse sign sketch), so κ ≲ 3 and LSMR converges in a few tens
    of steps (Meng, Saunders & Mahoney, 2014; Epperly, 2024).

    The inner solve is `lineax.LSMR` on the composed operator A_aug R⁻¹
    (the undamped path of `gaussx.LSMRSolver`), so gradients use lineax's
    implicit differentiation. Only A is sketched; the ridge block is exact.

    With ``damp == 0``, A must have full column rank (otherwise the solution
    is not unique and R is singular; ``throw=True`` raises); ``damp > 0``
    always has a unique solution. A matrix-free A is sketched with d
    transpose-matvecs, a batch of rows of S at a time, so the extra memory
    is O(M) per batch, not O(d M).

    **Scope.** The dense QR of the d × N sketch costs O(d N²) and O(d N)
    memory, so this targets N ≲ 10⁴ (e.g. Gauss-Newton steps on a Jacobian
    with 10⁶ pixels and a few hundred parameters). For larger N, solve the
    normal equations (Aᵀ A + β I) x = Aᵀ b with
    `gaussx.PreconditionedCGSolver` and a `gaussx.NystromPreconditioner`
    built from Aᵀ A with ``shift=β``.

    Attributes:
        sketch: ``"sparse_sign"`` (default; `gaussx.SparseSignSketch`),
            ``"srht"`` (`gaussx.SRHTSketch`) or ``"gaussian"``
            (`gaussx.GaussianSketch`).
        sampling_factor: Sampling factor γ ≥ 1; the sketch has
            d = min(⌈γ N⌉, M) rows.
        nnz: Non-zeros per column of the sparse sign sketch.
        damp: Tikhonov damping √β ≥ 0.
        atol: LSMR tolerance on the normal-equations residual. ``None``:
            1e-6 in float64, 1e-3 in float32, as `gaussx.LSMRSolver`
            (gh-327). The preconditioned system has κ ≲ 3, so 1e-6 is
            reachable in float32 too and costs only a few more steps; set it
            explicitly when float32 accuracy matters.
        btol: LSMR relative tolerance on the residual. ``None``: as
            ``atol``.
        max_steps: Maximum number of LSMR iterations.
        seed: The sketch S is drawn from ``jax.random.PRNGKey(seed)``; it is
            also the probe seed of `logdet`.
        throw: Raise when LSMR stops without converging, as
            `gaussx.LSMRSolver`.

    Raises:
        ValueError: If ``sketch`` is unknown, ``sampling_factor < 1`` or
            ``damp < 0``.

    Examples:
        A tall least-squares problem matches the dense solution:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jr.normal(jr.key(0), (2000, 10))
        >>> b = jr.normal(jr.key(1), (2000,))
        >>> solver = gaussx.SketchAndPrecondLSMR(atol=1e-6, btol=1e-6)
        >>> x = solver.solve(lx.MatrixLinearOperator(matrix), b)
        >>> bool(jnp.allclose(x, jnp.linalg.lstsq(matrix, b)[0], atol=1e-4))
        True

    References:
        Rokhlin, V. & Tygert, M. (2008). A fast randomized algorithm for
        overdetermined linear least-squares regression. *PNAS*, 105(36),
        13212-13217.

        Meng, X., Saunders, M. A. & Mahoney, M. W. (2014). LSRN: A parallel
        iterative solver for strongly over- or underdetermined systems.
        *SIAM J. Sci. Comput.*, 36(2), C95-C118.

        Epperly, E. N. (2024). Fast and forward stable randomized algorithms
        for linear least-squares problems. *SIAM J. Matrix Anal. Appl.*,
        45(4), 1782-1804.
    """

    sketch: SketchName = eqx.field(static=True, default="sparse_sign")
    sampling_factor: float = eqx.field(static=True, default=4.0)
    nnz: int = eqx.field(static=True, default=8)
    damp: float = eqx.field(static=True, default=0.0)
    atol: float | None = eqx.field(static=True, default=None)
    btol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=100)
    seed: int = eqx.field(static=True, default=0)
    throw: bool = eqx.field(static=True, default=True)

    def __check_init__(self) -> None:
        if self.sketch not in ("sparse_sign", "srht", "gaussian"):
            raise ValueError(
                "SketchAndPrecondLSMR.sketch must be 'sparse_sign', 'srht' or "
                f"'gaussian', got {self.sketch!r}."
            )
        if self.sampling_factor < 1:
            raise ValueError(
                "SketchAndPrecondLSMR needs sampling_factor >= 1, got "
                f"{self.sampling_factor}."
            )
        if self.damp < 0:
            raise ValueError(f"SketchAndPrecondLSMR needs damp >= 0, got {self.damp}.")

    def _sample_sketch(self, m: int, n: int) -> AbstractSketch:
        """Draw the sketch S with d = min(⌈γ N⌉, M) rows from ``seed``.

        Args:
            m: Number of rows M of A.
            n: Number of columns N of A.

        Returns:
            The sampled sketch S, d × M.
        """
        # d = min(⌈γ N⌉, M)
        sketch_size = min(math.ceil(self.sampling_factor * n), m)
        key = jax.random.PRNGKey(self.seed)
        if self.sketch == "sparse_sign":
            return SparseSignSketch.sample(key, sketch_size, m, nnz=self.nnz)
        if self.sketch == "srht":
            return SRHTSketch.sample(key, sketch_size, m)
        return GaussianSketch.sample(key, sketch_size, m)

    def _solve(
        self, operator: lx.AbstractLinearOperator, vector: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], lx.Solution]:
        """The solution and lineax's inner LSMR solution, with its stats.

            x = x₀ + R⁻¹ y,   y = argmin_y ‖A_aug R⁻¹ y − (b_aug − A_aug x₀)‖₂

        Algorithm:
            1. Sample S and factor [S A; √β I] = Q R.
            2. Warm-start with x₀ = R⁻¹ Q_1:dᵀ S b, checking R when β = 0.
            3. Run undamped `lineax.LSMR` on A_aug R⁻¹ against the residual
               b_aug − A_aug x₀.
            4. Map back, x = x₀ + R⁻¹ y.

        Args:
            operator: Operator A, M × N.
            vector: Right-hand side b, of length M.

        Returns:
            Tuple ``(solution, inner)``: x, and the `lineax.Solution` of
            the preconditioned LSMR solve.

        Raises:
            ValueError: If ``damp == 0`` and M < N.
        """
        m, n = operator.out_size(), operator.in_size()
        if self.damp == 0.0 and m < n:
            raise ValueError(
                "SketchAndPrecondLSMR needs a tall or square operator when "
                f"damp == 0, got shape ({m}, {n})."
            )
        sketch = self._sample_sketch(m, n)
        # [S A; √β I] = Q R
        orthogonal_top, triangular_factor = _sketch_qr(operator, sketch, self.damp)
        # x₀ = R⁻¹ Q_1:dᵀ S b
        initial_solution = _warm_start(
            orthogonal_top, triangular_factor, sketch, vector
        )
        if self.damp == 0.0 and self.throw:
            initial_solution = _check_full_rank(triangular_factor, initial_solution)
        damp = self.damp

        def precondition(y: Float[Array, " n"]) -> Float[Array, " n"]:
            # R⁻¹ y
            return jsl.solve_triangular(triangular_factor, y, lower=False)

        def augmented(x: Float[Array, " n"]) -> Float[Array, " mn"]:
            # A x
            operator_x = operator.mv(x)
            if damp == 0.0:
                return operator_x
            # A_aug x = [A x; √β x]
            return jnp.concatenate([operator_x, damp * x])

        dtype = operator_dtype(operator, vector)
        # A_aug R⁻¹
        preconditioned = lx.FunctionLinearOperator(
            lambda y: augmented(precondition(y)),
            jax.ShapeDtypeStruct((n,), triangular_factor.dtype),
        )
        # b_aug = [b; 0]
        augmented_rhs = (
            vector
            if damp == 0.0
            else jnp.concatenate([vector, jnp.zeros_like(initial_solution)])
        )
        # b_aug − A_aug x₀
        residual = augmented_rhs - augmented(initial_solution)
        atol = resolve_tolerance(self.atol, dtype, 1e-6)
        btol = resolve_tolerance(self.btol, dtype, 1e-6)
        # y = argmin_y ‖A_aug R⁻¹ y − (b_aug − A_aug x₀)‖₂
        inner = lx.linear_solve(
            preconditioned,
            residual,
            lx.LSMR(rtol=btol, atol=atol, max_steps=self.max_steps),
            throw=self.throw,
        )
        # x = x₀ + R⁻¹ y
        return initial_solution + precondition(inner.value), inner

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " m"],
    ) -> Float[Array, " n"]:
        """Solve the (damped) least-squares problem by sketch-preconditioned LSMR.

            x = argmin_x ‖A x − b‖₂² + β ‖x‖₂²,   β = damp²

        Algorithm:
            1. Sample S and factor [S A; √β I] = Q R.
            2. Warm-start with x₀ = R⁻¹ Q_1:dᵀ S b.
            3. Run LSMR on A_aug R⁻¹ and return x = x₀ + R⁻¹ y.

        Args:
            operator: Operator A, M × N; tall (M ≥ N) unless ``damp > 0``.
                Matrix-free operators are sketched with d transpose-matvecs.
            vector: Right-hand side b, of length M.

        Returns:
            The solution x, of length N.

        Raises:
            ValueError: If ``damp == 0`` and M < N.
            equinox.EquinoxRuntimeError: With ``throw=True``, if
                ``damp == 0`` and A has numerically dependent columns (the
                solution is not unique; pass ``damp > 0``), or LSMR does not
                converge.
        """
        return self._solve(operator, vector)[0]

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Stochastic log-determinant, delegated to `gaussx.LSMRSolver`'s SLQ.

            log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        Present for the `AbstractSolverStrategy` protocol only: the sketch
        plays no part. For the log-determinant of a square system, use a
        square-system strategy (`gaussx.SLQLogdet`, `gaussx.CGSolver`,
        `gaussx.DenseSolver`).

        Args:
            operator: Square symmetric PSD operator A, N × N.
            key: PRNG key for the probes z^(i); ``None`` means
                ``PRNGKey(seed)``.

        Returns:
            Scalar estimate of log |det A|.
        """
        return LSMRSolver(seed=self.seed).logdet(operator, key=key)
