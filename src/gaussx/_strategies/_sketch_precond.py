"""Sketch-and-precondition least squares (G15)."""

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
# peak extra memory is _ROW_BATCH * m (rows of S^T) rather than d * m.
_ROW_BATCH = 32


def _sketch(
    operator: lx.AbstractLinearOperator, sketch: AbstractSketch
) -> Float[Array, "d n"]:
    r"""$SA$, without materialising $S^\top$ for a matrix-free $A$.

    `AbstractSketch.sketch_operator` vmaps over all $d$ rows of $S$ at once,
    which forms the $(m, d)$ block $S^\top$: 4.8 GB in float32 at $m =
    10^6$, $d = 1200$. Row $i$ of $SA$ is $A^\top (S^\top e_i)$, so this
    maps over the rows of $S$ in batches of ``_ROW_BATCH``.
    """
    if operator.out_size() != sketch.in_size:
        raise ValueError(
            f"Cannot sketch an operator with {operator.out_size()} rows using a "
            f"sketch with in_size={sketch.in_size}."
        )
    if isinstance(operator, lx.MatrixLinearOperator):
        return sketch.apply(operator.matrix)
    transpose = operator.transpose()
    unit_rows = jnp.eye(sketch.out_size, dtype=operator.out_structure().dtype)
    return jax.lax.map(
        lambda e: transpose.mv(sketch.apply_transpose(e)),
        unit_rows,
        batch_size=min(_ROW_BATCH, sketch.out_size),
    )


def _check_full_rank(
    R: Float[Array, "n n"], x: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Raise (at run time) when the undamped triangular factor is singular."""
    magnitudes = jnp.abs(jnp.diagonal(R))
    threshold = R.shape[0] * jnp.finfo(R.dtype).eps * jnp.max(magnitudes)
    return eqx.error_if(
        x,
        jnp.min(magnitudes) <= threshold,
        "The sketch SA is numerically rank-deficient: A has (numerically) "
        "dependent columns, so the least-squares solution is not unique. "
        "Pass damp > 0 for the ridge solution.",
    )


def _sketch_qr(
    operator: lx.AbstractLinearOperator, sketch: AbstractSketch, damp: float
) -> tuple[Float[Array, "d n"], Float[Array, "n n"]]:
    r"""QR of the (augmented) sketch, $[SA;\ \delta I] = QR$.

    Returns the first $d$ rows of $Q$ (the block that meets $Sb$) and $R$.
    """
    SA = _sketch(operator, sketch)
    n = operator.in_size()
    stacked = SA
    if damp != 0.0:
        stacked = jnp.concatenate([SA, damp * jnp.eye(n, dtype=SA.dtype)])
    Q, R = jnp.linalg.qr(stacked)
    return Q[: sketch.out_size], R


def _warm_start(
    Q_top: Float[Array, "d n"],
    R: Float[Array, "n n"],
    sketch: AbstractSketch,
    vector: Float[Array, " m"],
) -> Float[Array, " n"]:
    r"""Sketch-and-solve, $x_0 = R^{-1} Q_{1:d}^\top S b$."""
    Sb = sketch.apply(vector)
    return jsl.solve_triangular(R, einsum(Q_top, Sb, "d n, d -> n"), lower=False)


def sketch_and_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " m"],
    *,
    sketch: AbstractSketch,
    damp: float = 0.0,
) -> Float[Array, " n"]:
    r"""Approximate least squares from a sketch.

    Solves the sketched problem $\min_x \|S(Ax - b)\|^2 + \delta^2\|x\|^2$
    exactly, through the QR factorisation
    $[SA;\ \delta I] = QR$ of the $d \times n$ sketch (stacked on
    $\delta I$ when ``damp`` $= \delta > 0$):

    $$
    \hat x = R^{-1} Q_{1:d}^\top S b .
    $$

    If $S$ is an $\varepsilon$-subspace embedding for
    $\operatorname{range}([A\ b])$, the residual is within a constant factor
    of the optimum $x^\star$ (Woodruff, 2014, §2):

    $$
    \|A\hat x - b\| \le \tfrac{1}{1-\varepsilon}\|S(A\hat x - b)\|
    \le \tfrac{1}{1-\varepsilon}\|S(Ax^\star - b)\|
    \le \frac{1+\varepsilon}{1-\varepsilon}\,\|Ax^\star - b\| .
    $$

    The solution itself is only $O(\varepsilon)$-accurate, so this is a
    function, not a solver strategy; `SketchAndPrecondLSMR` uses it as its
    warm start and then solves to tolerance.

    ```text
    SA = S A                        # d × n, matrix-free via sketch_operator
    Q, R = qr([SA; δ I])            # reduced QR, O(d n²)
    x̂ = R⁻¹ Q[:d]ᵀ (S b)
    ```

    Args:
        operator: Tall operator $A$, shape ``(m, n)``.
        vector: Right-hand side $b$, shape ``(m,)``.
        sketch: A sampled sketch with ``in_size == m`` and, for a unique
            solution when ``damp == 0``, ``out_size >= n``.
        damp: Ridge parameter $\delta \ge 0$.

    Returns:
        The sketch-and-solve estimate $\hat x$, shape ``(n,)``.

    Raises:
        ValueError: If ``damp < 0`` or the sketch's ``in_size`` is not $m$.
        equinox.EquinoxRuntimeError: If ``damp == 0`` and $SA$ is
            numerically rank-deficient ($A$ has dependent columns).

    References:
        Woodruff, D. P. (2014). Sketching as a tool for numerical linear
        algebra. *Foundations and Trends in Theoretical Computer Science*,
        10(1-2), 1-157.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (2000, 10))
        >>> b = A @ jnp.ones(10) + 0.1 * jr.normal(jr.key(1), (2000,))
        >>> S = gx.GaussianSketch.sample(jr.key(2), d=200, m=2000)
        >>> x = gx.sketch_and_solve(lx.MatrixLinearOperator(A), b, sketch=S)
        >>> bool(jnp.max(jnp.abs(x - 1.0)) < 0.05)
        True
    """
    if damp < 0:
        raise ValueError(f"sketch_and_solve needs damp >= 0, got {damp}.")
    Q_top, R = _sketch_qr(operator, sketch, damp)
    x = _warm_start(Q_top, R, sketch, vector)
    return x if damp != 0.0 else _check_full_rank(R, x)


class SketchAndPrecondLSMR(AbstractSolverStrategy):
    r"""Sketch-and-precondition LSMR for tall least squares (Rokhlin & Tygert, 2008).

    Solves

    $$
    \min_x \|Ax - b\|^2 + \delta^2 \|x\|^2, \qquad A \in \mathbb{R}^{m
    \times n},\ m \gg n,
    $$

    i.e. plain least squares on the augmented system $\tilde A = [A;\
    \delta I]$, $\tilde b = [b;\ 0]$ ($\tilde A = A$ when $\delta = 0$). A
    sketch $S \in \mathbb{R}^{d \times m}$, $d = \lceil \gamma n \rceil$,
    gives the QR factorisation $[SA;\ \delta I] = QR$, and $M = R^{-1}$ is a
    right preconditioner: if $S$ is an $\varepsilon$-embedding for
    $\operatorname{range}(A)$, then $\|[SA;\ \delta I]\,My\| = \|y\|$ and

    $$
    \|\tilde A M y\| \in \Big[\tfrac{1}{1+\varepsilon},
    \tfrac{1}{1-\varepsilon}\Big]\|y\|
    \quad\Longrightarrow\quad
    \kappa(\tilde A M) \le \frac{1+\varepsilon}{1-\varepsilon},
    $$

    so LSMR on $\tilde A M$ contracts the error by about
    $(\kappa - 1)/(\kappa + 1)$ per step, independently of $m$ and of
    $\kappa(A)$. The default $\gamma = 4$ gives $\varepsilon \approx
    \sqrt{1/\gamma} = 0.5$ for a Gaussian sketch (and comparable for the
    sparse sign sketch), so $\kappa \lesssim 3$ and LSMR converges in a few
    tens of steps (Meng, Saunders & Mahoney, 2014; Epperly, 2024).

    ```text
    S = sample(sketch, d = ceil(gamma n), m; PRNGKey(seed))
    Q, R = qr([S A; δ I])                    # O(d n²)
    x₀ = R⁻¹ Q[:d]ᵀ (S b)                    # sketch-and-solve warm start
    y = LSMR(Ã R⁻¹, [b; 0] − Ã x₀)           # undamped, lineax
    return x₀ + R⁻¹ y
    ```

    The inner solve is `lineax.LSMR` on the composed operator $\tilde A M$
    (the undamped path of `LSMRSolver`), so gradients use lineax's implicit
    differentiation. Only $A$ is sketched; the ridge block is exact.

    With ``damp == 0``, $A$ must have full column rank (otherwise the
    solution is not unique and $R$ is singular; ``throw=True`` raises);
    ``damp > 0`` always has a unique solution. A matrix-free $A$ is
    sketched with $d$ transpose-matvecs, a batch of rows of $S$ at a time,
    so the extra memory is $O(m)$ per batch, not $O(dm)$.

    **Scope.** The dense QR of the $d \times n$ sketch costs $O(d n^2)$ and
    $O(dn)$ memory, so this targets $n \lesssim 10^4$ (e.g. Gauss-Newton
    steps on a Jacobian with $10^6$ pixels and a few hundred parameters).
    For larger $n$, solve the normal equations $(A^\top A + \delta^2 I)x =
    A^\top b$ with `PreconditionedCGSolver` and a `NystromPreconditioner`
    built from $A^\top A$ with ``shift=δ²``.

    Attributes:
        sketch: ``"sparse_sign"`` (default; `SparseSignSketch`), ``"srht"``
            (`SRHTSketch`) or ``"gaussian"`` (`GaussianSketch`).
        sampling_factor: $\gamma \ge 1$; the sketch has $d = \min(\lceil
            \gamma n \rceil, m)$ rows.
        nnz: Non-zeros per column of the sparse sign sketch.
        damp: Ridge parameter $\delta \ge 0$.
        atol: LSMR tolerance on the normal-equations residual. ``None``:
            ``1e-6`` in float64, ``1e-3`` in float32, as `LSMRSolver`
            (gh-327). The preconditioned system has $\kappa \lesssim 3$, so
            ``1e-6`` is reachable in float32 too and costs only a few more
            steps; set it explicitly when float32 accuracy matters.
        btol: LSMR relative tolerance on the residual. ``None``: as
            ``atol``.
        max_steps: Maximum LSMR iterations.
        seed: The sketch is drawn from ``jax.random.PRNGKey(seed)``; it is
            also the probe seed of `logdet`.
        throw: Raise when LSMR stops without converging, as `LSMRSolver`.

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

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (2000, 10))
        >>> b = jr.normal(jr.key(1), (2000,))
        >>> solver = gx.SketchAndPrecondLSMR(atol=1e-6, btol=1e-6)
        >>> x = solver.solve(lx.MatrixLinearOperator(A), b)
        >>> bool(jnp.allclose(x, jnp.linalg.lstsq(A, b)[0], atol=1e-4))
        True
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
        d = min(math.ceil(self.sampling_factor * n), m)
        key = jax.random.PRNGKey(self.seed)
        if self.sketch == "sparse_sign":
            return SparseSignSketch.sample(key, d, m, nnz=self.nnz)
        if self.sketch == "srht":
            return SRHTSketch.sample(key, d, m)
        return GaussianSketch.sample(key, d, m)

    def _solve(
        self, operator: lx.AbstractLinearOperator, vector: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], lx.Solution]:
        """The solution and lineax's inner LSMR solution (with its stats)."""
        m, n = operator.out_size(), operator.in_size()
        if self.damp == 0.0 and m < n:
            raise ValueError(
                "SketchAndPrecondLSMR needs a tall or square operator when "
                f"damp == 0, got shape ({m}, {n})."
            )
        sketch = self._sample_sketch(m, n)
        Q_top, R = _sketch_qr(operator, sketch, self.damp)
        x0 = _warm_start(Q_top, R, sketch, vector)
        if self.damp == 0.0 and self.throw:
            x0 = _check_full_rank(R, x0)
        damp = self.damp

        def precondition(y: Float[Array, " n"]) -> Float[Array, " n"]:
            return jsl.solve_triangular(R, y, lower=False)

        def augmented(x: Float[Array, " n"]) -> Float[Array, " mn"]:
            Ax = operator.mv(x)
            if damp == 0.0:
                return Ax
            return jnp.concatenate([Ax, damp * x])

        dtype = operator_dtype(operator, vector)
        preconditioned = lx.FunctionLinearOperator(
            lambda y: augmented(precondition(y)),
            jax.ShapeDtypeStruct((n,), R.dtype),
        )
        residual = (
            vector if damp == 0.0 else jnp.concatenate([vector, jnp.zeros_like(x0)])
        ) - augmented(x0)
        atol = resolve_tolerance(self.atol, dtype, 1e-6)
        btol = resolve_tolerance(self.btol, dtype, 1e-6)
        solution = lx.linear_solve(
            preconditioned,
            residual,
            lx.LSMR(rtol=btol, atol=atol, max_steps=self.max_steps),
            throw=self.throw,
        )
        return x0 + precondition(solution.value), solution

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " m"],
    ) -> Float[Array, " n"]:
        r"""Least-squares solution $\arg\min_x \|Ax - b\|^2 + \delta^2\|x\|^2$.

        Args:
            operator: Operator $A$, shape ``(m, n)``; tall ($m \ge n$) unless
                ``damp > 0``. Matrix-free operators are sketched with $d$
                transpose-matvecs.
            vector: Right-hand side $b$, shape ``(m,)``.

        Returns:
            The solution $x$, shape ``(n,)``.

        Raises:
            ValueError: If ``damp == 0`` and $m < n$.
            equinox.EquinoxRuntimeError: With ``throw=True``, if ``damp == 0``
                and $A$ has numerically dependent columns (the solution is
                not unique; pass ``damp > 0``), or LSMR does not converge.
        """
        return self._solve(operator, vector)[0]

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        r"""Stochastic log-determinant, delegated to `LSMRSolver`'s SLQ path.

        Present for the `AbstractSolverStrategy` protocol only: the sketch
        plays no part. For the log-determinant of a square system, use a
        square-system strategy (`SLQLogdet`, `CGSolver`, `DenseSolver`).

        Args:
            operator: A square symmetric PSD linear operator.
            key: PRNG key for the probes; ``None`` means
                ``PRNGKey(seed)``.

        Returns:
            Scalar estimate of $\log|\det A|$.
        """
        return LSMRSolver(seed=self.seed).logdet(operator, key=key)
