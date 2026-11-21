"""Multigrid: V-cycles over a level hierarchy the caller builds.

A_ℓ is the operator on level ℓ = 0 (finest) … L − 1 (coarsest), P_ℓ the
prolongation from level ℓ + 1 to level ℓ, R_ℓ the restriction from ℓ to
ℓ + 1, and S_ℓ a smoother. One V-cycle u = V_ℓ(b), from u = 0, is

    u ← S_ℓ^{ν₁}(u, b);  r ← R_ℓ (b − A_ℓ u);  u ← u + P_ℓ V_ℓ₊₁(r);
    u ← S_ℓ^{ν₂}(u, b),

with an exact solve on the coarsest level. gaussx owns the algorithm; the
caller owns the levels (a grid library coarsens its mask and builds the
transfer stencils; `galerkin_levels` builds them from dense prolongations).

`MultigridPreconditioner` is one V-cycle as M⁻¹; `MultigridSolver` is CG
preconditioned by it (the robust form: CG fixes what the V-cycle misses),
and `MultigridSolver.cycles` runs plain V-cycles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._diag import matrix_free_diag
from gaussx._strategies._base import AbstractSolveStrategy, _SolveStats
from gaussx._strategies._cg import CGSolver


# Power iterations behind `WeightedJacobi.from_operator`'s spectral estimate,
# and the safety factor on it (an underestimate of ρ would make ω too large).
_POWER_STEPS = 50
_RHO_SAFETY = 1.1


class AbstractSmoother(eqx.Module):
    """A smoother u ← S(u, b) for A u = b on one level."""

    def __call__(
        self,
        operator: lx.AbstractLinearOperator,
        u: Float[Array, " n"],
        b: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Apply the smoother's sweeps.

        Args:
            operator: The level's operator A.
            u: The current iterate.
            b: The right-hand side.

        Returns:
            The smoothed iterate.
        """
        raise NotImplementedError


class WeightedJacobi(AbstractSmoother):
    """Weighted Jacobi: u ← u + ω D⁻¹ (b − A u), ``sweeps`` times.

        D = diag(A)

    Symmetric for a symmetric A, so a V-cycle with equal pre- and
    post-smoothing is a symmetric preconditioner.

    Attributes:
        inv_diagonal: D⁻¹ (0 where a row is not an unknown).
        omega: The relaxation weight ω (a leaf: `from_operator` estimates it
            from the operator's values).
        sweeps: Sweeps per application.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, -1.0], [-1.0, 2.0]]))
        >>> smoother = gaussx.WeightedJacobi(jnp.array([0.5, 0.5]), omega=1.0)
        >>> [round(float(v), 3) for v in smoother(A, jnp.zeros(2), jnp.ones(2))]
        [0.5, 0.5]
    """

    inv_diagonal: Float[Array, " n"]
    omega: float | Float[Array, ""] = 2.0 / 3.0
    sweeps: int = eqx.field(static=True, default=1)

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        *,
        omega: float | None = None,
        sweeps: int = 1,
    ) -> WeightedJacobi:
        """Build the smoother from A's diagonal, estimating ω when not given.

            ω = 4 / (3 ρ̂),  ρ̂ ≈ 1.1 λ_max(D⁻¹ A)   (0.61 for the 1-D Laplacian)

        Args:
            operator: The level's operator A.
            omega: The relaxation weight; ``None`` estimates it from 50 power
                iterations of D⁻¹ A.
            sweeps: Sweeps per application.

        Returns:
            The smoother.
        """
        diagonal = matrix_free_diag(operator, estimate=False)
        inv_diagonal = jnp.where(diagonal != 0, 1.0 / diagonal, 0.0)
        if omega is None:
            # ω changes how fast the V-cycle converges, not what it converges to
            rho = jax.lax.stop_gradient(_spectral_radius(operator, inv_diagonal))
            return cls(
                inv_diagonal, omega=4.0 / (3.0 * _RHO_SAFETY * rho), sweeps=sweeps
            )
        return cls(inv_diagonal, omega=omega, sweeps=sweeps)

    def __call__(
        self,
        operator: lx.AbstractLinearOperator,
        u: Float[Array, " n"],
        b: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Apply ``sweeps`` weighted-Jacobi sweeps.

        Args:
            operator: The level's operator A.
            u: The current iterate.
            b: The right-hand side.

        Returns:
            The smoothed iterate.
        """
        inv_diagonal = self.inv_diagonal.astype(u.dtype)
        omega = jnp.asarray(self.omega, dtype=u.dtype)

        def sweep(_: int, current: Float[Array, " n"]) -> Float[Array, " n"]:
            # u ← u + ω D⁻¹ (b − A u)
            return current + omega * inv_diagonal * (b - operator.mv(current))

        return jax.lax.fori_loop(0, self.sweeps, sweep, u)


def _spectral_radius(
    operator: lx.AbstractLinearOperator, inv_diagonal: Float[Array, " n"]
) -> Float[Array, ""]:
    """λ_max(D⁻¹ A) by power iteration from the all-ones vector."""

    def step(_: int, v: Float[Array, " n"]) -> Float[Array, " n"]:
        w = inv_diagonal * operator.mv(v)
        return w / jnp.linalg.norm(w)

    # a deterministic start with every frequency in it (the all-ones vector
    # misses the checkerboard mode, the top eigenvector of a Laplacian)
    v0 = jnp.sin(
        2.399963 * jnp.arange(1, inv_diagonal.shape[0] + 1, dtype=inv_diagonal.dtype)
    )
    v0 = v0 / jnp.linalg.norm(v0)
    v = jax.lax.fori_loop(0, _POWER_STEPS, step, v0)
    return jnp.abs(v @ (inv_diagonal * operator.mv(v)))


class MultigridLevel(eqx.Module):
    """One level of a hierarchy: A_ℓ, its transfers and its smoother.

    Attributes:
        operator: A_ℓ, N_ℓ × N_ℓ.
        prolongation: P_ℓ, N_ℓ × N_ℓ₊₁ (from the next coarser level);
            ``None`` on the coarsest level.
        restriction: R_ℓ, N_ℓ₊₁ × N_ℓ; ``None`` means Pᵀ.
        smoother: The smoother on this level (unused on the coarsest).
    """

    operator: lx.AbstractLinearOperator
    prolongation: lx.AbstractLinearOperator | None
    restriction: lx.AbstractLinearOperator | None
    smoother: AbstractSmoother | None


def galerkin_levels(
    operator: lx.AbstractLinearOperator,
    prolongations: Sequence[lx.AbstractLinearOperator],
    *,
    restriction_scale: float = 1.0,
    smoother: Callable[[lx.AbstractLinearOperator], AbstractSmoother] | None = None,
) -> tuple[MultigridLevel, ...]:
    """A hierarchy with Galerkin coarse operators A_ℓ₊₁ = R_ℓ A_ℓ P_ℓ.

        R_ℓ = c P_ℓᵀ,   A_ℓ₊₁ = c P_ℓᵀ A_ℓ P_ℓ

    The coarse operators are materialised once, as dense matrices: this
    builds a hierarchy offline from small or moderate prolongations. A grid
    library with stencils builds its own `MultigridLevel` s instead.

    Args:
        operator: The finest operator A₀.
        prolongations: P₀, P₁, …, finest first.
        restriction_scale: c, e.g. 1/4 for full weighting in 2-D.
        smoother: Builds each level's smoother from its operator; defaults
            to `WeightedJacobi.from_operator`.

    Returns:
        The levels, finest first, the last one without transfers.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> n = 7
        >>> L = 2 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)
        >>> A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)
        >>> # linear interpolation from 3 to 7 points
        >>> P = jnp.array([[0.5, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 1, 0],
        ...                [0, 0.5, 0.5], [0, 0, 1], [0, 0, 0.5]])
        >>> transfer = lx.MatrixLinearOperator(P)
        >>> levels = gaussx.galerkin_levels(A, [transfer])
        >>> [level.operator.in_size() for level in levels]
        [7, 3]
    """
    make_smoother = WeightedJacobi.from_operator if smoother is None else smoother
    levels = []
    current = operator
    for prolongation in prolongations:
        p = prolongation.as_matrix()
        # R = c Pᵀ, A_c = R A P
        restriction = lx.MatrixLinearOperator(
            restriction_scale * rearrange(p, "i c -> c i")
        )
        coarse = einsum(
            restriction.as_matrix(), current.as_matrix(), p, "c i, i j, j d -> c d"
        )
        levels.append(
            MultigridLevel(current, prolongation, restriction, make_smoother(current))
        )
        tags = lx.symmetric_tag if lx.is_symmetric(current) else ()
        current = lx.MatrixLinearOperator(coarse, tags)
    levels.append(MultigridLevel(current, None, None, None))
    return tuple(levels)


def _v_cycle(
    levels: Sequence[MultigridLevel],
    rhs: Float[Array, " n"],
    u: Float[Array, " n"],
    n_pre: int,
    n_post: int,
    bottom_solve: Callable[[Array], Array] | None,
) -> Float[Array, " n"]:
    """One V-cycle from the iterate u, recursing down the levels."""
    level = levels[0]
    if len(levels) == 1:
        if bottom_solve is not None:
            return bottom_solve(rhs)
        return jnp.linalg.solve(level.operator.as_matrix().astype(rhs.dtype), rhs)
    assert level.smoother is not None and level.prolongation is not None
    for _ in range(n_pre):
        u = level.smoother(level.operator, u, rhs)
    residual = rhs - level.operator.mv(u)
    restriction = level.restriction
    if restriction is None:
        restriction = level.prolongation.transpose()
    coarse_rhs = restriction.mv(residual)
    coarse = _v_cycle(
        levels[1:], coarse_rhs, jnp.zeros_like(coarse_rhs), n_pre, n_post, bottom_solve
    )
    u = u + level.prolongation.mv(coarse)
    for _ in range(n_post):
        u = level.smoother(level.operator, u, rhs)
    return u


def _project_out(
    u: Float[Array, " n"], null_space: Float[Array, "n k"] | None
) -> Float[Array, " n"]:
    """u − Z (Zᵀ Z)⁻¹ Zᵀ u: the component in the complement of span(Z)."""
    if null_space is None:
        return u
    z = null_space.astype(u.dtype)
    gram = einsum(z, z, "n i, n j -> i j")
    coefficients = jnp.linalg.solve(gram, einsum(z, u, "n i, n -> i"))
    return u - z @ coefficients


class MultigridPreconditioner(AbstractPreconditioner):
    """One V-cycle as the preconditioner M⁻¹ ≈ A⁻¹.

        z = M⁻¹ r = V₀(r)

    With equal pre- and post-smoothing, a symmetric smoother and R = c Pᵀ,
    the V-cycle is symmetric; for a negative-definite A it approximates the
    negative-definite A⁻¹, so `as_operator` returns −V₀, the PSD
    approximation of (−A)⁻¹ that CG on −A needs.

    Attributes:
        levels: The hierarchy, finest first.
        n_pre: Pre-smoothing applications per level.
        n_post: Post-smoothing applications per level.
        bottom_solve: The solve on the coarsest level; ``None`` is a dense
            solve of its operator.
        null_space: Z, a basis of A's null space (a Neumann problem), or
            ``None``; the cycle's output is projected onto its complement.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> n = 7
        >>> L = 2 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)
        >>> A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)
        >>> # linear interpolation from 3 to 7 points
        >>> P = jnp.array([[0.5, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 1, 0],
        ...                [0, 0.5, 0.5], [0, 0, 1], [0, 0, 0.5]])
        >>> transfer = lx.MatrixLinearOperator(P)
        >>> levels = gaussx.galerkin_levels(A, [transfer])
        >>> pre = gaussx.MultigridPreconditioner(levels)
        >>> pre(jnp.ones(n)).shape
        (7,)
    """

    levels: tuple[MultigridLevel, ...]
    n_pre: int = eqx.field(static=True, default=2)
    n_post: int = eqx.field(static=True, default=2)
    bottom_solve: Callable[[Array], Array] | None = None
    null_space: Float[Array, "n k"] | None = None

    def v_cycle(
        self, vector: Float[Array, " n"], initial: Float[Array, " n"] | None = None
    ) -> Float[Array, " n"]:
        """One V-cycle for A₀ u = b from ``initial`` (zero by default).

        Args:
            vector: The right-hand side b.
            initial: The starting iterate.

        Returns:
            The iterate after one V-cycle.
        """
        u0 = jnp.zeros_like(vector) if initial is None else initial
        u = _v_cycle(
            self.levels, vector, u0, self.n_pre, self.n_post, self.bottom_solve
        )
        return _project_out(u, self.null_space)

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return M⁻¹ = V₀ (or −V₀ for negative-definite levels) as a PSD operator.

        The sign is that of the finest level's operator A₀, which the
        V-cycle inverts, not of the system being solved: CG on A₀ and CG on
        −A₀ (as `gaussx.linear_solve` runs for a negative-definite A₀) both
        need the positive-definite approximation of (±A₀)⁻¹, which is −V₀
        when A₀ is negative definite.

        Args:
            operator: The system operator; unused, the sign comes from the
                levels.

        Returns:
            A positive-semidefinite operator applying one V-cycle.
        """
        finest = self.levels[0].operator
        negative = lx.is_negative_semidefinite(
            finest
        ) and not lx.is_positive_semidefinite(finest)
        sign = -1.0 if negative else 1.0
        structure = jax.ShapeDtypeStruct(
            (self.levels[0].operator.in_size(),),
            self.levels[0].operator.in_structure().dtype,
        )
        return lx.FunctionLinearOperator(
            lambda r: sign * self.v_cycle(r), structure, lx.positive_semidefinite_tag
        )


class MultigridSolver(AbstractSolveStrategy):
    """CG preconditioned by a V-cycle over the caller's levels.

        x = A⁻¹ b,   M⁻¹ = V₀ (one V-cycle)

    The solve is of the operator passed in (whose levels this hierarchy
    must approximate), so the answer and its gradients are those of any
    other strategy; the V-cycle only cuts the CG steps (from 175 to 5 on a
    64 × 64 masked Helmholtz problem, gh-856). `cycles` runs plain V-cycles.

    Attributes:
        levels: The hierarchy, finest first.
        n_pre: Pre-smoothing applications per level.
        n_post: Post-smoothing applications per level.
        bottom_solve: The coarsest-level solve; ``None`` is a dense solve.
        null_space: A basis of the null space of a singular A, or ``None``.
        rtol: CG's relative tolerance (``None``: per dtype).
        atol: CG's absolute tolerance (``None``: per dtype).
        max_steps: CG's step limit.
        throw: Raise when CG has not converged in ``max_steps``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> n = 7
        >>> L = 2 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)
        >>> A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)
        >>> # linear interpolation from 3 to 7 points
        >>> P = jnp.array([[0.5, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 1, 0],
        ...                [0, 0.5, 0.5], [0, 0, 1], [0, 0, 0.5]])
        >>> transfer = lx.MatrixLinearOperator(P)
        >>> solver = gaussx.MultigridSolver(gaussx.galerkin_levels(A, [transfer]))
        >>> x = solver.solve(A, jnp.ones(n))
        >>> round(float(jnp.max(jnp.abs(A.mv(x) - 1.0))), 4)
        0.0
    """

    levels: tuple[MultigridLevel, ...]
    n_pre: int = eqx.field(static=True, default=2)
    n_post: int = eqx.field(static=True, default=2)
    bottom_solve: Callable[[Array], Array] | None = None
    null_space: Float[Array, "n k"] | None = None
    rtol: float | None = eqx.field(static=True, default=None)
    atol: float | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=1000)
    throw: bool = eqx.field(static=True, default=True)

    def as_preconditioner(self) -> MultigridPreconditioner:
        """One V-cycle of this hierarchy as a preconditioner.

        Returns:
            The preconditioner.
        """
        return MultigridPreconditioner(
            self.levels,
            n_pre=self.n_pre,
            n_post=self.n_post,
            bottom_solve=self.bottom_solve,
            null_space=self.null_space,
        )

    def solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> Float[Array, " n"]:
        """Solve A x = b with V-cycle-preconditioned CG.

            x = A⁻¹ b

        Args:
            operator: The symmetric (positive or negative) definite A.
            vector: The right-hand side b.

        Returns:
            The solution x.

        Raises:
            equinox.EquinoxRuntimeError: If CG has not converged after
                ``max_steps`` steps and ``throw`` is set.
        """
        return self._solve_with_stats(operator, vector).value

    def _solve_with_stats(
        self,
        operator: lx.AbstractLinearOperator,
        vector: Float[Array, " n"],
    ) -> _SolveStats:
        """`solve`, with the step count and verdict of its CG.

        Args:
            operator: The symmetric (positive or negative) definite A.
            vector: Right-hand side b, of length N.

        Returns:
            The solution and how the solve ended.
        """
        cg = CGSolver(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            preconditioner=self.as_preconditioner(),
            throw=self.throw,
        )
        stats = cg._solve_with_stats(operator, vector)
        return stats._replace(value=_project_out(stats.value, self.null_space))

    def cycles(self, vector: Float[Array, " n"], n_cycles: int) -> Float[Array, " n"]:
        """``n_cycles`` plain V-cycles for A₀ u = b from zero.

        Args:
            vector: The right-hand side b.
            n_cycles: How many V-cycles (static).

        Returns:
            The iterate after ``n_cycles`` cycles.
        """
        pre = self.as_preconditioner()
        u = jnp.zeros_like(vector)
        for _ in range(n_cycles):
            u = pre.v_cycle(vector, u)
        return u
