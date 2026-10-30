"""Capacitance-matrix solver (Sherman-Morrison-Woodbury) for masked domains.

The capacitance-matrix method extends a fast base solve x ↦ A₀⁻¹ x (an
FFT, DST or DCT Helmholtz solve on a rectangle, say) to the same system
with N_b extra point constraints x_bₖ = 0 at the flat indices b_1, …, b_N_b
(the irregular boundary of a masked sub-domain). It is a low-rank Woodbury
correction around the base solve, through the N_b × N_b capacitance matrix
C with C_kl = [A₀⁻¹ e_bₗ]_bₖ, where e_bₗ is the bₗ-th unit vector.

This module owns only the generic linear algebra. The caller supplies:

* ``base_solve``, a callable applying A₀⁻¹ to a flat right-hand side;
* ``boundary_indices``, the flat indices b_1, …, b_N_b of the constrained
  degrees of freedom.

Which degrees of freedom are constrained (mask and boundary extraction) and
which base solver to use (the spectral transform) stay in the calling
package (finitevolX, spectraldiffx).

If A₀ is singular (a periodic FFT or Neumann DCT Poisson solve whose
``base_solve`` is the pseudo-inverse A₀⁺ with the constant mode projected
out), the plain method solves the wrong PDE: nothing fixes the null-mode
component, and the interior residual is a uniform constant. Passing
``null_vector`` borders the capacitance system with that null vector and a
solvability constraint (see `CapacitanceSolver`).

Reference: Buzbee, Golub & Nielson (1970), "On Direct Methods for Solving
Poisson's Equations", SIAM J. Numer. Anal. The null-vector augmentation is the
standard bordered-system treatment of a singular base operator.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float, Int


_BUILD_BATCH = 32


class CapacitanceSolver(eqx.Module):
    """Solve a base system subject to homogeneous point constraints.

        x = u − A₀⁻¹ ∑_k α_k e_bₖ,   u = A₀⁻¹ f,   α = C⁻¹ u_b

    Given a fast base solve x ↦ A₀⁻¹ x and N_b constrained indices
    b_1, …, b_N_b, this returns the x with x_bₖ = 0 for every k and
    (A₀ x)_i = f_i at every unconstrained index i. The capacitance matrix
    C ∈ ℝ^{N_b × N_b}, C_kl = [A₀⁻¹ e_bₗ]_bₖ, is LU-factorised once at
    construction, and u_b = [u_b₁; …; u_b_N_b] samples u at the constrained
    indices.

    Algorithm:
        1. Base solve: u = A₀⁻¹ f.
        2. Sample the constrained indices: u_b.
        3. Correction weights: α = C⁻¹ u_b.
        4. Subtract the point-source response: x = u − A₀⁻¹ ∑_k α_k e_bₖ.

    Cost and storage: only the factorised C is stored, O(N_b²) memory. The
    Green's functions A₀⁻¹ e_bₖ are never kept; step 4 is a second base
    solve with the point sources scattered into a zero field. Each call
    therefore costs two base solves plus an O(N_b²) back-substitution.
    Construction costs N_b base solves (one column at a time under ``jit``,
    so no dense N_b × N array is formed).

    **Singular base operators.** If A₀ is singular with right null vector r
    (the constant for a periodic or Neumann Poisson operator) and
    ``base_solve`` is the pseudo-inverse A₀⁺ that projects out that mode,
    the plain method leaves the null-mode component undetermined and the
    interior residual is a nonzero constant. Pass ``null_vector=r`` (and,
    for a non-symmetric A₀, ``left_null_vector=w`` with wᵀ A₀ = 0; it
    defaults to r). The solution is then x = u − A₀⁺ ∑_k α_k e_bₖ + β r,
    with (α, β) from the bordered system

        [ C      −r_b ] [α]   [ u_b  ]
        [ w_bᵀ    0   ] [β] = [ wᵀ f ]

    whose second row makes the effective source f − ∑_k α_k e_bₖ orthogonal
    to the left null space, so that A₀ x = f holds exactly away from the
    constrained indices. Without ``null_vector`` a singular base cannot be
    detected here, so callers must pass it.

    The solver works on flat vectors of length N. Any reshaping between
    fields and flat vectors, and any masking of the exterior, is the
    caller's job, which keeps grid and mask concepts out of this class.

    Args:
        base_solve: Callable applying the base inverse A₀⁻¹ (or the
            pseudo-inverse A₀⁺ for a singular A₀) to a flat right-hand side
            of length N. Stored as a static field, so it must be hashable
            (functions and closures are).
        boundary_indices: Flat constrained indices b_1, …, b_N_b.
        n: Length N of the flat solution vector.
        null_vector: Right null vector r of a singular A₀, length N;
            ``None`` (default) for a non-singular base.
        left_null_vector: Left null vector w of a singular A₀; defaults to
            ``null_vector`` (correct for a symmetric A₀). Ignored when
            ``null_vector`` is ``None``.
        keep_base_solve: When ``False``, ``base_solve`` is only used to
            build C and is not stored; pass it to `__call__` instead. A
            static closure over an operator hides the operator's arrays from
            autodiff and makes every instance a new ``jit`` cache key, so
            `gaussx.MaskedOperator` keeps its base as a pytree leaf and uses
            this.

    Attributes:
        base_solve: The base inverse callable (static), or ``None`` when
            built with ``keep_base_solve=False``.
        boundary_indices: The constrained indices b_1, …, b_N_b.
        capacitance_lu: LU factorisation ``(lu, pivots)`` of C, or of the
            bordered (N_b + 1) × (N_b + 1) system when a null vector is
            given.
        null_vector: r, or ``None``.
        left_null_vector: w, or ``None``.
        transposed: Whether this solver is the `transpose` of a built one
            (it then solves with A₀ᵀ through the transposed factorisation).

    Examples:
        Pin two entries of a 4 × 4 SPD solve to zero:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> A0 = 3.0 * jnp.eye(4) + jnp.ones((4, 4))
        >>> solver = gaussx.CapacitanceSolver(
        ...     lambda f: jnp.linalg.solve(A0, f), jnp.array([0, 3]), 4
        ... )
        >>> f = jnp.ones(4)
        >>> x = solver(f)
        >>> bool(jnp.allclose(x[jnp.array([0, 3])], 0.0, atol=1e-6))
        True

        Away from the constraints, A₀ x = f:

        >>> residual = A0 @ x - f
        >>> bool(jnp.allclose(residual[1:3], 0.0, atol=1e-5))
        True
    """

    base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]] | None = eqx.field(
        static=True
    )
    boundary_indices: Int[Array, " Nb"]
    capacitance_lu: tuple[Float[Array, "m m"], Int[Array, " m"]]
    null_vector: Float[Array, " n"] | None
    left_null_vector: Float[Array, " n"] | None
    transposed: bool = eqx.field(static=True)

    def __init__(
        self,
        base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]],
        boundary_indices: Int[Array, " Nb"],
        n: int,
        *,
        null_vector: Float[Array, " n"] | None = None,
        left_null_vector: Float[Array, " n"] | None = None,
        keep_base_solve: bool = True,
    ):
        indices = jnp.asarray(boundary_indices)
        # C_kl = [A₀⁻¹ e_bₗ]_bₖ
        capacitance = _capacitance_matrix(base_solve, indices, n)

        if null_vector is None:
            system = capacitance
            left_null_vector = None
        else:
            null_vector = jnp.asarray(null_vector)
            left_null_vector = (
                null_vector
                if left_null_vector is None
                else jnp.asarray(left_null_vector)
            )
            # r_b, r at the constrained indices
            boundary_null = null_vector[indices]
            # w_b, w at the constrained indices
            boundary_left_null = left_null_vector[indices]
            # [[C, −r_b], [w_bᵀ, 0]]
            # jnp, not einx: einx rejects the empty boundary axis when N_b = 0
            system = jnp.block(
                [
                    [capacitance, -boundary_null[:, None]],
                    [
                        boundary_left_null[None, :],
                        jnp.zeros((1, 1), dtype=capacitance.dtype),
                    ],
                ]
            )

        self.base_solve = base_solve if keep_base_solve else None
        self.boundary_indices = indices
        self.capacitance_lu = jsl.lu_factor(system)
        self.null_vector = null_vector
        self.left_null_vector = left_null_vector
        self.transposed = False

    def __call__(
        self,
        rhs: Float[Array, " n"],
        *,
        base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]] | None = None,
    ) -> Float[Array, " n"]:
        """Solve the constrained system for a flat right-hand side f.

            x = u − A₀⁻¹ ∑_k α_k e_bₖ (+ β r),   u = A₀⁻¹ f

        Algorithm:
            1. Base solve u = A₀⁻¹ f and sample u_b.
            2. Solve C α = u_b, or the bordered system for (α, β) when a
               null vector is set.
            3. Scatter the point sources ∑_k α_k e_bₖ into a zero field and
               subtract their base solve from u.
            4. With a null vector, add the null-mode component β r.

        Args:
            rhs: Flat right-hand side f, length N.
            base_solve: The base inverse to use; overrides the stored one
                and is required when the solver was built with
                ``keep_base_solve=False``. For a solver returned by
                `transpose` it must apply A₀⁻ᵀ.

        Returns:
            The constrained solution x, length N.

        Raises:
            ValueError: If no base solve is stored or passed.
        """
        base_solve = self.base_solve if base_solve is None else base_solve
        if base_solve is None:
            raise ValueError(
                "This CapacitanceSolver stores no base_solve; pass base_solve=."
            )
        # A transposed solver solves with the transposed factorisation. The
        # bordered system for A₀ᵀ is D Sᵀ D with D = diag(1, …, 1, −1).
        trans = 1 if self.transposed else 0
        # u = A₀⁻¹ f
        base_solution = base_solve(rhs)
        # u_b, u at the constrained indices
        boundary_values = base_solution[self.boundary_indices]
        if self.null_vector is None:
            # α = C⁻¹ u_b
            source_weights = jsl.lu_solve(
                self.capacitance_lu, boundary_values, trans=trans
            )
            null_weight = None
        else:
            assert self.left_null_vector is not None
            # wᵀ f
            load = jnp.dot(self.left_null_vector, rhs)
            sign = -1.0 if self.transposed else 1.0
            # [u_b; ±wᵀ f]
            bordered_rhs = jnp.append(boundary_values, sign * load)
            # (α, ±β) from the bordered system
            bordered_solution = jsl.lu_solve(
                self.capacitance_lu, bordered_rhs, trans=trans
            )
            source_weights = bordered_solution[:-1]
            null_weight = sign * bordered_solution[-1]
        # ∑_k α_k e_bₖ. Constrained indices are distinct; saying so keeps the
        # scatter transposable (reverse mode through a solve).
        sources = (
            jnp.zeros_like(base_solution)
            .at[self.boundary_indices]
            .set(source_weights, unique_indices=True)
        )
        # A₀⁻¹ ∑_k α_k e_bₖ
        correction = base_solve(sources)
        # x = u − A₀⁻¹ ∑_k α_k e_bₖ
        solution = base_solution - correction
        if null_weight is not None:
            assert self.null_vector is not None
            # β r
            null_component = null_weight * self.null_vector
            # x + β r
            solution = solution + null_component
        return solution

    def transpose(self) -> CapacitanceSolver:
        """The capacitance solver for A₀ᵀ, reusing this factorisation.

            C(A₀ᵀ) = C(A₀)ᵀ

        Cᵀ is the capacitance matrix of A₀ᵀ on the same indices, and A₀ᵀ's
        right and left null vectors are A₀'s left and right ones, so no base
        solve is needed. The result stores no ``base_solve``: pass A₀ᵀ's
        inverse to `__call__`.

        Returns:
            The transposed solver.
        """
        return _replace_fields(
            self,
            base_solve=None,
            null_vector=self.left_null_vector,
            left_null_vector=self.null_vector,
            transposed=not self.transposed,
        )


def _replace_fields(module: eqx.Module, **changes):
    """Copy an `equinox.Module`, replacing fields, without calling ``__init__``.

    ``dataclasses.replace`` would re-run the custom ``__init__``; this is the
    field-by-field construction equinox itself uses when unflattening.

    Args:
        module: The module to copy.
        **changes: New values for the named fields.

    Returns:
        A module of the same type with ``changes`` applied.
    """
    new = object.__new__(type(module))
    for field in dataclasses.fields(module):
        value = changes.get(field.name, getattr(module, field.name))
        object.__setattr__(new, field.name, value)
    return new


def _capacitance_matrix(
    base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]],
    indices: Int[Array, " Nb"],
    n: int,
) -> Float[Array, "Nb Nb"]:
    """Build the capacitance matrix one column at a time.

        C_kl = [A₀⁻¹ e_bₗ]_bₖ

    Each unit source e_bₗ is created inside the mapped function and only its
    N_b constrained samples are kept. Columns are processed in batches of
    ``_BUILD_BATCH`` (vectorised within a batch), so peak memory is
    O(``_BUILD_BATCH`` N + N_b²) rather than the O(N_b N) of a ``vmap``
    over all dense unit sources.

    Args:
        base_solve: Callable applying A₀⁻¹ to a flat vector.
        indices: Constrained indices b_1, …, b_N_b.
        n: Length N of the flat vectors.

    Returns:
        The N_b × N_b capacitance matrix C.
    """

    @jax.jit
    def build(idx: Int[Array, " Nb"]) -> Float[Array, "Nb Nb"]:
        def column(b_l: Int[Array, ""]) -> Float[Array, " Nb"]:
            # e_bₗ
            source = jnp.zeros(n).at[b_l].set(1.0)
            # [A₀⁻¹ e_bₗ]_bₖ for every k, column l of C
            return base_solve(source)[idx]

        # row l of the map is column l of C
        columns = jax.lax.map(column, idx, batch_size=_BUILD_BATCH)
        # C_kl, transposed back
        # jnp, not einx: einx rejects the empty boundary axis when N_b = 0
        return columns.T

    return build(indices)
