"""Capacitance-matrix solver (Sherman-Morrison / Woodbury) for masked domains.

The capacitance-matrix method extends a fast *base* solver ``B^{-1}`` (for which
an efficient inverse exists -- e.g. an FFT/DST/DCT Helmholtz solve on a
rectangle) to a problem with a small number of additional point constraints
(e.g. enforcing ``u = 0`` on the irregular boundary of a masked sub-domain). It
is a low-rank (Woodbury) correction around the base solve.

This module owns only the *generic linear algebra*. The caller supplies:

* ``base_solve`` -- a callable applying ``B^{-1}`` to a flat right-hand side;
* ``boundary_indices`` -- the flat indices of the constrained degrees of freedom.

Which degrees of freedom are constrained (mask / boundary extraction) and which
base solver to use (the spectral transform) stay in the calling package
(finitevolX / spectraldiffx).

If the base operator is singular (e.g. a periodic FFT or Neumann DCT Poisson
solve whose ``base_solve`` is a pseudo-inverse with the constant mode
projected out), the plain method solves the wrong PDE: nothing fixes the
null-mode component, and the interior residual is a uniform constant. Passing
``null_vector`` augments the capacitance system with that null vector and a
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
    r"""Solve a base system subject to homogeneous point constraints.

    Given a fast base solver ``B^{-1}`` and a set of ``N_b`` constrained indices
    ``b_1, …, b_{N_b}``, this returns ``x`` with ``x[b_k] = 0`` and
    ``B x = f`` at every unconstrained index, via the capacitance-matrix
    correction:

    1. Base solve:        ``u = B^{-1} f``
    2. Sample boundary:   ``u_b = u[boundary]``
    3. Correction:        ``α = C^{-1} u_b``
    4. Subtract:          ``x = u − B^{-1}(Σ_k α_k e_{b_k})``

    where ``C[k, l] = (B^{-1} e_{b_l})[b_k]`` is the ``N_b × N_b`` capacitance
    matrix, LU-factorized once at construction.

    Cost and storage: only the factorized ``C`` is stored — ``O(N_b²)``
    memory. The Green's functions ``B^{-1} e_{b_k}`` are never kept; step 4 is
    a second base solve with the point sources scattered into a zero field.
    Each call therefore costs two base solves plus an ``O(N_b²)``
    back-substitution. Construction costs ``N_b`` base solves (run one column
    at a time under ``jit``, so no dense ``(N_b, n)`` array is formed).

    **Singular base operators.** If ``B`` is singular with right null vector
    ``r`` (e.g. the constant for a periodic or Neumann Poisson operator) and
    ``base_solve`` is the pseudo-inverse that projects out that mode, the plain
    method leaves the null-mode component undetermined and the interior
    residual is a nonzero constant. Pass ``null_vector=r`` (and, for a
    non-symmetric ``B``, ``left_null_vector=w`` with ``wᵀ B = 0``; it defaults
    to ``r``). The solution is then ``x = u − B^{+}(Σ α_k e_{b_k}) + β r`` with
    ``(α, β)`` from the bordered system

    ``[ C      −r_b ] [α]   [ u_b  ]``
    ``[ w_bᵀ    0   ] [β] = [ wᵀ f ]``

    whose second row makes the effective source ``f − Σ α_k e_{b_k}``
    orthogonal to the left null space, so that ``B x = f`` holds exactly away
    from the constrained indices. Without ``null_vector`` a singular base
    cannot be detected here, so callers must pass it.

    The solver operates on **flat** vectors. Any reshaping between fields and
    flat vectors, and any masking of the exterior, is the caller's
    responsibility -- keeping grid/mask concepts out of this class.

    Args:
        base_solve: Callable applying the base inverse ``B^{-1}`` (or the
            pseudo-inverse ``B^{+}`` for a singular ``B``) to a flat
            right-hand side of length ``n``. Stored as a static field, so it
            must be hashable (functions and closures are).
        boundary_indices: Flat indices of the constrained degrees of freedom,
            shape ``(N_b,)``.
        n: Length of the flat solution vector.
        null_vector: Right null vector ``r`` of a singular base operator,
            shape ``(n,)``; ``None`` (default) for a non-singular base.
        left_null_vector: Left null vector ``w`` of a singular base operator;
            defaults to ``null_vector`` (correct for symmetric ``B``). Ignored
            when ``null_vector`` is ``None``.
        keep_base_solve: When ``False``, ``base_solve`` is only used to build
            the capacitance matrix and is not stored; pass it to `__call__`
            instead. A static closure over an operator hides the operator's
            arrays from autodiff and makes every instance a new ``jit``
            cache key, so `gaussx.MaskedOperator` keeps its base as a pytree
            leaf and uses this.

    Attributes:
        base_solve: The base inverse callable (static), or ``None`` when
            built with ``keep_base_solve=False``.
        boundary_indices: The constrained indices.
        capacitance_lu: LU factorization ``(lu, pivots)`` of ``C``, or of the
            bordered ``(N_b + 1) × (N_b + 1)`` system when a null vector is
            given.
        null_vector: ``r``, or ``None``.
        left_null_vector: ``w``, or ``None``.
        transposed: Whether this solver is the `transpose` of a built one
            (it then solves with ``Bᵀ`` using the transposed factorisation).
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
            r_b = null_vector[indices]
            l_b = left_null_vector[indices]
            system = jnp.block(
                [
                    [capacitance, -r_b[:, None]],
                    [l_b[None, :], jnp.zeros((1, 1), dtype=capacitance.dtype)],
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
        """Solve the constrained system for a flat right-hand side ``rhs``.

        Args:
            rhs: Flat right-hand side, shape ``(n,)``.
            base_solve: The base inverse to use; overrides the stored one
                and is required when the solver was built with
                ``keep_base_solve=False``. For a solver returned by
                `transpose` it must apply ``B^{-T}``.

        Returns:
            The constrained solution, shape ``(n,)``.
        """
        base_solve = self.base_solve if base_solve is None else base_solve
        if base_solve is None:
            raise ValueError(
                "This CapacitanceSolver stores no base_solve; pass base_solve=."
            )
        # A transposed solver solves with the transposed factorisation. The
        # bordered system for Bᵀ is D Sᵀ D with D = diag(1, …, 1, -1).
        trans = 1 if self.transposed else 0
        u = base_solve(rhs)
        u_b = u[self.boundary_indices]
        if self.null_vector is None:
            alpha = jsl.lu_solve(self.capacitance_lu, u_b, trans=trans)
            beta = None
        else:
            assert self.left_null_vector is not None
            load = jnp.dot(self.left_null_vector, rhs)
            sign = -1.0 if self.transposed else 1.0
            sol = jsl.lu_solve(
                self.capacitance_lu, jnp.append(u_b, sign * load), trans=trans
            )
            alpha, beta = sol[:-1], sign * sol[-1]
        # Constrained indices are distinct; saying so keeps the scatter
        # transposable (reverse mode through a solve).
        sources = (
            jnp.zeros_like(u).at[self.boundary_indices].set(alpha, unique_indices=True)
        )
        x = u - base_solve(sources)
        if beta is not None:
            assert self.null_vector is not None
            x = x + beta * self.null_vector
        return x

    def transpose(self) -> CapacitanceSolver:
        """The capacitance solver for ``Bᵀ``, reusing this factorisation.

        ``Cᵀ`` is the capacitance matrix of ``Bᵀ`` on the same indices, and
        ``Bᵀ``'s right/left null vectors are ``B``'s left/right ones, so no
        base solve is needed. The result stores no ``base_solve``: pass
        ``Bᵀ``'s inverse to `__call__`.
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
    """Build ``C[k, l] = (B^{-1} e_{b_l})[b_k]`` one column at a time.

    Each unit source is created inside the mapped function and only its
    ``N_b`` boundary samples are kept. Columns are processed in batches of
    ``_BUILD_BATCH`` (vectorized within a batch), so peak memory is
    ``O(_BUILD_BATCH · n + N_b²)`` rather than the ``O(N_b · n)`` of a
    ``vmap`` over all dense unit sources.
    """

    @jax.jit
    def build(idx: Int[Array, " Nb"]) -> Float[Array, "Nb Nb"]:
        def column(b_l: Int[Array, ""]) -> Float[Array, " Nb"]:
            source = jnp.zeros(n).at[b_l].set(1.0)
            return base_solve(source)[idx]

        # lax.map returns row l = column l of C; transpose to C[k, l].
        return jax.lax.map(column, idx, batch_size=_BUILD_BATCH).T

    return build(indices)
