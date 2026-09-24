"""Masked linear operator — row/column sub-selection of a base operator."""

from __future__ import annotations

import functools as ft
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._capacitance import CapacitanceSolver
from gaussx._operators._diagonalised import as_diagonalised


class MaskedOperator(lx.AbstractLinearOperator):
    """Row/column-masked view of a base operator.

    Given a base operator A of shape ``(N, N)`` and boolean masks,
    produces the sub-matrix ``A[row_mask][:, col_mask]``.

    Matvec is computed without materializing the sub-matrix:
    zero-pad input to full size, apply base matvec, then extract
    masked rows.

    **Fast masked solves (capacitance method).** For a square mask
    (``row_mask == col_mask == m``) whose base ``B`` has a fast solve (a
    `gaussx.DiagonalisedOperator`, `gaussx.Circulant`, or a Kronecker sum of
    them), pass ``coupling_indices``: the flat indices ``C`` of the
    masked-*out* degrees of freedom that ``B`` couples to masked-in rows
    (``B[m, j] ≠ 0``). Then ``gaussx.solve(op, f)`` solves
    ``B[m][:, m] x = f`` exactly by the capacitance method: it finds ``y`` on
    the full grid with ``y[C] = 0`` and ``(B y)[i] = f[i]`` at every ``i ∉ C``
    (point sources on ``C`` absorb the constraints), which on ``m`` is exactly
    the masked system because the remaining masked-out entries do not couple
    to ``m``. The ``|C| × |C|`` capacitance factorisation is built once, here
    in the constructor, so each solve costs two base solves.

    For a local stencil (e.g. a finite-difference Laplacian on a grid) ``C``
    is the one-cell ring outside the mask — use `grid_coupling_indices`, with
    ``periodic=True`` on axes where ``B`` wraps. Passing *all* masked-out
    indices is always exact (for any ``B``) but costs ``O(|C|)`` base solves
    to build.

    If ``B`` is singular (e.g. a periodic Poisson operator, whose solve is the
    pseudo-inverse), its null vector must enter the capacitance system (see
    `gaussx.CapacitanceSolver`). For a diagonalised ``B`` with a normal
    transform and exactly one zero eigenvalue this is derived automatically;
    otherwise pass ``null_vector`` (and ``left_null_vector`` if ``B`` is not
    symmetric).

    Args:
        base: The underlying ``(N, N)`` linear operator.
        row_mask: Boolean mask of length N selecting output rows.
        col_mask: Boolean mask of length N selecting input columns.
        coupling_indices: Optional flat indices ``C`` (see above); enables the
            capacitance solve. Requires ``row_mask == col_mask`` (concrete).
        null_vector: Right null vector of a singular ``base``; derived
            automatically for diagonalised bases when possible.
        left_null_vector: Left null vector of a singular non-symmetric
            ``base``; defaults to ``null_vector``.

    Attributes:
        capacitance: The precomputed `gaussx.CapacitanceSolver` on the full
            index space, or ``None`` when no ``coupling_indices`` were given
            (``solve`` then falls back to the dense path).
    """

    base: lx.AbstractLinearOperator
    row_mask: Bool[Array, " N"]
    col_mask: Bool[Array, " N"]
    capacitance: CapacitanceSolver | None
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        row_mask: Bool[Array, " N"],
        col_mask: Bool[Array, " N"],
        *,
        tags: object | frozenset[object] = frozenset(),
        coupling_indices: Int[Array, " Nc"] | None = None,
        null_vector: Float[Array, " N"] | None = None,
        left_null_vector: Float[Array, " N"] | None = None,
    ) -> None:
        if base.in_size() != base.out_size():
            raise ValueError(
                f"Base operator must be square, got in_size={base.in_size()}, "
                f"out_size={base.out_size()}."
            )
        n = base.in_size()
        if row_mask.shape != (n,) or col_mask.shape != (n,):
            raise ValueError(
                f"Masks must have shape ({n},), got row_mask={row_mask.shape}, "
                f"col_mask={col_mask.shape}."
            )
        self.base = base
        self.row_mask = jnp.asarray(row_mask, dtype=bool)
        self.col_mask = jnp.asarray(col_mask, dtype=bool)
        self._in_size = int(jnp.sum(col_mask))
        self._out_size = int(jnp.sum(row_mask))
        struct = base.out_structure()
        leaves = jax.tree.leaves(struct)
        self._dtype = str(leaves[0].dtype)
        self.tags = _to_frozenset(tags)
        self.capacitance = None
        if coupling_indices is not None:
            self.capacitance = _build_capacitance(
                base,
                self.row_mask,
                self.col_mask,
                jnp.asarray(coupling_indices),
                null_vector,
                left_null_vector,
            )

    def mv(self, vector: Float[Array, " m"]) -> Float[Array, " k"]:
        # Scatter input into full-size vector at col_mask positions
        n = self.base.in_size()
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        full_v = jnp.zeros(n, dtype=vector.dtype).at[col_indices].set(vector)
        # Apply base operator
        full_out = self.base.mv(full_v)
        # Gather output at row_mask positions
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        return full_out[row_indices]

    def as_matrix(self) -> Float[Array, "k m"]:
        full = self.base.as_matrix()
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        return full[jnp.ix_(row_indices, col_indices)]

    def transpose(self) -> MaskedOperator:
        return MaskedOperator(
            self.base.T,
            self.col_mask,
            self.row_mask,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _build_capacitance(
    base: lx.AbstractLinearOperator,
    row_mask: Bool[Array, " N"],
    col_mask: Bool[Array, " N"],
    coupling_indices: Int[Array, " Nc"],
    null_vector: Float[Array, " N"] | None,
    left_null_vector: Float[Array, " N"] | None,
) -> CapacitanceSolver:
    """Capacitance solver on the full index space for a square mask."""
    from gaussx._primitives._solve import solve

    try:
        same = bool(np.array_equal(np.asarray(row_mask), np.asarray(col_mask)))
    except jax.errors.TracerArrayConversionError as err:
        raise ValueError(
            "coupling_indices needs concrete masks (build the MaskedOperator "
            "outside jit)."
        ) from err
    if not same:
        raise ValueError(
            "coupling_indices requires a square mask (row_mask == col_mask)."
        )
    if null_vector is None:
        null_vector, left_null_vector = _derived_null_vectors(base)
    return CapacitanceSolver(
        ft.partial(solve, base),
        coupling_indices,
        base.in_size(),
        null_vector=null_vector,
        left_null_vector=left_null_vector,
    )


def _derived_null_vectors(
    base: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " N"] | None, Float[Array, " N"] | None]:
    """Null vectors of a singular diagonalised base, from its zero eigenvalue.

    ``A r = 0`` iff ``V r`` is supported on the zero eigenvalues, so with a
    single zero at coefficient ``k`` the right null vector is ``r = V⁻¹ e_k``.
    For a normal transform and a real operator ``Aᵀ`` has the same
    eigenvectors, so ``r`` is also the left null vector. Zero detection is
    exact (``λ == 0``), matching the pseudo-inverse used by ``solve``.
    """
    diagonalised = as_diagonalised(base)
    if diagonalised is None:
        return None, None
    try:
        lam = np.asarray(diagonalised.eigenvalues_flat())
    except jax.errors.TracerArrayConversionError:
        return None, None
    zeros = np.flatnonzero(lam == 0)
    if zeros.size == 0:
        return None, None
    if zeros.size > 1 or not (diagonalised.normal and diagonalised.real_output):
        raise ValueError(
            "The base operator is singular but its null vector cannot be derived "
            f"automatically ({zeros.size} zero eigenvalues, normal="
            f"{diagonalised.normal}, real_output={diagonalised.real_output}); "
            "pass null_vector (and left_null_vector) explicitly."
        )
    unit = jnp.zeros(lam.shape, dtype=lam.dtype).at[int(zeros[0])].set(1.0)
    r = jnp.real(diagonalised.inverse_flat(unit))
    return r, r


def grid_coupling_indices(
    mask: Bool[Array, " *grid"],
    *,
    periodic: bool | tuple[bool, ...] = False,
    connectivity: int = 1,
) -> Int[Array, " Nc"]:
    """Flat indices of masked-out grid cells adjacent to the masked-in region.

    This is the coupling set ``C`` that `MaskedOperator` needs for a local
    stencil operator on a regular grid: every cell outside ``mask`` that the
    stencil of some cell inside ``mask`` reaches. With ``connectivity=1``
    neighbours differ by ±1 along one axis (the 5-point / 7-point stencil);
    ``connectivity=mask.ndim`` includes diagonals (9-point / 27-point).

    Args:
        mask: Boolean grid, ``True`` for the unknowns (masked-in cells).
        periodic: Whether each axis wraps around (a bool applies to all
            axes). Must match the base operator: a periodic operator couples
            cells across the edge, so an edge-touching mask needs the wrapped
            neighbours in ``C``.
        connectivity: Maximum number of axes a neighbour offset may change.

    Returns:
        Flat (row-major) indices of the coupled masked-out cells.
    """
    m = np.asarray(mask, dtype=bool)
    ndim = m.ndim
    wrap = (periodic,) * ndim if isinstance(periodic, bool) else tuple(periodic)
    if len(wrap) != ndim:
        raise ValueError(f"periodic has {len(wrap)} entries for a {ndim}-D mask.")
    if not 1 <= connectivity <= ndim:
        raise ValueError(f"connectivity must be in [1, {ndim}], got {connectivity}.")
    reach = np.zeros_like(m)
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        if 0 < np.count_nonzero(offset) <= connectivity:
            reach |= _shift(m, offset, wrap)
    return jnp.asarray(np.flatnonzero((reach & ~m).ravel()))


def _shift(
    m: np.ndarray, offset: tuple[int, ...], wrap: tuple[bool, ...]
) -> np.ndarray:
    """``out[i] = m[i − offset]``, wrapping or zero-filling per axis."""
    out = m
    for axis, (step, periodic) in enumerate(zip(offset, wrap, strict=True)):
        if step == 0:
            continue
        out = np.roll(out, step, axis=axis)
        if not periodic:
            edge = [slice(None)] * m.ndim
            edge[axis] = slice(0, step) if step > 0 else slice(step, None)
            out = out.copy()
            out[tuple(edge)] = False
    return out
