"""Masked operator A = A₀[m_r, m_c]: rows and columns kept from a base A₀.

A₀ ∈ ℝ^{N × N} is the base operator, and the boolean masks m_r, m_c ∈
{0, 1}^N select the kept rows and columns, so A is N_r × N_c with N_r and
N_c the number of kept rows and columns. A square mask m_r = m_c = m gives
the masked system A₀[m, m] on the masked-in indices m. Its solve uses the
capacitance method of `CapacitanceSolver`, with the masked-out indices that
A₀ couples to m as the constrained indices b_1, …, b_N_b.
"""

from __future__ import annotations

import functools as ft
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.core import Tracer
from jaxtyping import Array, Bool, Float, Int

from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._capacitance import CapacitanceSolver, _replace_fields
from gaussx._operators._diagonalised import as_diagonalized


class MaskedOperator(lx.AbstractLinearOperator):
    """Masked view A = A₀[m_r, m_c] of a square base operator A₀.

        A x = (A₀ x̃)[m_r],   x̃[m_c] = x,   x̃ = 0 elsewhere

    The matvec never forms the sub-matrix: it zero-pads x to the full size
    N, applies A₀, and gathers the kept rows.

    **Fast masked solves (capacitance method).** For a square mask
    m_r = m_c = m whose base A₀ has a fast solve (a
    `gaussx.DiagonalizedOperator`, `gaussx.circulant`, or a Kronecker sum of
    them), pass ``coupling_indices``: the flat indices b_1, …, b_N_b of the
    masked-*out* degrees of freedom that A₀ couples to masked-in rows
    ([A₀]_ij ≠ 0 for some i ∈ m). Then ``gaussx.solve(op, f)`` solves
    A₀[m, m] x = f exactly by the capacitance method: it finds y on the full
    grid with y_bₖ = 0 for every k and (A₀ y)_i = f_i at every other index i
    (point sources on the b_k absorb the constraints), which on m is exactly
    the masked system because the remaining masked-out entries do not couple
    to m. The N_b × N_b capacitance factorisation is built once, here in the
    constructor, so each solve costs two base solves.

    For a local stencil (a finite-difference Laplacian on a grid, say) the
    b_k are the one-cell ring outside the mask; use `grid_coupling_indices`,
    with ``periodic=True`` on axes where A₀ wraps. Passing *all* masked-out
    indices is always exact (for any A₀) but costs O(N_b) base solves to
    build.

    If A₀ is singular (a periodic Poisson operator, whose solve is the
    pseudo-inverse A₀⁺), its null vector must enter the capacitance system
    (see `gaussx.CapacitanceSolver`). For a diagonalised A₀ with a normal
    transform and exactly one zero eigenvalue this is derived automatically;
    otherwise pass ``null_vector`` (and ``left_null_vector`` if A₀ is not
    symmetric).

    **The capacitance is a cache of** ``base``. Gradients with respect to
    ``base`` are exact, but replacing ``base`` in an existing operator
    (``eqx.apply_updates`` after an optimiser step, or ``eqx.tree_at``)
    leaves the factorisation describing the *old* base, and ``solve`` is
    then wrong. Rebuild with `with_base`, or construct the operator inside
    the loss from the current parameters. That also works under ``jit``,
    except that a traced base's null vector cannot be derived: for a
    singular base built under ``jit`` pass ``null_vector`` (and
    ``left_null_vector``) explicitly. A traced diagonalised base with a zero
    eigenvalue and no ``null_vector`` raises at run time rather than solving
    the wrong system.

    Args:
        base: The square base operator A₀, N × N.
        row_mask: Row mask m_r ∈ {0, 1}^N, ``True`` for the kept rows.
        col_mask: Column mask m_c ∈ {0, 1}^N, ``True`` for the kept columns.
        tags: Extra lineax tags for A.
        coupling_indices: Flat coupling indices b_1, …, b_N_b (see above);
            enables the capacitance solve. Requires concrete masks with
            m_r = m_c.
        null_vector: Right null vector r of a singular A₀, A₀ r = 0; derived
            automatically for a diagonalised A₀ when possible.
        left_null_vector: Left null vector w of a singular non-symmetric A₀,
            wᵀ A₀ = 0; defaults to ``null_vector``.

    Attributes:
        capacitance: The precomputed `gaussx.CapacitanceSolver` on the full
            index space, or ``None`` when no ``coupling_indices`` were given
            (``solve`` then falls back to the dense path). It stores no
            ``base_solve``: ``solve`` builds one from ``base`` at call time,
            so ``base`` stays a differentiable pytree leaf and ``solve`` is
            differentiated implicitly (not through the cached factorisation).

    Raises:
        ValueError: If A₀ is not square, a mask is not of length N,
            ``coupling_indices`` is given with traced or unequal masks, or a
            singular A₀'s null vector cannot be derived.

    Examples:
        Keep rows {0, 2} and columns {1, 2} of a 3 × 3 matrix:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        >>> A0 = lx.MatrixLinearOperator(matrix)
        >>> rows = jnp.array([True, False, True])
        >>> cols = jnp.array([False, True, True])
        >>> A = gaussx.MaskedOperator(A0, rows, cols)
        >>> A.as_matrix().tolist()
        [[1.0, 2.0], [7.0, 8.0]]
        >>> A.mv(jnp.array([1.0, -1.0])).tolist()
        [-1.0, -1.0]

        A masked periodic Laplacian solved by the capacitance method:

        >>> stencil = jnp.array([2.5, -1.0, 0.0, 0.0, 0.0, -1.0])
        >>> mask = jnp.array([False, True, True, True, False, False])
        >>> coupling = gaussx.grid_coupling_indices(mask, periodic=True)
        >>> coupling.tolist()
        [0, 4]
        >>> masked = gaussx.MaskedOperator(
        ...     gaussx.circulant(stencil, symmetric=True),
        ...     mask,
        ...     mask,
        ...     coupling_indices=coupling,
        ... )
        >>> f = jnp.ones(3)
        >>> x = gaussx.solve(masked, f)
        >>> bool(jnp.allclose(masked.mv(x), f, atol=1e-5))
        True
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
        # Counted with NumPy so constant masks also work inside ``jit``
        # (``jnp.sum`` would be staged); traced masks cannot give a static
        # size and raise.
        # N_c = ∑_j [m_c]_j
        self._in_size = int(np.sum(np.asarray(col_mask)))
        # N_r = ∑_i [m_r]_i
        self._out_size = int(np.sum(np.asarray(row_mask)))
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
        """Apply A to a vector through the base, without forming A.

            A x = (A₀ x̃)[m_r],   x̃[m_c] = x,   x̃ = 0 elsewhere

        Args:
            vector: Input x on the kept columns, of length N_c.

        Returns:
            A x on the kept rows, of length N_r.
        """
        n = self.base.in_size()
        # kept column indices; a static size, so an empty mask gives none
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        # x̃ ∈ ℝ^N, x̃[m_c] = x and zero elsewhere
        padded_vector = jnp.zeros(n, dtype=vector.dtype).at[col_indices].set(vector)
        # A₀ x̃
        full_output = self.base.mv(padded_vector)
        # kept row indices
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        # A x = (A₀ x̃)[m_r]
        return full_output[row_indices]

    def as_matrix(self) -> Float[Array, "k m"]:
        """Dense A = A₀[m_r, m_c], N_r × N_c.

        Returns:
            The materialised sub-matrix of the dense base.
        """
        full_matrix = self.base.as_matrix()
        row_indices = jnp.where(self.row_mask, size=self._out_size)[0]
        col_indices = jnp.where(self.col_mask, size=self._in_size)[0]
        # A₀[m_r, m_c]
        return full_matrix[jnp.ix_(row_indices, col_indices)]

    def with_base(
        self,
        base: lx.AbstractLinearOperator,
        *,
        null_vector: Float[Array, " N"] | None = None,
        left_null_vector: Float[Array, " N"] | None = None,
    ) -> MaskedOperator:
        """The same masked view of a new base, with its capacitance rebuilt.

            A = A₀′[m_r, m_c]

        Use this after updating the base (an optimiser step, say): the
        capacitance factorisation is a cache of the base and does not follow
        leaf updates. Costs the N_b base solves of construction.

        Args:
            base: The new square base operator A₀′, N × N.
            null_vector: Right null vector r of a singular A₀′. As in the
                constructor, it is derived automatically for a singular
                diagonalised base; the old one is *not* reused, since the
                update may change (or remove) the null space.
            left_null_vector: Left null vector w of a singular non-symmetric
                A₀′; defaults to ``null_vector``.

        Returns:
            A new `MaskedOperator` over A₀′ with the same masks, tags and
            coupling indices.

        Raises:
            ValueError: As in the constructor.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> mask = jnp.array([True, False, True])
            >>> A0 = lx.MatrixLinearOperator(jnp.eye(3))
            >>> A = gaussx.MaskedOperator(A0, mask, mask)
            >>> B = A.with_base(lx.MatrixLinearOperator(2.0 * jnp.eye(3)))
            >>> B.as_matrix().tolist()
            [[2.0, 0.0], [0.0, 2.0]]
        """
        if self.capacitance is None:
            return MaskedOperator(base, self.row_mask, self.col_mask, tags=self.tags)
        return MaskedOperator(
            base,
            self.row_mask,
            self.col_mask,
            tags=self.tags,
            coupling_indices=self.capacitance.boundary_indices,
            null_vector=null_vector,
            left_null_vector=left_null_vector,
        )

    def transpose(self) -> MaskedOperator:
        """Aᵀ = A₀ᵀ[m_c, m_r], still a `MaskedOperator`.

        Built field by field rather than through ``__init__``, which needs
        concrete masks, so it can run under a trace (lineax's solve VJP,
        say). The capacitance of A₀ᵀ is the transposed factorisation of
        A₀'s.

        Returns:
            The transposed operator, with swapped masks and transposed tags.
        """
        # (A₀[m_r, m_c])ᵀ = A₀ᵀ[m_c, m_r]
        return _replace_fields(
            self,
            base=self.base.T,
            row_mask=self.col_mask,
            col_mask=self.row_mask,
            capacitance=(
                None if self.capacitance is None else self.capacitance.transpose()
            ),
            _in_size=self._out_size,
            _out_size=self._in_size,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N_c,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N_r,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _build_capacitance(
    base: lx.AbstractLinearOperator,
    row_mask: Bool[Array, " N"],
    col_mask: Bool[Array, " N"],
    coupling_indices: Int[Array, " Nc"],
    null_vector: Float[Array, " N"] | None,
    left_null_vector: Float[Array, " N"] | None,
) -> CapacitanceSolver:
    """Capacitance solver on the full index space for a square mask.

    Algorithm:
        1. Check that the masks are concrete and equal, m_r = m_c.
        2. Without a given null vector, derive r and w from a concrete
           diagonalised A₀, or keep its traced eigenvalues γ for a run-time
           check.
        3. Build the solver from x ↦ A₀⁻¹ x and the coupling indices.
        4. For traced γ, guard the factorisation so any solve raises if some
           γ_i = 0.

    Args:
        base: The square base operator A₀, N × N.
        row_mask: Row mask m_r.
        col_mask: Column mask m_c.
        coupling_indices: Flat coupling indices b_1, …, b_N_b.
        null_vector: Right null vector r of a singular A₀, or ``None``.
        left_null_vector: Left null vector w of a singular A₀, or ``None``.

    Returns:
        The capacitance solver, without a stored base solve.

    Raises:
        ValueError: If the masks are traced or unequal, or a singular A₀'s
            null vector cannot be derived.
    """
    # lazy import, cycle: _primitives._solve -> _operators._masked
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
    traced_eigenvalues = None
    if null_vector is None:
        null_vector, left_null_vector = _derived_null_vectors(base)
        if null_vector is None:
            traced_eigenvalues = _traced_eigenvalues(base)
    solver = CapacitanceSolver(
        ft.partial(solve, base),
        coupling_indices,
        base.in_size(),
        null_vector=null_vector,
        left_null_vector=left_null_vector,
        keep_base_solve=False,
    )
    if traced_eigenvalues is None:
        return solver
    # A traced singular base cannot have its null vector derived, and the
    # unaugmented capacitance system would then solve the wrong problem.
    # Guard the factorisation (which every solve reads) at run time.
    lu, pivots = solver.capacitance_lu
    lu = eqx.error_if(
        lu,
        jnp.any(traced_eigenvalues == 0),
        "The base operator is singular, but its null vector cannot be derived "
        "under tracing; pass null_vector (and left_null_vector) explicitly.",
    )
    return eqx.tree_at(lambda s: s.capacitance_lu, solver, (lu, pivots))


def _traced_eigenvalues(base: lx.AbstractLinearOperator) -> Array | None:
    """A diagonalised base's eigenvalues γ when they are traced, else ``None``.

    Args:
        base: The base operator A₀.

    Returns:
        The flat eigenvalues γ of A₀ = W⁻¹ diag(γ) W if A₀ is diagonalisable
        and γ is a tracer, otherwise ``None``.
    """
    diagonalised = as_diagonalized(base)
    if diagonalised is None:
        return None
    eigenvalues = diagonalised.eigenvalues_flat()
    return eigenvalues if isinstance(eigenvalues, Tracer) else None


def _derived_null_vectors(
    base: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " N"] | None, Float[Array, " N"] | None]:
    """Null vectors of a singular diagonalised base, from its zero eigenvalue.

        A₀ r = 0,   r = W⁻¹ e_k,   γ_k = 0

    With A₀ = W⁻¹ diag(γ) W, A₀ r = 0 iff W r is supported on the zero
    eigenvalues, so with a single zero at coefficient k the right null
    vector is r = W⁻¹ e_k. For a normal transform and a real operator, A₀ᵀ
    has the same eigenvectors, so r is also the left null vector w. Zero
    detection is exact (γ_k == 0), matching the pseudo-inverse used by
    ``solve``.

    Args:
        base: The base operator A₀.

    Returns:
        Tuple ``(r, w)`` with w = r, or ``(None, None)`` when A₀ is not
        diagonalisable, γ is traced, or A₀ is non-singular.

    Raises:
        ValueError: If A₀ has several zero eigenvalues, or its transform is
            not normal or its output not real.
    """
    diagonalised = as_diagonalized(base)
    if diagonalised is None:
        return None, None
    try:
        eigenvalues = np.asarray(diagonalised.eigenvalues_flat())
    except jax.errors.TracerArrayConversionError:
        return None, None
    # {k : γ_k = 0}
    zeros = np.flatnonzero(eigenvalues == 0)
    if zeros.size == 0:
        return None, None
    if zeros.size > 1 or not (diagonalised.normal and diagonalised.real_output):
        raise ValueError(
            "The base operator is singular but its null vector cannot be derived "
            f"automatically ({zeros.size} zero eigenvalues, normal="
            f"{diagonalised.normal}, real_output={diagonalised.real_output}); "
            "pass null_vector (and left_null_vector) explicitly."
        )
    # e_k, the unit coefficient of the zero eigenvalue
    unit = jnp.zeros(eigenvalues.shape, dtype=eigenvalues.dtype)
    unit = unit.at[int(zeros[0])].set(1.0)
    # r = Re(W⁻¹ e_k)
    null_vector = jnp.real(diagonalised.inverse_flat(unit))
    return null_vector, null_vector


def grid_coupling_indices(
    mask: Bool[Array, " *grid"],
    *,
    periodic: bool | tuple[bool, ...] = False,
    connectivity: int = 1,
) -> Int[Array, " Nc"]:
    """Flat indices of masked-out grid cells adjacent to the masked-in region.

        {b_1, …, b_N_b} = {c ∉ m : c = c′ + δ for some c′ ∈ m, δ ∈ Δ}

    Δ ⊂ {−1, 0, 1}^D are the stencil offsets that change between 1 and
    ``connectivity`` of the D axes. The b_k are the coupling indices that
    `MaskedOperator` needs for a local stencil operator on a regular grid:
    every cell outside m that the stencil of some cell inside m reaches.
    With ``connectivity=1`` neighbours differ by ±1 along one axis (the
    5-point or 7-point stencil); ``connectivity=mask.ndim`` includes
    diagonals (9-point or 27-point).

    Args:
        mask: Boolean grid m, ``True`` for the unknowns (masked-in cells).
        periodic: Whether each axis wraps around (a bool applies to all
            axes). Must match the base operator: a periodic operator couples
            cells across the edge, so an edge-touching mask needs the wrapped
            neighbours among the b_k.
        connectivity: Maximum number of axes a neighbour offset may change.

    Returns:
        Flat (row-major) indices of the coupled masked-out cells, sorted.

    Raises:
        ValueError: If ``periodic`` has the wrong number of entries or
            ``connectivity`` is outside [1, D].

    Examples:
        The ring around a 2 × 2 block in a 4 × 4 grid, with and without
        diagonals:

        >>> import numpy as np
        >>> import gaussx
        >>> mask = np.zeros((4, 4), dtype=bool)
        >>> mask[1:3, 1:3] = True
        >>> gaussx.grid_coupling_indices(mask).tolist()
        [1, 2, 4, 7, 8, 11, 13, 14]
        >>> gaussx.grid_coupling_indices(mask, connectivity=2).shape
        (12,)

        A periodic axis wraps around the edge:

        >>> line = np.array([True, False, False, False])
        >>> gaussx.grid_coupling_indices(line, periodic=True).tolist()
        [1, 3]
    """
    grid_mask = np.asarray(mask, dtype=bool)
    ndim = grid_mask.ndim
    wrap = (periodic,) * ndim if isinstance(periodic, bool) else tuple(periodic)
    if len(wrap) != ndim:
        raise ValueError(f"periodic has {len(wrap)} entries for a {ndim}-D mask.")
    if not 1 <= connectivity <= ndim:
        raise ValueError(f"connectivity must be in [1, {ndim}], got {connectivity}.")
    reach = np.zeros_like(grid_mask)
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        # δ ∈ Δ: a non-zero offset changing at most ``connectivity`` axes
        if 0 < np.count_nonzero(offset) <= connectivity:
            # cells c′ + δ reached from some c′ ∈ m
            reach |= _shift(grid_mask, offset, wrap)
    # b_1, …, b_N_b = reached cells outside m, as flat row-major indices
    return jnp.asarray(np.flatnonzero((reach & ~grid_mask).ravel()))


def _shift(
    m: np.ndarray, offset: tuple[int, ...], wrap: tuple[bool, ...]
) -> np.ndarray:
    """Shift a boolean grid by an offset, wrapping or zero-filling per axis.

        out[c] = m[c − δ]

    Args:
        m: Boolean grid.
        offset: Offset δ, one step in {−1, 0, 1} per axis.
        wrap: Whether each axis wraps around; a non-wrapping axis fills the
            vacated edge with ``False``.

    Returns:
        The shifted grid, of the same shape.
    """
    out = m
    for axis, (step, periodic) in enumerate(zip(offset, wrap, strict=True)):
        if step == 0:
            continue
        # out[c] = m[c − δ] along this axis, wrapping
        out = np.roll(out, step, axis=axis)
        if not periodic:
            # clear the cells that wrapped across a non-periodic edge
            edge = [slice(None)] * m.ndim
            edge[axis] = slice(0, step) if step > 0 else slice(step, None)
            out = out.copy()
            out[tuple(edge)] = False
    return out
