"""Operators diagonal in a fast transform basis: ``A = V⁻¹ Λ V``.

Every constant-coefficient operator on a regular grid with periodic,
Dirichlet or Neumann boundaries — and every spectral-method Laplacian — is
diagonalised by a fast transform (FFT, DCT/DST, spherical harmonics, or a
precomputed eigenvector matrix). `DiagonalisedOperator` represents such an
operator by its transform pair and eigenvalue array, so `solve`, `logdet`,
`inv`, `sqrt` and `trace` are elementwise operations on the eigenvalues and
nothing is ever materialised.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Inexact, Shaped

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange
from gaussx._operators._block_diag import _to_frozenset


Transform = Callable[[Array], Array]


class DiagonalisedOperator(lx.AbstractLinearOperator):
    r"""Linear operator ``A x = V⁻¹(Λ ⊙ V x)`` given by a transform pair.

    ``forward`` applies ``V`` (field of shape ``in_shape`` → coefficients) and
    ``inverse`` applies ``V⁻¹`` (coefficients → field); ``eigenvalues`` holds
    ``Λ`` in the coefficient layout. The operator acts on **flat** vectors of
    length ``prod(in_shape)``, as every lineax operator does; ``mv`` reshapes
    internally.

    Examples of ``(forward, inverse, Λ)``:

    - Periodic grid: ``fftn`` / ``ifftn`` and the symbol of the stencil — see
      `Circulant` and `circulant_from_symbol`.
    - Dirichlet / Neumann grid: an orthonormal DST / DCT pair and the
      corresponding finite-difference or spectral eigenvalues.
    - A dense diagonalisable matrix ``M = V Λ V⁻¹`` (e.g. a Chebyshev
      collocation block): `DiagonalisedOperator.from_eigen_factorization`.

    **Algebra stays closed.** ``A + c·I``, ``A − c·I``, ``c·A``, ``A / c``,
    ``−A`` and ``A ± B`` (for ``B`` with the same transform pair) return
    another `DiagonalisedOperator` with updated eigenvalues, so a shifted
    Helmholtz operator ``A − λI`` is never materialised.

    **Structured primitives.** `gaussx.solve` divides by ``Λ`` (eigenvalues
    that are exactly zero get a zero coefficient, i.e. the pseudo-inverse for
    a normal operator); `gaussx.logdet` is ``Σ log|λ|`` (the ``slogdet``
    convention, so ``−inf`` for a singular operator); `gaussx.inv`,
    `gaussx.sqrt` and `gaussx.trace` are elementwise in ``Λ``. A
    `gaussx.KroneckerSum` whose factors are all `DiagonalisedOperator` s is
    solved through the composed per-axis transforms.

    **Transpose.** With ``normal=True`` (``V⁻¹ ∝ Vᴴ``, e.g. FFT or orthonormal
    DCT/DST), ``Aᵀ`` for a real operator (``real_output=True``) is the same
    pair with conjugated eigenvalues; for a complex one it is the same
    eigenvalues with the conjugated pair ``conj(V)``, ``conj(V)⁻¹``, since
    ``Aᵀ x = conj(Aᴴ conj(x))``. Otherwise pass
    ``transpose_pair=(forward_T, inverse_T)`` applying ``V⁻ᵀ`` and ``Vᵀ`` in
    the same roles; without either, `transpose` raises.

    Args:
        eigenvalues: ``Λ`` in the coefficient layout; ``eigenvalues.size``
            must equal ``prod(in_shape)``. May be complex.
        forward: ``V``: array of shape ``in_shape`` → coefficient array.
        inverse: ``V⁻¹``: coefficient array → array of shape ``in_shape``.
        in_shape: Field shape the transforms act on.
        real_output: Return the real part of ``mv`` (default ``True``), as
            needed when ``V`` is complex but ``A`` is real (FFT). The
            primitives work on ``Λ`` directly, so this is only consistent
            when ``V⁻¹ Λ V`` is itself real (for the FFT: a conjugate-even
            ``Λ``); otherwise ``mv`` applies ``Re(A)`` while ``solve`` and
            ``logdet`` describe ``A``. The generic class cannot check an
            arbitrary transform pair; `circulant_from_symbol` does.
        normal: Whether ``V⁻¹`` is proportional to ``Vᴴ`` (see Transpose).
        transpose_pair: Optional ``(forward_T, inverse_T)`` for ``Aᵀ``.
        tags: lineax tags. ``symmetric_tag`` is added automatically when
            ``normal`` and ``real_output`` are set and the eigenvalues have a
            real dtype. (Real eigenvalues in a unitary basis make ``A``
            Hermitian; it is also symmetric only when it is real.)

    The transform callables are stored as static fields, so they must be
    hashable (module-level functions, ``functools.partial`` of them, or
    closures); keep them stable between calls to avoid ``jit`` recompiles.
    """

    eigenvalues: Inexact[Array, " ..."]
    forward: Transform = eqx.field(static=True)
    inverse: Transform = eqx.field(static=True)
    in_shape: tuple[int, ...] = eqx.field(static=True)
    real_output: bool = eqx.field(static=True)
    normal: bool = eqx.field(static=True)
    transpose_pair: tuple[Transform, Transform] | None = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        eigenvalues: Inexact[ArrayLike, " ..."],
        forward: Transform,
        inverse: Transform,
        in_shape: tuple[int, ...],
        *,
        real_output: bool = True,
        normal: bool = False,
        transpose_pair: tuple[Transform, Transform] | None = None,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        eigenvalues = jnp.asarray(eigenvalues)
        in_shape = tuple(int(d) for d in in_shape)
        size = 1
        for d in in_shape:
            size *= d
        if eigenvalues.size != size:
            raise ValueError(
                f"eigenvalues has {eigenvalues.size} entries but in_shape "
                f"{in_shape} has {size}."
            )
        self.eigenvalues = eigenvalues
        self.forward = forward
        self.inverse = inverse
        self.in_shape = in_shape
        self.real_output = real_output
        self.normal = normal
        self.transpose_pair = transpose_pair
        tags = _to_frozenset(tags)
        if normal and real_output and not jnp.iscomplexobj(eigenvalues):
            tags = tags | {lx.symmetric_tag}
        self.tags = tags

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_eigen_factorization(
        cls,
        factorization,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> DiagonalisedOperator:
        """Operator ``M = V Λ V⁻¹`` from a `gaussx.EigenFactorization`.

        Uses the dense eigenvector matrices as the transform pair
        (``forward = V⁻¹ ·``, ``inverse = V ·``), so non-symmetric but
        diagonalisable factors — e.g. Chebyshev collocation second-derivative
        blocks — get an exact structured solve and transpose.
        """
        V = factorization.eigenvectors
        V_inv = factorization.eigenvectors_inv
        n = V.shape[0]
        return cls(
            factorization.eigenvalues,
            forward=_MatrixApply(V_inv),
            inverse=_MatrixApply(V),
            in_shape=(n,),
            transpose_pair=(_MatrixApply(V.T), _MatrixApply(V_inv.T)),
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Flat-vector transforms (used by solves and Kronecker-sum composition)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of degrees of freedom, ``prod(in_shape)``."""
        return int(self.eigenvalues.size)

    def forward_flat(self, vector: Shaped[Array, " n"]) -> Shaped[Array, " n"]:
        """``V`` on a flat vector, returning flat coefficients."""
        return _flatten(self.forward(_unflatten(vector, self.in_shape)))

    def inverse_flat(self, coeffs: Shaped[Array, " n"]) -> Shaped[Array, " n"]:
        """``V⁻¹`` on flat coefficients, returning a flat vector."""
        return _flatten(
            self.inverse(_unflatten(coeffs, self.eigenvalues.shape)),
        )

    def eigenvalues_flat(self) -> Inexact[Array, " n"]:
        """``Λ`` flattened to match `forward_flat`."""
        return _flatten(self.eigenvalues)

    def with_eigenvalues(
        self, eigenvalues: Inexact[ArrayLike, " ..."]
    ) -> DiagonalisedOperator:
        """Same transform pair with new eigenvalues.

        Structural tags are recomputed: the automatic ``symmetric_tag`` is
        re-derived, and definiteness tags (``positive_semidefinite_tag``,
        ``negative_semidefinite_tag``) are dropped because a shift or sign
        change can invalidate them. Re-tag the result if you know it holds.
        """
        user_tags = self.tags - {
            lx.symmetric_tag,
            lx.positive_semidefinite_tag,
            lx.negative_semidefinite_tag,
        }
        return DiagonalisedOperator(
            jnp.asarray(eigenvalues),
            self.forward,
            self.inverse,
            self.in_shape,
            real_output=self.real_output,
            normal=self.normal,
            transpose_pair=self.transpose_pair,
            tags=user_tags,
        )

    # ------------------------------------------------------------------
    # lineax interface
    # ------------------------------------------------------------------

    def mv(self, vector: Shaped[Array, " n"]) -> Inexact[Array, " n"]:
        coeffs = self.forward(_unflatten(vector, self.in_shape))
        out = _flatten(self.inverse(self.eigenvalues * coeffs))
        return jnp.real(out) if self.real_output else out

    def as_matrix(self) -> Inexact[Array, "n n"]:
        eye = jnp.eye(self.size, dtype=self.out_structure().dtype)
        return jax.vmap(self.mv, out_axes=1)(eye)

    def transpose(self) -> DiagonalisedOperator:
        tags = lx.transpose_tags(self.tags)
        if self.normal and not self.real_output:
            # Conjugated eigenvalues would give Aᴴ, not Aᵀ (gh-330).
            return DiagonalisedOperator(
                self.eigenvalues,
                _conjugated(self.forward),
                _conjugated(self.inverse),
                self.in_shape,
                real_output=False,
                normal=True,
                tags=tags,
            )
        if self.normal:
            return DiagonalisedOperator(
                jnp.conj(self.eigenvalues),
                self.forward,
                self.inverse,
                self.in_shape,
                real_output=self.real_output,
                normal=True,
                transpose_pair=self.transpose_pair,
                tags=tags,
            )
        if self.transpose_pair is None:
            raise NotImplementedError(
                "DiagonalisedOperator.transpose needs normal=True or a transpose_pair."
            )
        forward_t, inverse_t = self.transpose_pair
        return DiagonalisedOperator(
            self.eigenvalues,
            forward_t,
            inverse_t,
            self.in_shape,
            real_output=self.real_output,
            transpose_pair=(self.forward, self.inverse),
            tags=tags,
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.size,), self._dtype())

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.size,), self._dtype())

    def _dtype(self):
        dtype = self.eigenvalues.dtype
        if self.real_output:
            return jnp.real(jnp.zeros((), dtype)).dtype
        return jnp.result_type(dtype, jnp.complex64)

    # ------------------------------------------------------------------
    # Closed algebra: shifts, scalings, sums in the same basis
    # ------------------------------------------------------------------

    def _same_basis(self, other: object) -> bool:
        return (
            isinstance(other, DiagonalisedOperator)
            and other.forward is self.forward
            and other.inverse is self.inverse
            and other.in_shape == self.in_shape
            and other.eigenvalues.shape == self.eigenvalues.shape
        )

    def __add__(self, other):
        shift = _identity_multiple(other)
        if shift is not None:
            return self.with_eigenvalues(self.eigenvalues + shift)
        if self._same_basis(other):
            return self.with_eigenvalues(self.eigenvalues + other.eigenvalues)
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        shift = _identity_multiple(other)
        if shift is not None:
            return self.with_eigenvalues(self.eigenvalues - shift)
        if self._same_basis(other):
            return self.with_eigenvalues(self.eigenvalues - other.eigenvalues)
        return super().__sub__(other)

    def __mul__(self, other):
        if _is_scalar(other):
            return self.with_eigenvalues(self.eigenvalues * other)
        return super().__mul__(other)

    def __rmul__(self, other):
        if _is_scalar(other):
            return self.with_eigenvalues(self.eigenvalues * other)
        return super().__rmul__(other)

    def __truediv__(self, other):
        if _is_scalar(other):
            return self.with_eigenvalues(self.eigenvalues / other)
        return super().__truediv__(other)

    def __neg__(self):
        return self.with_eigenvalues(-self.eigenvalues)


# ----------------------------------------------------------------------
# Circulant operators
# ----------------------------------------------------------------------


def _fftn(x: Array) -> Array:
    return jnp.fft.fftn(x)


def _ifftn(c: Array) -> Array:
    return jnp.fft.ifftn(c)


def circulant_from_symbol(
    symbol: Inexact[ArrayLike, " ..."],
    *,
    real_output: bool = True,
    tags: object | frozenset[object] = frozenset(),
) -> DiagonalisedOperator:
    r"""(Block-)circulant operator from its DFT symbol.

    ``A x = ifftn(symbol ⊙ fftn(x))`` on a periodic grid of shape
    ``symbol.shape``. Use this when the eigenvalues are known in closed form,
    e.g. the five-point Laplacian ``Σ_d (2 cos(2π k_d / n_d) − 2) / h_d²`` or
    the spectral ``−|k|²``. A real-dtype symbol gives a symmetric operator
    (tagged automatically); add ``lx.positive_semidefinite_tag`` /
    ``negative_semidefinite_tag`` yourself when it applies.

    With ``real_output=True`` the symbol must be conjugate-even,
    ``symbol[k] = conj(symbol[−k mod n])`` (for a real symbol: even), since
    only then is ``ifftn(symbol ⊙ fftn(x))`` real for real ``x``. Any other
    symbol would make ``mv`` apply the real part of the operator while
    ``solve`` / ``logdet`` / ``inv`` / ``sqrt`` describe the complex one.

    Args:
        symbol: Eigenvalues in `numpy.fft` frequency order, shape = grid shape.
        real_output: Take the real part of ``mv`` (default ``True``).
        tags: Additional lineax tags.

    Returns:
        A `DiagonalisedOperator` with the FFT transform pair.

    Raises:
        ValueError: If ``real_output`` and a concrete ``symbol`` is not
            conjugate-even. A traced symbol is checked at run time and
            raises an ``EquinoxRuntimeError`` instead.
    """
    symbol = jnp.asarray(symbol)
    if real_output:
        symbol = _check_conjugate_even(symbol)
    return DiagonalisedOperator(
        symbol,
        _fftn,
        _ifftn,
        symbol.shape,
        real_output=real_output,
        normal=True,
        tags=tags,
    )


def Circulant(
    first_column: Inexact[ArrayLike, " ..."],
    *,
    symmetric: bool | None = None,
    tags: object | frozenset[object] = frozenset(),
) -> DiagonalisedOperator:
    r"""(Block-)circulant operator from its first column.

    A circulant matrix ``C[i, j] = c[(i − j) mod n]`` (and its n-D
    block-circulant analogue, with ``first_column`` of the grid's shape) is
    diagonalised by the DFT: ``C = F⁻¹ diag(F c) F``. Periodic
    finite-difference stencils and stationary covariances on periodic grids
    are circulant.

    Compare `gaussx.Toeplitz`, which represents a (non-periodic) symmetric
    Toeplitz matrix with an FFT matvec by circulant embedding but has no
    closed-form solve; a circulant matrix has exact elementwise
    solve / logdet / sqrt.

    A real even kernel, ``c[k] = c[−k mod n]``, has a real symbol and gives a
    symmetric operator. Say so with ``symmetric=True``: the decision then
    depends only on the flag, so construction under `jax.jit` gives the same
    eigenvalue dtype and tags as eager construction (gh-440).

    Args:
        first_column: ``c``, shape = grid shape (``(n,)`` in 1D).
        symmetric: ``True`` for a real even kernel: the symbol is real and
            the operator is tagged symmetric (a concrete column that is not
            even raises). ``False`` keeps the complex symbol. ``None`` (the
            deprecated default) infers it from the column's *values*, which
            only works eagerly — a traced column gives a complex symbol and
            no tag — and warns when it infers ``True``. The default will
            become ``False``.
        tags: Additional lineax tags.

    Returns:
        A `DiagonalisedOperator` with eigenvalues ``fftn(c)``.

    Raises:
        ValueError: If ``symmetric=True`` and the column is complex, or
            concrete and not even.
    """
    column = jnp.asarray(first_column)
    symbol = jnp.fft.fftn(column)
    if symmetric is None:
        symmetric = _is_even_real_kernel(column)
        if symmetric:
            warn_deprecated(
                "Circulant infers symmetry from the column's values, so a traced "
                "column gives a different operator (gh-440). Pass symmetric=True "
                "for a real even kernel; the default will become False."
            )
    elif symmetric:
        _check_even_real_kernel(column)
    if symmetric:
        # c[k] = c[−k] (mod n) and real ⇒ real symbol ⇒ symmetric operator.
        symbol = jnp.real(symbol)
    return circulant_from_symbol(
        symbol,
        real_output=not jnp.iscomplexobj(column),
        tags=tags,
    )


_NOT_CONJUGATE_EVEN = (
    "circulant_from_symbol: symbol is not conjugate-even "
    "(symbol[k] != conj(symbol[-k mod n])), so Re(ifftn(symbol * fftn(x))) is "
    "not diagonalised by the FFT; pass real_output=False, or symmetrise the "
    "symbol as (symbol[k] + conj(symbol[-k mod n])) / 2."
)


def _reflect(x: Array) -> Array:
    """``x[−k mod n]`` along every axis."""
    for axis in range(x.ndim):
        x = jnp.roll(jnp.flip(x, axis=axis), 1, axis=axis)
    return x


def _check_conjugate_even(symbol: Array) -> Array:
    """Reject a symbol whose real-output circulant is inconsistent (gh-325).

    The tolerance is relative to the largest eigenvalue, so a symbol built
    from ``cos`` in float32 passes despite its roundoff.
    """
    defect = jnp.max(jnp.abs(symbol - jnp.conj(_reflect(symbol))), initial=0.0)
    scale = jnp.max(jnp.abs(symbol), initial=0.0)
    tol = 1e3 * jnp.finfo(jnp.result_type(symbol, jnp.float32)).eps
    bad = defect > tol * scale
    try:
        concrete_bad = bool(bad)
    except jax.errors.ConcretizationTypeError:
        return eqx.error_if(symbol, bad, _NOT_CONJUGATE_EVEN)
    if concrete_bad:
        raise ValueError(_NOT_CONJUGATE_EVEN)
    return symbol


def _check_even_real_kernel(column: Array) -> None:
    """Reject a complex column, or a concrete one that is not even (gh-440)."""
    if jnp.iscomplexobj(column):
        raise ValueError("Circulant(symmetric=True) needs a real column.")
    try:
        c = np.asarray(column)
    except jax.errors.TracerArrayConversionError:
        return
    # Round-off tolerance (an explicit flag, unlike the exact inference).
    reflected = c[np.ix_(*(np.negative(np.arange(n)) % n for n in c.shape))]
    tol = (
        1e3
        * np.finfo(np.result_type(c.dtype, np.float32)).eps
        * np.max(np.abs(c), initial=0.0)
    )
    if not np.allclose(c, reflected, rtol=0.0, atol=tol):
        raise ValueError(
            "Circulant(symmetric=True) needs an even column, c[k] == c[-k mod n]."
        )


def _is_even_real_kernel(column: Array) -> bool:
    """Whether a concrete real kernel satisfies ``c[k] = c[−k mod n]``.

    Returns ``False`` for traced or complex input (the symbol then stays
    complex and the operator is not auto-tagged symmetric).
    """
    try:
        c = np.asarray(column)
    except jax.errors.TracerArrayConversionError:
        return False
    if np.iscomplexobj(c):
        return False
    reflected = c[np.ix_(*(np.negative(np.arange(n)) % n for n in c.shape))]
    return bool(np.allclose(c, reflected, rtol=0.0, atol=0.0))


# ----------------------------------------------------------------------
# Kronecker sums of diagonalised factors
# ----------------------------------------------------------------------


def as_diagonalised(operator: lx.AbstractLinearOperator) -> DiagonalisedOperator | None:
    """Express ``operator`` as one `DiagonalisedOperator`, if possible.

    Returns the operator itself for a `DiagonalisedOperator`, the composed
    operator for a `gaussx.KroneckerSum` whose factors are (recursively)
    diagonalisable, and ``None`` otherwise.

    For ``A ⊕ B`` with ``A = V_A⁻¹ Λ_A V_A`` and ``B = V_B⁻¹ Λ_B V_B``,
    ``A ⊕ B = (V_A ⊗ V_B)⁻¹ (Λ_A ⊕ Λ_B) (V_A ⊗ V_B)``: the forward transform
    applies ``V_A`` along the leading axis of the ``(n_A, n_B)`` field and
    ``V_B`` along the trailing one, and the eigenvalues are all pairwise sums
    ``λ^A_i + λ^B_j``.
    """
    from gaussx._operators._kronecker_sum import KroneckerSum

    if isinstance(operator, DiagonalisedOperator):
        return operator
    if not isinstance(operator, KroneckerSum):
        return None
    a = as_diagonalised(operator.A)
    b = as_diagonalised(operator.B)
    if a is None or b is None:
        return None
    lam = jnp.add.outer(a.eigenvalues_flat(), b.eigenvalues_flat())
    real_output = a.real_output and b.real_output
    return DiagonalisedOperator(
        lam,
        ft.partial(_kron_forward, a, b),
        ft.partial(_kron_inverse, a, b),
        (a.size, b.size),
        real_output=real_output,
        normal=a.normal and b.normal,
    )


def _kron_forward(a: DiagonalisedOperator, b: DiagonalisedOperator, x: Array) -> Array:
    x = _along_axis(a.forward_flat, x, axis=0)
    return _along_axis(b.forward_flat, x, axis=1)


def _kron_inverse(a: DiagonalisedOperator, b: DiagonalisedOperator, c: Array) -> Array:
    c = _along_axis(b.inverse_flat, c, axis=1)
    return _along_axis(a.inverse_flat, c, axis=0)


def _along_axis(transform: Transform, x: Array, axis: int) -> Array:
    """Apply a flat-vector transform to every 1-D fibre along ``axis``.

    Real-to-real transforms (DCT/DST, real eigenvector matrices) may not
    accept complex input, which appears once an FFT axis has been
    transformed; being linear, they are applied to the real and imaginary
    parts separately.
    """
    mapped = jax.vmap(transform, in_axes=1 - axis, out_axes=1 - axis)
    if not jnp.iscomplexobj(x):
        return mapped(x)
    probe = jax.eval_shape(
        transform, jax.ShapeDtypeStruct((x.shape[axis],), jnp.real(x).dtype)
    )
    if jnp.issubdtype(probe.dtype, jnp.complexfloating):
        return mapped(x)
    return mapped(jnp.real(x)) + 1j * mapped(jnp.imag(x))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _Conjugated:
    """Callable ``x ↦ conj(f(conj(x)))``: the transform with matrix ``conj(V)``.

    Compares by the wrapped transform, so repeated transposes hash equal and
    do not trigger ``jit`` recompiles.
    """

    __slots__ = ("transform",)

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def __call__(self, x: Array) -> Array:
        return jnp.conj(self.transform(jnp.conj(x)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Conjugated) and other.transform == self.transform

    def __hash__(self) -> int:
        return hash((_Conjugated, self.transform))


def _conjugated(transform: Transform) -> Transform:
    if isinstance(transform, _Conjugated):
        return transform.transform
    return _Conjugated(transform)


class _MatrixApply:
    """Callable ``x ↦ M @ x`` used as a static transform.

    Hashes and compares by identity, like a closure, so it can sit in a
    static field; the matrix is embedded as a constant when traced.
    """

    __slots__ = ("matrix",)

    def __init__(self, matrix: Float[Array, "n n"]) -> None:
        self.matrix = matrix

    def __call__(self, x: Array) -> Array:
        return self.matrix @ x


def _flatten(x: Array) -> Array:
    names = " ".join(f"d{i}" for i in range(x.ndim))
    return rearrange(x, f"{names} -> ({names})") if x.ndim > 1 else x


def _unflatten(x: Array, shape: tuple[int, ...]) -> Array:
    if len(shape) <= 1:
        return x
    names = [f"d{i}" for i in range(len(shape))]
    joined = " ".join(names)
    return rearrange(
        x, f"({joined}) -> {joined}", **dict(zip(names, shape, strict=True))
    )


def _is_scalar(x: object) -> bool:
    return isinstance(x, int | float | complex) or (
        isinstance(x, jax.Array | jnp.ndarray) and jnp.ndim(x) == 0
    )


def _identity_multiple(other: object):
    """``c`` if ``other`` is ``c · I`` in lineax form, else ``None``."""
    if isinstance(other, lx.IdentityLinearOperator):
        return 1.0
    if isinstance(other, lx.MulLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        return other.scalar
    if isinstance(other, lx.DivLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        return 1.0 / other.scalar
    if isinstance(other, lx.NegLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        return -1.0
    return None
