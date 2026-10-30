"""Operators diagonal in a fast transform basis: A = W⁻¹ diag(γ) W.

W is the forward transform (field → coefficients), W⁻¹ its inverse and γ the
eigenvalues in the coefficient layout, so A x = W⁻¹ (γ ⊙ W x). Every
constant-coefficient operator on a regular grid with periodic, Dirichlet or
Neumann boundaries, and every spectral-method Laplacian, is diagonalised by
a fast transform (the discrete Fourier transform (DFT), a DCT or DST,
spherical harmonics, or a precomputed eigenvector matrix).
`DiagonalizedOperator` holds the pair W, W⁻¹ and γ, so `solve`, `logdet`,
`inv`, `sqrt` and `trace` are elementwise in γ and nothing is materialised.
A circulant matrix C with first column c is the DFT case,
C = DFT⁻¹ diag(DFT(c)) DFT.
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
from gaussx._operators._kronecker_sum import KroneckerSum


Transform = Callable[[Array], Array]


class DiagonalizedOperator(lx.AbstractLinearOperator):
    """Linear operator A = W⁻¹ diag(γ) W given by a transform pair.

        A x = W⁻¹ (γ ⊙ W x)

    ``forward`` applies W (a field of shape ``in_shape`` to coefficients) and
    ``inverse`` applies W⁻¹ (coefficients to a field); ``eigenvalues`` holds
    γ in the coefficient layout. A acts on **flat** vectors of length
    N = ``prod(in_shape)``, as every lineax operator does; ``mv`` reshapes
    internally.

    Examples of (W, W⁻¹, γ):

    - Periodic grid: ``fftn`` / ``ifftn`` and the symbol of the stencil; see
      `circulant` and `circulant_from_symbol`.
    - Dirichlet or Neumann grid: an orthonormal DST or DCT pair and the
      matching finite-difference or spectral eigenvalues.
    - A dense diagonalisable matrix A = V diag(γ) V⁻¹ (a Chebyshev
      collocation block, say), with W = V⁻¹:
      `DiagonalizedOperator.from_eigen_factorization`.

    **Algebra stays closed.** A + c I, A − c I, c A, A / c, −A and A ± B
    (for B with the same transform pair) return another
    `DiagonalizedOperator` with updated eigenvalues, so a shifted Helmholtz
    operator A − c I is never materialised.

    **Structured primitives.** `gaussx.solve` divides by γ (eigenvalues that
    are exactly zero get a zero coefficient, the pseudo-inverse for a normal
    operator); `gaussx.logdet` is ∑_i log|γ_i| (the ``slogdet`` convention,
    so ``-inf`` for a singular operator); `gaussx.inv`, `gaussx.sqrt` and
    `gaussx.trace` are elementwise in γ. A `gaussx.KroneckerSum` whose
    factors are all `DiagonalizedOperator` s is solved through the composed
    per-axis transforms.

    **Transpose.** With ``normal=True`` (W⁻¹ ∝ Wᴴ, as for the FFT or an
    orthonormal DCT or DST), Aᵀ of a real operator (``real_output=True``)
    is the same pair with conjugated eigenvalues γ̄; for a complex one it is
    the same γ with the conjugated pair conj(W), conj(W)⁻¹, since
    Aᵀ x = conj(Aᴴ conj(x)). Otherwise pass
    ``transpose_pair=(forward_T, inverse_T)`` applying W⁻ᵀ and Wᵀ in the
    same roles; without either, `transpose` raises.

    Args:
        eigenvalues: Eigenvalues γ in the coefficient layout;
            ``eigenvalues.size`` must equal N = ``prod(in_shape)``. May be
            complex.
        forward: The forward transform W, from an array of shape
            ``in_shape`` to a coefficient array.
        inverse: The inverse transform W⁻¹, from a coefficient array to an
            array of shape ``in_shape``.
        in_shape: Field shape the transforms act on.
        real_output: Return the real part of ``mv`` (default ``True``), as
            needed when W is complex but A is real (the FFT). The primitives
            work on γ directly, so this is only consistent when
            W⁻¹ diag(γ) W is itself real (for the FFT: a conjugate-even γ);
            otherwise ``mv`` applies Re(A) while ``solve`` and ``logdet``
            describe A. The generic class cannot check an arbitrary
            transform pair; `circulant_from_symbol` does.
        normal: Whether W⁻¹ is proportional to Wᴴ (see Transpose).
        transpose_pair: Optional ``(forward_T, inverse_T)``, applying W⁻ᵀ
            and Wᵀ, for Aᵀ.
        tags: lineax tags. ``symmetric_tag`` is added automatically when
            ``normal`` and ``real_output`` are set and the eigenvalues have a
            real dtype. (Real eigenvalues in a unitary basis make A
            Hermitian; it is also symmetric only when it is real.)

    The transform callables are stored as static fields, so they must be
    hashable (module-level functions, ``functools.partial`` of them, or
    closures); keep them stable between calls to avoid ``jit`` recompiles.

    Raises:
        ValueError: If ``eigenvalues.size`` is not ``prod(in_shape)``.

    Examples:
        The periodic second difference plus a shift, diagonalised by the FFT:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> n = 8
        >>> frequencies = jnp.fft.fftfreq(n)
        >>> gamma = 2.0 * jnp.cos(2.0 * jnp.pi * frequencies) - 2.0 - 1.0
        >>> A = gaussx.DiagonalizedOperator(
        ...     gamma, jnp.fft.fftn, jnp.fft.ifftn, (n,), normal=True
        ... )
        >>> A.as_matrix().shape
        (8, 8)

        The solve divides by γ in the transform basis:

        >>> b = jnp.linspace(-1.0, 1.0, n)
        >>> x = gaussx.solve(A, b)
        >>> bool(jnp.allclose(A.mv(x), b, atol=1e-5))
        True

        Shifts stay in the same basis:

        >>> type(3.0 * A).__name__
        'DiagonalizedOperator'
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
        # N = ∏_d N_d
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
        # real γ in a unitary basis, real output ⇒ A = Aᵀ
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
    ) -> DiagonalizedOperator:
        """Operator A = V diag(γ) V⁻¹ from a `gaussx.EigenDecomposition`.

            A = W⁻¹ diag(γ) W,   W = V⁻¹

        The dense eigenvector matrices are the transform pair (W = V⁻¹ as
        ``forward``, W⁻¹ = V as ``inverse``), so non-symmetric but
        diagonalisable factors, such as Chebyshev collocation
        second-derivative blocks, get an exact structured solve and
        transpose (W⁻ᵀ = Vᵀ, Wᵀ = V⁻ᵀ).

        Args:
            factorization: The eigendecomposition A = V diag(γ) V⁻¹, with
                eigenvectors V and their inverse V⁻¹.
            tags: Extra lineax tags for A.

        Returns:
            A `DiagonalizedOperator` on vectors of length N, the size of V.

        Examples:
            A non-symmetric 2 × 2 matrix, transposed exactly:

            >>> import jax.numpy as jnp
            >>> import gaussx
            >>> M = jnp.array([[2.0, 1.0], [0.0, 3.0]])
            >>> eig = gaussx.EigenDecomposition.from_matrix(M)
            >>> A = gaussx.DiagonalizedOperator.from_eigen_factorization(eig)
            >>> bool(jnp.allclose(A.as_matrix(), M, atol=1e-5))
            True
            >>> bool(jnp.allclose(A.T.as_matrix(), M.T, atol=1e-5))
            True
        """
        V = factorization.eigenvectors
        V_inv = factorization.eigenvectors_inv
        n = V.shape[0]
        return cls(
            factorization.eigenvalues,
            # W = V⁻¹
            forward=_MatrixApply(V_inv),
            # W⁻¹ = V
            inverse=_MatrixApply(V),
            in_shape=(n,),
            # W⁻ᵀ = Vᵀ, Wᵀ = V⁻ᵀ
            transpose_pair=(_MatrixApply(V.T), _MatrixApply(V_inv.T)),
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Flat-vector transforms (used by solves and Kronecker-sum composition)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of degrees of freedom N = ``prod(in_shape)``."""
        return int(self.eigenvalues.size)

    def forward_flat(self, vector: Shaped[Array, " n"]) -> Shaped[Array, " n"]:
        """Forward transform W on a flat vector, returning flat coefficients.

        Args:
            vector: Flat field x, of length N.

        Returns:
            W x, flattened to length N.
        """
        return _flatten(self.forward(_unflatten(vector, self.in_shape)))

    def inverse_flat(self, coeffs: Shaped[Array, " n"]) -> Shaped[Array, " n"]:
        """Inverse transform W⁻¹ on flat coefficients, returning a flat vector.

        Args:
            coeffs: Flat coefficients, of length N.

        Returns:
            W⁻¹ applied to the coefficients, flattened to length N.
        """
        return _flatten(
            self.inverse(_unflatten(coeffs, self.eigenvalues.shape)),
        )

    def eigenvalues_flat(self) -> Inexact[Array, " n"]:
        """Eigenvalues γ flattened to match `forward_flat`.

        Returns:
            γ as a vector of length N.
        """
        return _flatten(self.eigenvalues)

    def with_eigenvalues(
        self, eigenvalues: Inexact[ArrayLike, " ..."]
    ) -> DiagonalizedOperator:
        """Same transform pair W, W⁻¹ with new eigenvalues γ′.

            A′ = W⁻¹ diag(γ′) W

        Structural tags are recomputed: the automatic ``symmetric_tag`` is
        re-derived, and definiteness tags (``positive_semidefinite_tag``,
        ``negative_semidefinite_tag``) are dropped because a shift or sign
        change can invalidate them. Re-tag the result if you know it holds.

        Args:
            eigenvalues: New eigenvalues γ′, in the layout of γ.

        Returns:
            The operator A′ in the same basis.
        """
        user_tags = self.tags - {
            lx.symmetric_tag,
            lx.positive_semidefinite_tag,
            lx.negative_semidefinite_tag,
        }
        return DiagonalizedOperator(
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
        """Apply A to a flat vector in the transform basis.

            A x = W⁻¹ (γ ⊙ W x)

        Args:
            vector: Flat input x, of length N.

        Returns:
            A x, of length N; its real part when ``real_output`` is set.
        """
        # W x, on the field of shape in_shape
        coeffs = self.forward(_unflatten(vector, self.in_shape))
        # γ ⊙ W x
        scaled = self.eigenvalues * coeffs
        # W⁻¹ (γ ⊙ W x), flattened
        out = _flatten(self.inverse(scaled))
        # Re(A x) when A is real
        return jnp.real(out) if self.real_output else out

    def as_matrix(self) -> Inexact[Array, "n n"]:
        """Dense A, N × N, one column A e_n per basis vector.

        Returns:
            The materialised operator.
        """
        # I_N
        eye = jnp.eye(self.size, dtype=self.out_structure().dtype)
        # [A e_1, …, A e_N]
        return jax.vmap(self.mv, out_axes=1)(eye)

    def transpose(self) -> DiagonalizedOperator:
        """Aᵀ in a transform basis, still a `DiagonalizedOperator`.

            Aᵀ = W⁻¹ diag(γ̄) W              normal, real output
            Aᵀ = conj(W)⁻¹ diag(γ) conj(W)   normal, complex output
            Aᵀ = Wᵀ diag(γ) W⁻ᵀ              with ``transpose_pair``

        Returns:
            The transposed operator, with transposed tags.

        Raises:
            NotImplementedError: If the operator is neither normal nor has a
                ``transpose_pair``.
        """
        tags = lx.transpose_tags(self.tags)
        if self.normal and not self.real_output:
            # Conjugated eigenvalues would give Aᴴ, not Aᵀ (gh-330).
            # Aᵀ = conj(W)⁻¹ diag(γ) conj(W)
            return DiagonalizedOperator(
                self.eigenvalues,
                _conjugated(self.forward),
                _conjugated(self.inverse),
                self.in_shape,
                real_output=False,
                normal=True,
                tags=tags,
            )
        if self.normal:
            # Aᵀ = W⁻¹ diag(γ̄) W
            return DiagonalizedOperator(
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
                "DiagonalizedOperator.transpose needs normal=True or a transpose_pair."
            )
        forward_t, inverse_t = self.transpose_pair
        # Aᵀ = Wᵀ diag(γ) W⁻ᵀ, with W⁻ᵀ forward and Wᵀ inverse
        return DiagonalizedOperator(
            self.eigenvalues,
            forward_t,
            inverse_t,
            self.in_shape,
            real_output=self.real_output,
            transpose_pair=(self.forward, self.inverse),
            tags=tags,
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self.size,), self._dtype())

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self.size,), self._dtype())

    def _dtype(self):
        """Real dtype of γ with ``real_output``, else its complex promotion."""
        dtype = self.eigenvalues.dtype
        if self.real_output:
            return jnp.real(jnp.zeros((), dtype)).dtype
        return jnp.result_type(dtype, jnp.complex64)

    # ------------------------------------------------------------------
    # Closed algebra: shifts, scalings, sums in the same basis
    # ------------------------------------------------------------------

    def _same_basis(self, other: object) -> bool:
        """Whether ``other`` is a `DiagonalizedOperator` with the same W, W⁻¹."""
        return (
            isinstance(other, DiagonalizedOperator)
            and other.forward is self.forward
            and other.inverse is self.inverse
            and other.in_shape == self.in_shape
            and other.eigenvalues.shape == self.eigenvalues.shape
        )

    def __add__(self, other):
        """A + c I = W⁻¹ diag(γ + c) W, or A + B = W⁻¹ diag(γ_A + γ_B) W."""
        shift = _identity_multiple(other)
        if shift is not None:
            # γ + c
            return self.with_eigenvalues(self.eigenvalues + shift)
        if self._same_basis(other):
            # γ_A + γ_B
            return self.with_eigenvalues(self.eigenvalues + other.eigenvalues)
        return super().__add__(other)

    def __radd__(self, other):
        """c I + A, the same as A + c I."""
        return self.__add__(other)

    def __sub__(self, other):
        """A − c I = W⁻¹ diag(γ − c) W, or A − B = W⁻¹ diag(γ_A − γ_B) W."""
        shift = _identity_multiple(other)
        if shift is not None:
            # γ − c
            return self.with_eigenvalues(self.eigenvalues - shift)
        if self._same_basis(other):
            # γ_A − γ_B
            return self.with_eigenvalues(self.eigenvalues - other.eigenvalues)
        return super().__sub__(other)

    def __mul__(self, other):
        """A c = W⁻¹ diag(c γ) W for a scalar c."""
        if _is_scalar(other):
            # γ c
            return self.with_eigenvalues(self.eigenvalues * other)
        return super().__mul__(other)

    def __rmul__(self, other):
        """c A = W⁻¹ diag(c γ) W for a scalar c."""
        if _is_scalar(other):
            # γ c
            return self.with_eigenvalues(self.eigenvalues * other)
        return super().__rmul__(other)

    def __truediv__(self, other):
        """A / c = W⁻¹ diag(γ / c) W for a scalar c."""
        if _is_scalar(other):
            # γ / c
            return self.with_eigenvalues(self.eigenvalues / other)
        return super().__truediv__(other)

    def __neg__(self):
        """−A = W⁻¹ diag(−γ) W."""
        return self.with_eigenvalues(-self.eigenvalues)


# ----------------------------------------------------------------------
# Circulant operators
# ----------------------------------------------------------------------


def _fftn(x: Array) -> Array:
    """Forward DFT over every axis, the W of a circulant."""
    return jnp.fft.fftn(x)


def _ifftn(c: Array) -> Array:
    """Inverse DFT over every axis, the W⁻¹ of a circulant."""
    return jnp.fft.ifftn(c)


def circulant_from_symbol(
    symbol: Inexact[ArrayLike, " ..."],
    *,
    real_output: bool = True,
    tags: object | frozenset[object] = frozenset(),
) -> DiagonalizedOperator:
    """(Block-)circulant operator C = DFT⁻¹ diag(γ) DFT from its symbol γ.

        C x = DFT⁻¹(γ ⊙ DFT(x))

    The grid is periodic with the shape of γ. Use this when the eigenvalues
    are known in closed form, such as the five-point Laplacian
    ∑_d (2 cos(2π κ_d / N_d) − 2) / h_d² for integer frequency κ_d and
    spacing h_d on axis d, or the spectral −‖κ‖². A real-dtype γ gives a
    symmetric operator (tagged automatically); add
    ``lx.positive_semidefinite_tag`` or ``negative_semidefinite_tag``
    yourself when it applies.

    With ``real_output=True`` the symbol must be conjugate-even,
    γ_κ = conj(γ_−κ mod N) (for a real symbol: even), since only then is
    DFT⁻¹(γ ⊙ DFT(x)) real for real x. Any other symbol would make ``mv``
    apply the real part of the operator while ``solve``, ``logdet``,
    ``inv`` and ``sqrt`` describe the complex one.

    Args:
        symbol: Eigenvalues γ in `numpy.fft` frequency order, with the grid's
            shape.
        real_output: Take the real part of ``mv`` (default ``True``).
        tags: Additional lineax tags.

    Returns:
        A `DiagonalizedOperator` with the FFT transform pair.

    Raises:
        ValueError: If ``real_output`` and a concrete ``symbol`` is not
            conjugate-even. A traced symbol is checked at run time and
            raises an ``EquinoxRuntimeError`` instead.

    Examples:
        The periodic second difference, shifted to be negative definite:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> gamma = 2.0 * jnp.cos(2.0 * jnp.pi * jnp.fft.fftfreq(6)) - 3.0
        >>> C = gaussx.circulant_from_symbol(gamma)
        >>> row = jnp.array([-3.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        >>> bool(jnp.allclose(C.as_matrix()[0], row, atol=1e-5))
        True

        A symbol that is not even is rejected for a real output:

        >>> gaussx.circulant_from_symbol(jnp.arange(1.0, 5.0))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: circulant_from_symbol: symbol is not conjugate-even
    """
    symbol = jnp.asarray(symbol)
    if real_output:
        symbol = _check_conjugate_even(symbol)
    return DiagonalizedOperator(
        symbol,
        _fftn,
        _ifftn,
        symbol.shape,
        real_output=real_output,
        normal=True,
        tags=tags,
    )


def circulant(
    first_column: Inexact[ArrayLike, " ..."],
    *,
    symmetric: bool | None = None,
    tags: object | frozenset[object] = frozenset(),
) -> DiagonalizedOperator:
    """(Block-)circulant operator C from its first column c.

        [C]_ij = c_(i−j) mod N,   C = DFT⁻¹ diag(DFT(c)) DFT

    The n-D block-circulant analogue takes ``first_column`` with the grid's
    shape. Periodic finite-difference stencils and stationary covariances on
    periodic grids are circulant.

    Compare `gaussx.Toeplitz`, which represents a (non-periodic) symmetric
    Toeplitz matrix with an FFT matvec by circulant embedding but has no
    closed-form solve; a circulant matrix has an exact elementwise solve,
    logdet and sqrt.

    A real even kernel, c_k = c_−k mod N, has a real symbol and gives a
    symmetric operator. Say so with ``symmetric=True``: the decision then
    depends only on the flag, so construction under `jax.jit` gives the same
    eigenvalue dtype and tags as eager construction (gh-440).

    Args:
        first_column: The first column c, with the grid's shape ((N,) in
            1-D).
        symmetric: ``True`` for a real even kernel: the symbol is real and
            the operator is tagged symmetric (a concrete column that is not
            even raises). ``False`` keeps the complex symbol. ``None`` (the
            deprecated default) infers it from the column's *values*, which
            only works eagerly (a traced column gives a complex symbol and
            no tag) and warns when it infers ``True``. The default becomes
            ``False`` in gaussx 0.7.0.
        tags: Additional lineax tags.

    Returns:
        A `DiagonalizedOperator` with eigenvalues γ = DFT(c).

    Raises:
        ValueError: If ``symmetric=True`` and the column is complex, or
            concrete and not even.

    Examples:
        A symmetric circulant covariance on a periodic grid of 6 points:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> c = jnp.array([4.0, 1.0, 0.5, 0.2, 0.5, 1.0])
        >>> C = gaussx.circulant(c, symmetric=True)
        >>> lx.is_symmetric(C)
        True
        >>> [round(float(v), 4) for v in C.as_matrix()[:, 0]]
        [4.0, 1.0, 0.5, 0.2, 0.5, 1.0]

        Its log-determinant is ∑_κ log|γ_κ|, with no dense factorisation:

        >>> dense = jnp.linalg.slogdet(C.as_matrix())[1]
        >>> bool(jnp.allclose(gaussx.logdet(C), dense, atol=1e-4))
        True
    """
    column = jnp.asarray(first_column)
    # γ = DFT(c)
    symbol = jnp.fft.fftn(column)
    if symmetric is None:
        symmetric = _is_even_real_kernel(column)
        if symmetric:
            warn_deprecated(
                "circulant infers symmetry from the column's values, so a traced "
                "column gives a different operator (gh-440). Pass symmetric=True "
                "for a real even kernel; the default becomes False in gaussx 0.7.0."
            )
    elif symmetric:
        _check_even_real_kernel(column)
    if symmetric:
        # c_k = c_−k mod N and real ⇒ γ real ⇒ C symmetric
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
    """Reflection x_−k mod N along every axis.

    Args:
        x: Array on a periodic grid.

    Returns:
        The reflected array, with the shape of x.
    """
    for axis in range(x.ndim):
        # jnp, not einx: einx has no flip or cyclic shift
        x = jnp.roll(jnp.flip(x, axis=axis), 1, axis=axis)
    return x


def _check_conjugate_even(symbol: Array) -> Array:
    """Reject a symbol whose real-output circulant is inconsistent (gh-325).

        max_κ |γ_κ − conj(γ_−κ mod N)| ≤ 10³ ε max_κ |γ_κ|

    The tolerance is relative to the largest eigenvalue, so a symbol built
    from ``cos`` in float32 passes despite its roundoff.

    Args:
        symbol: Eigenvalues γ in `numpy.fft` frequency order.

    Returns:
        γ unchanged, wrapped in a run-time check when it is traced.

    Raises:
        ValueError: If a concrete γ is not conjugate-even.
    """
    # max_κ |γ_κ − conj(γ_−κ)|
    defect = jnp.max(jnp.abs(symbol - jnp.conj(_reflect(symbol))), initial=0.0)
    # max_κ |γ_κ|
    scale = jnp.max(jnp.abs(symbol), initial=0.0)
    # 10³ ε
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
    """Reject a complex column, or a concrete one that is not even (gh-440).

        |c_k − c_−k mod N| ≤ 10³ ε max_k |c_k|

    Args:
        column: First column c of a circulant.

    Raises:
        ValueError: If c is complex, or concrete and not even.
    """
    if jnp.iscomplexobj(column):
        raise ValueError("circulant(symmetric=True) needs a real column.")
    try:
        c = np.asarray(column)
    except jax.errors.TracerArrayConversionError:
        return
    # c_−k mod N along every axis
    reflected = c[np.ix_(*(np.negative(np.arange(n)) % n for n in c.shape))]
    # Round-off tolerance (an explicit flag, unlike the exact inference).
    tol = (
        1e3
        * np.finfo(np.result_type(c.dtype, np.float32)).eps
        * np.max(np.abs(c), initial=0.0)
    )
    if not np.allclose(c, reflected, rtol=0.0, atol=tol):
        raise ValueError(
            "circulant(symmetric=True) needs an even column, c[k] == c[-k mod n]."
        )


def _is_even_real_kernel(column: Array) -> bool:
    """Whether a concrete real kernel satisfies c_k = c_−k mod N exactly.

    Args:
        column: First column c of a circulant.

    Returns:
        ``True`` for a concrete, real, exactly even c; ``False`` for traced
        or complex input (the symbol then stays complex and the operator is
        not auto-tagged symmetric).
    """
    try:
        c = np.asarray(column)
    except jax.errors.TracerArrayConversionError:
        return False
    if np.iscomplexobj(c):
        return False
    # c_−k mod N along every axis
    reflected = c[np.ix_(*(np.negative(np.arange(n)) % n for n in c.shape))]
    return bool(np.allclose(c, reflected, rtol=0.0, atol=0.0))


# ----------------------------------------------------------------------
# Kronecker sums of diagonalised factors
# ----------------------------------------------------------------------


def as_diagonalized(operator: lx.AbstractLinearOperator) -> DiagonalizedOperator | None:
    """Express an operator as one `DiagonalizedOperator`, if possible.

        A_1 ⊕ A_2 = (W_1 ⊗ W_2)⁻¹ diag(γ_1 ⊕ γ_2) (W_1 ⊗ W_2)

    A `DiagonalizedOperator` comes back as itself; a `gaussx.KroneckerSum`
    whose factors are (recursively) diagonalisable, A_d = W_d⁻¹ diag(γ_d) W_d,
    is composed into one operator on the (N_1, N_2) field whose forward
    transform applies W_1 along the leading axis and W_2 along the trailing
    one, with all pairwise eigenvalue sums γ_1,i + γ_2,j.

    Args:
        operator: The operator to express in a transform basis.

    Returns:
        The equivalent `DiagonalizedOperator`, or ``None`` when ``operator``
        is neither a `DiagonalizedOperator` nor a `gaussx.KroneckerSum` of
        diagonalisable factors.

    Examples:
        The 2-D periodic Laplacian as a Kronecker sum of two 1-D ones:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> gamma = 2.0 * jnp.cos(2.0 * jnp.pi * jnp.fft.fftfreq(4)) - 2.0
        >>> A_1 = gaussx.circulant_from_symbol(gamma)
        >>> A = gaussx.as_diagonalized(gaussx.KroneckerSum(A_1, A_1))
        >>> A.in_shape
        (4, 4)

        A dense factor has no transform basis:

        >>> import lineax as lx
        >>> print(gaussx.as_diagonalized(lx.MatrixLinearOperator(jnp.eye(2))))
        None
    """

    if isinstance(operator, DiagonalizedOperator):
        return operator
    if not isinstance(operator, KroneckerSum):
        return None
    a = as_diagonalized(operator.A)
    b = as_diagonalized(operator.B)
    if a is None or b is None:
        return None
    # γ_1,i + γ_2,j
    # jnp, not einx: einx rejects the empty axis of a zero-size factor
    lam = jnp.add.outer(a.eigenvalues_flat(), b.eigenvalues_flat())
    real_output = a.real_output and b.real_output
    return DiagonalizedOperator(
        lam,
        # W_1 ⊗ W_2
        ft.partial(_kron_forward, a, b),
        # (W_1 ⊗ W_2)⁻¹ = W_1⁻¹ ⊗ W_2⁻¹
        ft.partial(_kron_inverse, a, b),
        (a.size, b.size),
        real_output=real_output,
        normal=a.normal and b.normal,
    )


def _kron_forward(a: DiagonalizedOperator, b: DiagonalizedOperator, x: Array) -> Array:
    """Forward transform (W_1 ⊗ W_2) X = W_1 X W_2ᵀ of an (N_1, N_2) field.

    Args:
        a: The first factor A_1, with transform W_1.
        b: The second factor A_2, with transform W_2.
        x: Field X of shape (N_1, N_2).

    Returns:
        The coefficients W_1 X W_2ᵀ.
    """
    # W_1 X, along the leading axis
    x = _along_axis(a.forward_flat, x, axis=0)
    # (W_1 X) W_2ᵀ, along the trailing axis
    return _along_axis(b.forward_flat, x, axis=1)


def _kron_inverse(a: DiagonalizedOperator, b: DiagonalizedOperator, c: Array) -> Array:
    """Inverse transform W_1⁻¹ Ĉ W_2⁻ᵀ of (N_1, N_2) coefficients Ĉ.

    Args:
        a: The first factor A_1, with transform W_1.
        b: The second factor A_2, with transform W_2.
        c: Coefficients Ĉ of shape (N_1, N_2).

    Returns:
        The field W_1⁻¹ Ĉ W_2⁻ᵀ.
    """
    # Ĉ W_2⁻ᵀ, along the trailing axis
    c = _along_axis(b.inverse_flat, c, axis=1)
    # W_1⁻¹ (Ĉ W_2⁻ᵀ), along the leading axis
    return _along_axis(a.inverse_flat, c, axis=0)


def _along_axis(transform: Transform, x: Array, axis: int) -> Array:
    """Apply a flat-vector transform to every 1-D fibre along ``axis``.

    Real-to-real transforms (DCT/DST, real eigenvector matrices) may not
    accept complex input, which appears once an FFT axis has been
    transformed; being linear, they are applied to the real and imaginary
    parts separately, W x = W Re(x) + i W Im(x).

    Args:
        transform: The linear flat-vector transform W_d.
        x: A 2-D array.
        axis: The axis (0 or 1) whose fibres W_d acts on.

    Returns:
        x with W_d applied along ``axis``.
    """
    mapped = jax.vmap(transform, in_axes=1 - axis, out_axes=1 - axis)
    if not jnp.iscomplexobj(x):
        return mapped(x)
    probe = jax.eval_shape(
        transform, jax.ShapeDtypeStruct((x.shape[axis],), jnp.real(x).dtype)
    )
    if jnp.issubdtype(probe.dtype, jnp.complexfloating):
        return mapped(x)
    # W x = W Re(x) + i W Im(x)
    return mapped(jnp.real(x)) + 1j * mapped(jnp.imag(x))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _Conjugated:
    """Callable x ↦ conj(W conj(x)): the transform with matrix conj(W).

    Compares by the wrapped transform, so repeated transposes hash equal and
    do not trigger ``jit`` recompiles.

    Attributes:
        transform: The wrapped transform W.
    """

    __slots__ = ("transform",)

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def __call__(self, x: Array) -> Array:
        """conj(W conj(x))."""
        return jnp.conj(self.transform(jnp.conj(x)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Conjugated) and other.transform == self.transform

    def __hash__(self) -> int:
        return hash((_Conjugated, self.transform))


def _conjugated(transform: Transform) -> Transform:
    """conj(W) as a transform, unwrapping a double conjugation.

    Args:
        transform: The transform W.

    Returns:
        x ↦ conj(W conj(x)), or the original W if ``transform`` is already
        conjugated.
    """
    if isinstance(transform, _Conjugated):
        return transform.transform
    return _Conjugated(transform)


class _MatrixApply:
    """Callable x ↦ M x used as a static transform.

    Hashes and compares by identity, like a closure, so it can sit in a
    static field; the matrix is embedded as a constant when traced.

    Attributes:
        matrix: The dense matrix M.
    """

    __slots__ = ("matrix",)

    def __init__(self, matrix: Float[Array, "n n"]) -> None:
        self.matrix = matrix

    def __call__(self, x: Array) -> Array:
        """M x."""
        return self.matrix @ x


def _flatten(x: Array) -> Array:
    """Row-major flattening vec(x) of an array of any rank.

    Args:
        x: Array of shape (N_1, …, N_D).

    Returns:
        x as a vector of length ∏_d N_d (x itself when it is rank ≤ 1).
    """
    names = " ".join(f"d{i}" for i in range(x.ndim))
    return rearrange(x, f"{names} -> ({names})") if x.ndim > 1 else x


def _unflatten(x: Array, shape: tuple[int, ...]) -> Array:
    """Inverse of `_flatten`: a flat vector back to ``shape``.

    Args:
        x: Flat vector of length ∏_d N_d.
        shape: Target shape (N_1, …, N_D).

    Returns:
        x in row-major order with shape ``shape`` (x itself when ``shape``
        has at most one axis).
    """
    if len(shape) <= 1:
        return x
    names = [f"d{i}" for i in range(len(shape))]
    joined = " ".join(names)
    return rearrange(
        x, f"({joined}) -> {joined}", **dict(zip(names, shape, strict=True))
    )


def _is_scalar(x: object) -> bool:
    """Whether x is a Python number or a 0-d array."""
    return isinstance(x, int | float | complex) or (
        isinstance(x, jax.Array | jnp.ndarray) and jnp.ndim(x) == 0
    )


def _identity_multiple(other: object):
    """The scalar c if ``other`` is c I in lineax form, else ``None``.

    Args:
        other: Any operand of an operator sum.

    Returns:
        c for I, c I, I / c′ (c = 1 / c′) or −I (c = −1); otherwise ``None``.
    """
    if isinstance(other, lx.IdentityLinearOperator):
        return 1.0
    if isinstance(other, lx.MulLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        return other.scalar
    if isinstance(other, lx.DivLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        # I / c′ = (1 / c′) I
        return 1.0 / other.scalar
    if isinstance(other, lx.NegLinearOperator) and isinstance(
        other.operator, lx.IdentityLinearOperator
    ):
        return -1.0
    return None
