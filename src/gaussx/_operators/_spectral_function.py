"""Spectral functions f(A) = W⁻¹ diag(f(γ)) W of a factored eigenbasis.

A is symmetric with a factored eigenbasis, typically a Kronecker sum
A = A_1 ⊕ … ⊕ A_D of symmetric N_d × N_d factors
A_d = Q_d diag(γ_d) Q_dᵀ, d = 1, …, D, with orthonormal eigenvectors Q_d and
eigenvalues γ_d. A function of a symmetric operator shares its
eigenvectors, and a Kronecker sum's eigenvectors are the Kronecker product
of its factors', so with the forward transform W = Q_1ᵀ ⊗ … ⊗ Q_Dᵀ

    f(A) = W⁻¹ diag(f(γ)) W,   γ_i₁…i_D = γ_1,i₁ + … + γ_D,i_D

and every operation is a per-axis rotation, an elementwise function of the
eigenvalue tensor γ ∈ ℝ^{N_1 × … × N_D}, and the rotation back. The SPDE
(Matérn) precision on a regular grid is such a function
(`gaussx.spde_precision_grid`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._linalg._eigen_factorization import EigenDecomposition
from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._factored_eigen import FactoredEigen, factored_eigen
from gaussx._operators._kronecker_sum import KroneckerSum


class SpectralFunction(lx.AbstractLinearOperator):
    """f(A) = W⁻¹ diag(f(γ)) W for a symmetric A with a factored eigenbasis.

        f(A_1 ⊕ … ⊕ A_D) = W⁻¹ diag(f(γ)) W,   W = Q_1ᵀ ⊗ … ⊗ Q_Dᵀ

    A is typically a Kronecker sum A_1 ⊕ … ⊕ A_D of symmetric factors (a grid
    Laplacian L_H ⊕ L_W), so `mv`, `solve`, `logdet`, `diag_inv` and
    `sqrt_matmul` act on the (N_1, …, N_D) grid one axis at a time, in
    O(N ∑_d N_d) for N = ∏_d N_d, and the joint eigenbasis is never formed.
    The factor eigendecompositions are computed once, at construction.

    `gaussx.solve`, `gaussx.logdet`, `gaussx.diag_inv` (including
    ``pinv=True``) and `gaussx.diag` dispatch here, and a `SpectralFunction`
    factor of a shifted Kronecker product A_1 ⊗ f(A) + c I keeps its
    eigenbasis there too.

    Args:
        base: The symmetric operator A (a `gaussx.KroneckerSum`, a
            `gaussx.DiagonalizedOperator`, a diagonal or a symmetric dense
            operator). Its factors are eigendecomposed with ``eigh``; build
            it from concrete values (outside ``jit``), or use
            `from_eigen_factorizations` to supply the decompositions.
        fn: The elementwise function f, applied to the eigenvalue tensor γ
            of A. Must be real-valued on the spectrum. Pass an
            `equinox.Module` (or a `jax.tree_util.Partial`) when it carries
            parameters that are traced or differentiated, so they stay
            pytree leaves.
        tags: lineax tags. ``symmetric_tag`` is always added.

    Raises:
        ValueError: If ``base`` has no factored eigenbasis (it is not
            symmetric).

    Examples:
        (κ² I + L_H ⊕ L_W)², a Matérn ν = 1 precision on a 4 × 5 grid:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def path_laplacian(n):
        ...     off = -jnp.ones(n - 1)
        ...     main = jnp.diag(jnp.r_[1.0, 2.0 * jnp.ones(n - 2), 1.0])
        ...     L = main + jnp.diag(off, 1) + jnp.diag(off, -1)
        ...     return lx.MatrixLinearOperator(L, lx.symmetric_tag)
        >>> base = gaussx.KroneckerSum(path_laplacian(4), path_laplacian(5))
        >>> Q = gaussx.SpectralFunction(base, lambda gamma: (0.5 + gamma) ** 2)
        >>> Q.in_size()
        20
        >>> dense = jnp.linalg.matrix_power(0.5 * jnp.eye(20) + base.as_matrix(), 2)
        >>> bool(jnp.allclose(Q.as_matrix(), dense, atol=1e-4))
        True

        The diagonal of the inverse comes from the factor bases:

        >>> reference = jnp.diag(jnp.linalg.inv(dense))
        >>> bool(jnp.allclose(gaussx.diag_inv(Q), reference, rtol=1e-4))
        True

        A non-symmetric base has no factored eigenbasis:

        >>> M = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [0.0, 1.0]]))
        >>> gaussx.SpectralFunction(M, jnp.exp)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: SpectralFunction needs a symmetric base
    """

    base: lx.AbstractLinearOperator
    fn: Callable[[Array], Array]
    eigen: FactoredEigen
    _size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        fn: Callable[[Array], Array],
        *,
        tags: object | frozenset[object] = frozenset(),
        _eigen: FactoredEigen | None = None,
    ) -> None:
        if _eigen is None:
            # A = W⁻¹ diag(γ) W, factor by factor
            _eigen = factored_eigen(base)
            if _eigen is None:
                raise ValueError(
                    "SpectralFunction needs a symmetric base with a factored "
                    f"eigenbasis, got {type(base).__name__}."
                )
        self.base = base
        self.fn = fn
        self.eigen = _eigen
        # N = ∏_d N_d
        self._size = int(_eigen.eigenvalues.size)
        # dtype of f(γ), promoted with γ
        spectrum = jax.eval_shape(fn, _eigen.eigenvalues)
        self._dtype = str(jnp.result_type(spectrum.dtype, _eigen.eigenvalues.dtype))
        self.tags = _to_frozenset(tags) | {lx.symmetric_tag}

    @classmethod
    def from_eigen_factorizations(
        cls,
        factors: Sequence[EigenDecomposition],
        fn: Callable[[Array], Array],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> SpectralFunction:
        """f(A_1 ⊕ … ⊕ A_D) from precomputed symmetric factorizations.

            A_d = Q_d diag(γ_d) Q_dᵀ,   γ_i₁…i_D = γ_1,i₁ + … + γ_D,i_D

        Skips the ``eigh`` of `__init__`, so it is cheap inside ``jit``; the
        base A is the nested `gaussx.KroneckerSum` of the reassembled
        factors.

        Args:
            factors: One `gaussx.EigenDecomposition` A_d = Q_d diag(γ_d) Q_dᵀ
                per axis, in Kronecker (row-major) order, each of a
                symmetric matrix (orthonormal eigenvectors, as from
                ``from_matrix(..., symmetric=True)``).
            fn: The elementwise function f (see the class docstring).
            tags: lineax tags.

        Returns:
            The operator f(A).

        Raises:
            ValueError: If ``factors`` is empty.

        Examples:
            f(A) = I + A for a 2 × 3 grid Laplacian:

            >>> import jax.numpy as jnp
            >>> import gaussx
            >>> L2 = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
            >>> L3 = jnp.array(
            ...     [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
            ... )
            >>> factors = [
            ...     gaussx.EigenDecomposition.from_matrix(L, symmetric=True)
            ...     for L in (L2, L3)
            ... ]
            >>> fA = gaussx.SpectralFunction.from_eigen_factorizations(
            ...     factors, lambda gamma: 1.0 + gamma
            ... )
            >>> dense = jnp.eye(6) + jnp.kron(L2, jnp.eye(3)) + jnp.kron(jnp.eye(2), L3)
            >>> bool(jnp.allclose(fA.as_matrix(), dense, atol=1e-5))
            True
        """
        if not factors:
            raise ValueError("factors must contain at least one factorization.")
        # Q_1, …, Q_D
        bases = tuple(f.eigenvectors for f in factors)
        eigenvalues = None
        for factor in factors:
            if eigenvalues is None:
                # γ_1
                eigenvalues = factor.eigenvalues
                continue
            names = " ".join(f"a{m}" for m in range(eigenvalues.ndim))
            # γ_i₁…i_d = γ_i₁…i_d−1 + γ_d,i_d
            eigenvalues = einx.add(
                f"{names}, b -> {names} b", eigenvalues, factor.eigenvalues
            )
        assert eigenvalues is not None
        # A_d = Q_d diag(γ_d) Q_dᵀ, reassembled
        operators = [
            lx.MatrixLinearOperator(f.as_matrix(), lx.symmetric_tag) for f in factors
        ]
        # A = A_1 ⊕ (A_2 ⊕ (… ⊕ A_D))
        base = operators[-1]
        for operator in reversed(operators[:-1]):
            base = KroneckerSum(operator, base)
        return cls(base, fn, tags=tags, _eigen=FactoredEigen(bases, eigenvalues))

    def spectrum(self) -> Float[Array, " ..."]:
        """Eigenvalues f(γ) of f(A), with the grid shape (N_1, …, N_D).

        Returns:
            f applied to the eigenvalue tensor γ, in the operator's dtype.
        """
        # f(γ)
        return jnp.asarray(self.fn(self.eigen.eigenvalues), dtype=self._dtype)

    def factored_eigen(self) -> FactoredEigen:
        """Factored eigendecomposition f(A) = W⁻¹ diag(f(γ)) W.

        Returns:
            A `FactoredEigen` in the basis W of A with eigenvalues f(γ).
        """
        return FactoredEigen(self.eigen.bases, self.spectrum())

    def _apply(self, vector: Array, values: Array) -> Array:
        """W⁻¹ diag(v) W x for an eigenvalue tensor v in the basis of A.

        Args:
            vector: Input x, of length N.
            values: Eigenvalue tensor v, of shape (N_1, …, N_D).

        Returns:
            W⁻¹ (v ⊙ W x), of length N.
        """
        # W x
        coeffs = self.eigen.to_eigenbasis(vector)
        # v ⊙ W x
        scaled = coeffs * values
        # W⁻¹ (v ⊙ W x)
        return self.eigen._output(self.eigen.from_eigenbasis(scaled), vector)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply f(A) to a vector in the factored eigenbasis.

            f(A) x = W⁻¹ (f(γ) ⊙ W x)

        Args:
            vector: Input x, of length N.

        Returns:
            f(A) x, of length N.
        """
        return self._apply(vector, self.spectrum())

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Solve f(A) x = b in the factored eigenbasis.

            f(A)⁻¹ b = W⁻¹ (W b / f(γ))

        Args:
            vector: Right-hand side b, of length N.

        Returns:
            x = f(A)⁻¹ b, of length N.
        """
        # 1 / f(γ)
        reciprocal = 1.0 / self.spectrum()
        return self._apply(vector, reciprocal)

    def logdet(self) -> Float[Array, ""]:
        """log|det f(A)| = ∑ log|f(γ)|, in the ``slogdet`` convention.

        Returns:
            The log-absolute-determinant, a scalar.
        """
        # |f(γ)|
        magnitude = jnp.abs(self.spectrum())
        # ∑ log|f(γ)|
        return jnp.sum(jnp.log(magnitude))

    def diag_inv(self, *, pinv: bool = False) -> Float[Array, " n"]:
        """Diagonal of the inverse, diag(f(A)⁻¹), or diag(f(A)⁺) with ``pinv``.

            [f(A)⁻¹]_nn = ∑_k [W⁻¹]_nk [W]_kn / f(γ)_k

        Args:
            pinv: Use the pseudo-inverse: eigenvalues f(γ)_k below a round-off
                threshold contribute nothing.

        Returns:
            The diagonal, of length N, from the factor bases.
        """
        return self.factored_eigen().diag_inv(pinv=pinv)

    def diagonal(self) -> Float[Array, " n"]:
        """Diagonal of f(A) from the factor bases.

            [f(A)]_nn = ∑_k [W⁻¹]_nk [W]_kn f(γ)_k

        Returns:
            diag(f(A)), of length N.
        """
        # f(A) = (W⁻¹ diag(1 / f(γ)) W)⁻¹, so diag(f(A)) is its diag_inv
        reciprocal = 1.0 / self.spectrum()
        return FactoredEigen(self.eigen.bases, reciprocal).diag_inv()

    def sqrt_matmul(
        self, vector: Float[Array, " n"], *, inverse: bool = False
    ) -> Float[Array, " n"]:
        """Symmetric root f(A)^½ x, or f(A)^−½ x with ``inverse=True``.

            f(A)^½ x = W⁻¹ (√f(γ) ⊙ W x)

        With ``inverse=True`` and z ~ 𝒩(0, I), f(A)^−½ z is an exact draw
        with precision f(A).

        Args:
            vector: Input x, of length N.
            inverse: Apply the inverse root f(A)^−½ instead.

        Returns:
            f(A)^½ x (or f(A)^−½ x), of length N.
        """
        # √f(γ)
        root = jnp.sqrt(self.spectrum())
        # √f(γ), or 1 / √f(γ) for the inverse root
        return self._apply(vector, 1.0 / root if inverse else root)

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense f(A), N × N, one row f(A) e_n per basis vector.

        Returns:
            The materialised operator.
        """
        # I_N
        eye = jnp.eye(self._size, dtype=self._dtype)
        # Rows are f(A) e_n, i.e. columns of the symmetric matrix.
        return rearrange(jax.vmap(self.mv)(eye), "col row -> row col")

    def transpose(self) -> SpectralFunction:
        """f(A)ᵀ = f(A), since f(A) is symmetric.

        Returns:
            The operator itself.
        """
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))


# ---------------------------------------------------------------------------
# lineax registrations
# ---------------------------------------------------------------------------

lx.is_symmetric.register(SpectralFunction)(lambda operator: True)
for _check, _tag in (
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
):
    _check.register(SpectralFunction)(lambda operator, tag=_tag: tag in operator.tags)
for _check in (
    lx.is_diagonal,
    lx.is_tridiagonal,
    lx.is_lower_triangular,
    lx.is_upper_triangular,
    lx.has_unit_diagonal,
):
    _check.register(SpectralFunction)(lambda operator: False)

lx.linearise.register(SpectralFunction)(lambda operator: operator)
lx.materialise.register(SpectralFunction)(lambda operator: operator)
lx.diagonal.register(SpectralFunction)(lambda operator: operator.diagonal())


__all__ = ["SpectralFunction"]
