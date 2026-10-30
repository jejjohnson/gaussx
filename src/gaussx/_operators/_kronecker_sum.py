"""Kronecker sum A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2 and its symmetric square root.

A_1 is N_1 × N_1 and A_2 is N_2 × N_2, so A_1 ⊕ A_2 is N × N with
N = N_1 N_2. vec(Z) flattens Z ∈ ℝ^{N_1 × N_2} row by row, so
(A_1 ⊕ A_2) vec(Z) = vec(A_1 Z + Z A_2ᵀ). A symmetric factor has the
eigendecomposition A_d = Q_d diag(γ_d) Q_dᵀ, with orthonormal eigenvectors
Q_d and eigenvalues γ_d; the sum then has eigenvectors Q_1 ⊗ Q_2 and
eigenvalues γ_1,i + γ_2,j. Its symmetric square root S, S² = A_1 ⊕ A_2, has
the same eigenvectors and eigenvalues s_ij = √(γ_1,i + γ_2,j), and
Z̃ = Q_1ᵀ Z Q_2 is Z in that eigenbasis.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._tags import kronecker_sum_tag


_NEGATIVE_EIGENVALUE_TOLERANCE_FACTOR = 100


class KroneckerSum(lx.AbstractLinearOperator):
    """Kronecker sum A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2 of two square operators.

        (A_1 ⊕ A_2) vec(Z) = vec(A_1 Z + Z A_2ᵀ)

    Appears in separable PDEs, graph Laplacians on grids, and space-time GPs.
    For symmetric factors A_d = Q_d diag(γ_d) Q_dᵀ the sum has eigenvectors
    Q_1 ⊗ Q_2 and eigenvalues γ_1,i + γ_2,j, which the solve, logdet and
    eigendecomposition primitives use without forming the N × N matrix.

    Args:
        A: First factor A_1, N_1 × N_1.
        B: Second factor A_2, N_2 × N_2.
        tags: Extra lineax tags for A_1 ⊕ A_2; ``kronecker_sum_tag`` is
            always added.

    Raises:
        ValueError: If either factor is not square.

    Examples:
        A 4 × 4 Kronecker sum of two 2 × 2 factors:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> S = gaussx.KroneckerSum(A, B)
        >>> I2 = jnp.eye(2)
        >>> dense = jnp.kron(A.as_matrix(), I2) + jnp.kron(I2, B.as_matrix())
        >>> bool(jnp.allclose(S.as_matrix(), dense))
        True

        The logdet sums log(γ_1,i + γ_2,j) over the factor eigenvalues:

        >>> sign, expected = jnp.linalg.slogdet(dense)
        >>> bool(jnp.allclose(gaussx.logdet(S), expected, atol=1e-5))
        True
    """

    A: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _n_a: int = eqx.field(static=True)
    _n_b: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        A: lx.AbstractLinearOperator,
        B: lx.AbstractLinearOperator,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if A.in_size() != A.out_size():
            raise ValueError(
                f"A must be square, got in_size={A.in_size()}, out_size={A.out_size()}."
            )
        if B.in_size() != B.out_size():
            raise ValueError(
                f"B must be square, got in_size={B.in_size()}, out_size={B.out_size()}."
            )
        self.A = A
        self.B = B
        n_a = A.in_size()
        n_b = B.in_size()
        self._n_a = n_a
        self._n_b = n_b
        # N = N_1 N_2
        self._in_size = n_a * n_b
        self._out_size = n_a * n_b
        self._dtype = _resolve_dtype(A, B)

        self.tags = _to_frozenset(tags) | {kronecker_sum_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply A_1 ⊕ A_2 to a vector through its two factors.

            (A_1 ⊕ A_2) vec(Z) = vec(A_1 Z + Z A_2ᵀ)

        Args:
            vector: Input vec(Z), of length N = N_1 N_2.

        Returns:
            (A_1 ⊕ A_2) vec(Z), of length N.
        """
        # Zᵀ ∈ ℝ^{N_2 × N_1}
        grid = rearrange(vector, "(a b) -> b a", a=self._n_a, b=self._n_b)
        # A_2 Zᵀ = (Z A_2ᵀ)ᵀ: A_2 on each column
        second_term = jax.vmap(self.B.mv, in_axes=1, out_axes=1)(grid)
        # Zᵀ A_1ᵀ = (A_1 Z)ᵀ: A_1 on each row
        first_term = jax.vmap(self.A.mv)(grid)
        # (A_1 Z + Z A_2ᵀ)ᵀ
        result = second_term + first_term
        # vec(A_1 Z + Z A_2ᵀ)
        return rearrange(result, "b a -> (a b)")

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense A_1 ⊗ I + I ⊗ A_2, N × N.

        Returns:
            The materialised Kronecker sum.
        """
        first_factor = self.A.as_matrix()
        second_factor = self.B.as_matrix()
        identity_first = jnp.eye(self._n_a, dtype=jnp.dtype(self._dtype))
        identity_second = jnp.eye(self._n_b, dtype=jnp.dtype(self._dtype))
        # A_1 ⊗ I + I ⊗ A_2
        return jnp.kron(first_factor, identity_second) + jnp.kron(
            identity_first, second_factor
        )

    def transpose(self) -> KroneckerSum:
        """(A_1 ⊕ A_2)ᵀ = A_1ᵀ ⊕ A_2ᵀ, still a `KroneckerSum`.

        Returns:
            The transposed sum, with transposed tags.
        """
        return KroneckerSum(
            self.A.T,
            self.B.T,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def eigendecompose(
        self,
    ) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
        """Symmetric eigendecomposition from the per-factor eigendecompositions.

            A_1 ⊕ A_2 = (Q_1 ⊗ Q_2) diag(γ_1 ⊕ γ_2) (Q_1 ⊗ Q_2)ᵀ

        Assumes both factors are symmetric, so that Q = Q_1 ⊗ Q_2 is
        orthonormal; callers rely on A_1 ⊕ A_2 = Q diag(γ) Qᵀ. A diagonal
        factor takes a structural shortcut; any other is materialised and
        decomposed with ``jnp.linalg.eigh``, deliberately not through
        `gaussx.eig`, which falls back to the general ``jnp.linalg.eig`` for
        an untagged operator and would return non-orthonormal eigenvectors
        for a numerically symmetric matrix wrapped as a plain
        `lineax.MatrixLinearOperator`.

        Algorithm:
            1. Eigendecompose each factor, A_d = Q_d diag(γ_d) Q_dᵀ.
            2. Form every pairwise sum γ_1,i + γ_2,j, flattened row-major.
            3. Form Q = Q_1 ⊗ Q_2.

        Returns:
            Tuple ``(eigenvalues, Q)`` with the N eigenvalues γ_1,i + γ_2,j
            and the N × N eigenvectors Q = Q_1 ⊗ Q_2.
        """
        # A_1 = Q_1 diag(γ_1) Q_1ᵀ
        eigenvalues_first, eigenvectors_first = _eigh_factor(self.A)
        # A_2 = Q_2 diag(γ_2) Q_2ᵀ
        eigenvalues_second, eigenvectors_second = _eigh_factor(self.B)
        # jnp, not einx: einx cannot resolve an empty factor (N_d = 0).
        # γ_ij = γ_1,i + γ_2,j
        pairwise = eigenvalues_first[:, None] + eigenvalues_second[None, :]
        # vec(γ)
        eigenvalues = rearrange(pairwise, "a b -> (a b)")
        # Q = Q_1 ⊗ Q_2
        eigenvectors = jnp.kron(eigenvectors_first, eigenvectors_second)
        return eigenvalues, eigenvectors


class KroneckerSumSqrt(lx.AbstractLinearOperator):
    """Symmetric square root S of A_1 ⊕ A_2 from the factor eigenvectors.

        S = (Q_1 ⊗ Q_2) diag(s) (Q_1 ⊗ Q_2)ᵀ,   s_ij = √(γ_1,i + γ_2,j)

    S is never materialised: `mv` applies S and `solve` applies S⁻¹
    matrix-free, as Q_1 (s ⊙ Z̃) Q_2ᵀ and Q_1 (Z̃ ⊘ s) Q_2ᵀ with
    Z̃ = Q_1ᵀ Z Q_2, so the cost is O(N (N_1 + N_2)) rather than O(N²).
    Both go through custom JVPs written in terms of the factors, which stay
    exact at repeated factor eigenvalues (an isotropic factor c I, gh-295).

    Args:
        A: Symmetric positive-semidefinite factor A_1, N_1 × N_1.
        B: Symmetric positive-semidefinite factor A_2, N_2 × N_2.

    Raises:
        ValueError: If either factor is non-square or not tagged symmetric.
        EquinoxRuntimeError: If A_1 ⊕ A_2 is not positive semidefinite
            (checked with `equinox.error_if`, so also under ``jax.jit``).

    Examples:
        S² recovers the Kronecker sum:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> S = gaussx.KroneckerSumSqrt(A, B)
        >>> root = S.as_matrix()
        >>> dense = gaussx.KroneckerSum(A, B).as_matrix()
        >>> bool(jnp.allclose(root @ root, dense, atol=1e-4))
        True

        `solve` undoes `mv`:

        >>> v = jnp.array([1.0, -1.0, 0.5, 2.0])
        >>> bool(jnp.allclose(S.solve(S.mv(v)), v, atol=1e-4))
        True
    """

    a_factor: Float[Array, "a a"] | Float[Array, " a"]
    b_factor: Float[Array, "b b"] | Float[Array, " b"]
    eigenvectors_a: Float[Array, "a a"]
    eigenvectors_b: Float[Array, "b b"]
    sqrt_eigenvalues: Float[Array, "a b"]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _n_a: int = eqx.field(static=True)
    _n_b: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)

    def __init__(
        self,
        A: lx.AbstractLinearOperator,
        B: lx.AbstractLinearOperator,
    ) -> None:
        if A.in_size() != A.out_size():
            raise ValueError(
                f"A must be square, got in_size={A.in_size()}, out_size={A.out_size()}."
            )
        if B.in_size() != B.out_size():
            raise ValueError(
                f"B must be square, got in_size={B.in_size()}, out_size={B.out_size()}."
            )
        # The symmetric root is defined only for symmetric PSD factors, and
        # untagged, ``jnp.linalg.eigh`` would silently read one triangle of a
        # non-symmetric input and return wrong eigenvectors.
        if not lx.is_symmetric(A) or not lx.is_symmetric(B):
            raise ValueError(
                "KroneckerSumSqrt requires both factors to be symmetric "
                "(tag them with lx.symmetric_tag or lx.positive_semidefinite_tag)."
            )
        # The factors (a matrix, or the diagonal of a diagonal factor, which
        # is never materialised) are the differentiable leaves; the cached
        # eigendecomposition is a constant for autodiff. `mv` and `solve` go
        # through custom JVPs written in terms of the factors, because the
        # eigenvector derivative divides by eigenvalue gaps and is NaN at a
        # repeated eigenvalue (e.g. an isotropic factor c I), gh-295.
        first_factor = _factor_array(A)
        second_factor = _factor_array(B)
        # A_1 = Q_1 diag(γ_1) Q_1ᵀ
        eigenvalues_first, eigenvectors_first = _factor_eigh(
            jax.lax.stop_gradient(first_factor)
        )
        # A_2 = Q_2 diag(γ_2) Q_2ᵀ
        eigenvalues_second, eigenvectors_second = _factor_eigh(
            jax.lax.stop_gradient(second_factor)
        )
        # γ_ij = γ_1,i + γ_2,j, checked ≥ 0 up to round-off
        eigenvalues = _checked_kronecker_sum_spectrum(
            eigenvalues_first, eigenvalues_second
        )
        # s_ij = √max(γ_ij, 0)
        sqrt_eigenvalues = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))

        self.a_factor = first_factor
        self.b_factor = second_factor
        self.eigenvectors_a = eigenvectors_first
        self.eigenvectors_b = eigenvectors_second
        self.sqrt_eigenvalues = sqrt_eigenvalues
        self._n_a = A.in_size()
        self._n_b = B.in_size()
        # N = N_1 N_2
        self._in_size = self._n_a * self._n_b
        self._out_size = self._in_size
        self._dtype = str(
            jnp.result_type(eigenvectors_first, eigenvectors_second, sqrt_eigenvalues)
        )

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply the square root S to a vector.

            S vec(Z) = vec(Q_1 (s ⊙ Z̃) Q_2ᵀ),   Z̃ = Q_1ᵀ Z Q_2

        Args:
            vector: Input vec(Z), of length N = N_1 N_2.

        Returns:
            S vec(Z), of length N.
        """
        # Z as one slab, 1 × N_1 × N_2
        slabs = rearrange(vector, "(a b) -> 1 a b", a=self._n_a, b=self._n_b)
        # Q_1 (s ⊙ Z̃) Q_2ᵀ
        result = _kronecker_sum_root_apply(*self._parts(), slabs)
        return rearrange(result, "1 a b -> (a b)")

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply the inverse square root S⁻¹ to a vector.

            S⁻¹ vec(Z) = vec(Q_1 (Z̃ ⊘ s) Q_2ᵀ),   Z̃ = Q_1ᵀ Z Q_2

        Args:
            vector: Input vec(Z), of length N = N_1 N_2.

        Returns:
            S⁻¹ vec(Z), of length N.
        """
        # Z as one slab, 1 × N_1 × N_2
        slabs = rearrange(vector, "(a b) -> 1 a b", a=self._n_a, b=self._n_b)
        # Q_1 (Z̃ ⊘ s) Q_2ᵀ
        result = _kronecker_sum_root_solve(*self._parts(), slabs)
        return rearrange(result, "1 a b -> (a b)")

    def _parts(self):
        """The arguments of the custom-JVP kernels: factors, Q_1, Q_2 and s."""
        return (
            self.a_factor,
            self.b_factor,
            self.eigenvectors_a,
            self.eigenvectors_b,
            self.sqrt_eigenvalues,
        )

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense S, N × N, one column S e_n at a time.

        Returns:
            The materialised square root.
        """
        basis = jnp.eye(self._in_size, dtype=jnp.dtype(self._dtype))
        # S I, column by column
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(basis)

    def transpose(self) -> KroneckerSumSqrt:
        """Sᵀ = S, since the symmetric root is symmetric.

        Returns:
            This operator.
        """
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _checked_kronecker_sum_spectrum(
    evals_a: Float[Array, " a"], evals_b: Float[Array, " b"]
) -> Float[Array, "a b"]:
    """Spectrum γ_1,i + γ_2,j, with a run-time error if A_1 ⊕ A_2 is indefinite.

        γ_ij = γ_1,i + γ_2,j,   min γ_ij ≥ −100 ε √max(max|γ_ij|, 1)

    Shared by the constructor and the custom JVPs (which recompute the
    spectrum), so autodiff cannot bypass the check.

    Args:
        evals_a: Eigenvalues γ_1 of A_1.
        evals_b: Eigenvalues γ_2 of A_2.

    Returns:
        The N_1 × N_2 eigenvalues γ_ij, at least float32.
    """
    # jnp, not einx: einx cannot resolve an empty factor (N_d = 0), which
    # must still reach the ValueError of the reduction below.
    pairwise = evals_a[:, None] + evals_b[None, :]
    # γ_ij = γ_1,i + γ_2,j
    eigenvalues = pairwise.astype(jnp.result_type(evals_a, evals_b, jnp.float32))
    # The tolerance for numerically-zero negative eigenvalues scales with
    # √(spectrum magnitude): tight enough for large spectra yet admitting eigh
    # round-off. Linear scaling was too permissive: at a scale of 1e8,
    # 100 ε scale let genuinely indefinite operators (negatives near −1e3)
    # slip past the guard.
    # max(max|γ_ij|, 1)
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    # −100 ε √scale
    threshold = (
        -_NEGATIVE_EIGENVALUE_TOLERANCE_FACTOR
        * jnp.finfo(eigenvalues.dtype).eps
        * jnp.sqrt(scale)
    )
    # ``eqx.error_if`` rather than a Python branch, so the check also runs
    # (at run time) when the factors are traced.
    return eqx.error_if(
        eigenvalues,
        jnp.min(eigenvalues) < threshold,
        "A ⊕ B must be positive semidefinite (minimum eigenvalue of the "
        "Kronecker sum is below the round-off threshold).",
    )


def _factor_array(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "n n"] | Float[Array, " n"]:
    """A factor A_d as an array: its diagonal if diagonal (never materialised).

    Args:
        operator: The factor A_d.

    Returns:
        diag(A_d) for a diagonal factor, else the dense A_d.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    return operator.as_matrix()


def _factor_eigh(
    factor: Float[Array, "n n"] | Float[Array, " n"],
) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
    """Eigendecomposition A_d = Q_d diag(γ_d) Q_dᵀ of a `_factor_array`.

    Args:
        factor: A dense factor A_d, or the diagonal of a diagonal one.

    Returns:
        Tuple ``(γ_d, Q_d)``; Q_d = I for a diagonal factor.
    """
    if factor.ndim == 1:
        # A_d = I diag(d) Iᵀ
        return factor, jnp.eye(factor.shape[0], dtype=factor.dtype)
    return jnp.linalg.eigh(factor)


def _factor_tangent(tangent: Array) -> Float[Array, "n n"]:
    """A factor tangent dA_d as a matrix (a diagonal factor's is a vector).

    Args:
        tangent: The tangent of a `_factor_array`.

    Returns:
        dA_d as a dense matrix.
    """
    return jnp.diag(tangent) if tangent.ndim == 1 else tangent


def _spectral_parts(a, b):
    """Eigenbases Q_1, Q_2 and root spectrum s of A_1 ⊕ A_2, from the factors.

        A_d = Q_d diag(γ_d) Q_dᵀ,   s_ij = √max(γ_1,i + γ_2,j, 0)

    The custom JVPs below recompute these from the primal factors rather than
    use the cached (``stop_gradient``) ones, so that differentiating the JVP
    itself (a Hessian) still sees their dependence on the factors. The PSD
    check is re-applied, so autodiff cannot bypass it.

    First derivatives never differentiate ``eigh`` and are exact at repeated
    eigenvalues. Second and higher derivatives do differentiate it here, so,
    exactly like ``dense_symmetric_sqrt``, they are non-finite when a factor
    has a repeated eigenvalue. An exact second-order rule needs the Sylvester
    solve with a right-hand side that is no longer a Kronecker sum, a dense
    N × N intermediate.

    Args:
        a: Factor array of A_1 (`_factor_array`).
        b: Factor array of A_2.

    Returns:
        Tuple ``(Q_1, Q_2, s)``.
    """
    # A_1 = Q_1 diag(γ_1) Q_1ᵀ
    values_a, basis_a = _factor_eigh(a)
    # A_2 = Q_2 diag(γ_2) Q_2ᵀ
    values_b, basis_b = _factor_eigh(b)
    # γ_ij = γ_1,i + γ_2,j
    eigenvalues = _checked_kronecker_sum_spectrum(values_a, values_b)
    # s_ij = √max(γ_ij, 0)
    roots = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    return basis_a, basis_b, roots


def _to_eigenbasis(basis_a, basis_b, values):
    """Rotate each slab Z into the eigenbasis, Z̃ = Q_1ᵀ Z Q_2.

    Args:
        basis_a: Eigenvectors Q_1, N_1 × N_1.
        basis_b: Eigenvectors Q_2, N_2 × N_2.
        values: Slabs Z, of shape (slabs, N_1, N_2).

    Returns:
        The rotated slabs Z̃, same shape.
    """
    # Z̃ = Q_1ᵀ Z Q_2
    return einsum(basis_a, values, basis_b, "k i, s k l, l j -> s i j")


def _from_eigenbasis(basis_a, basis_b, values):
    """Rotate each slab Z̃ back from the eigenbasis, Z = Q_1 Z̃ Q_2ᵀ.

    Args:
        basis_a: Eigenvectors Q_1, N_1 × N_1.
        basis_b: Eigenvectors Q_2, N_2 × N_2.
        values: Slabs Z̃ in the eigenbasis, of shape (slabs, N_1, N_2).

    Returns:
        The slabs Z, same shape.
    """
    # Z = Q_1 Z̃ Q_2ᵀ
    return einsum(basis_a, values, basis_b, "i k, s k l, j l -> s i j")


def _kronecker_sum_root_tangent(basis_a, basis_b, roots, tangent_a, tangent_b, rotated):
    """Eigenbasis action of the root derivative dS on Z̃, for S² = A_1 ⊕ A_2.

        (dS̃ Z̃)_ij = ∑_k dÃ_1,ik Z̃_kj / (s_ij + s_kj)
                   + ∑_l dÃ_2,jl Z̃_il / (s_ij + s_il)

    As in ``dense_symmetric_sqrt``: in the eigenbasis of A_1 ⊕ A_2 the root
    derivative is the rotated tangent divided entrywise by *sums* of root
    eigenvalues rather than eigenvalue gaps, so repeated eigenvalues are
    harmless. With d(A_1 ⊕ A_2) = dA_1 ⊕ dA_2 the rotated tangent is
    dÃ_1,ik δ_jl + δ_ik dÃ_2,jl, which gives the two sums above at a cost of
    O(N (N_1 + N_2)) per slab, never forming A_1 ⊕ A_2. Entries whose root
    sum is zero (a doubly-degenerate zero eigenvalue) get a zero derivative,
    matching ``dense_symmetric_sqrt``.

    Algorithm:
        1. Symmetrise the tangents and rotate them, dÃ_d = Q_dᵀ dA_d Q_d.
        2. Form the reciprocal root sums, zero where a sum is zero.
        3. Fold them into the rotated tangents and contract with Z̃.

    Args:
        basis_a: Eigenvectors Q_1 of A_1, N_1 × N_1.
        basis_b: Eigenvectors Q_2 of A_2, N_2 × N_2.
        roots: Root eigenvalues s_ij = √(γ_1,i + γ_2,j), N_1 × N_2.
        tangent_a: Tangent dA_1, in the original basis.
        tangent_b: Tangent dA_2, in the original basis.
        rotated: Slabs Z̃ = Q_1ᵀ Z Q_2, of shape (slabs, N_1, N_2).

    Returns:
        The slabs dS̃ Z̃ in the eigenbasis, of shape (slabs, N_1, N_2).
    """
    # --- rotate the tangents ---
    # eigh reads one triangle, so project the tangents onto symmetric matrices.
    # (dA_1 + dA_1ᵀ) / 2
    sym_a = 0.5 * (tangent_a + rearrange(tangent_a, "i k -> k i"))
    # (dA_2 + dA_2ᵀ) / 2
    sym_b = 0.5 * (tangent_b + rearrange(tangent_b, "j l -> l j"))
    # dÃ_1 = Q_1ᵀ dA_1 Q_1
    tangent_a = einsum(basis_a, sym_a, basis_a, "p i, p q, q k -> i k")
    # dÃ_2 = Q_2ᵀ dA_2 Q_2
    tangent_b = einsum(basis_b, sym_b, basis_b, "p j, p q, q l -> j l")

    def safe_inverse(denominator):
        # 1 / x where x > 0, else 0
        positive = denominator > 0.0
        return jnp.where(positive, 1.0 / jnp.where(positive, denominator, 1.0), 0.0)

    # --- reciprocal root sums ---
    # s_ij + s_kj
    root_sums_a = einx.add("i j, k j -> i k j", roots, roots)
    # 1 / (s_ij + s_kj)
    inverse_a = safe_inverse(root_sums_a)
    # s_ij + s_il
    root_sums_b = einx.add("i j, i l -> i j l", roots, roots)
    # 1 / (s_ij + s_il)
    inverse_b = safe_inverse(root_sums_b)

    # --- contract ---
    # dÃ_1,ik / (s_ij + s_kj)
    weighted_a = einx.multiply("i k, i k j -> i k j", tangent_a, inverse_a)
    # dÃ_2,jl / (s_ij + s_il)
    weighted_b = einx.multiply("j l, i j l -> i j l", tangent_b, inverse_b)
    # ∑_k dÃ_1,ik Z̃_kj / (s_ij + s_kj)
    term_a = einsum(weighted_a, rotated, "i k j, s k j -> s i j")
    # ∑_l dÃ_2,jl Z̃_il / (s_ij + s_il)
    term_b = einsum(weighted_b, rotated, "i j l, s i l -> s i j")
    return term_a + term_b


@jax.custom_jvp
def _kronecker_sum_root_apply(a, b, basis_a, basis_b, roots, Z):
    """S Z per slab, Q_1 (s ⊙ Q_1ᵀ Z Q_2) Q_2ᵀ, from a cached eigendecomposition.

    ``a`` and ``b`` (`_factor_array` s) are unused by the primal; the JVP
    differentiates with respect to them and ignores the tangents of the
    cached Q_1, Q_2 and s, recomputing them from ``a`` and ``b``
    (`_spectral_parts`) so higher-order derivatives stay correct.

    Args:
        a: Factor array of A_1.
        b: Factor array of A_2.
        basis_a: Cached eigenvectors Q_1.
        basis_b: Cached eigenvectors Q_2.
        roots: Cached root eigenvalues s.
        Z: Slabs Z, of shape (slabs, N_1, N_2).

    Returns:
        The slabs S Z, same shape.
    """
    del a, b
    # Z̃ = Q_1ᵀ Z Q_2
    rotated = _to_eigenbasis(basis_a, basis_b, Z)
    # S Z = Q_1 (s ⊙ Z̃) Q_2ᵀ
    return _from_eigenbasis(basis_a, basis_b, roots * rotated)


@_kronecker_sum_root_apply.defjvp
def _kronecker_sum_root_apply_jvp(primals, tangents):
    """JVP of S Z, d(S Z) = dS Z + S dZ, without differentiating ``eigh``.

        d(S Z) = Q_1 (dS̃ Z̃ + s ⊙ dZ̃) Q_2ᵀ

    Args:
        primals: The primal arguments of `_kronecker_sum_root_apply`.
        tangents: Their tangents; those of Q_1, Q_2 and s are ignored.

    Returns:
        Tuple ``(S Z, d(S Z))``.
    """
    a, b, _, _, _, Z = primals
    tangent_a, tangent_b, _, _, _, tangent_Z = tangents
    basis_a, basis_b, roots = _spectral_parts(a, b)
    tangent_a, tangent_b = _factor_tangent(tangent_a), _factor_tangent(tangent_b)
    # Z̃ = Q_1ᵀ Z Q_2
    rotated = _to_eigenbasis(basis_a, basis_b, Z)
    # S Z = Q_1 (s ⊙ Z̃) Q_2ᵀ
    primal_out = _from_eigenbasis(basis_a, basis_b, roots * rotated)
    # dS̃ Z̃
    term = _kronecker_sum_root_tangent(
        basis_a, basis_b, roots, tangent_a, tangent_b, rotated
    )
    # dZ̃ = Q_1ᵀ dZ Q_2
    rotated_tangent_Z = _to_eigenbasis(basis_a, basis_b, tangent_Z)
    # Q_1 (s ⊙ dZ̃ + dS̃ Z̃) Q_2ᵀ
    tangent_out = _from_eigenbasis(basis_a, basis_b, roots * rotated_tangent_Z + term)
    return primal_out, tangent_out


@jax.custom_jvp
def _kronecker_sum_root_solve(a, b, basis_a, basis_b, roots, Z):
    """S⁻¹ Z per slab, Q_1 (Q_1ᵀ Z Q_2 ⊘ s) Q_2ᵀ, from a cached eigendecomposition.

    Args:
        a: Factor array of A_1 (unused by the primal, as in
            `_kronecker_sum_root_apply`).
        b: Factor array of A_2.
        basis_a: Cached eigenvectors Q_1.
        basis_b: Cached eigenvectors Q_2.
        roots: Cached root eigenvalues s.
        Z: Slabs Z, of shape (slabs, N_1, N_2).

    Returns:
        The slabs S⁻¹ Z, same shape.
    """
    del a, b
    # Z̃ = Q_1ᵀ Z Q_2
    rotated = _to_eigenbasis(basis_a, basis_b, Z)
    # S⁻¹ Z = Q_1 (Z̃ ⊘ s) Q_2ᵀ
    return _from_eigenbasis(basis_a, basis_b, rotated / roots)


@_kronecker_sum_root_solve.defjvp
def _kronecker_sum_root_solve_jvp(primals, tangents):
    """JVP of X = S⁻¹ Z, dX = S⁻¹ (dZ − dS X).

        dX = Q_1 ((dZ̃ − dS̃ X̃) ⊘ s) Q_2ᵀ,   X̃ = Z̃ ⊘ s

    Args:
        primals: The primal arguments of `_kronecker_sum_root_solve`.
        tangents: Their tangents; those of Q_1, Q_2 and s are ignored.

    Returns:
        Tuple ``(X, dX)``.
    """
    a, b, _, _, _, Z = primals
    tangent_a, tangent_b, _, _, _, tangent_Z = tangents
    basis_a, basis_b, roots = _spectral_parts(a, b)
    tangent_a, tangent_b = _factor_tangent(tangent_a), _factor_tangent(tangent_b)
    # X̃ = Z̃ ⊘ s
    rotated_out = _to_eigenbasis(basis_a, basis_b, Z) / roots
    # X = Q_1 X̃ Q_2ᵀ
    primal_out = _from_eigenbasis(basis_a, basis_b, rotated_out)
    # dS̃ X̃
    term = _kronecker_sum_root_tangent(
        basis_a, basis_b, roots, tangent_a, tangent_b, rotated_out
    )
    # dZ̃ = Q_1ᵀ dZ Q_2
    rotated_tangent_Z = _to_eigenbasis(basis_a, basis_b, tangent_Z)
    # dX = Q_1 ((dZ̃ − dS̃ X̃) ⊘ s) Q_2ᵀ
    tangent_out = _from_eigenbasis(basis_a, basis_b, (rotated_tangent_Z - term) / roots)
    return primal_out, tangent_out


def kronecker_sum_sample(
    A_op: lx.AbstractLinearOperator,
    B_op: lx.AbstractLinearOperator,
    *,
    key: jax.Array,
    num_samples: int = 1,
) -> Float[Array, "num_samples n_a n_b"]:
    """Sample from 𝒩(0, A_1 ⊕ A_2) through the factor eigendecompositions.

        z = S ε,   ε ~ 𝒩(0, I),   S² = A_1 ⊕ A_2

    Applies the matrix-free `KroneckerSumSqrt` S to standard normal noise,
    so the N × N covariance is never formed.

    Args:
        A_op: Symmetric positive-semidefinite factor A_1, N_1 × N_1.
        B_op: Symmetric positive-semidefinite factor A_2, N_2 × N_2.
        key: PRNG key for the standard normal draws ε.
        num_samples: Number of samples to draw.

    Returns:
        Samples of shape (num_samples, N_1, N_2), each one Z with vec(Z) ~
        𝒩(0, A_1 ⊕ A_2).

    Raises:
        ValueError: If ``num_samples`` is less than 1.

    Examples:
        Three draws on a 2 × 3 grid:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(2.0 * jnp.eye(2), psd)
        >>> B = lx.DiagonalLinearOperator(jnp.array([1.0, 0.5, 0.25]))
        >>> key = jax.random.PRNGKey(0)
        >>> gaussx.kronecker_sum_sample(A, B, key=key, num_samples=3).shape
        (3, 2, 3)
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    sqrt_op = KroneckerSumSqrt(A_op, B_op)
    # ε ~ 𝒩(0, I)
    noise = jax.random.normal(
        key,
        (num_samples, sqrt_op.in_size()),
        dtype=jnp.dtype(sqrt_op._dtype),
    )
    # z = S ε
    samples = jax.vmap(sqrt_op.mv)(noise)
    return rearrange(samples, "s (a b) -> s a b", a=sqrt_op._n_a, b=sqrt_op._n_b)


def _eigh_factor(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
    """Symmetric eigendecomposition A_d = Q_d diag(γ_d) Q_dᵀ of a factor.

    Always returns an ``eigh``-equivalent pair with orthonormal Q_d: callers
    (``_solve_kronecker_sum``, `KroneckerSum.eigendecompose`,
    `KroneckerSumSqrt`) rely on Q_dᵀ = Q_d⁻¹. A diagonal operator takes a
    free structural shortcut. Anything else is materialised and decomposed
    with ``jnp.linalg.eigh`` directly; routing through `gaussx.eig` is unsafe
    here, because for an untagged factor that primitive falls back to
    ``jnp.linalg.eig``, which returns general (non-orthonormal) eigenvectors.

    Args:
        operator: The symmetric factor A_d.

    Returns:
        Tuple ``(γ_d, Q_d)``.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        diagonal = lx.diagonal(operator)
        # A_d = I diag(d) Iᵀ
        return diagonal, jnp.eye(diagonal.shape[0], dtype=diagonal.dtype)
    return jnp.linalg.eigh(operator.as_matrix())
