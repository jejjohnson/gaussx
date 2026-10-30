"""Factored eigenbases of Kronecker-structured operators.

A Kronecker sum A_1 ⊕ A_2, a Kronecker product A_1 ⊗ A_2 and a shifted
product A_1 ⊗ A_2 + c I are all diagonalised by the Kronecker product of
their factors' eigenbases, A = W⁻¹ diag(γ) W with W = W_1 ⊗ … ⊗ W_D. W_d is
the forward transform of axis d, of size N_d: W_d = Q_dᵀ for a symmetric
factor A_d = Q_d diag(γ_d) Q_dᵀ with orthonormal eigenvectors Q_d, the
transform pair of a `DiagonalizedOperator` A_d = W_d⁻¹ diag(γ_d) W_d, or the
identity for a diagonal factor. `FactoredEigen` keeps W per axis next to the
eigenvalue tensor γ ∈ ℝ^{N_1 × … × N_D}, so ``solve``, ``logdet`` and the
diagonal of the inverse act on the grid one axis at a time and the joint
basis W is never formed.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Inexact

from gaussx._einx import einsum, rearrange
from gaussx._operators._diagonalised import (
    DiagonalizedOperator,
    _along_axis,
    _flatten,
    _unflatten,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum


# One axis of the basis: the dense orthonormal eigenvectors Q_d of
# A_d = Q_d diag(γ_d) Q_dᵀ, a `DiagonalizedOperator` A_d = W_d⁻¹ diag(γ_d) W_d,
# or ``None`` for the identity.
_AxisBasis = Float[Array, "n n"] | DiagonalizedOperator | None


class FactoredEigen(eqx.Module):
    """Eigendecomposition A = W⁻¹ diag(γ) W with W = W_1 ⊗ … ⊗ W_D kept per axis.

        A x = W⁻¹ (γ ⊙ W x),   W = W_1 ⊗ … ⊗ W_D

    Attributes:
        bases: Per-axis bases (see ``_AxisBasis``), in Kronecker order: the
            orthonormal Q_d (W_d = Q_dᵀ), a `DiagonalizedOperator` holding
            the pair W_d, W_d⁻¹, or ``None`` for W_d = I.
        eigenvalues: Eigenvalues γ as a tensor of shape (N_1, …, N_D); its
            row-major flattening matches the operator's vector layout.

    Examples:
        The factored eigenbasis of a Kronecker sum solves without forming it:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._operators._factored_eigen import factored_eigen
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.DiagonalLinearOperator(jnp.array([1.0, 0.5, 0.25]))
        >>> op = gaussx.KroneckerSum(A, B)
        >>> factorization = factored_eigen(op)
        >>> factorization.eigenvalues.shape
        (2, 3)
        >>> b = jnp.arange(1.0, 7.0)
        >>> x = factorization.solve(b)
        >>> bool(jnp.allclose(op.as_matrix() @ x, b, atol=1e-4))
        True
    """

    bases: tuple[_AxisBasis, ...]
    eigenvalues: Inexact[Array, " ..."]

    def _real(self) -> bool:
        """Whether every transform pair promises a real operator."""
        return all(
            basis.real_output
            for basis in self.bases
            if isinstance(basis, DiagonalizedOperator)
        )

    def _output(self, values: Array, like: Array | None = None) -> Array:
        """Drop the imaginary part of a result that is real by construction."""
        if self._real() and (like is None or not jnp.iscomplexobj(like)):
            return jnp.real(values)
        return values

    def to_eigenbasis(self, vector: Inexact[Array, " n"]) -> Inexact[Array, " ..."]:
        """Coefficients W x, as a tensor of shape (N_1, …, N_D).

            W x = (W_1 ⊗ … ⊗ W_D) x

        Args:
            vector: Input x, of length N = ∏_d N_d.

        Returns:
            The coefficient tensor W x.
        """
        # X ∈ ℝ^{N_1 × … × N_D}, vec(X) = x
        values = _unflatten(vector, self.eigenvalues.shape)
        for axis, basis in enumerate(self.bases):
            # X ×_d W_d
            values = _apply_axis(basis, values, axis, inverse=False)
        return values

    def from_eigenbasis(self, coeffs: Inexact[Array, " ..."]) -> Inexact[Array, " n"]:
        """W⁻¹ c for a coefficient tensor c, flattened.

            W⁻¹ c = (W_1⁻¹ ⊗ … ⊗ W_D⁻¹) vec(c)

        Args:
            coeffs: Coefficient tensor c, of shape (N_1, …, N_D).

        Returns:
            W⁻¹ vec(c), of length N.
        """
        for axis, basis in enumerate(self.bases):
            # c ×_d W_d⁻¹
            coeffs = _apply_axis(basis, coeffs, axis, inverse=True)
        return _flatten(coeffs)

    def shifted(self, shift: Array | float) -> FactoredEigen:
        """The factorisation of A + c I, in the same basis.

            A + c I = W⁻¹ diag(γ + c) W

        Args:
            shift: The scalar shift c.

        Returns:
            A `FactoredEigen` with eigenvalues γ + c.
        """
        dtype = self.eigenvalues.dtype
        # γ + c, in γ's dtype
        shifted_eigenvalues = self.eigenvalues + jnp.asarray(shift, dtype)
        return FactoredEigen(self.bases, shifted_eigenvalues)

    def solve(self, vector: Inexact[Array, " n"]) -> Inexact[Array, " n"]:
        """Solve A x = b in the factored eigenbasis.

            x = W⁻¹ ((W b) ⊘ γ)

        Args:
            vector: Right-hand side b, of length N.

        Returns:
            The solution x, real unless b or a transform pair is complex.
        """
        # W b
        coefficients = self.to_eigenbasis(vector)
        # x = W⁻¹ ((W b) ⊘ γ)
        solution = self.from_eigenbasis(coefficients / self.eigenvalues)
        return self._output(solution, vector)

    def logdet(self) -> Float[Array, ""]:
        """log|det A| = ∑ log|γ|, in the ``slogdet`` convention.

        Returns:
            The log absolute determinant.
        """
        # ∑ log|γ|
        return jnp.sum(jnp.log(jnp.abs(self.eigenvalues)))

    def diag_inv(self, *, pinv: bool = False) -> Float[Array, " n"]:
        """Diagonal of the inverse, diag(A⁻¹), or diag(A⁺) with ``pinv=True``.

            [A⁻¹]_hh = ∑_i ∏_d P_d[h_d, i_d] / γ_i,   P_d = W_d⁻¹ ∘ W_dᵀ

        P_d is Q_d ∘ Q_d for an orthonormal axis. The sum is one contraction
        per axis over the reciprocal eigenvalue tensor, O(N ∑_d N_d).

        Args:
            pinv: Treat eigenvalues below max_d N_d · ε · max|γ| as zero, so
                they contribute nothing (intrinsic priors).

        Returns:
            The N diagonal entries.
        """
        # 1 / γ, or the pseudo-inverse reciprocal
        values = _reciprocal(self.eigenvalues, pinv=pinv)
        for axis, basis in enumerate(self.bases):
            if basis is None:
                continue
            names = _axis_names(values.ndim)
            summed = " ".join("p" if m == axis else n for m, n in enumerate(names))
            joined = " ".join(names)
            # ∑_{i_d} P_d[h_d, i_d] (·)[…, i_d, …]
            values = einsum(
                _basis_weights(basis), values, f"{names[axis]} p, {summed} -> {joined}"
            )
        return self._output(_flatten(values))


def factored_eigen(operator: lx.AbstractLinearOperator) -> FactoredEigen | None:
    """The factored eigenbasis A = W⁻¹ diag(γ) W of an operator, if it has one.

        A_1 ⊕ A_2:  W = W_1 ⊗ W_2,  γ = γ_1 ⊕ γ_2
        A_1 ⊗ A_2:  W = W_1 ⊗ W_2,  γ = γ_1 ⊗ γ_2

    Kronecker sums and products recurse into their factors, so a factor that
    carries its own basis (a `DiagonalizedOperator`, a nested `KroneckerSum`,
    a `SpectralFunction` of one) keeps it, and a diagonal factor needs no
    basis at all. Any other factor must be symmetric, and is materialised and
    eigendecomposed with ``eigh``: only that factor, never the joint
    operator.

    Args:
        operator: Any lineax operator A.

    Returns:
        The factorisation, or ``None`` for a non-symmetric operator with no
        known eigenbasis (or a Kronecker product with a non-square factor).

    Examples:
        A Kronecker product of a dense and a diagonal factor:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> from gaussx._operators._factored_eigen import factored_eigen
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.DiagonalLinearOperator(jnp.array([1.0, 0.5, 0.25]))
        >>> factorization = factored_eigen(gaussx.Kronecker(A, B))
        >>> dense = gaussx.Kronecker(A, B).as_matrix()
        >>> sign, expected = jnp.linalg.slogdet(dense)
        >>> bool(jnp.allclose(factorization.logdet(), expected, atol=1e-4))
        True

        A non-symmetric dense operator has none:

        >>> upper = lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [0.0, 1.0]]))
        >>> print(factored_eigen(upper))
        None
    """
    # lazy import, cycle: _operators._spectral_function -> _operators._factored_eigen
    from gaussx._operators._spectral_function import SpectralFunction

    if isinstance(operator, lx.TaggedLinearOperator) and isinstance(
        operator.operator,
        KroneckerSum | Kronecker | DiagonalizedOperator | SpectralFunction,
    ):
        return factored_eigen(operator.operator)
    if isinstance(operator, SpectralFunction):
        return operator.factored_eigen()
    if isinstance(operator, lx.IdentityLinearOperator):
        dtype = operator.in_structure().dtype
        # I = I diag(1) I
        return FactoredEigen((None,), jnp.ones(operator.in_size(), dtype=dtype))
    if isinstance(operator, lx.DiagonalLinearOperator):
        # diag(d) = I diag(d) I
        return FactoredEigen((None,), lx.diagonal(operator))
    if isinstance(operator, DiagonalizedOperator):
        # A = W⁻¹ diag(γ) W
        return FactoredEigen((operator,), operator.eigenvalues_flat())
    if isinstance(operator, KroneckerSum):
        # γ = γ_1 ⊕ γ_2
        return _combine((operator.A, operator.B), "add")
    if isinstance(operator, Kronecker):
        if any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        # γ = γ_1 ⊗ … ⊗ γ_D
        return _combine(operator.operators, "multiply")
    if lx.is_symmetric(operator):
        # A = Q diag(γ) Qᵀ
        eigenvalues, basis = jnp.linalg.eigh(operator.as_matrix())
        return FactoredEigen((basis,), eigenvalues)
    return None


def kronecker_eigen(*factors: lx.AbstractLinearOperator) -> FactoredEigen | None:
    """`factored_eigen` of A_1 ⊗ … ⊗ A_D, without wrapping it in a `Kronecker`.

    Args:
        *factors: The factors A_1, …, A_D.

    Returns:
        The factorisation, or ``None`` if a factor has no eigenbasis.
    """
    return _combine(factors, "multiply")


def _combine(
    factors: tuple[lx.AbstractLinearOperator, ...], op: str
) -> FactoredEigen | None:
    """Join the factors' bases; their eigenvalues by outer sum or product.

        ⊕:  γ[i_1, …, i_D] = ∑_d γ_d[i_d]
        ⊗:  γ[i_1, …, i_D] = ∏_d γ_d[i_d]

    Args:
        factors: The factors A_1, …, A_D.
        op: ``"add"`` for a Kronecker sum, ``"multiply"`` for a product.

    Returns:
        The joint factorisation, or ``None`` if a factor has none.
    """
    parts = [factored_eigen(factor) for factor in factors]
    if any(part is None for part in parts):
        return None
    bases: tuple[_AxisBasis, ...] = ()
    eigenvalues = None
    for part in parts:
        assert part is not None
        bases = (*bases, *part.bases)
        if eigenvalues is None:
            eigenvalues = part.eigenvalues
            continue
        left = _axis_names(eigenvalues.ndim, "a")
        right = _axis_names(part.eigenvalues.ndim, "b")
        pattern = f"{' '.join(left)}, {' '.join(right)} -> {' '.join(left + right)}"
        # γ_1:d ⊕ γ_d+1 or γ_1:d ⊗ γ_d+1, an outer sum or product
        eigenvalues = getattr(einx, op)(pattern, eigenvalues, part.eigenvalues)
    assert eigenvalues is not None
    return FactoredEigen(bases, eigenvalues)


def _axis_names(ndim: int, prefix: str = "a") -> list[str]:
    """einx axis names ``a0, a1, …`` for a tensor of ``ndim`` axes."""
    return [f"{prefix}{m}" for m in range(ndim)]


def _apply_axis(basis: _AxisBasis, values: Array, axis: int, *, inverse: bool) -> Array:
    """Apply W_d (or W_d⁻¹) along tensor axis d.

        X ×_d W_d,   or   X ×_d W_d⁻¹

    Args:
        basis: The axis basis (see ``_AxisBasis``).
        values: The tensor X, of shape (N_1, …, N_D).
        axis: The axis d.
        inverse: Apply W_d⁻¹ instead of W_d.

    Returns:
        The transformed tensor, same shape.
    """
    if basis is None:
        # X ×_d I
        return values
    names = _axis_names(values.ndim)
    joined = " ".join(names)
    if isinstance(basis, DiagonalizedOperator):
        transform = basis.inverse_flat if inverse else basis.forward_flat
        rest = [n for m, n in enumerate(names) if m != axis]
        sizes = {n: values.shape[m] for m, n in enumerate(names) if m != axis}
        # mode-d fibres as rows
        fibres = rearrange(values, f"{joined} -> ({' '.join(rest)}) {names[axis]}")
        # W_d (or W_d⁻¹) on each fibre
        fibres = _along_axis(transform, fibres, axis=1)
        return rearrange(
            fibres, f"({' '.join(rest)}) {names[axis]} -> {joined}", **sizes
        )
    summed = " ".join("p" if m == axis else n for m, n in enumerate(names))
    # Q_d maps eigen-coefficients to values, so W_d = Q_dᵀ and W_d⁻¹ = Q_d.
    lhs = f"{names[axis]} p" if inverse else f"p {names[axis]}"
    # X ×_d Q_d or X ×_d Q_dᵀ
    return einsum(basis, values, f"{lhs}, {summed} -> {joined}")


def _basis_weights(basis: Float[Array, "n n"] | DiagonalizedOperator) -> Array:
    """Weights P_d = W_d⁻¹ ∘ W_dᵀ of one axis of diag(A⁻¹).

        P_d[h, i] = W_d⁻¹[h, i] W_d[i, h]   (Q_d ∘ Q_d for an orthonormal axis)

    Args:
        basis: An orthonormal Q_d or a `DiagonalizedOperator`.

    Returns:
        The N_d × N_d weights P_d.
    """
    if not isinstance(basis, DiagonalizedOperator):
        # Q_d ∘ Q_d
        return basis * basis
    eye = jnp.eye(basis.size, dtype=basis.in_structure().dtype)
    # row i: W_d⁻¹ e_i, so columns[i, h] = W_d⁻¹[h, i]
    columns = jax.vmap(basis.inverse_flat)(eye)
    # row h: W_d e_h, so forward[h, i] = W_d[i, h]
    forward = jax.vmap(basis.forward_flat)(eye)
    # W_d⁻¹[h, i] W_d[i, h]
    return rearrange(columns, "i h -> h i") * forward


def _reciprocal(eigenvalues: Array, *, pinv: bool) -> Array:
    """Reciprocal eigenvalues 1 / γ; with ``pinv``, numerically-zero ones map to 0.

        1 / γ_i,   or   0 where |γ_i| ≤ max_d N_d · ε · max|γ|

    Args:
        eigenvalues: Eigenvalue tensor γ.
        pinv: Use the pseudo-inverse reciprocal.

    Returns:
        The reciprocal tensor, same shape.
    """
    if not pinv:
        # 1 / γ
        return 1.0 / eigenvalues
    # |γ|
    magnitude = jnp.abs(eigenvalues)
    eps = jnp.finfo(magnitude.dtype).eps
    # τ = max_d N_d · ε · max|γ|
    tol = max(eigenvalues.shape) * eps * jnp.max(magnitude)
    zero = magnitude <= tol
    # 1 / γ, or 0 where |γ| ≤ τ
    return jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, eigenvalues))
