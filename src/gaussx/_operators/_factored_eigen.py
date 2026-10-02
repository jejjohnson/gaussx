"""Factored eigenbases of Kronecker-structured operators.

A Kronecker sum ``A ⊕ B``, a Kronecker product ``A ⊗ B`` and a shifted
product ``A ⊗ B + c·I`` are all diagonalised by the Kronecker product of
their factors' eigenbases. `FactoredEigen` keeps that basis per axis (a dense
orthonormal matrix, the transform pair of a `DiagonalisedOperator`, or the
identity) next to the eigenvalue tensor, so ``solve``, ``logdet`` and the
diagonal of the inverse act on the ``(n_1, …, n_k)`` grid one axis at a time
and the joint basis is never formed.
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
    DiagonalisedOperator,
    _along_axis,
    _flatten,
    _unflatten,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum


# One axis of the basis: a dense orthonormal ``Q`` (``A = Q Λ Qᵀ``), a
# `DiagonalisedOperator` (``A = V⁻¹ Λ V``), or ``None`` for the identity.
_AxisBasis = Float[Array, "n n"] | DiagonalisedOperator | None


class FactoredEigen(eqx.Module):
    r"""``K = V⁻¹ diag(Λ) V`` with ``V = V_1 ⊗ … ⊗ V_k`` kept per axis.

    Attributes:
        bases: Per-axis bases (see ``_AxisBasis``), in Kronecker order.
        eigenvalues: ``Λ`` as a tensor of shape ``(n_1, …, n_k)``; its
            row-major flattening matches the operator's vector layout.
    """

    bases: tuple[_AxisBasis, ...]
    eigenvalues: Inexact[Array, " ..."]

    def _real(self) -> bool:
        return all(
            basis.real_output
            for basis in self.bases
            if isinstance(basis, DiagonalisedOperator)
        )

    def _output(self, values: Array, like: Array | None = None) -> Array:
        if self._real() and (like is None or not jnp.iscomplexobj(like)):
            return jnp.real(values)
        return values

    def to_eigenbasis(self, vector: Inexact[Array, " n"]) -> Inexact[Array, " ..."]:
        """``V x`` as a tensor of shape ``(n_1, …, n_k)``."""
        values = _unflatten(vector, self.eigenvalues.shape)
        for axis, basis in enumerate(self.bases):
            values = _apply_axis(basis, values, axis, inverse=False)
        return values

    def from_eigenbasis(self, coeffs: Inexact[Array, " ..."]) -> Inexact[Array, " n"]:
        """``V⁻¹ c`` for a coefficient tensor ``c``, flattened."""
        for axis, basis in enumerate(self.bases):
            coeffs = _apply_axis(basis, coeffs, axis, inverse=True)
        return _flatten(coeffs)

    def shifted(self, shift: Array | float) -> FactoredEigen:
        """The factorisation of ``K + shift · I`` (same basis)."""
        dtype = self.eigenvalues.dtype
        return FactoredEigen(self.bases, self.eigenvalues + jnp.asarray(shift, dtype))

    def solve(self, vector: Inexact[Array, " n"]) -> Inexact[Array, " n"]:
        """``K⁻¹ b`` in the factored eigenbasis."""
        x = self.from_eigenbasis(self.to_eigenbasis(vector) / self.eigenvalues)
        return self._output(x, vector)

    def logdet(self) -> Float[Array, ""]:
        """``log|det K| = Σ log|λ|`` (``slogdet`` convention)."""
        return jnp.sum(jnp.log(jnp.abs(self.eigenvalues)))

    def diag_inv(self, *, pinv: bool = False) -> Float[Array, " n"]:
        r"""``diag(K⁻¹)``, or ``diag(K⁺)`` with ``pinv=True``.

        ``[K⁻¹]_{hh} = Σ_i Π_m P_m[h_m, i_m] / λ_i`` with
        ``P_m = V_m⁻¹ ∘ V_mᵀ`` (``Q_m ∘ Q_m`` for an orthonormal axis): one
        contraction per axis over the eigenvalue tensor, ``O(N Σ_m n_m)``.
        With ``pinv=True`` eigenvalues below ``max(n_m) · eps · max|λ|`` are
        treated as zero and contribute nothing (intrinsic priors).
        """
        values = _reciprocal(self.eigenvalues, pinv=pinv)
        for axis, basis in enumerate(self.bases):
            if basis is None:
                continue
            names = _axis_names(values.ndim)
            summed = " ".join("p" if m == axis else n for m, n in enumerate(names))
            joined = " ".join(names)
            values = einsum(
                _basis_weights(basis), values, f"{names[axis]} p, {summed} -> {joined}"
            )
        return self._output(_flatten(values))


def factored_eigen(operator: lx.AbstractLinearOperator) -> FactoredEigen | None:
    """The factored eigenbasis of ``operator``, or ``None`` if it has none.

    Kronecker sums and products recurse into their factors, so a factor that
    carries its own basis (a `DiagonalisedOperator`, a nested `KroneckerSum`,
    a `SpectralFunction` of one) keeps it; a diagonal factor needs no basis
    at all. Any other factor must be symmetric, and is materialised and
    eigendecomposed with ``eigh`` — only that factor, never the joint
    operator.

    Args:
        operator: Any lineax operator.

    Returns:
        The factorisation, or ``None`` for a non-symmetric operator with no
        known eigenbasis.
    """
    from gaussx._operators._spectral_function import SpectralFunction

    if isinstance(operator, lx.TaggedLinearOperator) and isinstance(
        operator.operator,
        KroneckerSum | Kronecker | DiagonalisedOperator | SpectralFunction,
    ):
        return factored_eigen(operator.operator)
    if isinstance(operator, SpectralFunction):
        return operator.factored_eigen()
    if isinstance(operator, lx.IdentityLinearOperator):
        dtype = operator.in_structure().dtype
        return FactoredEigen((None,), jnp.ones(operator.in_size(), dtype=dtype))
    if isinstance(operator, lx.DiagonalLinearOperator):
        return FactoredEigen((None,), lx.diagonal(operator))
    if isinstance(operator, DiagonalisedOperator):
        return FactoredEigen((operator,), operator.eigenvalues_flat())
    if isinstance(operator, KroneckerSum):
        return _combine((operator.A, operator.B), "add")
    if isinstance(operator, Kronecker):
        if any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        return _combine(operator.operators, "multiply")
    if lx.is_symmetric(operator):
        eigenvalues, basis = jnp.linalg.eigh(operator.as_matrix())
        return FactoredEigen((basis,), eigenvalues)
    return None


def kronecker_eigen(*factors: lx.AbstractLinearOperator) -> FactoredEigen | None:
    """`factored_eigen` of ``factors[0] ⊗ factors[1] ⊗ …`` without wrapping."""
    return _combine(factors, "multiply")


def _combine(
    factors: tuple[lx.AbstractLinearOperator, ...], op: str
) -> FactoredEigen | None:
    """Join the factors' bases; their eigenvalues by outer sum or product."""
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
        eigenvalues = getattr(einx, op)(pattern, eigenvalues, part.eigenvalues)
    assert eigenvalues is not None
    return FactoredEigen(bases, eigenvalues)


def _axis_names(ndim: int, prefix: str = "a") -> list[str]:
    return [f"{prefix}{m}" for m in range(ndim)]


def _apply_axis(basis: _AxisBasis, values: Array, axis: int, *, inverse: bool) -> Array:
    """Apply ``V_m`` (or ``V_m⁻¹``) along tensor axis ``axis``."""
    if basis is None:
        return values
    names = _axis_names(values.ndim)
    joined = " ".join(names)
    if isinstance(basis, DiagonalisedOperator):
        transform = basis.inverse_flat if inverse else basis.forward_flat
        rest = [n for m, n in enumerate(names) if m != axis]
        sizes = {n: values.shape[m] for m, n in enumerate(names) if m != axis}
        fibres = rearrange(values, f"{joined} -> ({' '.join(rest)}) {names[axis]}")
        fibres = _along_axis(transform, fibres, axis=1)
        return rearrange(
            fibres, f"({' '.join(rest)}) {names[axis]} -> {joined}", **sizes
        )
    summed = " ".join("p" if m == axis else n for m, n in enumerate(names))
    # ``Q`` maps eigen-coefficients to values, so ``V = Qᵀ`` and ``V⁻¹ = Q``.
    lhs = f"{names[axis]} p" if inverse else f"p {names[axis]}"
    return einsum(basis, values, f"{lhs}, {summed} -> {joined}")


def _basis_weights(basis: Float[Array, "n n"] | DiagonalisedOperator) -> Array:
    """``P[h, i] = V⁻¹[h, i] · V[i, h]`` — ``Q ∘ Q`` for an orthonormal axis."""
    if not isinstance(basis, DiagonalisedOperator):
        return basis * basis
    eye = jnp.eye(basis.size, dtype=basis.in_structure().dtype)
    columns = jax.vmap(basis.inverse_flat)(eye)  # rows: V⁻¹ e_i
    forward = jax.vmap(basis.forward_flat)(eye)  # rows: V e_h
    # columns[i, h] = V⁻¹[h, i]; forward[h, i] = V[i, h].
    return rearrange(columns, "i h -> h i") * forward


def _reciprocal(eigenvalues: Array, *, pinv: bool) -> Array:
    """``1/λ``; with ``pinv`` numerically-zero eigenvalues map to zero."""
    if not pinv:
        return 1.0 / eigenvalues
    magnitude = jnp.abs(eigenvalues)
    eps = jnp.finfo(magnitude.dtype).eps
    tol = max(eigenvalues.shape) * eps * jnp.max(magnitude)
    zero = magnitude <= tol
    return jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, eigenvalues))
