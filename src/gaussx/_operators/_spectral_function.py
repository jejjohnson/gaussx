"""Spectral functions of Kronecker sums: ``f(A₁ ⊕ … ⊕ A_d)``.

A function of a symmetric operator shares its eigenvectors, and a Kronecker
sum's eigenvectors are the Kronecker product of its factors'. So
``f(A₁ ⊕ … ⊕ A_d) = V diag(f(λ¹_i + … + λᵈ_k)) Vᵀ`` with
``V = V₁ ⊗ … ⊗ V_d``, and every operation is a per-axis rotation, an
elementwise function of the eigenvalue tensor, and the rotation back. The
SPDE (Matérn) precision on a regular grid is such a function
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
from gaussx._linalg._eigen_factorization import EigenFactorization
from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._factored_eigen import FactoredEigen, factored_eigen
from gaussx._operators._kronecker_sum import KroneckerSum


class SpectralFunction(lx.AbstractLinearOperator):
    r"""``f(B)`` for a symmetric ``B`` with a factored eigenbasis.

    ``B`` is typically a Kronecker sum ``A₁ ⊕ … ⊕ A_d`` of symmetric factors
    (a grid Laplacian ``L_H ⊕ L_W``). With ``A_m = V_m Λ_m V_mᵀ``,

    $$
    f(A_1 \oplus \cdots \oplus A_d)
    = (V_1 \otimes \cdots \otimes V_d)\,
      f(\Lambda_1 \oplus \cdots \oplus \Lambda_d)\,
      (V_1 \otimes \cdots \otimes V_d)^\top,
    $$

    so `mv`, `solve`, `logdet`, `diag_inv` and `sqrt_matmul` all act on the
    ``(n_1, …, n_d)`` grid one axis at a time, ``O(N Σ_m n_m)``, and the
    joint eigenbasis is never formed. The factor eigendecompositions are
    computed once, at construction.

    `gaussx.solve`, `gaussx.logdet`, `gaussx.diag_inv` (including
    ``pinv=True``) and `gaussx.diag` dispatch here, and a `SpectralFunction`
    factor of a shifted Kronecker product ``A ⊗ f(B) + c·I`` keeps its
    eigenbasis there too.

    Args:
        base: The symmetric operator ``B`` (a `gaussx.KroneckerSum`, a
            `gaussx.DiagonalisedOperator`, a diagonal or a symmetric dense
            operator). Its factors are eigendecomposed with ``eigh``; build
            it from concrete values (outside ``jit``), or use
            `from_eigen_factorizations` to supply the decompositions.
        fn: The elementwise function ``f``, applied to the eigenvalue tensor
            of ``B``. Must be real-valued on the spectrum. Pass an
            `equinox.Module` (or a `jax.tree_util.Partial`) when it carries
            parameters that are traced or differentiated, so they stay
            pytree leaves.
        tags: lineax tags. ``symmetric_tag`` is always added.

    Raises:
        ValueError: If ``base`` has no factored eigenbasis (it is not
            symmetric).

    Examples:
        ```python
        import jax.numpy as jnp
        import lineax as lx
        import gaussx

        def path_laplacian(n):
            off = -jnp.ones(n - 1)
            L = jnp.diag(jnp.r_[1.0, 2.0 * jnp.ones(n - 2), 1.0])
            return lx.MatrixLinearOperator(
                L + jnp.diag(off, 1) + jnp.diag(off, -1), lx.symmetric_tag
            )

        # (κ²I + L_H ⊕ L_W)², a Matérn nu = 1 precision on a 4 x 5 grid
        base = gaussx.KroneckerSum(path_laplacian(4), path_laplacian(5))
        Q = gaussx.SpectralFunction(base, lambda lam: (0.5 + lam) ** 2)
        dense = jnp.linalg.matrix_power(0.5 * jnp.eye(20) + base.as_matrix(), 2)
        assert jnp.allclose(Q.as_matrix(), dense, atol=1e-4)
        assert jnp.allclose(gaussx.diag_inv(Q), jnp.diag(jnp.linalg.inv(dense)))
        ```
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
            _eigen = factored_eigen(base)
            if _eigen is None:
                raise ValueError(
                    "SpectralFunction needs a symmetric base with a factored "
                    f"eigenbasis, got {type(base).__name__}."
                )
        self.base = base
        self.fn = fn
        self.eigen = _eigen
        self._size = int(_eigen.eigenvalues.size)
        spectrum = jax.eval_shape(fn, _eigen.eigenvalues)
        self._dtype = str(jnp.result_type(spectrum.dtype, _eigen.eigenvalues.dtype))
        self.tags = _to_frozenset(tags) | {lx.symmetric_tag}

    @classmethod
    def from_eigen_factorizations(
        cls,
        factors: Sequence[EigenFactorization],
        fn: Callable[[Array], Array],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> SpectralFunction:
        """``f(A₁ ⊕ … ⊕ A_d)`` from precomputed symmetric factorizations.

        Skips the ``eigh`` of `__init__`, so it is cheap inside ``jit``; the
        base is the nested `gaussx.KroneckerSum` of the reassembled factors.

        Args:
            factors: One `gaussx.EigenFactorization` per axis, in Kronecker
                (row-major) order, each of a symmetric matrix (orthonormal
                eigenvectors, as from ``from_matrix(..., symmetric=True)``).
            fn: The elementwise function ``f`` (see the class docstring).
            tags: lineax tags.

        Returns:
            The operator.
        """
        if not factors:
            raise ValueError("factors must contain at least one factorization.")
        bases = tuple(f.eigenvectors for f in factors)
        eigenvalues = None
        for factor in factors:
            if eigenvalues is None:
                eigenvalues = factor.eigenvalues
                continue
            names = " ".join(f"a{m}" for m in range(eigenvalues.ndim))
            eigenvalues = einx.add(
                f"{names}, b -> {names} b", eigenvalues, factor.eigenvalues
            )
        assert eigenvalues is not None
        operators = [
            lx.MatrixLinearOperator(f.as_matrix(), lx.symmetric_tag) for f in factors
        ]
        base = operators[-1]
        for operator in reversed(operators[:-1]):
            base = KroneckerSum(operator, base)
        return cls(base, fn, tags=tags, _eigen=FactoredEigen(bases, eigenvalues))

    def spectrum(self) -> Float[Array, " ..."]:
        """``f(λ)`` on the eigenvalue tensor, shape ``(n_1, …, n_d)``."""
        return jnp.asarray(self.fn(self.eigen.eigenvalues), dtype=self._dtype)

    def factored_eigen(self) -> FactoredEigen:
        """The factored eigendecomposition of ``f(B)`` (same basis as ``B``)."""
        return FactoredEigen(self.eigen.bases, self.spectrum())

    def _apply(self, vector: Array, values: Array) -> Array:
        coeffs = self.eigen.to_eigenbasis(vector)
        return self.eigen._output(self.eigen.from_eigenbasis(coeffs * values), vector)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        return self._apply(vector, self.spectrum())

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """``f(B)⁻¹ b``."""
        return self._apply(vector, 1.0 / self.spectrum())

    def logdet(self) -> Float[Array, ""]:
        """``log|det f(B)| = Σ log|f(λ)|``."""
        return jnp.sum(jnp.log(jnp.abs(self.spectrum())))

    def diag_inv(self, *, pinv: bool = False) -> Float[Array, " n"]:
        """``diag(f(B)⁻¹)`` (or of the pseudo-inverse), from the factor bases."""
        return self.factored_eigen().diag_inv(pinv=pinv)

    def diagonal(self) -> Float[Array, " n"]:
        """``diag(f(B))``, from the factor bases."""
        return FactoredEigen(self.eigen.bases, 1.0 / self.spectrum()).diag_inv()

    def sqrt_matmul(
        self, vector: Float[Array, " n"], *, inverse: bool = False
    ) -> Float[Array, " n"]:
        """``f(B)^{1/2} x``, or ``f(B)^{-1/2} x`` with ``inverse=True``.

        The symmetric square root: with ``inverse=True`` and ``z ~ N(0, I)``,
        ``f(B)^{-1/2} z`` is an exact sample with precision ``f(B)``.
        """
        root = jnp.sqrt(self.spectrum())
        return self._apply(vector, 1.0 / root if inverse else root)

    def as_matrix(self) -> Float[Array, "n n"]:
        eye = jnp.eye(self._size, dtype=self._dtype)
        # Rows are f(B) e_i, i.e. columns of the symmetric matrix.
        return rearrange(jax.vmap(self.mv)(eye), "i j -> j i")

    def transpose(self) -> SpectralFunction:
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
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
