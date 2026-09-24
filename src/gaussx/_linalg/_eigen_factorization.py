"""Precomputed eigenfactorizations and shifted Kronecker-sum solves.

These are the building blocks of the *matrix diagonalization* method for
separable operators (e.g. tensor-product spectral discretizations of
``∇² − α``): factor each 1D operator once, then every solve for any shift is
a handful of dense rotations plus a pointwise division.

Unlike `gaussx.KroneckerSum`'s structured solve, which needs symmetric
factors (it uses ``Qᵀ`` as the inverse rotation), `EigenFactorization`
stores the explicit inverse eigenvector matrix and therefore handles
non-symmetric but diagonalizable factors with a real spectrum — such as
Chebyshev collocation second-derivative matrices.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float

from gaussx._einx import einsum


_IMAG_TOLERANCE_FACTOR = 1e3


class EigenFactorization(eqx.Module):
    r"""Precomputed eigendecomposition ``A = V diag(λ) V⁻¹`` of a square matrix.

    Built once (on the host, outside ``jit``) and then reused for any number
    of shifted solves ``(A − σI)⁻¹ b``, each costing two dense mat-vecs —
    ``O(n²)`` — for *any* shift ``σ``, which may be a traced value (so solves
    are ``jit``/``grad``-compatible in ``σ``).

    Only real spectra are supported: the factorization is kept in real
    arithmetic, and `from_matrix` rejects matrices whose eigenvalues have
    non-negligible imaginary parts.

    Attributes:
        eigenvalues: Eigenvalues ``λ``, shape ``(n,)``.
        eigenvectors: Right eigenvectors ``V`` as columns, shape ``(n, n)``.
        eigenvectors_inv: ``V⁻¹``, shape ``(n, n)`` (``Vᵀ`` for symmetric ``A``).

    Example:
        ```python
        fac = gaussx.EigenFactorization.from_matrix(A)
        x = fac.solve_shifted(b, shift=2.0)  # (A - 2 I) x = b
        ```
    """

    eigenvalues: Float[Array, " n"]
    eigenvectors: Float[Array, "n n"]
    eigenvectors_inv: Float[Array, "n n"]

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike | lx.AbstractLinearOperator,
        *,
        symmetric: bool | None = None,
    ) -> EigenFactorization:
        """Eigendecompose a concrete square matrix on the host.

        Uses NumPy (``eigh`` for symmetric input, ``eig`` otherwise), so it
        works for non-symmetric matrices on every JAX backend. Because of
        that, ``matrix`` must be concrete: call this outside ``jit`` (e.g.
        when constructing a solver), not on traced values.

        Args:
            matrix: Square matrix, or a lineax operator (materialized via
                ``as_matrix()``).
            symmetric: Use the symmetric path. Defaults to
                ``lx.is_symmetric(matrix)`` for operators and ``False`` for
                arrays.

        Returns:
            The factorization.

        Raises:
            ValueError: If ``matrix`` is not square, is a tracer, or has a
                spectrum with non-negligible imaginary parts.
        """
        if isinstance(matrix, lx.AbstractLinearOperator):
            if symmetric is None:
                symmetric = lx.is_symmetric(matrix)
            matrix = matrix.as_matrix()
        try:
            mat = np.asarray(matrix)
        except jax.errors.TracerArrayConversionError as err:
            raise ValueError(
                "EigenFactorization.from_matrix needs a concrete matrix "
                "(it factorizes on the host with NumPy); call it outside jit."
            ) from err
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"matrix must be square, got shape {mat.shape}.")

        if symmetric:
            lam, V = np.linalg.eigh(mat)
            V_inv = V.T
        else:
            lam, V = np.linalg.eig(mat)
            scale = max(float(np.max(np.abs(lam))), 1.0)
            tol = _IMAG_TOLERANCE_FACTOR * np.finfo(mat.dtype).eps * scale
            if np.max(np.abs(lam.imag)) > tol:
                raise ValueError(
                    "EigenFactorization supports real spectra only; the largest "
                    f"imaginary part is {np.max(np.abs(lam.imag)):.3e}."
                )
            lam, V = lam.real, V.real
            V_inv = np.linalg.inv(V)
        return cls(jnp.asarray(lam), jnp.asarray(V), jnp.asarray(V_inv))

    def as_matrix(self) -> Float[Array, "n n"]:
        """Reassemble ``V diag(λ) V⁻¹``."""
        return (self.eigenvectors * self.eigenvalues) @ self.eigenvectors_inv

    def solve_shifted(
        self,
        rhs: Float[Array, " n *batch"],
        shift: float | Float[Array, ""] = 0.0,
        *,
        drop: Bool[Array, " n"] | None = None,
    ) -> Float[Array, " n *batch"]:
        """Solve ``(A − shift·I) x = rhs`` along the leading axis of ``rhs``.

        ``x = V diag(1 / (λ − shift)) V⁻¹ rhs``.

        Args:
            rhs: Right-hand side(s), shape ``(n,)`` or ``(n, *batch)``.
            shift: Scalar shift ``σ`` (may be traced).
            drop: Optional boolean mask over eigenmodes. Masked modes get a
                zero coefficient instead of ``1 / (λ − σ)`` — a restricted
                pseudo-inverse, e.g. to project out a known null space such
                as the constant mode of a pure-Neumann Laplacian.

        Returns:
            The solution, same shape as ``rhs``.
        """
        return kronecker_sum_solve((self,), rhs, shift, drop=drop)


def _apply_along_axis(matrix: Float[Array, "m n"], tensor: Array, axis: int) -> Array:
    """Contract ``matrix``'s column index with ``tensor``'s ``axis``.

    ``out[..., i, ...] = Σ_j matrix[i, j] · tensor[..., j, ...]``.
    """
    names = [f"a{k}" for k in range(tensor.ndim)]
    lhs = " ".join([*names[:axis], "col", *names[axis + 1 :]])
    out = " ".join([*names[:axis], "row", *names[axis + 1 :]])
    return einsum(matrix, tensor, f"row col, {lhs} -> {out}")


def kronecker_sum_solve(
    factors: Sequence[EigenFactorization],
    rhs: Float[Array, "*dims"],
    shift: float | Float[Array, ""] = 0.0,
    *,
    drop: Bool[Array, "*dims"] | None = None,
) -> Float[Array, "*dims"]:
    r"""Solve a shifted Kronecker-sum system in tensor (unvectorized) form.

    With ``A_k`` acting on axis ``k`` of the unknown tensor ``X`` (shape
    ``(n_0, …, n_{d−1})``, possibly followed by trailing batch axes), solves

    $$
    \sum_k X \times_k A_k \;-\; \sigma X \;=\; R,
    $$

    i.e. ``(A_0 ⊕ A_1 ⊕ … ⊕ A_{d−1} − σ I) vec(X) = vec(R)``. In 2D this is
    the Sylvester equation ``A_0 X + X A_1ᵀ − σX = R``. With
    ``A_k = V_k Λ_k V_k⁻¹``:

    1. ``X̂ = R ×_0 V_0⁻¹ ×_1 V_1⁻¹ …`` (rotate into the eigenbasis),
    2. ``X̂[i_0, …] /= λ⁰_{i_0} + λ¹_{i_1} + … − σ``,
    3. ``X = X̂ ×_0 V_0 ×_1 V_1 …`` (rotate back).

    Cost is ``O(N · Σ_k n_k)`` for ``N = Π_k n_k`` unknowns, and the
    ``N × N`` Kronecker matrix is never formed; factors need not be
    symmetric.

    Args:
        factors: One `EigenFactorization` per leading axis of ``rhs``.
        rhs: Right-hand side, shape ``(n_0, …, n_{d−1}, *batch)``.
        shift: Scalar shift ``σ`` (may be traced).
        drop: Optional boolean mask of shape ``(n_0, …, n_{d−1})``; masked
            eigen-combinations get a zero coefficient (restricted
            pseudo-inverse), as in `EigenFactorization.solve_shifted`.

    Returns:
        ``X``, same shape as ``rhs``.

    Raises:
        ValueError: If the factor sizes do not match the leading axes of
            ``rhs``.
    """
    d = len(factors)
    sizes = tuple(f.eigenvalues.shape[0] for f in factors)
    if rhs.shape[:d] != sizes:
        raise ValueError(
            f"rhs leading shape {rhs.shape[:d]} does not match factor sizes {sizes}."
        )

    x = rhs
    for k, fac in enumerate(factors):
        x = _apply_along_axis(fac.eigenvectors_inv, x, k)

    # Σ_k λᵏ broadcast over the d eigen axes: shape (n_0, …, n_{d−1}).
    lam_sum = jnp.zeros((), dtype=factors[0].eigenvalues.dtype)
    for k, fac in enumerate(factors):
        other_axes = tuple(j for j in range(d) if j != k)
        lam_sum = lam_sum + jnp.expand_dims(fac.eigenvalues, other_axes)
    denom = lam_sum - shift
    if drop is None:
        inv = 1.0 / denom
    else:
        inv = jnp.where(drop, 0.0, 1.0 / jnp.where(drop, 1.0, denom))
    inv = jnp.expand_dims(inv, tuple(range(d, x.ndim)))  # broadcast over batch axes
    x = x * inv

    for k, fac in enumerate(factors):
        x = _apply_along_axis(fac.eigenvectors, x, k)
    return x
