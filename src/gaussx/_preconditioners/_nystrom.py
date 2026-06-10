"""Randomized Nyström preconditioner."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._preconditioners._base import AbstractPreconditioner


class NystromPreconditioner(AbstractPreconditioner):
    r"""Low-rank approximate inverse from randomized operator probing.

    Builds a rank-``k`` Nyström approximation of a symmetric positive
    semidefinite operator ``A`` and uses it as an approximate inverse. Good when
    ``A`` is available only through matvecs and a handful of probes captures its
    dominant spectrum.

    Algorithm (for PSD ``A``):

    1. Draw a Gaussian probe matrix ``Omega in R^{n x k}`` and orthonormalise it
       via QR to get ``Q``.
    2. Form ``Y = A Q`` (``k`` matvecs) and the small matrix ``B = Q^T Y``.
    3. Eigendecompose ``B = U S U^T`` and set ``W = Q U`` (orthonormal columns).
    4. The approximate inverse is
       ``M^{-1} x = a x + W ((s_inv - a) (W^T x))``,
       where ``s_inv = 1 / |eig(B)|`` and the scalar fallback ``a`` keeps
       directions outside the captured subspace near the inverse of the smallest
       captured eigenvalue (so CG does not falsely converge in the
       preconditioned norm).

    Construct via `from_operator`.

    Attributes:
        basis: Orthonormal basis ``W``, shape ``(n, k)``.
        scale: Per-direction extra scaling ``s_inv - a``, shape ``(k,)``.
        shift: Scalar fallback ``a`` applied to the full space.
    """

    basis: Float[Array, "n k"]
    scale: Float[Array, " k"]
    shift: Float[Array, ""]

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        key: jax.Array | None = None,
    ) -> NystromPreconditioner:
        """Build a Nyström preconditioner by probing *operator*.

        Args:
            operator: A symmetric PSD operator ``A``.
            rank: Number of probe vectors (approximation rank).
            key: PRNG key for the probe matrix. Defaults to
                ``jax.random.PRNGKey(0)``.

        Returns:
            A ready-to-use `NystromPreconditioner`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        n = operator.in_size()
        k = min(rank, n)

        omega = jax.random.normal(key, (n, k))
        q, _ = jnp.linalg.qr(omega)

        y = eqx.filter_vmap(operator.mv, in_axes=1, out_axes=1)(q)
        b = einsum(q, y, "n k, n j -> k j")
        # Symmetrize before the eigendecomposition: b is symmetric in exact
        # arithmetic, but floating-point asymmetry in the off-diagonals can
        # perturb eigh. (Matches the convention in _distributions/_conditional.)
        b = symmetrize(b)
        eigvals, u = jnp.linalg.eigh(b)

        abs_eigvals = jnp.abs(eigvals)
        eps = jnp.finfo(abs_eigvals.dtype).eps * n
        s_inv = jnp.where(abs_eigvals > eps, 1.0 / abs_eigvals, 0.0)

        w = einsum(q, u, "n k, k j -> n j")
        # Fallback for uncaptured directions: random probing captures the
        # largest eigenvalues, so uncaptured directions have smaller eigenvalues
        # and larger inverses. The largest captured inverse is the best
        # available proxy and keeps the preconditioned spectrum near 1.
        shift = jnp.max(s_inv)
        scale = s_inv - shift
        return cls(basis=w, scale=scale, shift=shift)

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return the rank-``k`` approximate inverse as a PSD operator."""
        w = self.basis
        scale = self.scale
        shift = self.shift
        structure = jax.ShapeDtypeStruct((w.shape[0],), w.dtype)

        def matvec(x: Float[Array, " n"]) -> Float[Array, " n"]:
            coeffs = einsum(w, x, "n k, n -> k")
            return shift * x + einsum(w, scale * coeffs, "n k, k -> n")

        return lx.FunctionLinearOperator(
            matvec, structure, lx.positive_semidefinite_tag
        )
