"""Interpolated (SKI / KISS-GP) operator A = W_interp K_ℐℐ W_interpᵀ.

K_ℐℐ ∈ ℝ^{N_ℐ × N_ℐ} is the kernel matrix on the N_ℐ grid points of the
inducing set ℐ, and W_interp ∈ ℝ^{N × N_ℐ} interpolates grid values to N
target inputs, so K_XX ≈ W_interp K_ℐℐ W_interpᵀ. W_interp is stored
sparsely: row n has P nonzeros, the weights w_n,p at grid indices j_n,p,
p = 1, …, P, so [W_interp]_n,j_n,p = w_n,p (repeated indices add).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._operators._low_rank_update import _safe_query


class InterpolatedOperator(lx.AbstractLinearOperator):
    """Structured kernel interpolation A = W_interp K_ℐℐ W_interpᵀ ≈ K_XX.

        A x = W_interp (K_ℐℐ (W_interpᵀ x))

    W_interp is a sparse interpolation matrix with P nonzeros per row (cubic
    interpolation weights, say), and the base operator K_ℐℐ acts on the
    inducing grid (typically Toeplitz for a stationary kernel). A matvec
    costs O(N P + N_ℐ log N_ℐ) with a Toeplitz base, essentially linear in N.

    Args:
        base_operator: The kernel matrix K_ℐℐ on the grid, N_ℐ × N_ℐ.
        interp_indices: Grid indices j_n,p of the nonzeros of W_interp, one
            row of P per target.
        interp_values: Interpolation weights w_n,p, matching
            ``interp_indices``.
        tags: Extra lineax tags. Symmetry and positive or negative
            semi-definiteness are inherited from ``base_operator``
            structurally (W K_ℐℐ Wᵀ preserves them for any real W), so they
            need not be passed; caller tags are added on top.

    Raises:
        ValueError: If ``interp_indices`` and ``interp_values`` differ in
            shape, or ``base_operator`` is not square.

    Examples:
        A Toeplitz RBF kernel on 10 grid points, interpolated to 3 targets:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> grid = gaussx.create_grid([10], [(0.0, 1.0)])
        >>> psd = lx.positive_semidefinite_tag
        >>> K_grid = gaussx.Toeplitz(jnp.exp(-0.5 * (grid[0] / 0.3) ** 2), tags=psd)
        >>> X = jnp.array([[0.12], [0.5], [0.81]])
        >>> indices, weights = gaussx.cubic_interpolation_weights(X, grid)
        >>> A = gaussx.InterpolatedOperator(K_grid, indices, weights)
        >>> A.as_matrix().shape
        (3, 3)

        It inherits symmetry and semi-definiteness from K_ℐℐ:

        >>> lx.is_symmetric(A), lx.is_positive_semidefinite(A)
        (True, True)

        The matvec matches the dense product:

        >>> x = jnp.array([1.0, -1.0, 0.5])
        >>> bool(jnp.allclose(A.mv(x), A.as_matrix() @ x, atol=1e-5))
        True
    """

    base_operator: lx.AbstractLinearOperator
    interp_indices: Int[Array, "n p"]
    interp_values: Float[Array, "n p"]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _m: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base_operator: lx.AbstractLinearOperator,
        interp_indices: Int[Array, "n p"],
        interp_values: Float[Array, "n p"],
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if interp_indices.shape != interp_values.shape:
            raise ValueError(
                f"interp_indices and interp_values must have the same shape, "
                f"got {interp_indices.shape} and {interp_values.shape}."
            )
        # N_ℐ
        m = base_operator.in_size()
        out_m = base_operator.out_size()
        if m != out_m:
            raise ValueError(f"base_operator must be square, got shape ({out_m}, {m}).")
        # N
        n = interp_indices.shape[0]
        self.base_operator = base_operator
        self.interp_indices = jnp.asarray(interp_indices)
        self.interp_values = jnp.asarray(interp_values)
        self._in_size = n
        self._out_size = n
        self._m = m
        self._dtype = str(
            jnp.result_type(
                jnp.dtype(_resolve_dtype(base_operator)),
                self.interp_values.dtype,
            )
        )
        self.tags = _to_frozenset(tags) | _inherited_tags(base_operator)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply A to a vector by scatter, base matvec and gather.

            A x = W_interp (K_ℐℐ (W_interpᵀ x))

        Algorithm:
            1. Scatter: W_interpᵀ x adds w_n,p x_n into grid point j_n,p.
            2. Apply K_ℐℐ on the grid.
            3. Gather: [A x]_n = ∑_p w_n,p [K_ℐℐ W_interpᵀ x]_j_n,p.

        Args:
            vector: Input x, of length N.

        Returns:
            A x, of length N.
        """
        # --- scatter ---
        # 0_{N_ℐ}
        wt_v = jnp.zeros(self._m, dtype=vector.dtype)
        # w_n,p x_n
        # jnp, not einx: N = 0 targets or P = 0 weights leave an empty axis
        weighted = self.interp_values * vector[:, None]
        # W_interpᵀ x
        wt_v = wt_v.at[self.interp_indices].add(weighted)

        # --- base ---
        # K_ℐℐ W_interpᵀ x
        k_wt_v = self.base_operator.mv(wt_v)

        # --- gather ---
        # [K_ℐℐ W_interpᵀ x]_j_n,p
        gathered = k_wt_v[self.interp_indices]
        # w_n,p [K_ℐℐ W_interpᵀ x]_j_n,p
        weighted_gathered = self.interp_values * gathered
        # ∑_p w_n,p [K_ℐℐ W_interpᵀ x]_j_n,p
        # jnp, not einx: N = 0 targets or P = 0 weights leave an empty axis
        return jnp.sum(weighted_gathered, axis=-1)

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense A = W_interp K_ℐℐ W_interpᵀ, N × N.

        Returns:
            The materialised operator.
        """
        W = self._build_W()
        # K_ℐℐ
        K_uu = self.base_operator.as_matrix()
        # W_interp K_ℐℐ
        weighted_base = W @ K_uu
        # W_interp K_ℐℐ W_interpᵀ
        # jnp, not einx: N = 0 targets leave an empty axis
        return weighted_base @ W.T

    def transpose(self) -> InterpolatedOperator:
        """Aᵀ = W_interp K_ℐℐᵀ W_interpᵀ, still an `InterpolatedOperator`.

        A is its own transpose when K_ℐℐ is symmetric.

        Returns:
            The transposed operator, with transposed tags.
        """
        # (W K_ℐℐ Wᵀ)ᵀ = W K_ℐℐᵀ Wᵀ
        return InterpolatedOperator(
            self.base_operator.T,
            self.interp_indices,
            self.interp_values,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def _build_W(self) -> Float[Array, "n m"]:
        """Dense interpolation matrix W_interp, N × N_ℐ.

            [W_interp]_n,j_n,p += w_n,p

        Returns:
            W_interp, with repeated indices in a row summed.
        """
        n = self._in_size
        # 0_{N × N_ℐ}
        W = jnp.zeros((n, self._m), dtype=jnp.dtype(self._dtype))
        # n, broadcast against the P columns of j_n,p
        # jnp, not einx: N = 0 targets leave an empty axis
        rows = jnp.arange(n)[:, None]
        # [W_interp]_n,j_n,p += w_n,p
        W = W.at[rows, self.interp_indices].add(self.interp_values)
        return W


# W A Wᵀ preserves symmetry and semi-definiteness of A for any real W; it
# does not preserve definiteness or diagonality (gh-352).
_INHERITED_TAGS = (
    (lx.is_symmetric, lx.symmetric_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
)


def _inherited_tags(base: lx.AbstractLinearOperator) -> frozenset[object]:
    """Tags W A Wᵀ inherits from A = ``base`` (structural queries only).

    Args:
        base: The base operator A.

    Returns:
        The symmetric and semi-definite tags that hold for A.
    """
    return frozenset(tag for query, tag in _INHERITED_TAGS if _safe_query(query, base))
