"""`SparseCholeskyFactor` and `sparse_cholesky`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._operators._sparse import SparseOperator
from gaussx._sparse import _vjp
from gaussx._sparse._cholmod import cholmod_cholesky
from gaussx._sparse._numeric import lower_values, numeric_cholesky, solve_upper
from gaussx._sparse._symbolic import SymbolicCholesky, symbolic_cholesky
from gaussx._sparse._takahashi import takahashi


class SparseCholeskyFactor(eqx.Module):
    r"""Sparse Cholesky factor ``P Q Pᵀ = L Lᵀ`` on a static symbolic pattern.

    Built by `gaussx.sparse_cholesky` (or `gaussx.cholesky` on a
    `SparseOperator`). ``values`` are the entries of ``L`` on the CSC pattern
    of ``symbolic``; ``matrix_values`` are those of the factored matrix (its
    lower triangle, permuted, on the same pattern), the input that
    ``logdet`` and ``solve`` differentiate through their custom VJPs:

    - ``logdet``: $d\log|Q| = \operatorname{tr}(Q^{-1}dQ)$, so the cotangent is
      $Q^{-1}$ on ``pattern(Q)``, which Takahashi evaluates in one sweep;
    - ``solve``: $\bar b = Q^{-1}\bar x$ and $\bar Q = -\bar b\,x^\top$,
      symmetrised on the pattern.

    Gradients reach the operator's stored values per stored value: with
    ``symmetric=True`` storage an off-diagonal value sets ``Q_ij`` and
    ``Q_ji``, so its cotangent is doubled (``2 Z_ij``); general storage
    factors ``½(Q + Qᵀ)`` and each stored value gets ``Z_ij``. The other
    methods are differentiated by JAX through the factorisation.

    Attributes:
        values: ``L`` on its CSC pattern, shape ``(nnz_L,)``.
        matrix_values: Lower triangle of ``P Q Pᵀ`` on ``L``'s pattern.
        symbolic: The static symbolic analysis.

    Examples:
        ```python
        import jax.numpy as jnp
        import numpy as np
        import gaussx

        # Precision of a path graph 0 - 1 - 2 - 3 plus a unit shift
        n = 4
        Q = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(1, n)],
            np.r_[np.arange(n), np.arange(n - 1)],
            jnp.r_[jnp.array([2.0, 3.0, 3.0, 2.0]), -jnp.ones(n - 1)],
            (n, n),
            symmetric=True,
        )
        factor = gaussx.sparse_cholesky(Q)
        dense = Q.as_matrix()
        assert jnp.allclose(factor.logdet(), jnp.linalg.slogdet(dense)[1])
        b = jnp.ones(n)
        assert jnp.allclose(factor.solve(b), jnp.linalg.solve(dense, b))
        assert jnp.allclose(factor.diag_inv(), jnp.diag(jnp.linalg.inv(dense)))
        ```
    """

    values: Float[Array, " nnz_L"]
    matrix_values: Float[Array, " nnz_L"]
    symbolic: SymbolicCholesky = eqx.field(static=True)

    def _L(self) -> Array:
        # logdet and solve route their first-order cotangents to
        # ``matrix_values`` and give ``L`` none. ``L`` stays differentiable on
        # the JAX backend so that reverse-over-reverse (a Hessian) sees how
        # the Takahashi / adjoint cotangents in their backward passes move
        # with the values. CHOLMOD's callback has no derivative.
        if self.symbolic.backend == "cholmod":
            return jax.lax.stop_gradient(self.values)
        return self.values

    def solve(self, b: Float[Array, " n"]) -> Float[Array, " n"]:
        """``Q⁻¹ b = Pᵀ L⁻ᵀ L⁻¹ P b``.

        Args:
            b: Right-hand side, shape ``(n,)``.

        Returns:
            The solution, shape ``(n,)``.
        """
        sym = self.symbolic
        y = b[jnp.asarray(sym.perm)]
        z = _vjp.solve(sym, self.matrix_values, self._L(), y)
        return z[jnp.asarray(sym.iperm)]

    def logdet(self) -> Float[Array, ""]:
        """``log|Q| = 2 Σ_j log L_jj``.

        Returns:
            The log-determinant (NaN if ``Q`` is not positive definite).
        """
        return _vjp.logdet(self.symbolic, self.matrix_values, self._L())

    def solve_lower_transpose(self, z: Float[Array, " n"]) -> Float[Array, " n"]:
        """``x = Pᵀ L⁻ᵀ z``: with ``z ~ N(0, I)``, ``x ~ N(0, Q⁻¹)``.

        Args:
            z: Shape ``(n,)``.

        Returns:
            ``x``, shape ``(n,)``.
        """
        sym = self.symbolic
        return solve_upper(sym, self.values, z)[jnp.asarray(sym.iperm)]

    def selected_inverse(self) -> SparseOperator:
        """``Q⁻¹`` on the pattern of ``L + Lᵀ``, in the original order.

        The pattern contains ``pattern(Q)``. Like the block selected inverse,
        the result holds entries of ``Q⁻¹``; it is not an operator equal to
        ``Q⁻¹`` (whose other entries are not zero).

        Returns:
            A symmetric `SparseOperator` holding the selected entries.
        """
        pattern, index = self.symbolic.inverse_plan
        Z = takahashi(self.symbolic, self.values)
        return SparseOperator(Z[jnp.asarray(index)], pattern)

    def diag_inv(self) -> Float[Array, " n"]:
        """``diag(Q⁻¹)``, the marginal variances, by one Takahashi sweep.

        Returns:
            Shape ``(n,)``, in the original order.
        """
        sym = self.symbolic
        Z = takahashi(sym, self.values)
        return Z[jnp.asarray(sym.colptr[sym.iperm])]


def sparse_cholesky(
    op: SparseOperator, symbolic: SymbolicCholesky | None = None
) -> SparseCholeskyFactor:
    r"""Sparse Cholesky factorisation of a symmetric positive-definite operator.

    The symbolic analysis (ordering, elimination tree, pattern of ``L``) is
    host-side and cached per pattern; pass one from `gaussx.symbolic_cholesky`
    to choose the ordering or backend, or to reuse it explicitly. The numeric
    phase is traced: it ``jit``s, ``vmap``s over ``op.values`` and is
    differentiable. With the JAX backend it is a left-looking ``lax.scan``
    over columns, sequential in ``n`` and ``O(Σ_j |struct(L_{:,j})|²)`` work.

    Args:
        op: Symmetric positive-definite operator. A general (non-symmetric)
            pattern is factored as ``½(Q + Qᵀ)``.
        symbolic: Symbolic analysis of ``op.pattern``. Defaults to
            ``symbolic_cholesky(op.pattern)`` (RCM ordering, JAX backend).

    Returns:
        The factor.

    Raises:
        TypeError: If ``op`` is not a `SparseOperator`.
        ValueError: If ``symbolic`` was computed for a different pattern.

    Examples:
        ```python
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        import gaussx

        # A 1-D random-walk precision τ R + I: analyse once, factor for many τ
        n = 6
        R = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(1, n)],
            np.r_[np.arange(n), np.arange(n - 1)],
            jnp.r_[jnp.array([1.0] + [2.0] * (n - 2) + [1.0]), -jnp.ones(n - 1)],
            (n, n),
            symmetric=True,
        )
        sym = gaussx.symbolic_cholesky(R.pattern)  # host, once

        def logdet(log_tau):
            Q = eqx.tree_at(lambda op: op.values, R, jnp.exp(log_tau) * R.values)
            return gaussx.sparse_cholesky(Q.add_diagonal(jnp.ones(n)), sym).logdet()

        values = jax.vmap(logdet)(jnp.linspace(-1.0, 1.0, 4))
        slope = jax.grad(logdet)(0.0)  # through one Takahashi sweep
        ```
    """
    if not isinstance(op, SparseOperator):
        raise TypeError(
            f"sparse_cholesky needs a SparseOperator, got {type(op).__name__}."
        )
    if symbolic is None:
        symbolic = symbolic_cholesky(op.pattern)
    elif symbolic.pattern != op.pattern:
        raise ValueError(
            "symbolic was computed for a different sparsity pattern: "
            f"{symbolic.pattern} vs {op.pattern}."
        )
    a = lower_values(symbolic, op.values)
    if symbolic.backend == "cholmod":
        L = cholmod_cholesky(symbolic, a)
    else:
        L = numeric_cholesky(symbolic, a)
    return SparseCholeskyFactor(L, a, symbolic)
