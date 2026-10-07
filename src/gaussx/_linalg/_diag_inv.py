"""Diagonal of the inverse: compute diag(A⁻¹) without forming the full inverse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._linalg._safe_cholesky import safe_cholesky
from gaussx._linalg._selected_inverse import selected_inverse
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._factored_eigen import factored_eigen
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import SumOfKroneckers, _shifted_kronecker_eigen
from gaussx._primitives._diag import diag
from gaussx._primitives._inv import InverseOperator, inv
from gaussx._sparse._factor import SparseCholeskyFactor
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._dispatch import dispatch_solve
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


def diag_inv(
    operator: lx.AbstractLinearOperator,
    *,
    method: str = "auto",
    num_probes: int = 30,
    key: jax.Array | None = None,
    solver: AbstractSolveStrategy | None = None,
    pinv: bool = False,
) -> Float[Array, " N"]:
    """Compute the diagonal of the inverse of a linear operator.

    Returns ``diag(A⁻¹)`` without forming the full inverse matrix — the
    marginal variances of a Gaussian with precision ``A``.

    With ``method="auto"`` the operator's structure picks an exact path
    first:

    - ``DiagonalLinearOperator``: ``1 / d``; `BlockDiag`: per block, each
      dispatched in turn; a symmetric `LowRankUpdate`: the diagonal of its
      Woodbury inverse, ``O(N k²)``; ``c · A``, ``A / c`` and ``-A``: the
      structured path of ``A``, rescaled (gh-365).
    - `BlockTriDiag` (symmetric): the diagonal blocks of
      `gaussx.selected_inverse`, ``O(N d³)``.
    - `Kronecker` ``A ⊗ B``: ``diag_inv(A) ⊗ diag_inv(B)``, each factor
      dispatched in turn.
    - `KroneckerSum` ``A ⊕ B`` (and a `DiagonalizedOperator`, or a
      `SpectralFunction` ``f(A ⊕ B)``, with ``M_ij = 1/f(λ^A_i + λ^B_j)``): with
      ``A = U_A Λ_A U_Aᵀ``, ``B = U_B Λ_B U_Bᵀ``,
      ``diag((A ⊕ B)⁻¹) = (U_A ∘ U_A) M (U_B ∘ U_B)ᵀ`` with
      ``M_ij = 1/(λ^A_i + λ^B_j)`` — two small matrix products,
      ``O(H²W + HW²)`` on an ``H × W`` grid. A factor that carries its own
      eigenbasis keeps it; others are eigendecomposed individually.
    - Shifted Kronecker products ``A ⊗ B + c·I`` (a `SumOfKroneckers` or the
      equivalent lineax sum): the same formula with
      ``M_ij = 1/(λ^A_i λ^B_j + c)``.
    - `SparseOperator` with ``solver=SparseCholeskySolver(...)``: Takahashi's
      selected inverse through the sparse Cholesky factor, exact at
      ``O(Σ_j |struct(L_{:,j})|²)``, at any size.

    Anything else uses Cholesky for ``N ≤ 2048`` (the sparse factor for a
    `SparseOperator`, dense otherwise) and Hutchinson above.

    Args:
        operator: A linear operator representing A.
        method: Algorithm to use. One of ``"cholesky"`` (exact via
            dense Cholesky), ``"solve"`` (exact via repeated solves),
            ``"hutchinson"`` (stochastic estimator),
            or ``"auto"`` (the structured paths above; otherwise cholesky
            for N ≤ 2048, hutchinson above). ``"cholesky"`` on a
            `SparseOperator` is the sparse factor's Takahashi sweep.
        num_probes: Number of Rademacher probe vectors for the
            hutchinson method.
        key: PRNG key for probe generation in the hutchinson method.
            When ``None``, defaults to ``jax.random.PRNGKey(0)``.
        solver: Optional solve strategy for ``"solve"`` and
            ``"hutchinson"`` methods.
        pinv: Return the diagonal of the pseudo-inverse instead: eigenvalues
            below ``max(n_k) · eps · max|λ|`` contribute nothing. For
            intrinsic (singular) precisions on grids, e.g. the exact ICAR /
            BYM2 scaling constant on a raster. Only the eigenvalue-based
            paths (Kronecker sums and products of such operators, shifted
            Kronecker products, `DiagonalizedOperator`, `SpectralFunction`)
            support it.

    Returns:
        1D array of shape ``(N,)`` with the diagonal entries of A⁻¹.

    Raises:
        ValueError: For an unknown ``method``, or ``pinv=True`` on an
            operator without an eigenvalue-based path.

    Examples:
        ```python
        import jax.numpy as jnp
        import lineax as lx
        import gaussx

        # Prior sd of an intrinsic field on a 4 x 5 grid: L_H ⊕ L_W is
        # singular (constants), so take the pseudo-inverse.
        def path_laplacian(n):
            off = -jnp.ones(n - 1)
            L = jnp.diag(jnp.r_[1.0, 2.0 * jnp.ones(n - 2), 1.0])
            L = L + jnp.diag(off, 1) + jnp.diag(off, -1)
            return lx.MatrixLinearOperator(L, lx.symmetric_tag)

        Q = gaussx.KroneckerSum(path_laplacian(4), path_laplacian(5))
        variances = gaussx.diag_inv(Q, pinv=True)
        dense = jnp.diag(jnp.linalg.pinv(Q.as_matrix()))
        assert jnp.allclose(variances, dense, atol=1e-5)
        ```
    """
    n = operator.in_size()

    if method == "auto":
        structured = _diag_inv_structured(
            operator, pinv=pinv, num_probes=num_probes, key=key, solver=solver
        )
        if structured is not None:
            return structured
        method = "cholesky" if n <= 2048 else "hutchinson"

    if pinv:
        msg = (
            "pinv=True needs an eigenvalue-based structure (KroneckerSum, "
            "Kronecker, a shifted Kronecker product or DiagonalizedOperator) "
            f"and method='auto'; got {type(operator).__name__} with "
            f"method={method!r}."
        )
        raise ValueError(msg)

    if method == "cholesky":
        if isinstance(operator, SparseOperator):
            return _sparse_factor(operator, solver).diag_inv()
        return _diag_inv_cholesky(operator)
    if method == "solve":
        return _diag_inv_solve(operator, solver=solver)
    if method == "hutchinson":
        return _diag_inv_hutchinson(
            operator, num_probes=num_probes, key=key, solver=solver
        )

    msg = (
        f"Unknown method {method!r}; expected 'cholesky', 'solve', "
        "'hutchinson', or 'auto'."
    )
    raise ValueError(msg)


def _diag_inv_structured(
    operator: lx.AbstractLinearOperator,
    *,
    pinv: bool,
    num_probes: int,
    key: jax.Array | None,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"] | None:
    """The exact structural path for ``operator``, or ``None`` if none applies."""
    if isinstance(operator, lx.TaggedLinearOperator):
        return _diag_inv_structured(
            operator.operator,
            pinv=pinv,
            num_probes=num_probes,
            key=key,
            solver=solver,
        )
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        # diag((c A)⁻¹) = diag(A⁻¹) / c, also for the pseudo-inverse (gh-365).
        inner = _diag_inv_structured(
            operator.operator,
            pinv=pinv,
            num_probes=num_probes,
            key=key,
            solver=solver,
        )
        if inner is None:
            return None
        if isinstance(operator, lx.DivLinearOperator):
            return inner * operator.scalar
        if pinv:
            # (0 · A)⁺ = 0, so a zero multiplier gives a zero diagonal.
            scalar = operator.scalar
            is_zero = scalar == 0
            return jnp.where(is_zero, 0.0, inner / jnp.where(is_zero, 1.0, scalar))
        return inner / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        inner = _diag_inv_structured(
            operator.operator,
            pinv=pinv,
            num_probes=num_probes,
            key=key,
            solver=solver,
        )
        return None if inner is None else -inner
    if isinstance(operator, lx.DiagonalLinearOperator):
        return None if pinv else 1.0 / lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        if pinv or any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        return jnp.concatenate(
            [
                diag_inv(op, num_probes=num_probes, key=key, solver=solver)
                for op in operator.operators
            ]
        )
    if isinstance(operator, LowRankUpdate):
        # Woodbury keeps inv(A) a LowRankUpdate, whose diagonal is O(N k). It
        # solves against the base, so only a base known to be invertible and
        # cheap qualifies: an identity, or a concrete nonzero diagonal. A
        # singular base (e.g. the zero base of `ensemble_covariance`) or a
        # dense one keeps the general path.
        if pinv or not _invertible_diagonal_base(operator.base):
            return None
        inverse = inv(operator)
        return None if isinstance(inverse, InverseOperator) else diag(inverse)
    if isinstance(operator, SparseOperator):
        if pinv or not isinstance(solver, SparseCholeskySolver):
            return None
        return solver.diag_inv(operator)
    if isinstance(operator, BlockTriDiag):
        if pinv or not operator.symmetric:
            return None
        blocks = selected_inverse(operator).diagonal
        return rearrange(blocks, "N d d -> (N d)")
    if isinstance(operator, Kronecker):
        if any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        result = None
        for factor in operator.operators:
            factor_diag = diag_inv(
                factor, pinv=pinv, num_probes=num_probes, key=key, solver=solver
            )
            result = (
                factor_diag
                if result is None
                else einsum(result, factor_diag, "a, b -> (a b)")
            )
        return result
    if isinstance(operator, KroneckerSum | DiagonalizedOperator | SpectralFunction):
        factorization = factored_eigen(operator)
        return None if factorization is None else factorization.diag_inv(pinv=pinv)
    if isinstance(operator, SumOfKroneckers | lx.AddLinearOperator):
        factorization = _shifted_kronecker_eigen(operator)
        return None if factorization is None else factorization.diag_inv(pinv=pinv)
    return None


def _invertible_diagonal_base(base: lx.AbstractLinearOperator) -> bool:
    if isinstance(base, lx.IdentityLinearOperator):
        return base.in_size() == base.out_size()
    if not isinstance(base, lx.DiagonalLinearOperator):
        return False
    try:
        d = np.asarray(lx.diagonal(base))
    except jax.errors.TracerArrayConversionError:
        return False
    return bool(np.all(d != 0))


def _sparse_factor(
    operator: SparseOperator, solver: AbstractSolveStrategy | None
) -> SparseCholeskyFactor:
    """The sparse factor, with ``solver``'s ordering and backend if it has them."""
    if not isinstance(solver, SparseCholeskySolver):
        solver = SparseCholeskySolver()
    return solver.factor(operator)


def _diag_inv_cholesky(operator: lx.AbstractLinearOperator) -> Float[Array, " N"]:
    """Exact diagonal of A⁻¹ via Cholesky factorisation.

    Uses `safe_cholesky` for robustness on ill-conditioned
    matrices, then computes L⁻¹ via triangular solve against the
    identity and returns ``sum_j (L⁻¹)_{j,i}²`` for each column i.
    """
    L = safe_cholesky(operator)
    n = L.shape[0]
    I = jnp.eye(n, dtype=L.dtype)
    L_inv = jax.scipy.linalg.solve_triangular(L, I, lower=True)
    return reduce(L_inv**2, "M N -> N", "sum")


def _diag_inv_solve(
    operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"]:
    """Exact diagonal of A⁻¹ via repeated solves against basis vectors."""
    n = operator.in_size()
    dtype = operator.out_structure().dtype
    I = jnp.eye(n, dtype=dtype)

    def _single_basis(e_i: Float[Array, " N"]) -> Float[Array, ""]:
        Ainv_ei = dispatch_solve(operator, e_i, solver)
        return jnp.sum(e_i * Ainv_ei)

    return jax.vmap(_single_basis)(I)


def _diag_inv_hutchinson(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    key: jax.Array | None,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"]:
    """Stochastic diagonal estimator via Hutchinson's trick.

    Generates Rademacher probe vectors z, solves A⁻¹ z, and
    estimates ``diag(A⁻¹) ≈ mean(z ⊙ A⁻¹z)`` over probes.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    dtype = operator.out_structure().dtype
    keys = jax.random.split(key, num_probes)

    def _single_probe(k: jax.Array) -> Float[Array, " N"]:
        z = 2.0 * jax.random.bernoulli(k, shape=(n,)).astype(dtype) - 1.0
        Ainv_z = dispatch_solve(operator, z, solver)
        return z * Ainv_z

    samples = jax.vmap(_single_probe)(keys)
    return jnp.mean(samples, axis=0)
