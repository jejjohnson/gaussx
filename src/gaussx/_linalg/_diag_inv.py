"""Diagonal of the inverse: compute diag(A⁻¹) without forming the full inverse."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
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
    """Diagonal of the inverse, diag(A⁻¹), without forming A⁻¹.

        diag(A⁻¹)_i = [A⁻¹]_ii

    For a Gaussian with precision A these are the marginal variances.

    With ``method="auto"`` the structure of A picks an exact path first:

    - ``DiagonalLinearOperator`` D: 1 / D_ii. `BlockDiag`: block by block,
      each dispatched in turn. A symmetric `LowRankUpdate`: the diagonal of
      its Woodbury inverse, O(N k²). c A, A / c and −A: the structured path
      of A, rescaled (gh-365).
    - `BlockTriDiag` (symmetric): the diagonal blocks of
      `gaussx.selected_inverse`, O(N d³).
    - `Kronecker` A ⊗ B: diag(A⁻¹) ⊗ diag(B⁻¹), each factor dispatched in
      turn.
    - `KroneckerSum` A ⊕ B (and a `DiagonalizedOperator`, or a
      `SpectralFunction` f(A ⊕ B), with M_ij = 1 / f(γ^A_i + γ^B_j)): with
      A = U_A diag(γ^A) U_Aᵀ and B = U_B diag(γ^B) U_Bᵀ,
      diag((A ⊕ B)⁻¹) = (U_A ⊙ U_A) M (U_B ⊙ U_B)ᵀ with
      M_ij = 1 / (γ^A_i + γ^B_j), two small matrix products, O(H² W + H W²)
      on an H × W grid. A factor that carries its own eigenbasis keeps it;
      the others are eigendecomposed one by one.
    - A shifted Kronecker product A ⊗ B + c I (a `SumOfKroneckers` or the
      equivalent lineax sum): the same formula with
      M_ij = 1 / (γ^A_i γ^B_j + c).
    - `SparseOperator` with ``solver=SparseCholeskySolver(...)``: Takahashi's
      selected inverse through the sparse Cholesky factor, exact at
      O(∑_j |struct(L_:,j)|²), at any size.

    Anything else uses Cholesky for N ≤ 2048 (the sparse factor for a
    `SparseOperator`, dense otherwise) and Hutchinson above.

    XDiag (``method="xdiag"``, Epperly, Tropp & Webber, 2024) is the
    variance-reduced estimator for operators too large to factor (the
    Laplace Hessian of a GMRF on a 10⁶-node mesh, say). For B = A⁻¹, probes
    Ω = [ω_1, …, ω_k] (the letters of the paper) and Q = orth(B Ω),

        diag(B) = diag(Q Qᵀ B) + diag((I − Q Qᵀ) B)

    with the first term exact and the second estimated by Hutchinson,
    diag(M) ≈ ω ⊙ M ω. XDiag uses every probe for both terms: the i-th
    estimate builds Q_−i from the other k − 1 probes and applies Hutchinson
    with ω_i, and the result is the mean of the k leave-one-out estimates.
    A GMRF covariance has a few large-variance (smooth, large-scale)
    directions where plain Hutchinson is noisy, and the low-rank part
    captures exactly those. It costs 2k solves (B Ω and Bᵀ Q), so compare
    it with Hutchinson at ``num_probes = 2k``. With 2k ≥ N it returns the
    exact diagonal from N solves instead.

    It wraps matfree's ``stochtrace.leave_one_out_xdiag``, which reads a
    rank-deficient A⁻¹ Ω as a low-rank A⁻¹ and returns the projected
    diagonal as exact. k random-sign vectors in N dimensions are dependent
    with probability about k² 2⁻ᴺ (two equal or opposite columns), so
    signs, which estimate a diagonal with less variance, are used only when
    that is below 2⁻⁶⁰; smaller problems use Gaussian probes, dependent with
    probability zero. The solves go through ``solver`` (any solve strategy;
    for a precision matrix, Jacobi-preconditioned CG, since the Nyström and
    RPCholesky preconditioners are for covariance form). Prefer an exact
    path (the structured ones above, or Takahashi through
    ``SparseCholeskySolver``) whenever the operator factors; XDiag beats
    Hutchinson whenever the spectrum of A⁻¹ decays, and matches it when the
    spectrum is flat.

    Algorithm:
        1. With ``method="auto"``, try the exact structured paths above;
           otherwise pick Cholesky for N ≤ 2048 and Hutchinson above.
        2. Cholesky: L Lᵀ = A, then diag(A⁻¹)_i = ∑_j [L⁻¹]_ji².
        3. Solve: diag(A⁻¹)_i = e_iᵀ A⁻¹ e_i, one solve per basis vector.
        4. Hutchinson: diag(A⁻¹) ≈ mean over probes of z ⊙ A⁻¹ z with
           Rademacher z.
        5. XDiag: Y = A⁻¹ Ω and Q R = Y (k solves), Z = A⁻ᵀ Q (k more), then
           the mean over i of diag(Q_−i Q_−iᵀ B) + ω_i ⊙ (I − Q_−i Q_−iᵀ) B ω_i,
           in closed form.

    Args:
        operator: The operator A.
        method: Algorithm to use. One of ``"cholesky"`` (exact via dense
            Cholesky), ``"solve"`` (exact via repeated solves),
            ``"hutchinson"`` (stochastic estimator), ``"xdiag"`` (the
            variance-reduced estimator above), or ``"auto"`` (the
            structured paths above; otherwise cholesky for N ≤ 2048,
            hutchinson above). ``"cholesky"`` on a `SparseOperator` is the
            sparse factor's Takahashi sweep.
        num_probes: Number of probe vectors k for the hutchinson and xdiag
            methods (xdiag: 1 ≤ k ≤ N, and 2k solves).
        key: PRNG key for the probes of the hutchinson and xdiag methods.
            When ``None``, defaults to ``jax.random.PRNGKey(0)``.
        solver: Optional solve strategy for the ``"solve"``,
            ``"hutchinson"`` and ``"xdiag"`` methods.
        pinv: Return the diagonal of the pseudo-inverse A⁺ instead:
            eigenvalues below max(n_k) · eps · max|γ| contribute nothing.
            For intrinsic (singular) precisions on grids, such as the exact
            ICAR / BYM2 scaling constant on a raster. Only the
            eigenvalue-based paths (Kronecker sums and products of such
            operators, shifted Kronecker products, `DiagonalizedOperator`,
            `SpectralFunction`) support it.

    Returns:
        The diagonal of A⁻¹ (or of A⁺ with ``pinv=True``).

    Raises:
        ValueError: For an unknown ``method``, ``pinv=True`` on an
            operator without an eigenvalue-based path, or ``method="xdiag"``
            with ``num_probes`` outside [1, N].

    References:
        Epperly, E. N., Tropp, J. A. & Webber, R. J. (2024). XTrace: making
        the most of every sample in stochastic trace estimation. *SIAM J.
        Matrix Anal. Appl.*, 45(1), 1-23.

    Examples:
        Prior variances of an intrinsic field on a 4 × 5 grid. The
        Kronecker sum of two path-graph Laplacians is singular (the
        constants), so take the pseudo-inverse:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def path_laplacian(size):
        ...     off_diagonal = -jnp.ones(size - 1)
        ...     degrees = jnp.r_[1.0, 2.0 * jnp.ones(size - 2), 1.0]
        ...     laplacian = jnp.diag(degrees) + jnp.diag(off_diagonal, 1)
        ...     laplacian = laplacian + jnp.diag(off_diagonal, -1)
        ...     return lx.MatrixLinearOperator(laplacian, lx.symmetric_tag)
        >>> precision = gaussx.KroneckerSum(path_laplacian(4), path_laplacian(5))
        >>> variances = gaussx.diag_inv(precision, pinv=True)
        >>> dense = jnp.diag(jnp.linalg.pinv(precision.as_matrix()))
        >>> bool(jnp.allclose(variances, dense, atol=1e-5))
        True

        XDiag with every probe it can take (k = N) is exact, through any
        solve strategy:

        >>> import jax
        >>> precision = lx.MatrixLinearOperator(
        ...     jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]),
        ...     lx.positive_semidefinite_tag,
        ... )
        >>> estimate = gaussx.diag_inv(
        ...     precision, method="xdiag", num_probes=3, key=jax.random.key(0)
        ... )
        >>> exact = jnp.diag(jnp.linalg.inv(precision.as_matrix()))
        >>> bool(jnp.allclose(estimate, exact, atol=1e-4))
        True

        A diagonal operator takes its exact path:

        >>> diagonal = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> gaussx.diag_inv(diagonal).tolist()
        [0.5, 0.25]
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
    if method == "xdiag":
        return _diag_inv_xdiag(operator, num_probes=num_probes, key=key, solver=solver)

    msg = (
        f"Unknown method {method!r}; expected 'cholesky', 'solve', "
        "'hutchinson', 'xdiag', or 'auto'."
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
    """The exact structured path for diag(A⁻¹), or ``None`` if none applies.

    Also used by `gaussx.MultivariateNormalPrecision` for its marginal variances.

    Args:
        operator: The operator A.
        pinv: Return the diagonal of A⁺ instead.
        num_probes: Passed on to the recursive calls on blocks and factors.
        key: Passed on to the recursive calls.
        solver: Solve strategy; a `SparseCholeskySolver` enables the sparse
            Takahashi path.

    Returns:
        diag(A⁻¹) (or diag(A⁺)), or ``None`` when no structured path applies.
    """
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
            # diag((A / c)⁻¹) = c diag(A⁻¹)
            return inner * operator.scalar
        if pinv:
            # (0 · A)⁺ = 0, so a zero multiplier gives a zero diagonal.
            scalar = operator.scalar
            is_zero = scalar == 0
            safe_scalar = jnp.where(is_zero, 1.0, scalar)
            # diag((c A)⁺) = diag(A⁺) / c, or 0 for c = 0
            return jnp.where(is_zero, 0.0, inner / safe_scalar)
        # diag((c A)⁻¹) = diag(A⁻¹) / c
        return inner / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        inner = _diag_inv_structured(
            operator.operator,
            pinv=pinv,
            num_probes=num_probes,
            key=key,
            solver=solver,
        )
        # diag((−A)⁻¹) = −diag(A⁻¹)
        return None if inner is None else -inner
    if isinstance(operator, lx.DiagonalLinearOperator):
        # diag(D⁻¹)_i = 1 / D_ii
        return None if pinv else 1.0 / lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        if pinv or any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        # diag(A⁻¹) = [diag(A_1⁻¹); diag(A_2⁻¹); …], block by block
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
        # (A₀ + U D Vᵀ)⁻¹ by Woodbury, still a low-rank update
        inverse = inv(operator)
        return None if isinstance(inverse, InverseOperator) else diag(inverse)
    if isinstance(operator, SparseOperator):
        if pinv or not isinstance(solver, SparseCholeskySolver):
            return None
        # Takahashi through the sparse Cholesky factor
        return solver.diag_inv(operator)
    if isinstance(operator, BlockTriDiag):
        if pinv or not operator.symmetric:
            return None
        # Σ_k,k, the diagonal blocks of Σ = A⁻¹
        blocks = selected_inverse(operator).diagonal
        # diag(Σ_k,k), stacked over the blocks
        return rearrange(blocks, "N d d -> (N d)")
    if isinstance(operator, Kronecker):
        if any(op.in_size() != op.out_size() for op in operator.operators):
            return None
        result = None
        for factor in operator.operators:
            factor_diag = diag_inv(
                factor, pinv=pinv, num_probes=num_probes, key=key, solver=solver
            )
            # diag((A ⊗ B)⁻¹) = diag(A⁻¹) ⊗ diag(B⁻¹)
            result = (
                factor_diag
                if result is None
                else einsum(result, factor_diag, "a, b -> (a b)")
            )
        return result
    if isinstance(operator, KroneckerSum | DiagonalizedOperator | SpectralFunction):
        # (U_A ⊙ U_A) M (U_B ⊙ U_B)ᵀ, M_ij = 1 / f(γ^A_i + γ^B_j)
        factorization = factored_eigen(operator)
        return None if factorization is None else factorization.diag_inv(pinv=pinv)
    if isinstance(operator, SumOfKroneckers | lx.AddLinearOperator):
        # (U_A ⊙ U_A) M (U_B ⊙ U_B)ᵀ, M_ij = 1 / (γ^A_i γ^B_j + c)
        factorization = _shifted_kronecker_eigen(operator)
        return None if factorization is None else factorization.diag_inv(pinv=pinv)
    return None


def _invertible_diagonal_base(base: lx.AbstractLinearOperator) -> bool:
    """Whether a low-rank base A₀ is an identity or a concrete nonzero diagonal.

    Args:
        base: The base A₀.

    Returns:
        ``True`` when A₀ is known to be cheap to invert; ``False`` for a
        traced, singular or non-diagonal base.
    """
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
    """The sparse Cholesky factor of A, with the ordering of ``solver`` if any.

    Args:
        operator: The sparse operator A.
        solver: A `SparseCholeskySolver` whose ordering and backend to use;
            anything else gets the default one.

    Returns:
        The sparse factor of A.
    """
    if not isinstance(solver, SparseCholeskySolver):
        solver = SparseCholeskySolver()
    return solver.factor(operator)


def _diag_inv_cholesky(operator: lx.AbstractLinearOperator) -> Float[Array, " N"]:
    """Exact diagonal of A⁻¹ through a dense Cholesky factor.

        diag(A⁻¹)_i = ∑_j [L⁻¹]_ji²,   A = L Lᵀ

    Uses `safe_cholesky` for robustness on ill-conditioned matrices.

    Args:
        operator: The operator A.

    Returns:
        The diagonal of A⁻¹.
    """
    # A = L Lᵀ
    factor = safe_cholesky(operator)
    size = factor.shape[0]
    identity = jnp.eye(size, dtype=factor.dtype)
    # L⁻¹
    factor_inverse = jax.scipy.linalg.solve_triangular(factor, identity, lower=True)
    # diag(A⁻¹)_i = ∑_j [L⁻¹]_ji²
    return reduce(factor_inverse**2, "M N -> N", "sum")


def _diag_inv_solve(
    operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"]:
    """Exact diagonal of A⁻¹ by one solve per basis vector.

        diag(A⁻¹)_i = e_iᵀ A⁻¹ e_i

    Args:
        operator: The operator A.
        solver: Solve strategy; ``None`` uses structural dispatch.

    Returns:
        The diagonal of A⁻¹.
    """
    size = operator.in_size()
    dtype = operator.out_structure().dtype
    identity = jnp.eye(size, dtype=dtype)

    def _single_basis(basis_vector: Float[Array, " N"]) -> Float[Array, ""]:
        # A⁻¹ e_i
        solved = dispatch_solve(operator, basis_vector, solver)
        # e_iᵀ A⁻¹ e_i
        return jnp.sum(basis_vector * solved)

    return jax.vmap(_single_basis)(identity)


def _diag_inv_hutchinson(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    key: jax.Array | None,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"]:
    """Hutchinson estimate of diag(A⁻¹) from Rademacher probes.

        diag(A⁻¹) ≈ (1 / k) ∑_e z⁽ᵉ⁾ ⊙ A⁻¹ z⁽ᵉ⁾,   z⁽ᵉ⁾ ∈ {−1, 1}ᴺ

    Args:
        operator: The operator A.
        num_probes: Number of probes k.
        key: PRNG key for the probes; ``None`` uses
            ``jax.random.PRNGKey(0)``.
        solver: Solve strategy; ``None`` uses structural dispatch.

    Returns:
        The estimated diagonal of A⁻¹.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    size = operator.in_size()
    dtype = operator.out_structure().dtype
    probe_keys = jax.random.split(key, num_probes)

    def _single_probe(probe_key: jax.Array) -> Float[Array, " N"]:
        # z⁽ᵉ⁾ ∈ {−1, 1}ᴺ
        probe = 2.0 * jax.random.bernoulli(probe_key, shape=(size,)).astype(dtype) - 1.0
        # A⁻¹ z⁽ᵉ⁾
        solved = dispatch_solve(operator, probe, solver)
        # z⁽ᵉ⁾ ⊙ A⁻¹ z⁽ᵉ⁾
        return probe * solved

    samples = jax.vmap(_single_probe)(probe_keys)
    # (1 / k) ∑_e z⁽ᵉ⁾ ⊙ A⁻¹ z⁽ᵉ⁾
    return reduce(samples, "member row -> row", "mean")


def _is_symmetric(operator: lx.AbstractLinearOperator) -> bool:
    """Whether lineax knows A is symmetric; ``False`` when it cannot tell.

    Args:
        operator: The operator A.

    Returns:
        ``True`` when A is known to be symmetric.
    """
    try:
        return lx.is_symmetric(operator)
    except NotImplementedError:
        return False


def _diag_inv_xdiag(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int,
    key: jax.Array | None,
    solver: AbstractSolveStrategy | None,
) -> Float[Array, " N"]:
    """XDiag estimate of diag(A⁻¹) (Epperly, Tropp & Webber, 2024), via matfree.

    Args:
        operator: The operator A.
        num_probes: Number of probes k, 1 ≤ k ≤ N.
        key: PRNG key for the probes; ``None`` uses
            ``jax.random.PRNGKey(0)``.
        solver: Solve strategy; ``None`` uses structural dispatch.

    Returns:
        The estimated diagonal of A⁻¹.

    Raises:
        ValueError: If ``num_probes`` is outside [1, N].
    """
    n = operator.in_size()
    if not 1 <= num_probes <= n:
        msg = (
            f"diag_inv(method='xdiag') needs 1 <= num_probes <= N = {n}, "
            f"got {num_probes}."
        )
        raise ValueError(msg)
    if key is None:
        key = jax.random.PRNGKey(0)
    symmetric = _is_symmetric(operator)

    def transpose_solve(_: object, b: Float[Array, " N"]) -> Float[Array, " N"]:
        # A⁻ᵀ b
        return dispatch_solve(operator.T, b, solver)

    def apply_inverse(v: Float[Array, " N"]) -> Float[Array, " N"]:
        # XDiag needs A⁻ᵀ too (matfree takes the linear transpose). A custom
        # linear solve transposes by solving again (with A when symmetric,
        # Aᵀ otherwise) instead of transposing the solver's internals, so
        # any solve strategy works, including non-lineax loops.
        # A⁻¹ v
        return jax.lax.custom_linear_solve(
            operator.mv,
            v,
            solve=lambda _, b: dispatch_solve(operator, b, solver),
            transpose_solve=None if symmetric else transpose_solve,
            symmetric=symmetric,
        )

    template = jnp.zeros((n,), dtype=operator.out_structure().dtype)
    estimate = matfree.stochtrace.estimator_leave_one_out(
        matfree.stochtrace.leave_one_out_xdiag(),
        _xdiag_sampler(template, num_probes),
    )
    return estimate(apply_inverse, key)


# k sign vectors in N dimensions are dependent with probability ~ k² 2^-N
# (pairs that are equal or opposite dominate); signs are used below 2^-60.
_SIGN_DEPENDENCE_LOG2 = -60


def _xdiag_sampler(template: Float[Array, " N"], num_probes: int):
    """Random signs when they are (almost surely) independent, else Gaussian.

    matfree's XDiag treats a rank-deficient probe image as a low-rank
    operator and returns the projected diagonal as exact, so dependent
    probes would bias the estimate (gh-485 review).

    Args:
        template: A vector with the size N and dtype of the probes.
        num_probes: Number of probes k.

    Returns:
        A matfree sampler of k sign probes when k² 2⁻ᴺ ≤ 2⁻⁶⁰, else of k
        Gaussian probes.
    """
    n = template.shape[0]
    # log₂(k² 2⁻ᴺ) = 2 log₂ k − N
    if 2 * math.log2(max(num_probes, 1)) - n <= _SIGN_DEPENDENCE_LOG2:
        return matfree.stochtrace.sampler_signs(template, num=num_probes)
    return matfree.stochtrace.sampler_normal(template, num=num_probes)
