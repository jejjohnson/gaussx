"""Structured eigendecomposition with dispatch on operator type."""

from __future__ import annotations

import functools as ft
import warnings
from typing import Literal

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig
import numpy as np
from jax.core import Tracer
from jax.scipy.linalg import block_diag as _block_diag, cho_solve, solve_triangular
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._randomized._svd import randomized_eigh


def eig(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    method: Literal["lanczos", "randomized"] = "lanczos",
    key: jax.Array | None = None,
) -> tuple[Array, Array]:
    """Compute eigenvalues and eigenvectors.

    For symmetric operators returns real eigenvalues via ``eigh``.
    When ``rank`` is given (and the operator has no exploitable
    structure), computes a partial eigendecomposition of a symmetric
    operator without matrix materialization, by ``method``:

    - ``"lanczos"`` (default): matfree Lanczos.
    - ``"randomized"``: `randomized_eigh` with its defaults
      (``oversample=10``, ``n_power_iter=2``, ``which="largest"``); call it
      directly to tune them. Use ``n_power_iter >= 2`` for slowly decaying
      spectra.

    Randomized methods target the **top** of the spectrum (the eigenvalues
    of largest magnitude).

    Args:
        operator: A square linear operator.
        rank: Number of eigenvalues to compute. If ``None``,
            computes the full eigendecomposition.
        method: Partial-eig algorithm, ``"lanczos"`` or ``"randomized"``.
            Only used when ``rank`` is given.
        key: PRNG key for the initial random vector (Lanczos) or the
            Gaussian test matrix (randomized) when using partial eig.
            If ``None``, uses ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` where eigenvalues has
        shape ``(K,)`` and eigenvectors has shape ``(N, K)``.

    Raises:
        ValueError: If ``method`` is invalid, or ``"randomized"`` without
            ``rank``.
    """
    if method not in ("lanczos", "randomized"):
        raise ValueError(f"method must be 'lanczos' or 'randomized', got {method!r}.")
    if method == "randomized" and rank is None:
        raise ValueError("method='randomized' needs a rank.")
    return _eig(operator, rank, method, key, symmetric=False)


def _eig(
    operator: lx.AbstractLinearOperator,
    rank: int | None,
    method: Literal["lanczos", "randomized"],
    key: jax.Array | None,
    *,
    symmetric: bool,
) -> tuple[Array, Array]:
    """`eig` dispatch; ``symmetric`` carries an outer wrapper's tags (gh-314)."""
    if isinstance(operator, lx.TaggedLinearOperator):
        # Tags cannot change the spectrum, but a PSD/NSD tag implies a real
        # symmetric operator, so the inner one takes ``eigh`` (gh-314).
        return _eig(
            operator.operator,
            rank,
            method,
            key,
            symmetric=symmetric or _is_real_symmetric(operator),
        )
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _eig_diagonal(operator)
    if isinstance(operator, BlockDiag):
        return _eig_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _eig_kronecker(operator)
    if isinstance(operator, KroneckerSum):
        return _eig_kronecker_sum(operator)
    if rank is not None:
        if method == "randomized":
            return randomized_eigh(operator, rank, key=key)
        return _eig_partial(operator, rank, key)
    return _eig_dense(operator, symmetric=symmetric)


def eigvals(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jax.Array | None = None,
) -> Array:
    """Compute eigenvalues only.

    When ``rank`` is given, returns the top-k eigenvalues via
    matfree Lanczos without matrix materialization.

    Args:
        operator: A square linear operator.
        rank: Number of eigenvalues to compute.
        key: PRNG key for partial eigendecomposition.

    Returns:
        Eigenvalues array of shape ``(K,)``.
    """
    return _eigvals(operator, rank, key, symmetric=False)


def _eigvals(
    operator: lx.AbstractLinearOperator,
    rank: int | None,
    key: jax.Array | None,
    *,
    symmetric: bool,
) -> Array:
    """`eigvals` dispatch; ``symmetric`` as in ``_eig`` (gh-314)."""
    if isinstance(operator, lx.TaggedLinearOperator):
        return _eigvals(
            operator.operator,
            rank,
            key,
            symmetric=symmetric or _is_real_symmetric(operator),
        )
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return jnp.concatenate([eigvals(op) for op in operator.operators])
    if isinstance(operator, Kronecker):
        return _eigvals_kronecker(operator)
    if isinstance(operator, KroneckerSum):
        return _eigvals_kronecker_sum(operator)
    if rank is not None:
        vals, _ = _eig_partial(operator, rank, key)
        return vals
    return _eigvals_dense(operator, symmetric=symmetric)


def _eig_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[Array, Array]:
    d = lx.diagonal(operator)
    n = d.shape[0]
    return d, jnp.eye(n, dtype=d.dtype)


def _eig_block_diag(
    operator: BlockDiag,
) -> tuple[Array, Array]:
    vals_list = []
    vecs_list = []
    for op in operator.operators:
        v, V = eig(op)
        vals_list.append(v)
        vecs_list.append(V)
    vals = jnp.concatenate(vals_list)
    vecs = _block_diag(*vecs_list)
    return vals, vecs


def _eig_kronecker(
    operator: Kronecker,
) -> tuple[Array, Array]:
    """eig(A kron B) = (kron(eigvals), kron(eigvecs))."""
    factor_eigs = [eig(op) for op in operator.operators]
    vals = ft.reduce(jnp.kron, (v for v, _ in factor_eigs))
    vecs = ft.reduce(jnp.kron, (V for _, V in factor_eigs))
    return vals, vecs


def _eigvals_kronecker(operator: Kronecker) -> Array:
    """eigvals(A kron B) = kron(eigvals(A), eigvals(B))."""
    return ft.reduce(jnp.kron, (eigvals(op) for op in operator.operators))


def _eig_kronecker_sum(
    operator: KroneckerSum,
) -> tuple[Array, Array]:
    """eig(A (+) B) via per-factor eigendecomposition.

    A (+) B = (Q_A ⊗ Q_B) diag(λ^A_i + λ^B_j) (Q_A ⊗ Q_B)^T.
    """
    evals_a, evecs_a = eig(operator.A)
    evals_b, evecs_b = eig(operator.B)
    eigenvalues = jnp.reshape(
        evals_a[:, None] + evals_b[None, :],
        (-1,),
    )
    Q = jnp.kron(evecs_a, evecs_b)
    return eigenvalues, Q


def _eigvals_kronecker_sum(operator: KroneckerSum) -> Array:
    """eigvals(A (+) B) = sum-pairs of eigvals — no factor materialization."""
    evals_a = eigvals(operator.A)
    evals_b = eigvals(operator.B)
    return jnp.reshape(evals_a[:, None] + evals_b[None, :], (-1,))


def _eig_partial(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
) -> tuple[Array, Array]:
    """Partial eigendecomposition via matfree Lanczos."""
    if key is None:
        key = jr.PRNGKey(0)

    n = operator.in_size()
    rank = min(rank, n)
    v0 = jr.normal(key, (n,), dtype=operator.in_structure().dtype)

    tridiag = matfree.decomp.tridiag_sym(rank, reortho="full")
    eigh_fn = matfree.eig.eigh_partial(tridiag)

    # matfree returns vals: (k,), vecs: (k, n)
    vals, vecs = eigh_fn(operator.mv, v0)
    return vals, vecs.T


def _is_real_symmetric(operator: lx.AbstractLinearOperator) -> bool:
    """Symmetric, or PSD/NSD-tagged with a real dtype (lineax's own rule).

    lineax reports ``is_symmetric(Tagged(X, positive_semidefinite_tag))`` as
    ``False``, yet a real PSD/NSD operator is symmetric (gh-314).
    """
    if lx.is_symmetric(operator):
        return True
    definite = lx.is_positive_semidefinite(operator) or lx.is_negative_semidefinite(
        operator
    )
    return definite and not jnp.issubdtype(
        operator.in_structure().dtype, jnp.complexfloating
    )


def _eig_dense(
    operator: lx.AbstractLinearOperator,
    *,
    symmetric: bool = False,
) -> tuple[Array, Array]:
    mat = operator.as_matrix()
    if symmetric or _is_real_symmetric(operator):
        return jnp.linalg.eigh(mat)
    return jnp.linalg.eig(mat)


def _eigvals_dense(
    operator: lx.AbstractLinearOperator,
    *,
    symmetric: bool = False,
) -> Array:
    mat = operator.as_matrix()
    if symmetric or _is_real_symmetric(operator):
        return jnp.linalg.eigvalsh(mat)
    return jnp.linalg.eigvals(mat)


# ---------------------------------------------------------------------------
# Generalised symmetric-definite eigenproblem A v = λ B v
# ---------------------------------------------------------------------------


def eigh_generalized(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    which: Literal["smallest", "largest"] = "smallest",
    rcond: float | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " K"], Float[Array, "N K"]]:
    r"""Solve $A v = \lambda B v$ for symmetric $A$ and symmetric PSD $B$.

    The eigenvectors are $B$-orthonormal, $V^\top B V = I$, so the
    smallest $K$ of them minimise $\operatorname{tr}(Y^\top A Y)$ subject
    to $Y^\top B Y = I$ (graph embeddings, LPP, manifold alignment).
    Dispatch is on the structure of $B$:

    - `lineax.DiagonalLinearOperator` with positive entries: scale,
      $S = B^{-1/2} A B^{-1/2}$, $v = B^{-1/2} u$. This is matrix-free in
      $A$: with `rank=` it runs Lanczos on $x \mapsto B^{-1/2} A B^{-1/2} x$
      and never materialises $A$. A diagonal with zero entries (only
      checked outside `jit`) is sent to the singular path below.
    - Positive definite, i.e. tagged `lineax.positive_semidefinite_tag`
      (lineax's tag for Cholesky-factorisable operators): Cholesky
      whitening $C^{-1} A C^{-\top}$ with $B = C C^\top$.
    - Otherwise $B$ is treated as PSD and possibly singular. $B$ is
      eigendecomposed, $B = U_+ S_+ U_+^\top$, and directions with
      eigenvalue $\le$ `rcond` $\cdot \max$ span $\ker B$. Writing
      $A_{\cdot\cdot}$ for the blocks of $U^\top A U$, the $\ker B$ rows give
      $v_0 = -A_{00}^{-1} A_{0+} v_+$, and the finite eigenpairs solve the
      Schur-complement pencil
      $(A_{++} - A_{+0} A_{00}^{-1} A_{0+}) v_+ = \lambda S_+ v_+$.
      $A_{00}$ must be positive definite, otherwise the trace minimisation
      is unbounded along $\ker B$ and a `ValueError` is raised. At most
      $\operatorname{rank}(B)$ finite eigenpairs exist; asking for more
      warns and returns fewer. The numerical rank decides output shapes,
      so this path cannot run under `jax.jit` (tag $B$ positive definite
      for a traceable path).

    Krylov subspaces are shift-invariant, so the Lanczos path needs no
    spectral shift for `which="smallest"`: it runs an oversampled Krylov
    space of dimension `min(N, max(2 * rank + 1, rank + 20))` and keeps
    the `rank` extreme Ritz pairs at the requested end.

    Args:
        A: Symmetric `(N, N)` operator.
        B: Symmetric positive semidefinite `(N, N)` operator.
        rank: Number of eigenpairs $K$. `None` returns all finite ones.
        which: `"smallest"` or `"largest"` eigenvalues.
        rcond: Relative threshold on $B$'s eigenvalues below which a
            direction counts as $\ker B$ (singular path only). `None`
            means `N * eps` of $B$'s dtype.
        key: PRNG key for the Lanczos start vector (diagonal path with
            `rank=` only). `None` means `jax.random.PRNGKey(0)`.

    Returns:
        `(eigenvalues, eigenvectors)` of shapes `(K,)` and `(N, K)`,
        eigenvalues in ascending order, eigenvectors $B$-orthonormal.

    Raises:
        ValueError: If `which` is invalid, `rank < 1`, $B$ is zero, or
            $A_{00}$ is not positive definite.

    Examples:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> # Laplacian eigenmaps on a path graph: L y = λ D y.
        >>> # Its eigenvalues are 1 - cos(πk/4); rank= runs Lanczos.
        >>> W = jnp.diag(jnp.ones(4), 1) + jnp.diag(jnp.ones(4), -1)
        >>> degree = W @ jnp.ones(5)
        >>> L = lx.MatrixLinearOperator(jnp.diag(degree) - W, lx.symmetric_tag)
        >>> D = lx.DiagonalLinearOperator(degree)
        >>> lam, Y = gaussx.eigh_generalized(L, D, rank=2, which="smallest")
        >>> bool(jnp.allclose(lam, 1 - jnp.cos(jnp.pi * jnp.arange(2) / 4), atol=1e-5))
        True
        >>> # A singular B: one finite eigenpair fewer than N.
        >>> B = lx.MatrixLinearOperator(jnp.diag(jnp.array([1.0, 2.0, 0.0])))
        >>> A = lx.MatrixLinearOperator(jnp.eye(3) + 0.5 * jnp.ones((3, 3)))
        >>> lam, V = gaussx.eigh_generalized(A, B)
        >>> lam.shape
        (2,)
    """
    if which not in ("smallest", "largest"):
        msg = f"which must be 'smallest' or 'largest', got {which!r}."
        raise ValueError(msg)
    if rank is not None and rank < 1:
        msg = f"rank must be a positive integer, got {rank}."
        raise ValueError(msg)
    if A.in_size() != B.in_size() or A.in_size() != A.out_size():
        msg = (
            f"A and B must be square and of the same size, got A "
            f"{A.out_size()}x{A.in_size()} and B {B.out_size()}x{B.in_size()}."
        )
        raise ValueError(msg)

    if isinstance(B, lx.DiagonalLinearOperator):
        d = lx.diagonal(B)
        if isinstance(d, Tracer) or bool(jnp.min(d) > _null_tol(d, rcond, d.shape[0])):
            return _eigh_generalized_diagonal(A, d, rank, which, key)
    elif lx.is_positive_semidefinite(B):
        return _eigh_generalized_cholesky(A, B, rank, which)
    return _eigh_generalized_singular(A, B, rank, which, rcond)


def _null_tol(s: Array, rcond: float | None, n: int) -> Array:
    """Absolute threshold below which an eigenvalue of ``B`` counts as zero."""
    if rcond is None:
        rcond = n * float(jnp.finfo(s.dtype).eps)
    return rcond * jnp.max(jnp.abs(s))


def _select(vals: Array, vecs: Array, k: int, which: str) -> tuple[Array, Array]:
    """Keep the ``k`` eigenpairs at the requested end of ascending ``vals``."""
    if which == "smallest":
        return vals[:k], vecs[:, :k]
    n = vals.shape[0]
    return vals[n - k :], vecs[:, n - k :]


def _eigh_generalized_diagonal(
    A: lx.AbstractLinearOperator,
    d: Array,
    rank: int | None,
    which: str,
    key: PRNGKeyArray | None,
) -> tuple[Array, Array]:
    """Positive diagonal ``B``: eig of ``B^{-1/2} A B^{-1/2}``, then rescale."""
    n = d.shape[0]
    s = 1.0 / jnp.sqrt(d)
    if rank is None:
        S = einx.multiply(
            "i, i j -> i j", s, einx.multiply("i j, j -> i j", A.as_matrix(), s)
        )
        lam, U = jnp.linalg.eigh(S)
        k = n
    else:
        k = min(rank, n)
        if key is None:
            key = jr.PRNGKey(0)
        depth = min(n, max(2 * k + 1, k + 20))
        v0 = jr.normal(key, (n,), dtype=s.dtype)
        eigh_fn = matfree.eig.eigh_partial(
            matfree.decomp.tridiag_sym(depth, reortho="full")
        )
        lam, U = eigh_fn(lambda x: s * A.mv(s * x), v0)
        U = rearrange(U, "k n -> n k")
    lam, U = _select(lam, U, k, which)
    return lam, einx.multiply("i, i k -> i k", s, U)


def _eigh_generalized_cholesky(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    rank: int | None,
    which: str,
) -> tuple[Array, Array]:
    """Positive definite ``B = C Cᵀ``: eig of ``C⁻¹ A C⁻ᵀ``, ``v = C⁻ᵀ u``."""
    n = B.in_size()
    C = jnp.linalg.cholesky(B.as_matrix())
    X = solve_triangular(C, A.as_matrix(), lower=True)  # C⁻¹ A
    M = solve_triangular(C, rearrange(X, "i j -> j i"), lower=True)  # C⁻¹ A C⁻ᵀ
    lam, U = jnp.linalg.eigh(M)
    lam, U = _select(lam, U, n if rank is None else min(rank, n), which)
    return lam, solve_triangular(C, U, lower=True, trans="T")


def _eigh_generalized_singular(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    rank: int | None,
    which: str,
    rcond: float | None,
) -> tuple[Array, Array]:
    """PSD, possibly singular ``B``: Schur-complement elimination of ``ker B``."""
    n = B.in_size()
    Am = A.as_matrix()
    s, Ub = jnp.linalg.eigh(B.as_matrix())
    pos = np.asarray(s > _null_tol(s, rcond, n))
    r = int(pos.sum())
    if r == 0:
        msg = "B is numerically zero: the pencil has no finite eigenvalues."
        raise ValueError(msg)
    k = r if rank is None else min(rank, r)
    if rank is not None and rank > r:
        warnings.warn(
            f"eigh_generalized: B has numerical rank {r} < rank={rank}, so "
            f"only {r} finite eigenpairs exist; returning {r}.",
            stacklevel=2,
        )

    Up, sp = Ub[:, pos], s[pos]
    S = einsum(Up, Am, Up, "i a, i j, j b -> a b")
    if r < n:
        U0 = Ub[:, ~pos]
        A00 = einsum(U0, Am, U0, "i a, i j, j b -> a b")
        A0p = einsum(U0, Am, Up, "i a, i j, j b -> a b")
        e00 = jnp.linalg.eigvalsh(A00)
        if not bool(e00[0] > _null_tol(e00, None, n - r)):
            msg = (
                "eigh_generalized: A restricted to ker B (A₀₀) is not positive "
                f"definite (smallest eigenvalue {float(e00[0]):.3e}), so "
                "tr(Yᵀ A Y) s.t. Yᵀ B Y = I is unbounded below along ker B."
            )
            raise ValueError(msg)
        Z = cho_solve((jnp.linalg.cholesky(A00), True), A0p)  # A₀₀⁻¹ A₀₊
        S = S - einsum(A0p, Z, "a p, a q -> p q")

    w = 1.0 / jnp.sqrt(sp)
    M = einx.multiply("p, p q -> p q", w, einx.multiply("p q, q -> p q", S, w))
    lam, W = jnp.linalg.eigh(M)
    lam, W = _select(lam, W, k, which)
    vp = einx.multiply("p, p k -> p k", w, W)
    V = Up @ vp
    if r < n:
        V = V - U0 @ (Z @ vp)  # v₀ = −A₀₀⁻¹ A₀₊ v₊
    return lam, V
