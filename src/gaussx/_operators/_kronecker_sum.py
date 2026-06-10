"""Kronecker sum operator: A (+) B = A (x) I_b + I_a (x) B."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset


_NEGATIVE_EIGENVALUE_TOLERANCE_FACTOR = 100


class KroneckerSum(lx.AbstractLinearOperator):
    r"""Kronecker sum ``A \oplus B = A \otimes I_b + I_a \otimes B``.

    Appears in separable PDEs, graph Laplacians, and space-time GPs.
    If ``A = Q_A \Lambda_A Q_A^T`` and ``B = Q_B \Lambda_B Q_B^T``,
    the Kronecker sum has eigenvectors ``Q_A \otimes Q_B`` with
    eigenvalues ``\lambda^A_i + \lambda^B_j``.

    Args:
        A: First operator, shape ``(n_a, n_a)``.
        B: Second operator, shape ``(n_b, n_b)``.
    """

    A: lx.AbstractLinearOperator
    B: lx.AbstractLinearOperator
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _n_a: int = eqx.field(static=True)
    _n_b: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        A: lx.AbstractLinearOperator,
        B: lx.AbstractLinearOperator,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if A.in_size() != A.out_size():
            raise ValueError(
                f"A must be square, got in_size={A.in_size()}, out_size={A.out_size()}."
            )
        if B.in_size() != B.out_size():
            raise ValueError(
                f"B must be square, got in_size={B.in_size()}, out_size={B.out_size()}."
            )
        self.A = A
        self.B = B
        n_a = A.in_size()
        n_b = B.in_size()
        self._n_a = n_a
        self._n_b = n_b
        self._in_size = n_a * n_b
        self._out_size = n_a * n_b
        self._dtype = _resolve_dtype(A, B)
        from gaussx._tags import kronecker_sum_tag

        self.tags = _to_frozenset(tags) | {kronecker_sum_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # (A (x) I_b + I_a (x) B) vec(X) = vec(B X + X A^T)
        # where X is (n_b, n_a)
        X = rearrange(vector, "(a b) -> b a", a=self._n_a, b=self._n_b)
        # I_a (x) B: apply B to each column of X
        BX = jax.vmap(self.B.mv, in_axes=1, out_axes=1)(X)
        # A (x) I_b: apply A^T to each row of X (= apply A to rows of X^T)
        XAt = jax.vmap(self.A.mv)(X)  # (n_b, n_a): apply A to rows
        result = BX + XAt
        return rearrange(result, "b a -> (a b)")

    def as_matrix(self) -> Float[Array, "n n"]:
        A_mat = self.A.as_matrix()
        B_mat = self.B.as_matrix()
        I_a = jnp.eye(self._n_a, dtype=jnp.dtype(self._dtype))
        I_b = jnp.eye(self._n_b, dtype=jnp.dtype(self._dtype))
        return jnp.kron(A_mat, I_b) + jnp.kron(I_a, B_mat)

    def transpose(self) -> KroneckerSum:
        return KroneckerSum(
            self.A.T,
            self.B.T,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def eigendecompose(
        self,
    ) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
        """Symmetric eigendecomposition via per-factor decomposition.

        Assumes both factors are symmetric so the returned
        ``Q = Q_A ⊗ Q_B`` is orthonormal — callers rely on
        ``self == Q @ diag(eigenvalues) @ Q.T``. Diagonal factors get a
        structural shortcut; other operators are materialized and
        decomposed via ``jnp.linalg.eigh``. We deliberately avoid
        routing untagged factors through `gaussx.eig` because
        that primitive falls back to ``jnp.linalg.eig`` for untagged
        operators and would return general (non-orthonormal)
        eigenvectors — breaking the ``Q.T == Q^{-1}`` contract for the
        common case of numerically symmetric matrices wrapped as plain
        `lineax.MatrixLinearOperator`.

        Returns:
            Tuple ``(eigenvalues, Q)`` where ``Q = Q_A ⊗ Q_B`` and the
            eigenvalues are ``lambda^A_i + lambda^B_j`` for all pairs.
        """

        evals_a, evecs_a = _eigh_factor(self.A)
        evals_b, evecs_b = _eigh_factor(self.B)
        # Eigenvalues: lambda_a_i + lambda_b_j for all (i, j) pairs
        eigenvalues = rearrange(evals_a[:, None] + evals_b[None, :], "a b -> (a b)")
        # Eigenvectors: Q_A (x) Q_B
        Q = jnp.kron(evecs_a, evecs_b)
        return eigenvalues, Q


class KroneckerSumSqrt(lx.AbstractLinearOperator):
    r"""Symmetric square root of ``A \oplus B`` via per-factor eigenvectors.

    Represents the symmetric matrix ``S`` with ``S @ S = A \oplus B``
    (where ``\oplus`` is the Kronecker sum ``A ⊗ I + I ⊗ B``). The square
    root is never materialized: `mv` and `solve` apply ``S`` and
    ``S^{-1}`` matrix-free using the per-factor eigendecompositions, so the
    cost is governed by the factor sizes rather than the full ``n_a · n_b``
    dimension.

    Args:
        A: Symmetric PSD factor, shape ``(n_a, n_a)``.
        B: Symmetric PSD factor, shape ``(n_b, n_b)``.

    Raises:
        ValueError: If either factor is non-square, untagged as symmetric, or
            if ``A \oplus B`` is not positive semidefinite.
    """

    eigenvectors_a: Float[Array, "a a"]
    eigenvectors_b: Float[Array, "b b"]
    sqrt_eigenvalues: Float[Array, "a b"]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _n_a: int = eqx.field(static=True)
    _n_b: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)

    def __init__(
        self,
        A: lx.AbstractLinearOperator,
        B: lx.AbstractLinearOperator,
    ) -> None:
        if A.in_size() != A.out_size():
            raise ValueError(
                f"A must be square, got in_size={A.in_size()}, out_size={A.out_size()}."
            )
        if B.in_size() != B.out_size():
            raise ValueError(
                f"B must be square, got in_size={B.in_size()}, out_size={B.out_size()}."
            )
        # The symmetric sqrt is well-defined only when A and B are
        # symmetric PSD. Without these tags, ``jnp.linalg.eigh`` would
        # silently use only the lower triangle and return wrong
        # eigenvectors for non-symmetric inputs.
        if not lx.is_symmetric(A) or not lx.is_symmetric(B):
            raise ValueError(
                "KroneckerSumSqrt requires both factors to be symmetric "
                "(tag them with lx.symmetric_tag or lx.positive_semidefinite_tag)."
            )
        evals_a, evecs_a = _eigh_factor(A)
        evals_b, evecs_b = _eigh_factor(B)
        eigenvalues = (evals_a[:, None] + evals_b[None, :]).astype(
            jnp.result_type(evals_a, evals_b, jnp.float32)
        )
        # Tolerance for "numerically zero" negative eigenvalues. We scale
        # by ``sqrt(spectrum magnitude)`` so the threshold stays tight
        # enough for large-magnitude spectra while still admitting eigh
        # roundoff. Linear scaling becomes too permissive: with
        # ``scale ~ 1e8`` and the previous ``100 * eps * scale`` formula,
        # genuinely-indefinite operators (negatives on the order of
        # ``-1e3``) could slip past the guard.
        scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
        threshold = (
            -_NEGATIVE_EIGENVALUE_TOLERANCE_FACTOR
            * jnp.finfo(eigenvalues.dtype).eps
            * jnp.sqrt(scale)
        )
        min_eigenvalue = jnp.min(eigenvalues)
        if bool(min_eigenvalue < threshold):
            raise ValueError(
                "A ⊕ B must be positive semidefinite; "
                f"minimum eigenvalue {float(min_eigenvalue):.2e} is below "
                f"threshold {float(threshold):.2e}."
            )
        sqrt_eigenvalues = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))

        self.eigenvectors_a = evecs_a
        self.eigenvectors_b = evecs_b
        self.sqrt_eigenvalues = sqrt_eigenvalues
        self._n_a = A.in_size()
        self._n_b = B.in_size()
        self._in_size = self._n_a * self._n_b
        self._out_size = self._in_size
        self._dtype = str(jnp.result_type(evecs_a, evecs_b, sqrt_eigenvalues))

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        X = rearrange(vector, "(a b) -> b a", a=self._n_a, b=self._n_b)
        C = self.eigenvectors_b.T @ X @ self.eigenvectors_a
        C = self.sqrt_eigenvalues.T * C
        result = self.eigenvectors_b @ C @ self.eigenvectors_a.T
        return rearrange(result, "b a -> (a b)")

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply the inverse square root ``S^{-1}`` to ``vector``.

        Args:
            vector: Input vector, shape ``(n_a · n_b,)``.

        Returns:
            ``S^{-1} @ vector``, shape ``(n_a · n_b,)``.
        """
        X = rearrange(vector, "(a b) -> b a", a=self._n_a, b=self._n_b)
        C = self.eigenvectors_b.T @ X @ self.eigenvectors_a
        C = C / self.sqrt_eigenvalues.T
        result = self.eigenvectors_b @ C @ self.eigenvectors_a.T
        return rearrange(result, "b a -> (a b)")

    def as_matrix(self) -> Float[Array, "n n"]:
        basis = jnp.eye(self._in_size, dtype=jnp.dtype(self._dtype))
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(basis)

    def transpose(self) -> KroneckerSumSqrt:
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def kronecker_sum_sample(
    A_op: lx.AbstractLinearOperator,
    B_op: lx.AbstractLinearOperator,
    *,
    key: jax.Array,
    num_samples: int = 1,
) -> Float[Array, "num_samples n_a n_b"]:
    """Sample from ``𝒩(0, A ⊕ B)`` using per-factor eigendecompositions.

    Draws zero-mean samples with covariance ``A ⊕ B`` by applying the
    matrix-free `KroneckerSumSqrt` to standard normal noise, avoiding
    materialization of the full ``(n_a · n_b, n_a · n_b)`` covariance.

    Args:
        A_op: Symmetric PSD factor, shape ``(n_a, n_a)``.
        B_op: Symmetric PSD factor, shape ``(n_b, n_b)``.
        key: PRNG key for the standard normal draws.
        num_samples: Number of samples to draw.

    Returns:
        Samples of shape ``(num_samples, n_a, n_b)``.

    Raises:
        ValueError: If ``num_samples`` is less than 1.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    sqrt_op = KroneckerSumSqrt(A_op, B_op)
    eps = jax.random.normal(
        key,
        (num_samples, sqrt_op.in_size()),
        dtype=jnp.dtype(sqrt_op._dtype),
    )
    samples = jax.vmap(sqrt_op.mv)(eps)
    return rearrange(samples, "s (a b) -> s a b", a=sqrt_op._n_a, b=sqrt_op._n_b)


def _eigh_factor(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
    """Symmetric eigendecomposition of a Kronecker-sum factor.

    Always returns an ``eigh``-equivalent ``(eigenvalues, Q)`` with
    orthonormal ``Q`` — callers (``_solve_kronecker_sum``,
    ``KroneckerSum.eigendecompose``, ``KroneckerSumSqrt``) rely on
    ``Q.T == Q^{-1}``.

    Diagonal operators get a free structural shortcut. For anything
    else we materialize and call ``jnp.linalg.eigh`` directly — routing
    through ``gaussx.eig`` is unsafe here because for untagged factors
    that primitive falls back to ``jnp.linalg.eig``, which returns
    general (non-orthonormal) eigenvectors.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        d = lx.diagonal(operator)
        return d, jnp.eye(d.shape[0], dtype=d.dtype)
    return jnp.linalg.eigh(operator.as_matrix())
