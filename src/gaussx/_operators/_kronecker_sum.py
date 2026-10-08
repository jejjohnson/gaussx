"""Kronecker sum operator: A (+) B = A (x) I_b + I_a (x) B."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
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

    Examples:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> S = gaussx.KroneckerSum(A, B)  # A ⊗ I + I ⊗ B
        >>> I2 = jnp.eye(2)
        >>> dense = jnp.kron(A.as_matrix(), I2) + jnp.kron(I2, B.as_matrix())
        >>> bool(jnp.allclose(S.as_matrix(), dense))
        True
        >>> sign, expected = jnp.linalg.slogdet(dense)
        >>> bool(jnp.allclose(gaussx.logdet(S), expected, atol=1e-5))  # λ_i + μ_j
        True
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
        ValueError: If either factor is non-square or untagged as symmetric.
        EquinoxRuntimeError: If ``A \oplus B`` is not positive semidefinite
            (checked with `equinox.error_if`, so also under ``jax.jit``).
    """

    a_factor: Float[Array, "a a"] | Float[Array, " a"]
    b_factor: Float[Array, "b b"] | Float[Array, " b"]
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
        # The factors (a matrix, or the diagonal of a diagonal factor, which
        # is never materialised) are the differentiable leaves; the cached
        # eigendecomposition is a constant for autodiff. `mv` and `solve` go
        # through custom JVPs written in terms of the factors, because the
        # eigenvector derivative divides by eigenvalue gaps and is NaN at a
        # repeated eigenvalue (e.g. an isotropic factor ``s I``), gh-295.
        a_factor = _factor_array(A)
        b_factor = _factor_array(B)
        evals_a, evecs_a = _factor_eigh(jax.lax.stop_gradient(a_factor))
        evals_b, evecs_b = _factor_eigh(jax.lax.stop_gradient(b_factor))
        eigenvalues = _checked_kronecker_sum_spectrum(evals_a, evals_b)
        sqrt_eigenvalues = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))

        self.a_factor = a_factor
        self.b_factor = b_factor
        self.eigenvectors_a = evecs_a
        self.eigenvectors_b = evecs_b
        self.sqrt_eigenvalues = sqrt_eigenvalues
        self._n_a = A.in_size()
        self._n_b = B.in_size()
        self._in_size = self._n_a * self._n_b
        self._out_size = self._in_size
        self._dtype = str(jnp.result_type(evecs_a, evecs_b, sqrt_eigenvalues))

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        Z = rearrange(vector, "(a b) -> 1 a b", a=self._n_a, b=self._n_b)
        result = _kronecker_sum_root_apply(*self._parts(), Z)
        return rearrange(result, "1 a b -> (a b)")

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Apply the inverse square root ``S^{-1}`` to ``vector``.

        Args:
            vector: Input vector, shape ``(n_a · n_b,)``.

        Returns:
            ``S^{-1} @ vector``, shape ``(n_a · n_b,)``.
        """
        Z = rearrange(vector, "(a b) -> 1 a b", a=self._n_a, b=self._n_b)
        result = _kronecker_sum_root_solve(*self._parts(), Z)
        return rearrange(result, "1 a b -> (a b)")

    def _parts(self):
        return (
            self.a_factor,
            self.b_factor,
            self.eigenvectors_a,
            self.eigenvectors_b,
            self.sqrt_eigenvalues,
        )

    def as_matrix(self) -> Float[Array, "n n"]:
        basis = jnp.eye(self._in_size, dtype=jnp.dtype(self._dtype))
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(basis)

    def transpose(self) -> KroneckerSumSqrt:
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _checked_kronecker_sum_spectrum(
    evals_a: Float[Array, " a"], evals_b: Float[Array, " b"]
) -> Float[Array, "a b"]:
    """``λ^A_i + λ^B_j``, with a run-time error if ``A ⊕ B`` is indefinite.

    Shared by the constructor and the custom JVPs (which recompute the
    spectrum), so autodiff cannot bypass the check.
    """
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
    # ``eqx.error_if`` rather than a Python branch, so the check also
    # runs (at run time) when the factors are traced.
    return eqx.error_if(
        eigenvalues,
        jnp.min(eigenvalues) < threshold,
        "A ⊕ B must be positive semidefinite (minimum eigenvalue of the "
        "Kronecker sum is below the round-off threshold).",
    )


def _factor_array(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "n n"] | Float[Array, " n"]:
    """A factor as an array: its diagonal if diagonal (never materialised)."""
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    return operator.as_matrix()


def _factor_eigh(
    factor: Float[Array, "n n"] | Float[Array, " n"],
) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
    """``(eigenvalues, Q)`` of a `_factor_array` (identity basis if diagonal)."""
    if factor.ndim == 1:
        return factor, jnp.eye(factor.shape[0], dtype=factor.dtype)
    return jnp.linalg.eigh(factor)


def _factor_tangent(tangent: Array) -> Float[Array, "n n"]:
    """A factor tangent as a matrix (a diagonal factor's tangent is a vector)."""
    return jnp.diag(tangent) if tangent.ndim == 1 else tangent


def _spectral_parts(a, b):
    """Eigenbases and root spectrum of ``A ⊕ B``, differentiably from the factors.

    The custom JVPs below recompute these from the primal factors rather than
    use the cached (``stop_gradient``) ones, so that differentiating the JVP
    itself -- a Hessian -- still sees their dependence on the factors. The
    PSD check is re-applied, so autodiff cannot bypass it.

    First derivatives never differentiate ``eigh`` and are exact at repeated
    eigenvalues. Second and higher derivatives do differentiate it here, so
    -- exactly like ``dense_symmetric_sqrt`` -- they are non-finite
    when a factor has a repeated eigenvalue. (An exact second-order rule
    needs the Sylvester solve with a right-hand side that is no longer a
    Kronecker sum, i.e. a dense ``(n_a n_b)²`` intermediate.)
    """
    values_a, basis_a = _factor_eigh(a)
    values_b, basis_b = _factor_eigh(b)
    eigenvalues = _checked_kronecker_sum_spectrum(values_a, values_b)
    roots = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    return basis_a, basis_b, roots


def _to_eigenbasis(basis_a, basis_b, values):
    """``Q_Aᵀ Z Q_B`` for each slab ``Z``."""
    return einsum(basis_a, values, basis_b, "k i, s k l, l j -> s i j")


def _from_eigenbasis(basis_a, basis_b, values):
    """``Q_A Z̃ Q_Bᵀ`` for each slab ``Z̃``."""
    return einsum(basis_a, values, basis_b, "i k, s k l, j l -> s i j")


def _kronecker_sum_root_tangent(basis_a, basis_b, roots, tangent_a, tangent_b, rotated):
    r"""Eigenbasis action of ``d√K`` on ``z̃`` for ``K = A ⊕ B``.

    As in ``dense_symmetric_sqrt``: in ``K``'s eigenbasis ``dS̃`` is
    ``dK̃ / (r_p + r_q)`` entrywise, dividing by *sums* of root eigenvalues
    rather than eigenvalue gaps, so repeated eigenvalues are harmless. With
    ``dK = dA ⊕ dB`` the rotated tangent is ``dÃ_ik δ_jl + δ_ik dB̃_jl``, so

    $$
    (dS̃\, \tilde z)_{ij} = \sum_k \frac{dÃ_{ik} \tilde z_{kj}}{r_{ij} + r_{kj}}
      + \sum_l \frac{dB̃_{jl} \tilde z_{il}}{r_{ij} + r_{il}},
    $$

    which costs ``O(n_A n_B (n_A + n_B))`` per slab and never forms ``K``.
    Entries whose root sum is zero (a doubly-degenerate zero eigenvalue) get
    a zero derivative, matching ``dense_symmetric_sqrt``.

    Args:
        basis_a: Eigenvectors of ``A``, shape ``(n_a, n_a)``.
        basis_b: Eigenvectors of ``B``, shape ``(n_b, n_b)``.
        roots: ``√(λ^A_i + λ^B_j)``, shape ``(n_a, n_b)``.
        tangent_a: Tangent of ``A`` (in the original basis).
        tangent_b: Tangent of ``B`` (in the original basis).
        rotated: ``z̃ = Q_Aᵀ Z Q_B`` slabs, shape ``(s, n_a, n_b)``.

    Returns:
        ``(dS̃ z̃)`` slabs in the eigenbasis, shape ``(s, n_a, n_b)``.
    """
    # eigh reads one triangle, so project the tangents onto symmetric matrices.
    sym_a = 0.5 * (tangent_a + rearrange(tangent_a, "i k -> k i"))
    sym_b = 0.5 * (tangent_b + rearrange(tangent_b, "j l -> l j"))
    tangent_a = einsum(basis_a, sym_a, basis_a, "p i, p q, q k -> i k")
    tangent_b = einsum(basis_b, sym_b, basis_b, "p j, p q, q l -> j l")

    def safe_inverse(denominator):
        positive = denominator > 0.0
        return jnp.where(positive, 1.0 / jnp.where(positive, denominator, 1.0), 0.0)

    # inverse_a[i, k, j] = 1 / (r_ij + r_kj); inverse_b[i, j, l] = 1 / (r_ij + r_il)
    inverse_a = safe_inverse(roots[:, None, :] + roots[None, :, :])
    inverse_b = safe_inverse(roots[:, :, None] + roots[:, None, :])
    # Fold the elementwise weights into the tangents, then contract.
    weighted_a = tangent_a[:, :, None] * inverse_a
    weighted_b = tangent_b[None, :, :] * inverse_b
    term_a = einsum(weighted_a, rotated, "i k j, s k j -> s i j")
    term_b = einsum(weighted_b, rotated, "i j l, s i l -> s i j")
    return term_a + term_b


@jax.custom_jvp
def _kronecker_sum_root_apply(a, b, basis_a, basis_b, roots, Z):
    """``√(A ⊕ B) Z`` per slab, from a cached eigendecomposition.

    ``a`` and ``b`` (`_factor_array` s) are unused by the primal; the JVP
    differentiates with respect to them and ignores the tangents of the
    cached ``basis_*`` / ``roots``, recomputing them from ``a`` and ``b``
    (`_spectral_parts`) so higher-order derivatives stay correct.
    """
    del a, b
    return _from_eigenbasis(
        basis_a, basis_b, roots * _to_eigenbasis(basis_a, basis_b, Z)
    )


@_kronecker_sum_root_apply.defjvp
def _kronecker_sum_root_apply_jvp(primals, tangents):
    a, b, _, _, _, Z = primals
    tangent_a, tangent_b, _, _, _, tangent_Z = tangents
    basis_a, basis_b, roots = _spectral_parts(a, b)
    tangent_a, tangent_b = _factor_tangent(tangent_a), _factor_tangent(tangent_b)
    rotated = _to_eigenbasis(basis_a, basis_b, Z)
    primal_out = _from_eigenbasis(basis_a, basis_b, roots * rotated)
    term = _kronecker_sum_root_tangent(
        basis_a, basis_b, roots, tangent_a, tangent_b, rotated
    )
    rotated_tangent_Z = _to_eigenbasis(basis_a, basis_b, tangent_Z)
    tangent_out = _from_eigenbasis(basis_a, basis_b, roots * rotated_tangent_Z + term)
    return primal_out, tangent_out


@jax.custom_jvp
def _kronecker_sum_root_solve(a, b, basis_a, basis_b, roots, Z):
    """``√(A ⊕ B)⁻¹ Z`` per slab, from a cached eigendecomposition."""
    del a, b
    return _from_eigenbasis(
        basis_a, basis_b, _to_eigenbasis(basis_a, basis_b, Z) / roots
    )


@_kronecker_sum_root_solve.defjvp
def _kronecker_sum_root_solve_jvp(primals, tangents):
    """``d(S⁻¹ z) = S⁻¹ (dz - dS S⁻¹ z)``."""
    a, b, _, _, _, Z = primals
    tangent_a, tangent_b, _, _, _, tangent_Z = tangents
    basis_a, basis_b, roots = _spectral_parts(a, b)
    tangent_a, tangent_b = _factor_tangent(tangent_a), _factor_tangent(tangent_b)
    rotated_out = _to_eigenbasis(basis_a, basis_b, Z) / roots
    primal_out = _from_eigenbasis(basis_a, basis_b, rotated_out)
    term = _kronecker_sum_root_tangent(
        basis_a, basis_b, roots, tangent_a, tangent_b, rotated_out
    )
    rotated_tangent_Z = _to_eigenbasis(basis_a, basis_b, tangent_Z)
    tangent_out = _from_eigenbasis(basis_a, basis_b, (rotated_tangent_Z - term) / roots)
    return primal_out, tangent_out


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
