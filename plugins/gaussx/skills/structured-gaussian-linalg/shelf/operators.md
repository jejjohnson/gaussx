# Structured operators

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: structures and primitives.

Open this when you have a covariance or a precision whose structure you can name, and you are about to build it as a dense array.
Build the operator instead; the verbs in [solves](solves.md) then use the structure.

## Names

Every operator is a `lineax.AbstractLinearOperator`, so `op.mv(x)`, `op.as_matrix()`, `A + B` and `2.0 * A` work on all of them.

| Name | What it is |
|---|---|
| `Kronecker(*ops)` | `A₁ ⊗ A₂ ⊗ … ⊗ Aₖ`. Solves, log-determinants and Cholesky factors work per factor. |
| `KroneckerSum(A, B)` | `A ⊕ B = A ⊗ I + I ⊗ B`, solved through the two eigendecompositions. |
| `KroneckerSumSqrt(A, B)` | The symmetric square root of `A ⊕ B`. |
| `SumOfKroneckers(kron1, kron2, *krons)` | `Σₖ Aₖ ⊗ Bₖ` of `Kronecker` operators, for example a grid kernel plus noise. |
| `is_eigen_reducible(op)` | Whether a `SumOfKroneckers` has the exact two-term reduction. |
| `BlockDiag(*ops)` | `diag(A₁, …, Aₖ)`. |
| `BlockTriDiag(diagonal, sub_diagonal)` | A symmetric block-tridiagonal matrix, the precision of a Markov chain. |
| `LowerBlockTriDiag`, `UpperBlockTriDiag` | The block-bidiagonal Cholesky factor of a `BlockTriDiag`, and its transpose. |
| `low_rank_plus_diag(diag, U, d=None, V=None)` | `diag(diag) + U diag(d) Vᵀ`. Woodbury solve, determinant-lemma log-determinant. |
| `low_rank_plus_identity(U, d=None, V=None, scale=...)` | `scale · I + U diag(d) Vᵀ`. |
| `svd_low_rank_plus_diag(diag, U, S, V)` | The same from a truncated SVD. |
| `LowRankUpdate(base, U, d, V)` | `L + U diag(d) Vᵀ` on any base operator `L`. |
| `Toeplitz(column)` | A symmetric Toeplitz matrix from its first column: a stationary kernel on a regular 1-D grid. |
| `ToeplitzCholesky(column, embedding_factor=2)` | A circulant-embedding sample factor for a `Toeplitz`. |
| `circulant(first_column)`, `circulant_from_symbol(symbol)` | A circulant or block-circulant operator, diagonal in the Fourier basis. |
| `DiagonalizedOperator(eigenvalues, forward, inverse, ...)` | `A x = V⁻¹(Λ ⊙ V x)` from a transform pair. |
| `as_diagonalized(op)` | `op` as one `DiagonalizedOperator`, or `None`. |
| `SpectralFunction(base, fn)` | `f(B)` for a symmetric `B` with a factored eigenbasis. |
| `MaskedOperator(base, row_mask, col_mask)` | A row and column masked view, for missing observations. |
| `InterpolatedOperator(base_operator, interp_indices, interp_values)` | Structured kernel interpolation, `K ≈ W K_uu Wᵀ`. |
| `create_grid`, `grid_data` | The grid for an `InterpolatedOperator`. |
| `interpolation_operator(indices, weights, grid_size)` | The interpolation matrix `W` as a matrix-free operator, acting on the flattened grid values. |
| `masked_obs_covariance(obs_cov, mask)` | An observation-error covariance with the masked entries marginalised out; diagonal and block-diagonal inputs keep their structure. |
| `cubic_interpolation_weights`, `linear_interpolation_weights` | Deprecated, removed in 0.7.0. Build the stencil with functax, whose `Plan.to_gaussx()` returns the `(indices, weights)` that `InterpolatedOperator` and `interpolation_operator` take. |
| `grid_coupling_indices(mask)` | Flat indices of masked-out grid cells next to the masked-in region. |
| `CapacitanceSolver(base_solve, boundary_indices, n)` | A base solve under homogeneous point constraints. |
| `sum_operator(*ops)`, `product_operator(A, B)`, `scaled_operator(A, c)` | Lazy `A + B`, `A B` and `c A` that keep the tags you pass. |
| `kronecker_sum_sample`, `sum_of_kroneckers_sample`, `toeplitz_sample` | Draws from `𝒩(0, op)` that use the structure. |
| `SVDLowRankUpdate`, `SumKronecker`, `sumkronecker_sample` | Deprecated; use `LowRankUpdate`, `SumOfKroneckers` and `sum_of_kroneckers_sample`. |

## Use it

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag


def rbf(x, lengthscale=0.3):  # (n,) → (n, n)
    d = x[:, None] - x[None, :]
    return jnp.exp(-0.5 * (d / lengthscale) ** 2) + 1e-4 * jnp.eye(x.shape[0])


K1 = lx.MatrixLinearOperator(rbf(jnp.linspace(0, 1, 6)), PSD)  # (6, 6)
K2 = lx.MatrixLinearOperator(rbf(jnp.linspace(0, 1, 5)), PSD)  # (5, 5)
K = gaussx.Kronecker(K1, K2)  # (30, 30), never formed
b = jr.normal(jr.key(0), (30,))

x = gaussx.solve(K, b)  # two small solves
ld = gaussx.logdet(K)  # 5 · logdet(K1) + 6 · logdet(K2)
dense = jnp.kron(K1.as_matrix(), K2.as_matrix())
assert jnp.allclose(x, jnp.linalg.solve(dense, b), rtol=1e-2, atol=1e-2)
assert jnp.allclose(ld, jnp.linalg.slogdet(dense)[1], rtol=1e-3)

U = jr.normal(jr.key(1), (200, 3))  # (n, k)
A = gaussx.low_rank_plus_diag(jnp.full(200, 0.5), U, psd=True)  # D + U Uᵀ
y = gaussx.solve(A, jnp.ones(200))  # Woodbury, O(n k²)
assert jnp.allclose(A.mv(y), jnp.ones(200), atol=1e-3)
```

## Rules

- Pass operators as factors, not arrays: `Kronecker(K1, K2)` takes two `lineax` operators.
- Tag each factor with what it is (`lx.positive_semidefinite_tag`); `Kronecker`, `BlockDiag` and `KroneckerSum` then answer lineax's symmetric and PSD predicates from their factors. See [tags](tags.md).
- A covariance on a full grid is a `Kronecker`; with a noise term it is a `SumOfKroneckers` or a `KroneckerSum`, not `Kronecker + Diagonal`.
- A time series prior is a `BlockTriDiag` precision, not a dense covariance.

## Goes wrong

- `jnp.kron(K1, K2)` builds the `(N, N)` array and every later verb pays `O(N³)`.
- `op.as_matrix()` before a solve throws the structure away. A `DenseFallbackWarning` says a path densified.
- An untagged factor sends a `Kronecker` solve down the LU path instead of Cholesky.

## Next

- The verbs that use these: [solves](solves.md), [identities](identities.md).
- A sparse precision: [sparse](sparse.md). A GMRF precision: [gmrf](gmrf.md).
- To add an operator to gaussx itself, use its contributor skill `add-operator`.
