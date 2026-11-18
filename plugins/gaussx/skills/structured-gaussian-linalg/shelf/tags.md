# Tags and predicates

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: structures and primitives.

Open this when a verb is slow or warns about a dense fallback, or when you need to ask an operator what it is.
Tags are how gaussx and lineax choose a method; the operators that carry them are in [operators](operators.md).

## Names

The first eight rows are lineax tags and predicates, re-exported by gaussx so one import covers both.
The five gaussx tags and their predicates mark structure that lineax has no name for.

| Name | What it is |
|---|---|
| `symmetric_tag`, `is_symmetric(op)` | `A = Aᵀ`. |
| `positive_semidefinite_tag`, `is_positive_semidefinite(op)` | PSD; `solve` then uses a Cholesky factorisation on the dense path. |
| `negative_semidefinite_tag`, `is_negative_semidefinite(op)` | NSD. |
| `diagonal_tag`, `is_diagonal(op)` | Diagonal; `solve` divides by the diagonal. |
| `lower_triangular_tag`, `is_lower_triangular(op)` | Lower triangular; `solve` uses a triangular solve. |
| `upper_triangular_tag`, `is_upper_triangular(op)` | Upper triangular. |
| `tridiagonal_tag` | A lineax tag for tridiagonal structure. |
| `unit_diagonal_tag` | A lineax tag for an operator with a unit diagonal. |
| `kronecker_tag`, `is_kronecker(op)` | A Kronecker product; `Kronecker` adds it itself. |
| `kronecker_sum_tag`, `is_kronecker_sum(op)` | `A ⊗ I + I ⊗ B`; `KroneckerSum` adds it itself. |
| `block_diagonal_tag`, `is_block_diagonal(op)` | Block diagonal; `BlockDiag` adds it itself. |
| `block_tridiagonal_tag`, `is_block_tridiagonal(op)` | Block tridiagonal; `BlockTriDiag` adds it itself. |
| `low_rank_tag`, `is_low_rank(op)` | `L + U diag(d) Vᵀ`; `LowRankUpdate` adds it itself. |

## Use it

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag
x = jnp.linspace(0, 1, 5)
dense = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.3) ** 2) + 1e-2 * jnp.eye(
    5
)  # (5, 5)

tagged = lx.MatrixLinearOperator(dense, PSD)
bare = lx.MatrixLinearOperator(dense)  # same numbers, no claim
assert gaussx.is_positive_semidefinite(tagged)
assert not gaussx.is_positive_semidefinite(bare)

K = gaussx.Kronecker(tagged, tagged)  # (25, 25)
assert gaussx.is_kronecker(K) and not gaussx.is_kronecker(tagged)
assert gaussx.kronecker_tag in K.tags
assert gaussx.is_symmetric(K) and gaussx.is_positive_semidefinite(
    K
)  # follows from the factors
assert not gaussx.is_positive_semidefinite(
    gaussx.Kronecker(bare, bare)
)  # one untagged factor is enough to lose it

U = jnp.ones((5, 2))
A = gaussx.low_rank_plus_diag(jnp.ones(5), U)
assert gaussx.is_low_rank(A) and gaussx.low_rank_tag in A.tags

b = jnp.arange(25.0)
assert jnp.allclose(
    gaussx.solve(K, b), jnp.linalg.solve(K.as_matrix(), b), rtol=1e-2, atol=1e-2
)
```

## Rules

- A lineax tag on a leaf is a claim you make, not a check of the values, so tag only what is true.
- Tag each leaf operator (`lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)`); `Kronecker`, `BlockDiag` and `KroneckerSum` answer the lineax predicates from their factors.
- The gaussx predicates (`is_kronecker` and the rest) are registered per operator class, so only the matching gaussx operator answers `True`.
- `BlockTriDiag` is symmetric by default (`symmetric=True`); a `LowRankUpdate` is symmetric when its factors are shared.
- `sum_operator`, `scaled_operator` and `product_operator` take `tags=` when you want the result to carry a tag lineax would not infer.
- Test structure with the predicates or `op.tags`, not with `isinstance` on an operator you did not build.

## Goes wrong

- Tagging an operator PSD when it is not sends the dense `solve` down the Cholesky path, which is wrong for it.
- Leaving a leaf untagged makes every composite above it lose PSD and fall back to LU.
- `is_kronecker` on a `lx.TaggedLinearOperator(M, kronecker_tag)` returns `False`, because the predicate looks at the class.

## Next

- The operators that set these tags themselves: [operators](operators.md).
- What a tag unlocks in a solve or log-determinant: [solves](solves.md).
- A sparse operator also carries tags: [sparse](sparse.md).
