---
name: structured-gaussian-linalg
description: Write covariance and precision code in JAX on gaussx instead of dense jnp.linalg — structured operators (Kronecker, Kronecker sums, block-diagonal, block-tridiagonal, low-rank plus diagonal, Toeplitz, circulant, sparse, GMRF precisions), tags, and structure-aware solve / logdet / Cholesky / sqrt / inv / diag / trace, Woodbury and Schur identities. Use whenever a task builds a covariance or precision matrix, solves a linear system, takes a log-determinant or factorises a matrix in JAX, in a project that uses (or could use) gaussx. The front door to gaussx; it routes to gaussx:solvers-and-preconditioners and gaussx:gaussians-and-recipes.
---

# Structured Gaussian linear algebra with gaussx

Your covariance matrix has structure.
`jnp.linalg.solve`, `cho_solve` and `slogdet` throw it away and pay O(N³); gaussx keeps it and pays for the structure instead.
A Kronecker product of two 100 × 100 factors costs two small solves, not one 10 000 × 10 000 solve.

This skill covers the two lowest rungs of gaussx: ⓪ structures (the operators) and ① primitives (the verbs on them).
The rungs are those of the `role-ladder` skill, which says what a structure, a primitive and a strategy are; this skill does not repeat it.

## Where to enter

| You have | Enter at |
|---|---|
| A matrix with structure and a right-hand side | here: build the operator, then call the verb |
| A system too big for an exact factorisation, a bare `v ↦ A v`, or a preconditioner to tune | `gaussx:solvers-and-preconditioners` |
| A Gaussian to evaluate, condition or sample; a GP, a Kalman filter, a quadrature or an ensemble update | `gaussx:gaussians-and-recipes` |

## The three rules

1. **Operators, not arrays.** A covariance or a precision is a `lineax.AbstractLinearOperator`. A dense matrix enters as `lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)`; the tag is what lets gaussx and lineax take the Cholesky and PSD paths instead of LU.
2. **Say what the structure is.** `gaussx.Kronecker(K1, K2)`, not `jnp.kron(K1, K2)`. `gaussx.low_rank_plus_diag(d, U, psd=True)`, not `jnp.diag(d) + U @ U.T`. Lineax's algebra (`A + B`, `2.0 * A`, `A @ B`) keeps operators lazy.
3. **Never materialise to solve.** `op.as_matrix()` followed by a dense solve undoes both rules. A `DenseFallbackWarning` means a path had to densify, and names the matrix-free alternative.

## Nouns and verbs

- **Nouns.** `Kronecker`, `KroneckerSum`, `SumOfKroneckers`, `BlockDiag`, `BlockTriDiag`, `Toeplitz`, `circulant`, `SparseOperator`, `low_rank_plus_diag`, `LowRankUpdate`, `MaskedOperator`, `InterpolatedOperator`.
- **Verbs.** `solve`, `logdet`, `cholesky`, `sqrt`, `inv`, `diag`, `trace`, `eig`, `svd`; `solve_matrix`, `woodbury_solve`, `schur_complement`, `safe_cholesky`.
- **Adverb.** `gaussx.solve(op, b, solver=...)`. Left out, the verb dispatches on the structure. The strategies are in `gaussx:solvers-and-preconditioners`.

## Worked example

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import gaussx

PSD = lx.positive_semidefinite_tag


def rbf(x, lengthscale):  # (n,) → (n, n)
    d = x[:, None] - x[None, :]
    return jnp.exp(-0.5 * (d / lengthscale) ** 2) + 1e-4 * jnp.eye(x.shape[0])


x1, x2 = jnp.linspace(0, 1, 8), jnp.linspace(0, 1, 6)  # an 8 × 6 grid
y = jr.normal(jr.key(0), (8 * 6,))  # (N,), N = 48


def quad_and_logdet(lengthscale):
    K = gaussx.Kronecker(
        lx.MatrixLinearOperator(rbf(x1, lengthscale), PSD),  # (8, 8)
        lx.MatrixLinearOperator(rbf(x2, lengthscale), PSD),  # (6, 6)
    )  # (N, N), never formed: O(n₁³ + n₂³), not O(N³)
    return y @ gaussx.solve(K, y) + gaussx.logdet(K)  # ()


value, grad = jax.jit(jax.value_and_grad(quad_and_logdet))(0.3)
assert jnp.isfinite(value) and jnp.isfinite(grad)

U = jr.normal(jr.key(1), (200, 3))  # (n, k)
A = gaussx.low_rank_plus_diag(jnp.full(200, 0.5), U, psd=True)  # D + U Uᵀ
x = gaussx.solve(A, jnp.ones(200))  # Woodbury, O(n k²)
assert jnp.allclose(A.mv(x), jnp.ones(200), atol=1e-3)
```

## Don't write it

| Don't write | Use |
|---|---|
| `jnp.linalg.solve(K, b)`, `cho_solve(cho_factor(K), b)`, `solve_triangular` pairs | `gaussx.solve(lx.MatrixLinearOperator(K, PSD), b)`, or on the structured operator |
| `jnp.linalg.slogdet(K)[1]`, `2 * jnp.sum(jnp.log(jnp.diag(L)))` | `gaussx.logdet(op)`; from a factor you already have, `gaussx.cholesky_logdet(L)` |
| `jnp.linalg.inv(K) @ B` | `gaussx.solve_matrix(op, B)`, or `gaussx.inv(op)`, which stays lazy |
| A jitter-retry loop around `jnp.linalg.cholesky` | `gaussx.safe_cholesky(op)` |
| `jnp.kron(A, B)` then a solve, a log-determinant or a sample | `gaussx.Kronecker(A, B)`; `A ⊗ I + I ⊗ B` is `KroneckerSum`; `Σᵢ Aᵢ ⊗ Bᵢ` is `SumOfKroneckers` |
| The Woodbury identity or the matrix-determinant lemma by hand | `gaussx.low_rank_plus_diag`, `low_rank_plus_identity`, `LowRankUpdate`, `woodbury_solve` |

A dense `jnp.linalg` call is fine on a small matrix with no structure to use.
The rule is not "never"; it is "not on a matrix gaussx can do better".

## The shelf

Each entry is one file with the names it covers, a worked example, the rules and what goes wrong.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ⓪ structure | [Structured operators](shelf/operators.md) | the Kronecker family, block, low-rank, Toeplitz and circulant, masked and interpolated operators | 34 |
| ⓪ structure | [Tags and predicates](shelf/tags.md) | PSD, symmetric, Kronecker and the other tags, and what each unlocks | 24 |
| ⓪ structure | [Sparse operators](shelf/sparse.md) | `SparseOperator`, patterns, symbolic and numeric sparse Cholesky | 6 |
| ⓪ structure | [GMRF precisions](shelf/gmrf.md) | SPDE, random-walk, AR1, Besag and BYM2 precision builders | 12 |
| ① primitive | [Solves and factorisations](shelf/solves.md) | `solve`, `logdet`, `cholesky`, `sqrt`, `inv`, `diag`, `trace`, `eig`, `svd` and the root decompositions | 28 |
| ① primitive | [Linear-algebra identities](shelf/identities.md) | Woodbury, Schur complements, Lyapunov, tridiagonal solves, selected inverse | 20 |
<!-- shelf:end -->

## Before you finish

- Every covariance or precision is a lineax operator, PSD-tagged where it is PSD, and built with the structure it has.
- No `as_matrix()` feeds a solve, a log-determinant, a Cholesky or a sample, and no `DenseFallbackWarning` is ignored.
- The result matches a dense reference on a small case, and the code runs under `jax.jit` and `jax.grad`.

If gaussx lacks what you need, keep the addition small and shaped like gaussx: an operator in, an operator or an array out, pure, with `solver=` where it solves.

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Nothing else in the stack: this is a base skill.

Used by (pointers, not skills to load): argminax, filterax, finitevolX, gauss_flows, integrix, kernellib, manipy, plumax, pyrox, radtrax, spectraldiffx, vardax.

<!-- skill-tree:end -->
