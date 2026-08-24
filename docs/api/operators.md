# Operators & Tags

Layer 1: structured linear operators extending
[`lineax.AbstractLinearOperator`](https://docs.kidger.site/lineax/api/operators/).
All are immutable `equinox.Module` pytrees, so they compose freely with `jit`,
`grad`, and `vmap`. The [primitives](primitives.md) dispatch on these types: a
`solve` against a `Kronecker` factorizes per Kronecker factor, a `logdet` of a
`BlockDiag` sums per block, a `LowRankUpdate` solve applies Woodbury.

## Structured products & sums

The Kronecker product $A_1 \otimes A_2 \otimes \cdots$ gives $O(\sum_i n_i^3)$
solves on a $\prod_i n_i$ grid; the Kronecker *sum* $A \otimes I + I \otimes B$
diagonalises in the joint eigenbasis with eigenvalues $\lambda_i + \mu_j$.

!!! warning "`KroneckerSum` and `SumOfKroneckers` are different operators"
    They are not word-order variants of one idea:

    | | Math | Eigendecomposition |
    |---|---|---|
    | `KroneckerSum(A, B)` | $A \otimes I + I \otimes B$ | Closed form — eigenvalues $\lambda_i + \mu_j$, eigenvectors $V_A \otimes V_B$ |
    | `SumOfKroneckers(K_1, K_2, …)` | $\sum_k A_k \otimes B_k$ | None in general. `solve` / `logdet` reduce the *two-term* case with one term positive definite — $B \otimes C + \sigma^2 I$ and $B_1 \otimes C_1 + B_2 \otimes C_2$ — to one eigendecomposition per factor, $O(n_c^3 + n_d^3)$; `eigendecompose` covers the same two-term symmetric case more generally by densifying a $(n_c n_d)^2$ block |

    `SumOfKroneckers` was called `SumKronecker` until gh-136. The old name
    still imports and subclasses the new one — so `isinstance` checks keep
    working — but emits a `DeprecationWarning` on construction and will be
    removed in a future release.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Kronecker, BlockDiag, KroneckerSum, KroneckerSumSqrt, SumOfKroneckers, SumKronecker]

## Low-rank updates

$L + U\,\mathrm{diag}(d)\,V^\top$ with Woodbury-efficient solves and
matrix-determinant-lemma logdets. The factories build the common special cases
directly from arrays. Pass `orthonormal=True` when $U$ and $V$ have orthonormal
columns (truncated SVD, Nyström, ensemble factors) to unlock the stronger
symmetry / PSD tag inference.

!!! warning "`SVDLowRankUpdate` is deprecated"
    It remains a `LowRankUpdate` subclass — so `isinstance` checks and
    `singledispatch` registrations keyed on it still work — but it forces
    `orthonormal=True` and emits a `DeprecationWarning` on construction. New
    code should use `LowRankUpdate(base, U, S, V, orthonormal=True)` or
    [`svd_low_rank_plus_diag`](#gaussx.svd_low_rank_plus_diag). It will be
    removed in a future release.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LowRankUpdate, SVDLowRankUpdate, low_rank_plus_diag, low_rank_plus_identity, svd_low_rank_plus_diag]

## Banded & Toeplitz

Block-tridiagonal operators solve in $O(N d^3)$ via block-banded Cholesky — the
precision structure of Markovian (state-space) GPs. Symmetric Toeplitz
operators get $O(n \log n)$ matvecs and sampling via FFT circulant embedding.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [BlockTriDiag, LowerBlockTriDiag, UpperBlockTriDiag, Toeplitz, ToeplitzCholesky]

## Kernel operators

Kernel matrices as operators, plus grid-interpolated (KISS-GP style) and masked
variants.

### Choosing between the three kernel operators

All three are **matrix-free and scan-based** — none of them ever materializes
its kernel block. The word *implicit* in two of the names is therefore not the
distinction it appears to be. What actually separates them is the **shape
contract**, whether a **noise term is fused in**, and the **scan granularity**:

| | `KernelOperator` | `ImplicitKernelOperator` | `ImplicitCrossKernelOperator` |
|---|---|---|---|
| **Shape** | Rectangular $(N, M)$ | Square $(N, N)$ | Rectangular $(N, M)$ |
| **Points** | `X1`, `X2` (independent) | `X` (one set) | `X_data`, `X_inducing` |
| **Noise term** | — | **fused** `+ noise_var * I` | — |
| **Scan step** | one row of `X1` | one row of `X` | `batch_size` rows (default 1024) |
| **Peak memory/step** | $O(M)$ | $O(N)$ | $O(\texttt{batch\_size} \times M)$ |
| **Kernel signature** | `k(params, x, x')` (required) | `k(x, x')` or `k(params, x, x')` | `k(x, z)` or `k(params, x, z)` |
| **`jax.custom_jvp`** | always | only with `params=` | only with `params=` |

!!! warning "The custom JVP follows `params`, not the class"
    `KernelOperator` takes `params` as a required argument, so its matvec
    always runs under a `jax.custom_jvp` that differentiates the kernel
    without materializing Jacobians. The two `Implicit*` operators default to
    `params=None`, and in that mode `mv` runs the ordinary scan with no custom
    rule — autodiff falls back to differentiating straight through the scan.
    Pass a `params` pytree (and use the `k(params, x, x')` signature) if you
    are differentiating w.r.t. hyperparameters and want the efficient path.

Practical guidance:

- Building a **training covariance** you will pass to CG or BBMM? Use
  `ImplicitKernelOperator` — the fused noise term means $K$ and $\sigma^2 I$
  never exist as separate operators.
- Need a **general kernel block** between two point sets? `KernelOperator`.
- Need a **data-inducing block** and want to trade peak memory for throughput?
  `ImplicitCrossKernelOperator`, and tune `batch_size`.

!!! note "Naming rule for future kernel operators"
    Every kernel operator in gaussx is matrix-free, so *implicit*, *lazy*, and
    *matrix-free* carry no information in a class name — and neither would
    *scan*, since all three scan. Name a new kernel operator after the part of
    its **contract** that differs: what shape it produces, what it fuses in, or
    what point sets it relates. The existing `Implicit*` names predate this rule
    and are kept for compatibility; see
    [#135](https://github.com/jejjohnson/gaussx/issues/135) for the rename
    discussion.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [KernelOperator, ImplicitKernelOperator, ImplicitCrossKernelOperator, implicit_cross_kernel, InterpolatedOperator, MaskedOperator]

## Lazy algebra & sampling

Sum / scale / compose operators without materializing, sample
$\varepsilon \sim \mathcal{N}(0, A)$ for the structured families, and solve
bordered systems through the capacitance (Schur-complement) form.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [SumOperator, ScaledOperator, ProductOperator, kronecker_sum_sample, sumkronecker_sample, toeplitz_sample, CapacitanceSolver]

## Structural tags & predicates

Tags mark structure and properties on operators; the `is_*` predicates are what
the primitives consult when choosing an algorithm. The property tags
(`positive_semidefinite_tag`, `symmetric_tag`, the triangular tags, …) are
re-exported from lineax so user code only needs one import.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - is_kronecker
        - is_kronecker_sum
        - is_block_diagonal
        - is_block_tridiagonal
        - is_low_rank
        - is_diagonal
        - is_symmetric
        - is_positive_semidefinite
        - is_negative_semidefinite
        - is_lower_triangular
        - is_upper_triangular
        - kronecker_tag
        - kronecker_sum_tag
        - block_diagonal_tag
        - block_tridiagonal_tag
        - low_rank_tag
        - diagonal_tag
        - symmetric_tag
        - positive_semidefinite_tag
        - negative_semidefinite_tag
        - lower_triangular_tag
        - upper_triangular_tag
        - tridiagonal_tag
        - unit_diagonal_tag
