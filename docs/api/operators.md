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

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Kronecker, BlockDiag, KroneckerSum, KroneckerSumSqrt, SumKronecker]

## Low-rank updates

$L + U\,\mathrm{diag}(d)\,V^\top$ with Woodbury-efficient solves and
matrix-determinant-lemma logdets. The factories build the common special cases
directly from arrays.

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

Kernel matrices as operators — dense (`KernelOperator`), matrix-free
(`ImplicitKernelOperator`, rows generated on the fly per matvec), rectangular
cross-kernels, and grid-interpolated (KISS-GP style) variants.

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
