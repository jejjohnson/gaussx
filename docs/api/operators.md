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

## Fast-diagonalisable operators

`DiagonalisedOperator` is an operator diagonal in a fast transform basis,
$A = V^{-1}\Lambda V$, given by a forward/inverse transform pair (FFT,
orthonormal DCT/DST, spherical harmonics, or a dense eigenvector matrix) and
the eigenvalue array $\Lambda$. `solve`, `logdet`, `inv`, `sqrt` and `trace`
are elementwise in $\Lambda$, and shifts/scalings such as $A - \lambda I$
stay diagonalised, so a spectral Helmholtz solve is never materialised.
`Circulant` / `circulant_from_symbol` are the FFT special case (periodic
stencils, stationary covariances on periodic grids), and a `KroneckerSum` whose
factors are all diagonalised solves through the composed per-axis transforms.
Use `DiagonalisedOperator.from_eigen_factorization` for dense non-symmetric
diagonalisable factors such as Chebyshev collocation blocks.

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

n = 128
k = 2 * jnp.pi * jnp.fft.fftfreq(n)
symbol = (2 * jnp.cos(k) - 2)[:, None] + (2 * jnp.cos(k) - 2)[None, :]
laplacian = gaussx.circulant_from_symbol(symbol)          # periodic 5-point ∇²
helmholtz = laplacian - 1.0 * lx.IdentityLinearOperator(laplacian.in_structure())
psi = gaussx.solve(helmholtz, f)                          # two FFTs, no matrix
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DiagonalisedOperator, Circulant, circulant_from_symbol, as_diagonalised]

## Interpolated & masked operators

Grid-interpolated (KISS-GP style) and masked operators, plus the grid and
cubic-interpolation helpers that build them. Kernel operators themselves
(`KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`)
moved to [kernellib](https://github.com/jejjohnson/kernellib) in gaussx 0.2.0;
they are lineax operators, so every gaussx primitive and solver strategy works
on them unchanged.

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [InterpolatedOperator, MaskedOperator, grid_coupling_indices, create_grid, grid_data, cubic_interpolation_weights]

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
