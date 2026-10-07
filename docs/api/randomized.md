# Randomized Linear Algebra

Randomized factorisations that touch a matrix only through a few columns or
matvecs. They take an explicit PRNG `key`; `key=None` means
`jax.random.PRNGKey(0)`.

## Range finder, QB, SVD and eigh

Randomized subspace iteration (Halko, Martinsson & Tropp, 2011) finds an
orthonormal basis $Q$ for the dominant range of a matrix-free operator from a
block of $\ell = k + p$ matvecs. For a Gaussian test matrix,

$$
\mathbb E\,\|A - QQ^\top A\|_2 \le \Big(1+\sqrt{\tfrac{k}{p-1}}\Big)\sigma_{k+1}
+ \frac{e\sqrt{k+p}}{p}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}.
$$

The tail term hurts for slowly decaying spectra (Matérn-½ Gram matrices, most
geophysical fields); `n_power_iter=q` applies the bound to $(AA^\top)^qA$,
whose singular values are $\sigma_j^{2q+1}$, for $2q$ more passes. Use
`n_power_iter >= 2` there. These methods target the **top** of the spectrum;
the small end (e.g. the smallest Laplacian eigenvalues) is Lanczos / LOBPCG
territory.

`qb` returns $Q$ and $B = Q^\top A$, `randomized_svd` lifts the SVD of $B$,
and `randomized_eigh` is the Rayleigh–Ritz projection $Q^\top A Q$ for
symmetric, possibly indefinite, operators. `svd(op, rank=k,
method="randomized")` and `eig(op, rank=k, method="randomized")` route here;
Lanczos stays the default.

```python
# 50 EOFs of a (100k pixels × 3650 days) anomaly matrix, available only as a matvec
U, s, Vt = gx.randomized_svd(anomalies_op, 50, oversample=10, n_power_iter=2, key=key)
eofs, pcs = U, einx.multiply("k, k t -> k t", s, Vt)
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [range_finder, qb, randomized_svd, randomized_eigh]

## Randomized Nyström

For a PSD operator, `randomized_nystrom` returns the Nyström approximation
$\hat A = (A\Omega)(\Omega^\top A\Omega)^{+}(A\Omega)^\top$ from one pass of
$\ell$ matvecs (Tropp, Yurtsever, Udell & Cevher, 2017, Algorithm 3). It
satisfies $0 \preceq \hat A \preceq A$ and, for the same $\ell$, is more
accurate than the Rayleigh–Ritz projection of `randomized_eigh`. The result
is an orthonormal [`LowRankUpdate`](operators.md#gaussx.LowRankUpdate)
$U\hat\Lambda U^\top$, so the same factors on a $\sigma^2 I$ base solve and
take log-determinants of $\hat A + \sigma^2 I$ through the Woodbury rules. It
is also the sketch behind
[`NystromPreconditioner`](solvers.md#gaussx.NystromPreconditioner).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [randomized_nystrom]

## Randomly pivoted Cholesky

`rp_cholesky` builds a partial Cholesky factor from the diagonal and a
`column(j)` callable, picking each pivot with probability proportional to the
residual diagonal (Chen, Epperly, Tropp & Webber, 2023). It returns the
pivots too, so they serve as landmark indices for Nyström, Falkon or SVGP
inducing points without ever forming the kernel matrix. `pivoting="greedy"`
is the classic pivoted Cholesky behind
[`PartialCholeskyPreconditioner`](solvers.md#gaussx.PartialCholeskyPreconditioner).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [rp_cholesky]

## Interpolative and CUR decompositions

`column_id` returns $k$ **actual** columns $J$ of $A$ and an interpolation
matrix $X$ ($X_{:,J} = I$) with $A \approx A_{:,J}X$, from column-pivoted QR
of a randomized sketch of the row space (`range_finder` on $A^\top$; Voronin
& Martinsson, 2017). `cur` adds a row ID of $C = A_{:,J}$ and returns
$A \approx CUR$ with $C = A_{:,J}$, $R = A_{I,:}$ and $U = XR^{+}$. Unlike
the SVD factors, $C$ and $R$ are columns and rows of $A$: they keep its
sparsity, sign and units, which makes them interpretable as representative
stations, time steps or landmark points.

```python
# 40 representative stations (columns) of a (days × stations) anomaly matrix
cols, X = gx.column_id(anomalies_op, 40, key=key)
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [column_id, cur, ColumnID, CUR]
