# Randomized Linear Algebra

Randomized factorisations that touch a matrix only through a few columns or
matvecs. They take an explicit PRNG `key`; `key=None` means
`jax.random.PRNGKey(0)`.

## Range finder, QB, SVD and eigh

Randomized subspace iteration (Halko, Martinsson & Tropp, 2011) finds an
orthonormal basis $\mathbf{Q}$ for the dominant range of a matrix-free operator $\mathbf{A}$ from a
block of $\ell = k + p$ matvecs. For a Gaussian test matrix,

$$
\mathbb{E} \Vert \mathbf{A} - \mathbf{Q}\mathbf{Q}^\top \mathbf{A} \Vert_2 \le \left(1 + \sqrt{\frac{k}{p-1}}\right) s_{k+1} + \frac{e\sqrt{k+p}}{p} \left(\sum_{j>k} s_j^2\right)^{1/2}
$$

with $s_j$ the singular values of $\mathbf{A}$.

The tail term hurts for slowly decaying spectra (Matérn-½ Gram matrices, most
geophysical fields); `n_power_iter=q` applies the bound to $(\mathbf{A}\mathbf{A}^\top)^q \mathbf{A}$,
whose singular values are $s_j^{2q+1}$, for $2q$ more passes. Use
`n_power_iter >= 2` there. These methods target the **top** of the spectrum;
the small end (e.g. the smallest Laplacian eigenvalues) is Lanczos / LOBPCG
territory.

`qb` returns $\mathbf{Q}$ and $\mathbf{B} = \mathbf{Q}^\top \mathbf{A}$, `randomized_svd` lifts the SVD of $\mathbf{B}$,
and `randomized_eigh` is the Rayleigh–Ritz projection $\mathbf{Q}^\top \mathbf{A} \mathbf{Q}$ for
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
$\hat{\mathbf{A}} = (\mathbf{A}\boldsymbol{\Omega})(\boldsymbol{\Omega}^\top \mathbf{A} \boldsymbol{\Omega})^{+}(\mathbf{A}\boldsymbol{\Omega})^\top$ from one pass of
$\ell$ matvecs (Tropp, Yurtsever, Udell & Cevher, 2017, Algorithm 3). It
satisfies $0 \preceq \hat{\mathbf{A}} \preceq \mathbf{A}$ and, for the same $\ell$, is more
accurate than the Rayleigh–Ritz projection of `randomized_eigh`. The result
is an orthonormal [`LowRankUpdate`](operators.md#gaussx.LowRankUpdate)
$\mathbf{U} \mathrm{diag}(\hat{\boldsymbol{\gamma}}) \mathbf{U}^\top$, so the same factors on a $\sigma^2 \mathbf{I}$ base solve and
take log-determinants of $\hat{\mathbf{A}} + \sigma^2 \mathbf{I}$ through the Woodbury rules. It
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

`column_id` returns $k$ **actual** columns $J$ of $\mathbf{A}$ and an interpolation matrix $\mathbf{X}$ ($[\mathbf{X}]_{:,J} = \mathbf{I}_k$) with $\mathbf{A} \approx [\mathbf{A}]_{:,J} \mathbf{X}$, from column-pivoted QR of a randomized sketch of the row space (`range_finder` on $\mathbf{A}^\top$; Voronin & Martinsson, 2017). `cur` adds a row ID of $\mathbf{C} = [\mathbf{A}]_{:,J}$ and returns $\mathbf{A} \approx \mathbf{C}\mathbf{U}\mathbf{R}$ with $\mathbf{C} = [\mathbf{A}]_{:,J}$, $\mathbf{R} = [\mathbf{A}]_{I,:}$ and $\mathbf{U} = \mathbf{X}\mathbf{R}^{+}$. Unlike
the SVD factors, $\mathbf{C}$ and $\mathbf{R}$ are columns and rows of $\mathbf{A}$: they keep its
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
