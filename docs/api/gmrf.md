# GMRF Precision Builders

The components of a latent Gaussian model, as precision operators. Each is a
quadratic form $\tfrac{\tau}{2}\|Dx\|^2$ for a sparse difference operator $D$,
so $Q = \tau D^\top D$ is sparse, and each builder returns the structured
operator that sends `solve`, `logdet`, `diag_inv` and sampling to their
cheapest exact path:

| Builder | Returns | Exact path |
|---|---|---|
| `iid_precision` | `lineax.DiagonalLinearOperator` | elementwise |
| `rw1_structure`, `rw2_structure`, `ar1_precision` | `BlockTriDiag` (`SparseOperator` when cyclic) | block Cholesky and the block selected inverse, $O(N d^3)$ |
| `besag_structure`, `bym2_precision` | `SparseOperator` | sparse Cholesky (dense below `AutoSolver`'s threshold until it lands) |
| `spde_precision` (+ `fem_matrices`) | `SparseOperator` | sparse Cholesky, as above |
| `spde_precision_grid` | `SpectralFunction` of a `KroneckerSum` | factor eigenvectors, $O(\sum_m n_m^3)$ once and $O(N\sum_m n_m)$ per call |

gaussx never builds graphs or meshes: a graph's structure matrix and null
space arrive as an operator and an array (kernellib's
`Graph.laplacian_operator()` and `graph_null_space`), and meshes come from
fmesher, pygmsh or meshio.

**The maths.**

- **RW1 / RW2.** Increments $x_{i+1}-x_i$ (second differences
  $x_{i+1}-2x_i+x_{i-1}$) are $\mathcal N(0,\tau^{-1})$; the structure matrix
  has null space $\{\mathbf 1\}$ ($\{\mathbf 1, t\}$). RW2 is stored with
  $2\times 2$ blocks, so an odd $n$ gets one decoupled unit-precision padding
  node; strip it from results.
- **AR(1).**
  $Q = \frac{\tau}{1-\rho^2}\operatorname{tridiag}(-\rho,\ 1+\rho^2,\ -\rho)$
  with 1 in the corners: every marginal variance is $1/\tau$.
- **BYM2** (Riebler et al., 2016). The pair $(b, u^*)$ with
  $b = (\sqrt{1-\phi}\,v + \sqrt\phi\,u^*)/\sqrt\tau$ has a sparse joint
  precision whose pattern does not depend on $(\tau, \phi)$. $u^*$ is scaled
  by `generalized_variance_scale`, the geometric mean of the constrained
  marginal variances (Sørbye & Rue, 2014); it matches R-INLA's
  `inla.scale.model` (golden fixture, `scripts/golden/inla/`).
- **SPDE** (Lindgren, Rue & Lindström, 2011).
  $(\kappa^2-\Delta)^{\alpha/2}(\tau x) = \mathcal W$ has Matérn covariance
  with $\nu = \alpha - d/2$. With P1 elements, $K = \kappa^2\tilde C + G$
  and $Q_\alpha = \tau^2 K(\tilde C^{-1}K)^{\alpha-1}$; on a grid with
  spacing $h$, $Q_\alpha = \tau^2 h^d(\kappa^2 I + h^{-2}(L_1\oplus\cdots\oplus L_d))^\alpha$.
  `matern_spde_params` converts (range, $\sigma$, $\nu$) into
  $(\kappa, \tau, \alpha)$.

**Boundary effects and domain extension.** The SPDE on a bounded domain,
mesh or grid, has natural (Neumann) boundary conditions. They inflate the
marginal variance within about one practical range of the boundary: up to
about $2\sigma^2$ on an edge and $4\sigma^2$ in a corner of a raster. Extend
the domain by at least one range beyond the region of interest and discard
the extension: a larger raster, or a mesh with an outer ring of coarser
triangles. Periodic axes of a grid (`periodic=(False, True)` for longitude on
a global raster) and closed surfaces such as a sphere have no boundary. On
the matching right-triangle mesh, `spde_precision_grid` equals
`spde_precision` exactly at nodes at least $\alpha$ cells from the boundary;
nearer the boundary the mesh's lumped mass and half-weight boundary edges
differ, which is the same boundary effect.

**Example.**

```python
import jax.numpy as jnp
import numpy as np

import gaussx as gx

# Temporal: a daily RW2 trend and an AR(1) nuisance
R_trend = gx.rw2_structure(364)  # null space {1, t}
Q_ar = gx.ar1_precision(365, rho=0.8, tau=10.0)
sd_ar = jnp.sqrt(gx.diag_inv(Q_ar))  # block selected inverse, O(N)

# Areal: BYM2 on a graph (here the path 0 - 1 - 2 - 3; a county graph from
# kernellib in practice)
senders, receivers = np.array([1, 2, 3]), np.array([0, 1, 2])
degree = np.bincount(np.r_[senders, receivers], minlength=4).astype(float)
R = gx.besag_structure(
    gx.SparseOperator.from_coo(
        np.r_[np.arange(4), senders],
        np.r_[np.arange(4), receivers],
        jnp.asarray(np.r_[degree, -np.ones(3)]),
        (4, 4),
        symmetric=True,
    )
)
s = gx.generalized_variance_scale(R, jnp.ones(4))
Q_bym2 = gx.bym2_precision(s * R, tau=1.5, phi=0.7)  # sparse (b, u*) stack

# Continuous space: Matérn ν = 1 on a mesh, range 0.5, sd 2
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
C, G = gx.fem_matrices(vertices, triangles)
kappa, tau, alpha = gx.matern_spde_params(range=0.5, sigma=2.0, nu=1.0, d=2)
Q_spde = gx.spde_precision(C, G, kappa, tau, alpha)
stations = np.array([[0.2, 0.1], [0.7, 0.6]])
A = gx.fem_projector(vertices, triangles, stations)  # 3 non-zeros per row

# ...or on a global raster, with no mesh at all
Q_grid = gx.spde_precision_grid(
    (90, 180), kappa=0.3, tau=1.0, alpha=2, periodic=(False, True)  # wrap longitude
)
sd_grid = jnp.sqrt(gx.diag_inv(Q_grid))  # exact, two small matrix products per axis
```

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - iid_precision
        - rw1_structure
        - rw2_structure
        - ar1_precision
        - besag_structure
        - generalized_variance_scale
        - bym2_precision
        - spde_precision
        - spde_precision_grid
        - matern_spde_params
        - fem_matrices
        - fem_projector
