# GMRF precisions

Role: ⓪ structure (`role-ladder`, `references/structure.md`).
Unit: structures and primitives.

Open this when the model is a Gaussian Markov random field and you need its sparse precision matrix: a time series, a graph, or a Matérn field on a mesh or a raster.
These builders return operators, so you factor or solve them with the verbs in [solves](solves.md) and never form a covariance.

## Names

| Name | What it is |
|---|---|
| `iid_precision(n, tau)` | `τ I` for `n` independent effects, as a `lineax` diagonal operator. |
| `rw1_structure(n, *, spacing=None, cyclic=False)` | The structure matrix `R` of a first-order random walk, singular with the constants as null space. A `BlockTriDiag`, or a `SparseOperator` when `cyclic=True`. |
| `rw2_structure(n, *, cyclic=False)` | The structure matrix of a second-order random walk, with null space `span{1, t}`. For odd `n` the open version has one extra decoupled node. |
| `ar1_precision(n, rho, tau)` | The precision of a stationary AR(1) process with marginal precision `τ`, as a `BlockTriDiag` with `1 × 1` blocks. |
| `besag_structure(laplacian_op)` | Checks a graph Laplacian is square, symmetric and has zero row sums, then tags it PSD. |
| `generalized_variance_scale(structure, null_space, *, eps=None)` | The scalar `s` such that `s R` has generalised variance one, so `τ` means the same on every graph. |
| `bym2_precision(structure_scaled, tau, phi)` | The joint precision of the BYM2 pair `(b, u*)`, a `SparseOperator` of size `2n`. |
| `matern_spde_params(range, sigma, nu, d)` | Returns `(kappa, tau, alpha)` for a Matérn field. `nu + d/2` must be a positive integer. |
| `fem_matrices(vertices, triangles)` | The lumped mass `C̃` as a diagonal operator and the stiffness `G` as a `SparseOperator`, for P1 elements on a triangle mesh. |
| `fem_projector(vertices, triangles, points, *, triangle_index=None)` | The `(n, V)` sparse matrix of barycentric weights that maps vertex values to observation points. |
| `spde_precision(C_lumped, G, kappa, tau, alpha)` | The sparse SPDE precision `τ² K (C̃⁻¹K)^{α−1}` with `K = κ² C̃ + G`. |
| `spde_precision_grid(shape, kappa, tau, alpha, *, spacing=1.0, periodic=False)` | The same precision on a regular grid, as a `SpectralFunction` of a Kronecker sum. |

## Use it

```python
import jax.numpy as jnp
import numpy as np

import gaussx

Q = gaussx.ar1_precision(20, rho=0.8, tau=10.0)  # (20, 20) BlockTriDiag
assert jnp.allclose(
    Q.as_matrix() @ jnp.linalg.inv(Q.as_matrix()), jnp.eye(20), atol=1e-3
)
assert jnp.allclose(
    gaussx.diag_inv(Q), 0.1, atol=1e-3
)  # every marginal variance is 1 / tau

R = gaussx.rw1_structure(6)  # singular: constants are in the null space
assert jnp.allclose(R.mv(jnp.ones(6)), 0.0, atol=1e-5)
s = gaussx.generalized_variance_scale(
    R, jnp.ones(6)
)  # use s * R as the scaled structure

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
triangles = np.array([[0, 1, 2], [0, 2, 3]])
C, G = gaussx.fem_matrices(vertices, triangles)
kappa, tau, alpha = gaussx.matern_spde_params(
    range=0.5, sigma=1.0, nu=1.0, d=2
)  # alpha == 2
Qm = gaussx.spde_precision(C, G, kappa, tau, alpha)  # (4, 4) SparseOperator
K = G.as_matrix() + kappa**2 * C.as_matrix()
dense = tau**2 * K @ jnp.linalg.inv(C.as_matrix()) @ K
assert jnp.allclose(Qm.as_matrix(), dense, rtol=1e-3, atol=1e-3)

Qg = gaussx.spde_precision_grid((5, 6), kappa=1.0, tau=1.0, alpha=2)  # (30, 30)
assert jnp.allclose(
    gaussx.diag_inv(Qg), jnp.diag(jnp.linalg.inv(Qg.as_matrix())), rtol=1e-2
)
```

## Rules

- Prefer a time-series prior as `ar1_precision`, `rw1_structure` or `rw2_structure`; they are banded and stay `BlockTriDiag`.
- The random-walk structures are singular, so scale them by `τ` and handle the null space before you ask for a log-determinant; see `pseudo_logdet` in [solves](solves.md).
- Scale `R` with `generalized_variance_scale` before `bym2_precision`; `bym2_precision` needs a `SparseOperator` or a scalar multiple of one.
- Get `(kappa, tau, alpha)` from `matern_spde_params` and pass them on; `nu` and `d` must be concrete.
- Build mesh matrices once with `fem_matrices`, then vary `kappa` and `tau` under `jit` or `grad` without redoing the host work.
- On a regular raster use `spde_precision_grid`; its solves and `diag_inv` go through the factor eigenvectors with no Cholesky.
- Gaussx builds no meshes or graphs; bring the triangles or the Laplacian from another package.

## Goes wrong

- `matern_spde_params` raises `ValueError` when `nu + d/2` is not a positive integer, so `nu=0.5` in two dimensions fails.
- `spde_precision` and `spde_precision_grid` impose natural boundaries, which inflate the variance near an edge; extend the domain by about one range and discard the extension.
- `rw2_structure(n)` with odd `n` has size `n + 1`; slice results with `[:n]`.
- `besag_structure` raises when a row of the Laplacian does not sum to zero.

## Next

- The operators these return: [operators](operators.md), [sparse](sparse.md).
- Marginal variances and selected inverses of them: [identities](identities.md).
- The tags they carry: [tags](tags.md).
