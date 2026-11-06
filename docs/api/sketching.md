# Sketching

Random sketching operators $\mathbf{S} \in \mathbb{R}^{d \times M}$ that compress a tall matrix $\mathbf{A} \in \mathbb{R}^{M \times N}$ to $\mathbf{S}\mathbf{A} \in \mathbb{R}^{d \times N}$ while approximately preserving the geometry of its range. $\mathbf{S}$ is an $\varepsilon$-subspace embedding for $\mathrm{range}(\mathbf{A})$ if

$$
(1 - \varepsilon) \Vert \mathbf{A}\mathbf{x} \Vert_2 \le \Vert \mathbf{S}\mathbf{A}\mathbf{x} \Vert_2 \le (1 + \varepsilon) \Vert \mathbf{A}\mathbf{x} \Vert_2
$$

for every $\mathbf{x} \in \mathbb{R}^N$.

Sketches are the foundation of the randomized linear-algebra stack (range
finders, randomized SVD, sketch-and-precondition least squares).

| Sketch | Size `d` for an $\varepsilon$-embedding | Cost to apply |
|---|---|---|
| `GaussianSketch` | $O(N/\varepsilon^2)$ | $O(dMN)$ |
| `SparseSignSketch` (default for tall problems) | $O(N\log N/\varepsilon^2)$ | $O(\mathrm{nnz}\cdot MN)$ |
| `SRHTSketch` | $O((N+\log M)\log N/\varepsilon^2)$ | $O(MN\log M)$ |

A sketch is **sampled once**, with an explicit PRNG key (`key=None` means
`jax.random.PRNGKey(0)`), and its random draws live in the module: `apply`
($\mathbf{S}\mathbf{A}$) and `apply_transpose` ($\mathbf{S}^\top \mathbf{Y}$) always refer to the same $\mathbf{S}$.
`sketch_operator` sketches a matrix-free lineax operator with $d$
transpose-matvecs, and `as_operator` returns $\mathbf{S}$ itself as a lineax operator.

```python
# Sketch a tall Jacobian (10⁶ residuals × 200 parameters) down to 800 rows
S = gx.SparseSignSketch.sample(key, d=800, m=1_000_000, nnz=8)
SJ = S.sketch_operator(J_op)  # (800, 200), matrix-free
sv = jnp.linalg.svd(SJ, compute_uv=False)  # J's singular values, within (1 ± ε)
```

## Abstract interface

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractSketch]

## Sketches

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianSketch, OrthonormalSketch, SparseSignSketch, SRHTSketch, RowSamplingSketch]

## Fast transforms

The unnormalised fast Walsh–Hadamard transform behind `SRHTSketch` (and
kernellib's FastFood features).

::: gaussx
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [hadamard_transform]
