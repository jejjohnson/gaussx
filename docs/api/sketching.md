# Sketching

Random sketching operators $S \in \mathbb{R}^{d \times m}$ that compress a tall
matrix $A \in \mathbb{R}^{m \times n}$ to $SA \in \mathbb{R}^{d \times n}$
while approximately preserving the geometry of its range. $S$ is an
$\varepsilon$-subspace embedding for $\operatorname{range}(A)$ if

$$
(1-\varepsilon)\|Ax\| \le \|SAx\| \le (1+\varepsilon)\|Ax\| \qquad \forall x .
$$

Sketches are the foundation of the randomized linear-algebra stack (range
finders, randomized SVD, sketch-and-precondition least squares).

| Sketch | Size `d` for an $\varepsilon$-embedding | Cost to apply |
|---|---|---|
| `GaussianSketch` | $O(n/\varepsilon^2)$ | $O(dmn)$ |
| `SparseSignSketch` (default for tall problems) | $O(n\log n/\varepsilon^2)$ | $O(\text{nnz}\cdot mn)$ |
| `SRHTSketch` | $O((n+\log m)\log n/\varepsilon^2)$ | $O(mn\log m)$ |

A sketch is **sampled once**, with an explicit PRNG key (`key=None` means
`jax.random.PRNGKey(0)`), and its random draws live in the module: `apply`
($SA$) and `apply_transpose` ($S^\top Y$) always refer to the same $S$.
`sketch_operator` sketches a matrix-free lineax operator with $d$
transpose-matvecs, and `as_operator` returns $S$ itself as a lineax operator.

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
