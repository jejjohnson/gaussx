# gaussx

> Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX.

Built on [lineax](https://github.com/patrick-kidger/lineax), [equinox](https://github.com/patrick-kidger/equinox), and [matfree](https://github.com/pnkraemer/matfree).

## Installation

```bash
pip install gaussx
```

Or with `uv`:

```bash
uv add gaussx
```

## Quickstart

```python
import jax.numpy as jnp
import lineax as lx
import gaussx

# Structured operators
A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
B = lx.DiagonalLinearOperator(jnp.array([3.0, 4.0]))
K = gaussx.Kronecker(A, B)

# Primitives exploit structure automatically
v = jnp.ones(4)
x = gaussx.solve(K, v)       # per-factor solve
ld = gaussx.logdet(K)         # n2*logdet(A) + n1*logdet(B)
L = gaussx.cholesky(K)        # Kronecker(chol(A), chol(B))
t = gaussx.trace(K)           # trace(A) * trace(B)
```

## Examples

- [Basics](notebooks/basics.ipynb) — operators, primitives, JAX transforms
- [Kernel Regression](notebooks/kernel_regression.ipynb) — GP regression with hyperparameter optimization
- [Structured GP](notebooks/structured_gp.ipynb) — Kronecker and low-rank structure for scalable GPs

## Links

- [API Reference](api/reference.md)
- [GitHub](https://github.com/jejjohnson/gaussx)
