# gaussx

[![Tests](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/gaussx/actions/workflows/typecheck.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Structured linear algebra, Gaussian distributions, and exponential family primitives for JAX.**

Built on top of [lineax](https://github.com/patrick-kidger/lineax), [equinox](https://github.com/patrick-kidger/equinox), and [matfree](https://github.com/pnkraemer/matfree).

## Installation

```bash
pip install gaussx
```

Or with `uv`:

```bash
uv add gaussx
```

## Quick Start

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

# Structured operators
A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
B = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
K = gaussx.Kronecker(A, B)

# Primitives with structural dispatch
v = jnp.ones(6)
x = gaussx.solve(K, v)       # Per-factor solve (efficient)
ld = gaussx.logdet(K)         # n_B * logdet(A) + n_A * logdet(B)
L = gaussx.cholesky(K)        # Kronecker(chol(A), chol(B))
```

## API Notes

Recent Layer 1 and Layer 3 additions come with a few usage details that are easy to miss:

- `gaussx.kronecker_posterior_predictive(...)` now requires `K_test_diag_factors=` so predictive variances use the exact prior diagonal at the test points instead of reconstructing it from cross-covariances.
- `gaussx.ssm_to_naturals(A, Q, mu_0, P_0)` expects `Q[0]` to equal `P_0`; the function raises a `ValueError` if the initial covariance is inconsistent.
- `gaussx.ImplicitKernelOperator(...)` only reports `lineax` structure such as symmetry or PSD when those tags are passed explicitly.

```python
import jax.numpy as jnp
import lineax as lx

import gaussx

# Kronecker GP posterior moments need exact test-point prior diagonals.
mean, var = gaussx.kronecker_posterior_predictive(
	K_factors=[Kx, Ky],
	y=y,
	noise_var=1e-2,
	grid_shape=(nx, ny),
	K_cross_factors=[Kx_star, Ky_star],
	K_test_diag_factors=[jnp.ones(nx_star), jnp.ones(ny_star)],
)

# Initial covariance must be represented consistently.
theta_1, theta_2 = gaussx.ssm_to_naturals(A, Q, mu_0, P_0=Q[0])

# Structural tags for matrix-free kernels are opt-in.
kernel_op = gaussx.ImplicitKernelOperator(
	kernel_fn,
	X,
	noise_var=1e-3,
	tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
)
```

## Architecture

Four-layer stack:

| Layer | Contents |
|-------|----------|
| 0 — Primitives | `solve`, `logdet`, `cholesky`, `diag`, `trace`, `sqrt`, `inv` |
| 1 — Operators | `Kronecker`, `BlockDiag`, `BlockTriDiag`, `LowRankUpdate`, `ImplicitKernelOperator` |
| 2 — Distributions | `MultivariateNormal` + sugar ops |
| 3 — Recipes | Kalman filter, Kronecker GP recipes, LOVE, CVI, SSM natural params |

## Development

```bash
git clone https://github.com/jejjohnson/gaussx.git
cd gaussx
make install      # install all dependency groups
make test         # run tests
make docs-serve   # preview docs locally
```

## License

MIT
