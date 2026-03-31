---
applyTo: "docs/**/*.py,docs/**/*.md,notebooks/**/*.py"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **jupytext percent-format `.py` files**. They are the single source of truth for all figures, tables, and timing data shown in the documentation.

**Execution model**: `mkdocs-jupyter` renders notebooks with `execute: false`. Authors run notebooks locally via jupytext to produce `.ipynb` files with inline cell outputs, and commit both the `.py` source and the executed `.ipynb`.

## Directory Layout

```
docs/
├── notebooks/
│   ├── demo_foo.py            # jupytext percent-format (source of truth)
│   ├── demo_foo.ipynb         # executed notebook (committed, contains inline outputs)
│   └── benchmark_bar.py
└── guide.md                   # markdown page
```

## Workflow

1. Write or edit the `.py` source file
2. Execute via jupytext: `jupytext --to notebook --execute foo.py -o foo.ipynb`
3. Commit both `.py` and `.ipynb`

Figures render inline from `plt.show()` — the executed `.ipynb` cell outputs are the single source of rendered figures. Do **not** use `savefig` or commit separate PNG files.

## Jupytext Header

Every notebook `.py` file must start with this header:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Cell Markers

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import jax.numpy as jnp
```

## Notebook Structure

Every example notebook should follow this order:

1. **Title & overview** (markdown) — what the notebook demonstrates, prerequisites
2. **Imports** (code)
3. **Problem setup** (markdown + code) — grid, parameters, initial conditions
4. **Core computation** (markdown + code) — the actual demonstration
5. **Figures & tables** (code) — generate and display via `plt.show()`
6. **Summary / takeaways** (markdown)

## Figures

Just use `plt.show()`. The executed `.ipynb` captures the inline output. No `savefig`, no `IMG_DIR`, no `Path(__file__)`.

```python
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x, y, "k-", lw=1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()
```

## Timing & Benchmarks

For performance comparisons, use a JIT-warm-then-measure pattern:

```python
import time
import jax

def time_fn(fn, warmup=2, repeats=5):
    """JIT-compile, warm up, then time *repeats* calls (seconds)."""
    jitted = jax.jit(fn)
    for _ in range(warmup):
        jitted().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        jitted().block_until_ready()
    return (time.perf_counter() - t0) / repeats
```

Key rules:
- **Always warm up** JIT-compiled functions before timing
- **Use `block_until_ready()`** for JAX async dispatch
- **Collect results into a dict/list**, then produce a summary figure or table at the end

## Tables & Statistics

For comparison tables, print them in a code cell:

```python
# %%
for name, stats in results.items():
    print(f"{name:20s}  time={stats['time_ms']:8.2f} ms  residual={stats['residual']:.2e}")
```

The printed output is captured in the `.ipynb` cell output and rendered by mkdocs.

## Adding to the Docs Site

Every new notebook must be added to the `nav` section in `mkdocs.yml`:

```yaml
nav:
  - Examples:
      - My New Notebook: notebooks/my_new_notebook.ipynb
```

Without this entry the notebook won't appear in the site navigation.

## Checklist for New Notebooks

- [ ] Jupytext header present
- [ ] Executed `.ipynb` committed alongside `.py` source
- [ ] Figures use `plt.show()` (no `savefig`)
- [ ] Timing uses warm-up + `block_until_ready()` (if JAX)
- [ ] Notebook added to `mkdocs.yml` nav
