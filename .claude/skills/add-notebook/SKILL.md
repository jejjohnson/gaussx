---
name: add-notebook
description: Add or update an example notebook in gaussx's docs — a jupytext percent-format .py source plus its executed .ipynb, rendered by mystmd — following the docs-examples standards. Use when asked to write a tutorial, example, demo, walkthrough or benchmark notebook, or after changing an API a notebook uses.
---

# Add an example notebook

The full standards are `.github/instructions/docs-examples.instructions.md`;
this is the workflow.

## 1. Plan it

- One question per notebook, answered with gaussx's public API only
  (`import gaussx`; no `gaussx._*` imports). Check the existing notebooks
  in the `toc` of `docs/myst.yml` ("Examples") so you extend one rather than duplicate it.
- Show the structure paying off: the structured call next to the dense
  reference it matches, and the cost it saves (O(·), or a timed comparison).

## 2. Write the source (`docs/notebooks/<name>.py`)

- The jupytext percent header from the standards file, then `# %%` code
  cells and `# %% [markdown]` cells.
- Order: title and overview → imports → problem setup → core computation →
  figures and tables → takeaways.
- Shapes in comments (`# (N, N) float64`), equations in markdown cells
  (KaTeX: a bare `_` inside `\text{}` or `\texttt{}` fails the build), keys explicit (`jr.key(0)`).
- Figures with `plt.show()` only: no `savefig`, no committed PNGs.
- Timing: warm up every jitted function first and call
  `block_until_ready()`.
- The file is linted with the repo (`ruff check .`), so it must be clean.

## 3. Execute and commit both files

```bash
uv run --group docs jupytext --to notebook --execute docs/notebooks/<name>.py -o docs/notebooks/<name>.ipynb
```

Commit the `.py` **and** the executed `.ipynb`.
`tests/test_notebooks_in_sync.py` compares their cell sources, so re-execute
after every edit to the `.py`.

## 4. Publish it

- Add it to the topic it belongs to under "Examples" in the `toc` of `docs/myst.yml`, with a short `title:` for the nav. It is served at `/<file-name>/`, with `_` turned into `-`.
- `make docs` (needs mystmd: `npm install -g mystmd`). The notebooks are pre-executed; mystmd renders the saved outputs and does not run them.
