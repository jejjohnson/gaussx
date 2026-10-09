---
name: pre-pr-check
description: Run gaussx's full pre-PR verification — lint and format on the whole repo, ty, the fast lane and the float32 lane, the slow tier for touched code, doctests, the capability index, the lockfile, the strict docs build and the sdist contents. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo: src, tests, scripts, docs/notebooks
uv run --group lint ruff format --check .
make typecheck                            # ty check src/gaussx
make test-fast                            # fast tier, x64 on (PR CI, without coverage)
make test-no-x64                          # float32 lane (PR CI)
```

While iterating, run the tests you touched (`uv run pytest -q tests/<area>`),
but both lanes must pass before the commit. CI also gates coverage at
`fail_under`: for a large change, `uv run pytest -n auto -m "not slow and not integration" --cov=src/gaussx`
and check the total did not drop.

The fast lane already includes the convention tests (dispatch table, API
coverage, capability index, docstring signatures, doctests, naming,
deprecations, einx ceilings, code size, lazy imports); when one fails, read
its docstring, which says how to fix it, rather than loosening it.

## When you touched slow code

SSM filters, distributions, numpyro paths, long scans and `jit`+`grad`+`vmap`
sweeps have `slow` / `integration` tests:

```bash
uv run pytest -n auto -m "slow or integration" tests/<area>
```

and add the `run-slow` label to the PR so "Extended Tests" runs the heavy
lane on it.

## When the public API changed

```bash
make capabilities                          # regenerate docs/capabilities.md
uv run pytest -q tests/test_capabilities.py tests/test_docs_api_coverage.py tests/test_doctests.py
```

## When dependencies or the version changed

```bash
uv lock && uv lock --check                 # commit uv.lock with the change
```

Runtime dependencies are `>=` floors with no upper bound. A new floor is
exercised weekly by the latest-deps `lowest-direct` job; check it locally
with `uv lock --upgrade --resolution lowest-direct` in a scratch copy if the
change depends on a new upstream feature (and don't commit that lock).

## When docstrings, docs or `mkdocs.yml` changed

```bash
uv run --group docs mkdocs build --strict
```

A notebook `.py` edit needs its `.ipynb` re-executed (`add-notebook` skill).

## When packaging changed

```bash
uv build
tar tzf dist/*.tar.gz | grep -E '\.ipynb$|/\.github/|/\.claude/|/uv\.lock$|/CLAUDE\.md$|/AGENTS\.md$|/\.env\.example$' && echo "sdist ships tooling"
```

The sdist is an allow-list (`[tool.hatch.build.targets.sdist]`); CI fails if
it ships notebooks or agent files or exceeds 1 MB, and if the wheel lacks
`gaussx/py.typed`.

## Before pushing

- `git status` shows no stray files (scratch notebooks, `.plans/`, caches).
- Conventional Commits title with a lowercase subject; `deprecate:` for a
  deprecation-only PR; `!` and a `BREAKING CHANGE:` footer for a breaking
  change.
- Push only to your feature branch, only when asked.
