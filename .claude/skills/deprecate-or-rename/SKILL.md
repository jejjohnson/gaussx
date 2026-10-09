---
name: deprecate-or-rename
description: Rename, deprecate or remove a public gaussx name, keyword argument or option under the deprecation policy (a GaussxDeprecationWarning naming the replacement and the removal version, docs moved to "Deprecated aliases", the deprecate commit type), or carry out the scheduled removals for a minor release. Use when asked to rename, deprecate, remove or clean up an API in gaussx.
---

# Rename, deprecate or remove a public name

gaussx is pre-1.0, but pyrox and other downstream code pin against it, so
public names change on a schedule, never silently. The policy is in
[`docs/api/index.md`](../../../docs/api/index.md#deprecation-policy); the
naming rules a rename usually serves are in its "Naming" table.

## 1. Pick the new name

Check it against the naming rules (US spelling, CamelCase only for classes,
`*Result` / `*State` / `*Cache` / `*Decomposition` / `*Params`,
`solve_<rhs-shape>` vs `<structure>_solve`, KL(first ‖ second)) and against
`docs/capabilities.md` (it must not collide with an existing gaussx, lineax
or optimistix name).

## 2. Keep the old spelling working, with a warning

Every warning goes through `gaussx._deprecation`, so it is a
`GaussxDeprecationWarning` attributed to the caller's line, and names the
replacement **and** the removal version (the first `0.x.0` after the release
that first ships the warning; check `RENAMED_REMOVAL` and the open
deprecations for the current one).

| What changes | How |
|---|---|
| A public function or class name | Rename it, export the new name, remove the old name from the imports and `__all__` in `src/gaussx/__init__.py`, and add `"old": "new"` to `RENAMED` in `_deprecation.py`; `gaussx.__getattr__` returns the *same* object with a warning, so `isinstance` keeps working |
| A class that must stay distinct (e.g. a subclass with its own behaviour), or a wrapper with a different signature | A thin subclass whose `__init__` calls `warn_deprecated(...)` (see `SumKronecker` in `_operators/_sum_kronecker.py`) |
| A keyword argument | `@renamed_kwargs(old="new")` on the function |
| An option value or behaviour | `warn_deprecated("... is deprecated and will be removed in gaussx X.Y.0; use ...")` at the point of use |

Internal callers switch to the new name in the same PR, so the package
itself never warns (gaussx DeprecationWarnings are errors in the test suite,
gh-332).

## 3. Docs and index

- The new name goes into its page's `members:` list.
- A `RENAMED` alias is not exported, so it leaves the docs entirely (listing
  it fails `test_no_documented_symbol_is_stale` and the strict build);
  record the rename as a row in the "Naming" table of `docs/api/index.md`.
- An exported deprecated subclass or wrapper moves to the page's
  "Deprecated aliases" block (create it at the end of the page if missing),
  with a docstring summary starting "Deprecated".
- `make capabilities`.

## 4. Tests

- A test that the old spelling warns (`pytest.warns(GaussxDeprecationWarning)`)
  and returns the same object / result as the new one.
- `tests/test_deprecations.py` reads every `warn_deprecated` call: it fails
  when a message names no removal version, and once `gaussx.__version__`
  reaches a promised version.

## 5. The PR

A PR that only deprecates uses the `deprecate:` type (release-please lists
it under "Deprecations"). A `feat` / `fix` PR that also deprecates adds a
`deprecate(...)` line to its squash commit through a
`BEGIN_COMMIT_OVERRIDE` block in the PR body.

## Removing deprecations on schedule

When `tests/test_deprecations.py` goes red on a release, or when cutting the
minor release that is due: delete each alias, `RENAMED` entry,
`renamed_kwargs` mapping and "Deprecated aliases" docs entry that names that
version, remove their tests, update `docs/capabilities.md`, and land it as a
breaking change (`feat!:` / `refactor!:` with a `BREAKING CHANGE:` footer
listing the removed names and their replacements).
