# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/capabilities.md` (every public gaussx name,
  plus lineax, matfree and optimistix) before writing a helper, an operator
  or a solver; the "Reuse before you write" table in `AGENTS.md` maps the
  usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-operator`, `add-dispatch-path`, `add-solver-strategy`,
    `add-recipe`, `deprecate-or-rename`, `add-notebook`;
  - shipping: `pre-pr-check`, `gaussx-review`, `squash-commit`,
    `triage-ci-failure`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `gaussx-review`; run them on any diff that adds code, before committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/capabilities.md` (gaussx, lineax, matfree, optimistix)?
  - `numerics-reviewer`: traced control flow, dtype promotion, densified
    fast paths, PRNG misuse, broken gradients, silent non-convergence.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
