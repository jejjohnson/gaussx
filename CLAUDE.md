# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/capabilities.md` (every public gaussx name,
  plus lineax, matfree and optimistix) before writing a helper, an operator
  or a solver; the "Reuse before you write" table in `AGENTS.md` maps the
  usual hand-rolled code to what already exists.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
