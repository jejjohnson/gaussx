---
name: create-gh-issue
description: Open GitHub issues on gaussx from the templates in .github/ISSUE_TEMPLATE/ (feature, bug, design, research, epic), with the repo's labels and relationships. Use when asked to file / open an issue, turn a finding or a review comment into an issue, or publish drafted backlog issues.
---

# Open a gaussx issue

## 1. Pick the template

| Template | Use when |
|---|---|
| `feature.md` | One deliverable: a primitive, operator, strategy, recipe, notebook, docs page |
| `bug.md` | A wrong result, a crash, a failure under `jit` / `grad` / `vmap`, a dtype or performance regression |
| `design.md` | An open API or architecture question to decide |
| `research.md` | Mapping an external library or paper onto gaussx → follow-up issues |
| `epic.md` | Grouping issues that ship together |

Default for "add X" is `feature.md`; ask when unsure.

## 2. Check for duplicates

Search open and closed issues first (`gh issue list --search "<terms>" --state all`
or the GitHub MCP `search_issues`), and `docs/capabilities.md` for a feature
request: the thing may already exist.

## 3. Draft the body

Read the template, strip its YAML front matter and the guidance comments,
and fill every section that applies:

- **Title**: `<area>(<scope>): <description>` with a lowercase subject, as
  the repo's issues do (`primitives(solve): …`, `ssm(kalman): …`,
  `distributions(conditional): …`, `tests(config): …`); `[Design] …` and
  `[EPIC] …` for those templates.
- **Code first**: lead "Proposed API" / "Reproduction" with the exact code.
- **Math as spec**: algorithmic issues keep "Mathematical Notes" with the
  defining equations, conventions, the structured cost and the invariants to
  test (GitHub `$…$` math or Unicode: σ², Λ⁻¹, ⊗, O(n³)).
- **Reuse**: name the gaussx / lineax / matfree pieces it composes.
- **Concrete steps**: each implementation step names a file and a symbol.

## 4. Labels

List the repo's labels first (`gh label list`); apply only ones that exist.
The repo uses one `type:*` (`type:feature`, `type:chore`, `type:research`,
`type:epic-theme`, or the `bug` label), one or more `area:*` (e.g.
`area:operators`, `area:sugar`, `area:maintenance`, `area:research`), a
`wave:*` and a `priority:*` (`p1` high, `p2` normal), plus `epic` on
epics. `ci-failure` is reserved for the scheduled-workflow reporter, and
`run-slow` is a PR label.

## 5. Create it

Write the body to a temp file (never pass code fences through `--body`):

```bash
gh issue create --title "<title>" --body-file /tmp/issue.md \
  --label "type:feature,area:operators,priority:p2"
```

Without the `gh` CLI, use the GitHub MCP `issue_write` tool with the same
title, body and labels.

## 6. Relationships and report

Apply parent / blocked-by links with the `link-gh-issues` skill (keep the
prose `## Relationships` lines too). Report the issue URL with a one-line
summary; for several issues, a table of draft → number.
