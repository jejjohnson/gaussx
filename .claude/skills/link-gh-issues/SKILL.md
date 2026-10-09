---
name: link-gh-issues
description: Apply native GitHub issue relationships on gaussx (sub-issues, blocked-by) via the GraphQL API or the GitHub MCP tools. Use when asked to link issues as parent/child, mark one blocked by another, wire up an epic's children, or inspect an issue's relationships.
---

# Link gaussx issues

The prose `## Relationships` block in an issue body is the human-readable
record; this skill applies the matching native link so GitHub's sub-issue
panel and dependency graph work. Keep both.

## With the GitHub MCP tools

`sub_issue_write` adds, removes or reprioritises a sub-issue (parent issue
number + child issue id). Use it when the `gh` CLI is unavailable.

## With `gh`

Resolve issue numbers to node ids:

```bash
id() { gh api repos/jejjohnson/gaussx/issues/"$1" --jq .node_id; }
PARENT=$(id 511); CHILD=$(id 512)
```

Sub-issue (parent ↔ child):

```bash
gh api graphql -f query='
mutation($parent: ID!, $child: ID!) {
  addSubIssue(input: {issueId: $parent, subIssueId: $child}) {
    subIssue { number title }
  }
}' -f parent="$PARENT" -f child="$CHILD"
```

Pass `replaceParent: true` in the input to move a child that already has a
parent.

Blocked-by ("A is blocked by B", which is the same link as "B blocks A":
`issueId` is A, `blockingIssueId` is B):

```bash
gh api graphql -f query='
mutation($this: ID!, $blocker: ID!) {
  addBlockedBy(input: {issueId: $this, blockingIssueId: $blocker}) {
    issue { number }
  }
}' -f this="$(id <A>)" -f blocker="$(id <B>)"
```

Read back:

```bash
gh api graphql -f query='
query($number: Int!) {
  repository(owner: "jejjohnson", name: "gaussx") {
    issue(number: $number) {
      title
      parent { number title }
      subIssues(first: 50) { nodes { number title state } }
      subIssuesSummary { total completed percentCompleted }
      blockedBy(first: 50) { nodes { number title state } }
      blocking(first: 50) { nodes { number title state } }
    }
  }
}' -F number=<N>
```

## From an epic's body

Fetch the epic (`gh issue view <N> --json body --jq .body`), make every
`#NNN` in its "Issues" checklist a sub-issue, apply its "Blocked by" /
"Blocks" lines, and leave "Related" as prose (no native feature).

## Errors

A 422 "already a sub-issue" / "already blocked by" is a no-op; report it.
"Node not found" means a wrong number or repository.
