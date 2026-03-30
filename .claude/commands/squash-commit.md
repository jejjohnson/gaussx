Generate a squash commit message for a GitHub PR.

## Instructions

1. If an argument is provided (PR number or URL), fetch the PR details using `gh`. Otherwise, detect the current branch and find its open PR.
2. Fetch all individual commit messages in the PR using `gh api` or `git log`.
3. Combine them into a single squash commit message following these rules:

### Format

```
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits.
Focus on the "why" and overall effect, not per-commit details.>

Co-authored-by: <preserve all unique Co-authored-by lines from the individual commits>
```

### Rules

- Use conventional commit format (`fix`, `feat`, `docs`, `refactor`, `chore`, etc.)
- The type/scope should reflect the dominant change across all commits
- Collapse redundant or incremental commits into a single coherent description
- Preserve ALL unique `Co-authored-by` lines from the individual commits
- If commits span multiple types (e.g. a fix + a chore), use the most significant type and mention the rest in the body
- Keep the summary line under 72 characters
- Do NOT include the individual commit messages as bullet points — this is a squash, not a merge

### Output

Print ONLY the final squash commit message in a code block so the user can copy it directly. Do not add explanation before or after.
