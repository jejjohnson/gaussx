# pypackage_template

[![Tests](https://github.com/jejjohnson/pypackage_template/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pypackage_template/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pypackage_template/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/pypackage_template/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/pypackage_template/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/pypackage_template/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pypackage_template)
[![PyPI version](https://img.shields.io/pypi/v/mypackage.svg)](https://pypi.org/project/mypackage/)
[![Python versions](https://img.shields.io/pypi/pyversions/mypackage.svg)](https://pypi.org/project/mypackage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson
Repo: [https://github.com/jejjohnson/pypackage_template](https://github.com/jejjohnson/pypackage_template)
Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

An opinionated, modern Python package template with best-practice tooling already wired up. When you use this template, you get linting, formatting, type checking, testing with coverage gates, auto-generated documentation, automated releases, security scanning, and AI agent instructions — all configured and integrated from day one. No boilerplate to write; just rename the package and start coding.

---

## 📂 Repository Layout

```
pypackage_template/
├── src/mypackage/                    # Main package code (src layout)
├── tests/                            # pytest test suite
├── docs/                             # MkDocs documentation source
├── notebooks/                        # Jupyter notebooks
├── .github/
│   ├── workflows/                    # GitHub Actions CI/CD workflows (9 total)
│   ├── instructions/                 # Copilot custom instructions
│   ├── copilot-instructions.md       # Copilot behavioural config
│   ├── dependabot.yml                # Automated dependency PRs
│   └── labeler.yml                   # Automatic PR labelling rules
├── pyproject.toml                    # Single source of truth for project metadata & tools
├── uv.lock                           # Fully reproducible lockfile
├── Makefile                          # Self-documenting task runner
├── mkdocs.yml                        # Documentation site configuration
├── .pre-commit-config.yaml           # Git hook definitions
├── release-please-config.json        # Automated release & changelog config
├── .release-please-manifest.json     # Tracks the current released version
├── .env.example                      # Template for local environment variables
├── AGENTS.md                         # Standing instructions for AI coding agents
├── CODE_REVIEW.md                    # Code review standards
└── CHANGELOG.md                      # Auto-generated changelog
```

---

## 🚀 Quick Start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/pypackage_template.git
cd pypackage_template
make install      # install all dependency groups
make test         # run tests
make docs-serve   # preview docs locally
```

---

## ✨ Features

### 📦 Package Management — `uv` + `uv.lock`

**Files:** `pyproject.toml`, `uv.lock`

[uv](https://github.com/astral-sh/uv) is a Rust-based Python package manager that is 10–100× faster than pip. It handles virtual environments, dependency resolution, and installation in a single tool.

- `uv.lock` provides a fully reproducible environment (like `package-lock.json` but for Python) — every developer and every CI run installs the exact same versions.
- Dependency groups (`dev`, `lint`, `typecheck`, `docs`) let you install only what you need for a given task.

> **What:** A Rust-based package manager and virtual environment tool — runs `uv sync` once and every developer gets a bit-for-bit identical environment from `uv.lock`.

> **Why uv?** Eliminates "works on my machine" problems. Faster CI. A single binary replaces pip, pip-tools, virtualenv, and pyenv.

---

### 🏗️ Project Metadata — `pyproject.toml`

**File:** `pyproject.toml`

`pyproject.toml` is the single source of truth, replacing `setup.py`, `setup.cfg`, and `requirements.txt`. Everything lives in one place:

- **`[project]`** (PEP 621): name, version, description, authors, license, classifiers, `requires-python`
- **`[dependency-groups]`**: `dev`, `lint`, `typecheck`, `docs`
- **`[build-system]`**: `hatchling` as the build backend
- **`[tool.ruff]`**, **`[tool.ty]`**, **`[tool.pytest.ini_options]`**, **`[tool.coverage.*]`**: all tool configs co-located

> **What:** A single `pyproject.toml` replaces `setup.py`, `setup.cfg`, and every scattered `*.cfg` config file — one place for project metadata and every tool's configuration.

> **Why one file?** Standardised (PEP 517/518/621), no scattered config files, tooling reads from one canonical location.

---

### 📁 `src/` Layout

**Directory:** `src/mypackage/`

Source code lives under `src/mypackage/` rather than at the repo root. This forces the package to be properly installed before it can be imported, which means:

- Packaging bugs (missing files, incorrect paths) are caught early rather than hidden by the flat-layout import shortcut.
- Tests always exercise the installed package, not an accidentally-importable source directory.

> **What:** All source code lives under `src/mypackage/` instead of at the repo root, so the package must be installed before it can be imported.

> **Why `src/` layout?** Industry best practice. See writings by Brett Cannon and Hynek Schlawack on why flat layouts silently mask packaging errors.

---

### 🔨 Build Backend — `hatchling`

**File:** `pyproject.toml` (`[build-system]`, `[tool.hatch.build.targets.wheel]`)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
```

Zero-config, PEP 517 compliant. No `MANIFEST.in`, no surprises. Integrates cleanly with uv.

> **What:** `hatchling` is the PEP 517 build backend — run `uv build` and it produces a wheel and sdist with zero extra config.

> **Why hatchling?** Minimal configuration, actively maintained, excellent uv integration, and handles the `src/` layout without extra config.

---

### 🧹 Linting & Formatting — `ruff`

**Files:** `pyproject.toml` (`[tool.ruff]`), `.pre-commit-config.yaml`, `.github/workflows/lint.yml`

[ruff](https://github.com/astral-sh/ruff) replaces flake8, isort, and black in a single Rust binary.

Rule sets enabled:

| Code | What it checks |
|------|---------------|
| `E`, `F` | pycodestyle + pyflakes (core correctness) |
| `I` | Import sorting (replaces isort) |
| `UP` | pyupgrade — modernise syntax automatically |
| `B` | flake8-bugbear — common gotchas |
| `SIM` | flake8-simplify — simplify expressions |
| `RUF` | Ruff-native rules |

Pre-commit hooks run both `ruff` (lint + autofix) and `ruff-format`. CI enforces with `ruff check` and `ruff format --check`.

> **What:** A single Rust binary that lints, sorts imports, and formats Python code — replaces flake8 + isort + black with one tool and one config block.

> **Why ruff?** ~100× faster than the black + flake8 + isort combination; single config block; same results.

---

### 🔬 Type Checking — `ty`

**Files:** `pyproject.toml` (`[tool.ty]`), `.github/workflows/typecheck.yml`

```toml
[tool.ty.environment]
python-version = "3.12"
```

[ty](https://github.com/astral-sh/ty) is Astral's new Rust-based type checker (same team as ruff and uv). It is extremely fast and designed to integrate with the rest of the Astral toolchain.

> **What:** A Rust-based static type checker from the Astral team — run `make typecheck` to catch type errors without executing any code.

> **Why ty?** Catches whole classes of bugs before runtime. Faster than mypy or pyright. Part of the same ecosystem as ruff and uv.

---

### 🧪 Testing — `pytest` + `pytest-cov`

**Files:** `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.*]`), `.github/workflows/ci.yml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/mypackage --cov-report=term-missing --cov-report=xml:coverage.xml"

[tool.coverage.report]
fail_under = 80
```

- Tests live in `tests/`.
- Fast local run (no coverage): `make test` (skips coverage collection for speed)
- Coverage runs (with reports + gate): `uv run pytest` or `make test-cov`, and in CI
- Coverage report (when enabled): terminal summary + `coverage.xml` for Codecov upload
- **Coverage gate:** `fail_under = 80` — coverage-enabled runs and CI fail if coverage drops below 80%
- CI matrix: Python 3.12 and 3.13

> **What:** pytest with an 80% coverage gate wired in — `make test` is fast (no coverage), while CI and `make test-cov` enforce the gate and upload results to Codecov.

> **Why a coverage gate?** Prevents silent regressions. The matrix catches version-specific bugs that single-version CI misses.

---

### 🪝 Pre-commit — `.pre-commit-config.yaml`

**Files:** `.pre-commit-config.yaml`, `.github/workflows/pre-commit-autoupdate.yml`

Hooks that run on every `git commit`:

| Hook | What it does |
|------|-------------|
| `end-of-file-fixer` | Ensures files end with a newline |
| `trailing-whitespace` | Removes trailing whitespace |
| `check-yaml` | Validates YAML syntax |
| `check-added-large-files` | Blocks accidentally committed large files |
| `ruff` | Lint + autofix |
| `ruff-format` | Format |

Run manually: `make precommit`. Hook versions are bumped automatically weekly via `.github/workflows/pre-commit-autoupdate.yml`.

> **What:** Six git hooks that run automatically on every `git commit` to catch formatting, whitespace, and YAML issues before they reach CI.

> **Why pre-commit?** Stops bad commits at the source rather than relying on CI to catch them after a push. Weekly autoupdate keeps hooks on patched versions.

---

### 📖 Documentation — MkDocs + Material + mkdocstrings + mkdocs-jupyter

**Files:** `mkdocs.yml`, `pyproject.toml` (`[dependency-groups] docs`), `.github/workflows/pages.yml`

| Plugin | Role |
|--------|------|
| `mkdocs-material` | Responsive theme with dark/light toggle, tabs, copy buttons |
| `mkdocstrings[python]` | Auto-generates API docs from Google-style docstrings |
| `mkdocs-jupyter` | Renders `.ipynb` notebooks directly in the docs site |
| `jupytext` | Stores notebooks as `.py` files for clean git diffs |

Commands:

```bash
make docs          # build static site
make docs-serve    # preview locally at http://127.0.0.1:8000
make docs-deploy   # deploy to GitHub Pages
```

Auto-deploy: `.github/workflows/pages.yml` deploys on every push to `main`.

Nav structure: **Home → API Reference → Changelog**

> **What:** A versioned docs site auto-generated from Google-style docstrings and Jupyter notebooks, deployed to GitHub Pages on every push to the default branch.

> **Why docs-as-code?** Documentation that lives next to code gets updated with it. Auto-API-docs from docstrings means zero duplication between source and docs.

---

### 🏷️ Conventional Commits

**File:** `.github/workflows/conventional-commits.yml`

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

The `conventional-commits.yml` workflow validates every **PR title** automatically using [`amannn/action-semantic-pull-request`](https://github.com/amannn/action-semantic-pull-request).

> **What:** A CI check that enforces Conventional Commits format on every PR title, ensuring the commit history is machine-readable for automated changelogs and version bumps.

> **Why Conventional Commits?** Enables automated changelogs, drives semantic versioning decisions, and makes `git log` readable at a glance.

---

### 🚀 Automated Releases — Release Please

**Files:** `release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/release-please.yml`

On every push to the default branch, [Release Please](https://github.com/googleapis/release-please) opens or updates a "release PR" that:

1. Bumps the version in `pyproject.toml` based on commit types
2. Updates `CHANGELOG.md` with grouped entries

Merging that PR creates a GitHub Release and a git tag — no manual version bumping or changelog editing required.

Changelog sections are driven by `release-please-config.json`. The current mapping:

| Commit type | CHANGELOG section | Visible |
|-------------|------------------|---------|
| `feat` | Features | ✅ |
| `fix` | Bug Fixes | ✅ |
| `perf` | Performance Improvements | ✅ |
| `revert` | Reverts | ✅ |
| `docs` | Documentation | hidden |
| `style` | Styles | hidden |
| `refactor` | Code Refactoring | hidden |
| `test` | Tests | hidden |
| `build` | Build System | hidden |
| `ci` | Continuous Integration | hidden |
| `chore` | Miscellaneous | hidden |

> **What:** Fully automated releases — every push to the default branch keeps a "release PR" up to date; merging it tags a release, bumps the version, and publishes a changelog.

> **Why Release Please?** Zero-friction releases. The changelog writes itself from your commit messages.

---

### 🔒 Security — CodeQL

**File:** `.github/workflows/codeql.yml`

GitHub's free static analysis tool scans for security vulnerabilities on every push, every PR, and on a weekly schedule. Zero configuration required.

> **What:** GitHub's built-in static analysis — scans the codebase for security vulnerabilities on push, PR, and a weekly schedule.

> **Why CodeQL?** Catches common security issues automatically. Free for public repos. No maintenance overhead.

---

### 🤖 Dependabot — `.github/dependabot.yml`

Dependabot monitors both **GitHub Actions** versions and **Python (pip) dependencies**, and opens automated PRs to update them whenever a new version is released.

> **What:** Automated dependency PRs every week — both GitHub Actions and Python packages are kept up to date without manual intervention.

> **Why Dependabot?** Keeps CI actions and Python dependencies on recent, patched versions without any manual tracking.

---

### 🏷️ Auto PR Labeling — `.github/labeler.yml` + `label-pr.yml`

File path patterns are mapped to labels automatically. For example:

- Changes in `src/` → `source` label
- Changes in `.github/workflows/` → `ci` label
- Changes in dependency files (e.g. `pyproject.toml`, `.pre-commit-config.yaml`) → `dependencies` label

> **What:** Automatic label application on every PR based on which files changed — no manual labelling required.

> **Why auto-labeling?** At-a-glance PR categorization in the GitHub UI with zero manual effort.

---

### 🛠️ Makefile — Self-documenting Task Runner

**File:** `Makefile`

All common tasks are available via `make`:

| Target | Description |
|--------|-------------|
| `make help` | Print all targets with descriptions |
| `make install` | Install all dependency groups via uv |
| `make init` | Bootstrap `.env` from `.env.example` |
| `make test` | Run tests (no coverage) |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Lint with ruff (no autofix) |
| `make format` | Format with ruff (format + autofix) |
| `make typecheck` | Type-check with ty |
| `make precommit` | Run pre-commit hooks on all files |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artefacts and caches |
| `make docs` | Build documentation site |
| `make docs-serve` | Preview documentation locally |
| `make docs-deploy` | Deploy documentation to GitHub Pages |
| `make version` | Display package version and git hash |

The Makefile auto-loads `.env` and exports its variables. The `check-env-VARNAME` guard pattern lets targets declare required env vars as prerequisites.

> **What:** A self-documenting task runner — `make help` prints every available command with its description, so you never need to remember toolchain invocations.

> **Why a Makefile?** Single interface regardless of the underlying toolchain. Onboarding takes seconds: `make help` shows everything.

---

### 🔐 Environment Variables — `.env.example`

**File:** `.env.example`

`.env.example` is committed to git as a template. The actual `.env` is gitignored. Run `make init` to copy the example to `.env`.

Documented variables:

| Variable | Purpose |
|----------|---------|
| `PKGROOT` | Path to package source (default: `src/mypackage`) |
| `PYPI_TOKEN` | PyPI token for publishing |
| `GITHUB_TOKEN` | GitHub personal access token |

> **What:** A committed `.env.example` documents every supported environment variable; the real `.env` is gitignored and auto-loaded by the Makefile.

> **Why `.env.example`?** Local overrides without committing secrets. The Makefile auto-loads and exports vars, so all targets see them automatically.

---

### 🤖 AI Agent Support — `AGENTS.md` + `.github/copilot-instructions.md`

**Files:** `AGENTS.md`, `.github/copilot-instructions.md`, `.github/instructions/`

`AGENTS.md` contains standing instructions for all AI coding agents (Copilot, Claude, Gemini, etc.):

- **Karpathy Coding Principles:** Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution
- **Pre-commit checklist:** tests, lint, format, type checks must pass before every commit
- **Git safety rules:** never push to `main` without explicit instruction
- **Conventional Commits** requirement for all commit messages

`.github/copilot-instructions.md` provides GitHub Copilot-specific behavioural config: project overview, build commands, key directories, and review guidelines.

> **What:** Explicit behavioural contracts for AI coding agents — `AGENTS.md` for all agents, `copilot-instructions.md` for GitHub Copilot — covering coding principles, quality gates, and git safety rules.

> **Why explicit agent instructions?** LLMs working on repos need explicit contracts about style, safety, and quality gates — the same way human contributors need a `CONTRIBUTING.md`.

---

### 📋 Code Review Standards — `CODE_REVIEW.md`

**File:** `CODE_REVIEW.md`

Defines the review checklist (style, idioms, packaging, docs, error handling, testing, performance, security), Python-specific checks, output format, and suggestion priorities. Referenced by both `AGENTS.md` and `copilot-instructions.md` so that humans and AI agents apply the same criteria.

> **What:** A shared code-review rubric that both human reviewers and AI agents reference, ensuring consistent feedback quality across the project.

---

## 🔄 CI/CD Workflows

| Workflow | File | Trigger | What it does |
|----------|------|---------|-------------|
| Tests | `ci.yml` | push / PR to default branch | pytest matrix (3.12, 3.13) + Codecov upload |
| Lint | `lint.yml` | push / PR to default branch | `ruff check` + `ruff format --check` |
| Type Check | `typecheck.yml` | push / PR to default branch | `ty check` |
| Deploy Docs | `pages.yml` | push to default branch | `mkdocs gh-deploy` |
| Release Please | `release-please.yml` | push to default branch | automated release PR + changelog |
| CodeQL | `codeql.yml` | push / PR / schedule | security static analysis |
| Conventional Commits | `conventional-commits.yml` | PR | validates PR title format |
| PR Labeler | `label-pr.yml` | PR | applies path-based labels |
| Pre-commit Autoupdate | `pre-commit-autoupdate.yml` | weekly schedule | bumps hook revisions, opens PR |

---

## 🛠️ Adapting This Template

Follow this checklist when using this repo as a base for a new project:

1. **Search-and-replace** `mypackage` with your package name everywhere (source, config, docs)
2. **Update `[project]` in `pyproject.toml`**: name, description, authors, keywords, classifiers, `requires-python`
3. **Update `mkdocs.yml`**: `site_name`, `site_description`, `repo_url`, `repo_name`
4. **Rename `src/mypackage/`** to `src/<yourpackage>/`
5. **Copy `.env.example` to `.env`** (`make init`) and fill in values
6. **Run `make install`** then **`make test`** to verify the baseline works
7. **Set up Codecov** and add `CODECOV_TOKEN` to GitHub Secrets if you want coverage tracking
8. **Update badge URLs** in this README to point at your repository
9. **Update `[project.urls]`** in `pyproject.toml` to your repository URL
10. Delete or update `CHANGELOG.md` and `.release-please-manifest.json` to start fresh

---

## 📚 Further Reading

| Tool | Documentation |
|------|--------------|
| uv | <https://docs.astral.sh/uv/> |
| ruff | <https://docs.astral.sh/ruff/> |
| ty | <https://github.com/astral-sh/ty> |
| hatchling | <https://hatch.pypa.io/latest/> |
| pytest | <https://docs.pytest.org/> |
| MkDocs Material | <https://squidfunk.github.io/mkdocs-material/> |
| mkdocstrings | <https://mkdocstrings.github.io/> |
| pre-commit | <https://pre-commit.com/> |
| Release Please | <https://github.com/googleapis/release-please> |
| Conventional Commits | <https://www.conventionalcommits.org/> |