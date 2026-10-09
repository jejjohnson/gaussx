"""Generate, or check, the capability index: every public name in gaussx.

``docs/capabilities.md`` lists every name in ``gaussx.__all__`` once, grouped
exactly as the API reference groups it (page, then section, read from the
``members:`` blocks in ``docs/api/*.md``), with the first line of its
docstring. It then lists the public API of the libraries gaussx builds on
(lineax, matfree, optimistix) and the private toolkits contributors share.
Agents and people search it before writing a helper ("Reuse before you
write" in ``AGENTS.md``).

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check in the fast lane. The upstream
sections depend on the installed versions, which the index records; when they
differ from the installed ones (the weekly latest-deps run), only the gaussx
part is compared.

``--check`` also fails when a gaussx name shadows a lineax or optimistix name
with a different object, unless ``ALLOWED_SHADOWS`` gives the reason.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import inspect
import re
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "capabilities.md"
DOCS_API = ROOT / "docs" / "api"
SRC = ROOT / "src"

# Upstream libraries whose public API is listed, and whose names gaussx must
# not shadow with a different object. matfree has no top-level API: its
# public functions live in submodules.
UPSTREAM = {
    "lineax": ["lineax"],
    "matfree": [
        "matfree.decomp",
        "matfree.eig",
        "matfree.funm",
        "matfree.low_rank",
        "matfree.lstsq",
        "matfree.stochtrace",
        "matfree.bounds",
    ],
    "optimistix": ["optimistix"],
}
UPSTREAM_BLURB = {
    "lineax": (
        "The operator protocol, tags and solvers every gaussx operator extends."
        " Use these before wrapping or re-implementing an operator or solver."
    ),
    "matfree": (
        "Matrix-free Krylov decompositions, matrix functions and stochastic"
        " trace / diagonal estimators. Never hand-roll Lanczos, Arnoldi or"
        " Hutchinson."
    ),
    "optimistix": (
        "Root finding, fixed points, minimisation and least squares with"
        " implicit-differentiation adjoints. Use it instead of a hand-written"
        " ``while_loop`` when gradients must flow through the iteration."
    ),
}

# gaussx names that deliberately reuse an upstream name for a different
# object, and why.
ALLOWED_SHADOWS: dict[str, str] = {
    "linear_solve": (
        "gaussx's front door (gh-376): takes an operator or a (matvec, shape)"
        " pair, a gaussx strategy and a preconditioner; lineax.linear_solve"
        " stays the low-level lineax call"
    ),
}

# Private modules whose helpers the rest of the package builds on. Listed by
# path and read with ``ast`` (nothing imported).
TOOLKITS = [
    "gaussx/_einx.py",
    "gaussx/_testing.py",
    "gaussx/_deprecation.py",
    "gaussx/_operators/_utils.py",
    "gaussx/_strategies/_dispatch.py",
    "gaussx/_strategies/_tolerances.py",
    "gaussx/_strategies/_lineax.py",
    "gaussx/_distributions/_utils.py",
    "gaussx/_ssm/_utils.py",
]

_INLINE_MEMBERS = re.compile(r"members:\s*\[([^\]]*)\]")
_BLOCK_MEMBERS = re.compile(r"members:\s*\n((?:[ \t]*-[ \t]*\w+[ \t]*\n)+)")
_BLOCK_ITEM = re.compile(r"-[ \t]*(\w+)")
_PAGE_LINK = re.compile(r"\]\((\w+\.md)\)")
_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s+(?=[A-Z])")


def _pages() -> list[Path]:
    """Reference pages in the order docs/api/index.md links them."""
    index = (DOCS_API / "index.md").read_text(encoding="utf-8")
    order = list(dict.fromkeys(_PAGE_LINK.findall(index)))
    pages = [DOCS_API / name for name in order if (DOCS_API / name).is_file()]
    rest = sorted(p for p in DOCS_API.glob("*.md") if p.name != "index.md")
    return pages + [p for p in rest if p not in pages]


def _sections(page: Path) -> tuple[str, list[tuple[str, list[str]]]]:
    """The page title and its ``## section -> members`` in document order."""
    title = page.stem
    sections: list[tuple[str, list[str]]] = []
    current = ""
    lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
    text_by_section: list[tuple[str, str]] = []
    buffer: list[str] = []
    for line in lines:
        if line.startswith("# ") and title == page.stem:
            title = line[2:].strip()
        elif line.startswith("## "):
            text_by_section.append((current, "".join(buffer)))
            current, buffer = line[3:].strip(), []
            continue
        buffer.append(line)
    text_by_section.append((current, "".join(buffer)))
    for heading, text in text_by_section:
        names: list[str] = []
        for group in _INLINE_MEMBERS.findall(text):
            names += [n.strip() for n in group.split(",") if n.strip()]
        for group in _BLOCK_MEMBERS.findall(text):
            names += _BLOCK_ITEM.findall(group)
        if names:
            sections.append((heading, names))
    return title, sections


def _attribute_docs(module_name: str) -> dict[str, str]:
    """``name = value`` followed by a string literal: attribute docstrings."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return {}
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    docs = {}
    for node, nxt in zip(tree.body, tree.body[1:], strict=False):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[node.targets[0].id] = nxt.value.value
    return docs


def _first_line(text: str) -> str:
    """The docstring's summary: its first paragraph, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    # mkdocs-autorefs links ("[text][target]") resolve only inside the docs
    # of the package that wrote them; keep the text.
    line = _AUTOREF.sub(r"\1", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return line.replace("|", "\\|")


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str) or (not callable(obj) and not inspect.isclass(obj)):
        module = type(obj).__module__
    return module


def _summary(name: str, obj: object, *, home: str) -> str:
    module = _origin(obj)
    if module.split(".")[0] != home:
        return f"Re-exported from `{module.split('.')[0]}`."
    if callable(obj):
        return _first_line(inspect.getdoc(obj) or "")
    doc = _attribute_docs(module).get(name, "")
    if not doc and name.endswith("_tag"):
        return "Structural / property tag for operators."
    return _first_line(doc)


def _gaussx_part() -> tuple[list[str], int]:
    import gaussx

    out: list[str] = []
    seen: set[str] = set()
    for page in _pages():
        title, sections = _sections(page)
        rel = page.relative_to(ROOT / "docs").as_posix()
        out += [f"## [{title}]({rel})", ""]
        for heading, names in sections:
            if heading:
                out += [f"### {heading}", ""]
            out += ["| Name | Kind | What it does |", "|---|---|---|"]
            for name in names:
                if name in seen or name not in gaussx.__all__:
                    continue
                seen.add(name)
                obj = getattr(gaussx, name)
                summary = _summary(name, obj, home="gaussx")
                out.append(f"| `{name}` | {_kind(obj)} | {summary} |")
            out.append("")
    missing = sorted(set(gaussx.__all__) - seen)
    if missing:
        out += [
            "## Not yet in the API reference",
            "",
            "| Name | Kind | What it does |",
            "|---|---|---|",
        ]
        for name in missing:
            obj = getattr(gaussx, name)
            out.append(
                f"| `{name}` | {_kind(obj)} | {_summary(name, obj, home='gaussx')} |"
            )
        out.append("")
    return out, len(seen) + len(missing)


def _public_names(module: types.ModuleType, *, own: bool) -> list[str]:
    names = getattr(module, "__all__", None) or [
        n for n in dir(module) if not n.startswith("_")
    ]
    keep = []
    for name in sorted(names):
        obj = getattr(module, name)
        if isinstance(obj, types.ModuleType):
            continue
        if own and not _origin(obj).startswith(module.__name__):
            continue
        keep.append(name)
    return keep


def _upstream_part() -> list[str]:
    out: list[str] = []
    for package, modules in UPSTREAM.items():
        out += [f"## Upstream: `{package}`", "", UPSTREAM_BLURB[package], ""]
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                # Absent at this version (the floors run); only the gaussx
                # part is compared there.
                continue
            if len(modules) > 1:
                out += [f"### `{module_name}`", ""]
            out += ["| Name | Kind | What it does |", "|---|---|---|"]
            for name in _public_names(module, own=len(modules) > 1):
                obj = getattr(module, name)
                summary = _summary(name, obj, home=package)
                out.append(f"| `{name}` | {_kind(obj)} | {summary} |")
            out.append("")
    return out


def _toolkit_part() -> list[str]:
    out = [
        "## Shared private toolkits",
        "",
        "Not public API: the plumbing gaussx's own modules share. Contributors",
        "use these instead of writing new helpers; a helper two modules need is",
        "added here, not inline in one caller.",
        "",
        "| Module | Purpose | Helpers |",
        "|---|---|---|",
    ]
    for rel in TOOLKITS:
        path = SRC / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        purpose = _first_line(ast.get_docstring(tree) or "")
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
            and not node.name.startswith("__")
        ]
        module = ".".join(Path(rel).with_suffix("").parts)
        helpers = ", ".join(f"`{n}`" for n in names)
        out.append(f"| `{module}` | {purpose} | {helpers} |")
    out.append("")
    return out


def _versions() -> str:
    return ", ".join(f"{name} {importlib.metadata.version(name)}" for name in UPSTREAM)


def shadows() -> list[str]:
    """gaussx names bound to a different object than lineax / optimistix's."""
    import gaussx

    found = []
    for package in ("lineax", "optimistix"):
        upstream = importlib.import_module(package)
        for name in gaussx.__all__:
            if name in ALLOWED_SHADOWS or not hasattr(upstream, name):
                continue
            if getattr(gaussx, name) is not getattr(upstream, name):
                found.append(f"gaussx.{name} shadows {package}.{name}")
    return found


UPSTREAM_MARKER = "<!-- upstream -->"


def render() -> str:
    gaussx_part, total = _gaussx_part()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in gaussx ({total} of them), grouped as the API",
        "reference groups it, with the first line of its docstring; then the",
        "public API of the libraries gaussx builds on, and the private toolkits",
        "gaussx's own modules share. Search this page before writing a helper:",
        "if what you need is here, compose it; if it is almost here, extend it",
        "where it lives. Regenerate with `make capabilities` after changing a",
        "public API (`tests/test_capabilities.py` checks it is current).",
        "",
        *gaussx_part,
        UPSTREAM_MARKER,
        "",
        f"Listed at {_versions()}.",
        "",
        *_upstream_part(),
        *_toolkit_part(),
    ]
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether ``current`` differs from ``text`` where it is comparable.

    The upstream sections are compared only when the recorded versions match
    the installed ones; the gaussx part and the toolkits always are.
    """
    if current == text:
        return False
    if f"Listed at {_versions()}." in current:
        return True
    head, _, _ = current.partition(UPSTREAM_MARKER)
    new_head, _, _ = text.partition(UPSTREAM_MARKER)
    toolkits = "## Shared private toolkits"
    return (
        head != new_head
        or current.partition(toolkits)[2] != text.partition(toolkits)[2]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    found = shadows()
    if found:
        print("gaussx names that shadow an upstream name with another object:")
        print("\n".join(f"  {s}" for s in found))
        print("Rename the gaussx one, or add it to ALLOWED_SHADOWS with a reason.")
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
