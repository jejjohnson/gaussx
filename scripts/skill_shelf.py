"""Generate, or check, the generated parts of the plugin's skills.

The plugin (``plugins/gaussx/``) ships gaussx's guidance as a shelf:
``tree/shelf.toml`` lists three listed skills, seven units and one entry per
job, each entry with the public names it covers. An entry is a file in its
skill's ``shelf/`` folder; a unit has a contract card in ``tree/cards/``.

This script writes the two parts of each ``SKILL.md`` that follow from the
shelf file, so nobody edits them by hand:

- the table of the skill's entries, between ``shelf:begin`` and ``shelf:end``;
- the cards of the units of gaussx below the skill, between
  ``shelf:below:begin`` and ``shelf:below:end``.

``--check`` fails when either is stale, and on the shelf's gates:

- GX-1: every name in ``gaussx.__all__`` belongs to exactly one entry, and
  the entry's file names it;
- GX-2: every ``gaussx.X`` a unit card writes is a public name;
- GX-3: the authored part of a ``SKILL.md`` has at most 150 lines, an entry
  at most 120, a card at most 15.

The "Builds on" block at the end of each ``SKILL.md`` (between
``skill-tree:begin`` and ``skill-tree:end``) is not written here: it follows
from the dependency graph of the whole stack, and geostack's
``scripts/skill_tree.py`` generates it.

Usage::

    make skill-shelf                                    # rewrite the parts
    uv run python scripts/skill_shelf.py --check        # fail if stale

``tests/test_plugin_skill.py`` runs the check in the fast lane, and every
``python`` block of the plugin in the slow lane. The public names are read
from the source of ``src/gaussx/__init__.py``, so the script imports nothing.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "gaussx"
SHELF = PLUGIN / "tree" / "shelf.toml"
INIT = ROOT / "src" / "gaussx" / "__init__.py"

TABLE = ("<!-- shelf:begin -->", "<!-- shelf:end -->")
BELOW = ("<!-- shelf:below:begin -->", "<!-- shelf:below:end -->")
TREE_MARK = "<!-- skill-tree:begin"

SKILL_MAX_LINES = 150
ENTRY_MAX_LINES = 120
CARD_MAX_LINES = 15

ROLE_MARKS = {
    "structure": "⓪ structure",
    "primitive": "① primitive",
    "strategy": "② strategy",
    "setup": "setup",
    "model": "③ model",
    "algorithm": "④ algorithm",
    "step": "⑤ step",
    "recipe": "⑥ recipe",
}


@dataclass(frozen=True)
class Unit:
    """A group of entries the same dependents use; it has one card."""

    key: str
    title: str
    rungs: str
    skill: str


@dataclass(frozen=True)
class Entry:
    """One entry: a file in a listed skill's ``shelf/``, and the names it covers."""

    file: str
    title: str
    role: str
    unit: str
    skill: str
    covers: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class Shelf:
    """The shelf file: listed skills (with the units below each), units, entries."""

    skills: dict[str, tuple[str, ...]]
    units: dict[str, Unit]
    entries: tuple[Entry, ...]


def load(path: Path = SHELF) -> Shelf:
    """Read the shelf file."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Shelf(
        skills={
            k: tuple(v.get("builds_on_units", ())) for k, v in raw["skills"].items()
        },
        units={
            k: Unit(k, u["title"], u["rungs"], u["skill"])
            for k, u in raw["units"].items()
        },
        entries=tuple(
            Entry(
                e["file"],
                e["title"],
                e["role"],
                e["unit"],
                e["skill"],
                e["covers"],
                tuple(e["names"]),
            )
            for e in raw["entries"]
        ),
    )


def public_names(init: Path = INIT) -> set[str]:
    """Every name ``__init__.py`` puts in ``__all__``, extras included.

    The names that need the numpyro extra are appended to ``__all__`` inside a
    ``try``; reading the source finds them whether or not numpyro is installed.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(init.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            names |= {
                c.value
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            }
    return names


def skill_file(skill: str) -> Path:
    return PLUGIN / "skills" / skill / "SKILL.md"


def entry_file(entry: Entry) -> Path:
    return PLUGIN / "skills" / entry.skill / "shelf" / f"{entry.file}.md"


def card_file(unit: str) -> Path:
    return PLUGIN / "tree" / "cards" / f"{unit}.md"


def table(shelf: Shelf, skill: str) -> str:
    """The table of a listed skill's entries."""
    rows = ["| Role | Entry | Covers | Names |", "|---|---|---|---|"]
    for e in shelf.entries:
        if e.skill == skill:
            role = ROLE_MARKS.get(e.role, e.role)
            link = f"[{e.title}](shelf/{e.file}.md)"
            rows.append(f"| {role} | {link} | {e.covers} | {len(e.names)} |")
    return "\n".join(rows)


def below(shelf: Shelf, skill: str) -> str:
    """The cards of the units of gaussx that a listed skill builds on."""
    parts = [
        "## The units below this skill",
        "",
        "A card is enough to call a unit.",
        "Load its skill only when the card's last line applies.",
    ]
    for key in shelf.skills[skill]:
        unit = shelf.units[key]
        parts += [
            "",
            f"### {unit.title} · {unit.rungs} · skill `gaussx:{unit.skill}`",
            "",
            card_file(key).read_text(encoding="utf-8").strip(),
        ]
    return "\n".join(parts)


def _replace(text: str, markers: tuple[str, str], body: str, where: str) -> str:
    begin, end = markers
    if begin not in text or end not in text.split(begin, 1)[1]:
        raise ValueError(f"{where}: needs {begin} and {end}")
    head, rest = text.split(begin, 1)
    return f"{head}{begin}\n{body}\n{end}{rest.split(end, 1)[1]}"


def render(shelf: Shelf, skill: str, text: str) -> str:
    """``text`` (a listed ``SKILL.md``) with its generated parts rewritten."""
    text = _replace(text, TABLE, table(shelf, skill), skill)
    if shelf.skills[skill]:
        text = _replace(text, BELOW, below(shelf, skill), skill)
    return text


def authored_lines(text: str) -> int:
    """Lines of a ``SKILL.md`` an author wrote: all but the generated parts."""
    text = text.split(TREE_MARK, 1)[0]
    for begin, end in (TABLE, BELOW):
        if begin in text and end in text:
            text = text.split(begin, 1)[0] + text.split(end, 1)[1]
    return len(text.rstrip("\n").splitlines())


def violations(shelf: Shelf, names: set[str]) -> list[str]:
    """Every stale part and every gate that fails; empty when all hold."""
    out: list[str] = []
    for skill in shelf.skills:
        text = skill_file(skill).read_text(encoding="utf-8")
        if render(shelf, skill, text) != text:
            out.append(f"stale: {skill}/SKILL.md (run make skill-shelf)")
        if authored_lines(text) > SKILL_MAX_LINES:
            out.append(f"GX-3 {skill}: {authored_lines(text)} authored lines")
    seen: dict[str, str] = {}
    for e in shelf.entries:
        unit = shelf.units.get(e.unit)
        if unit is None or e.skill not in shelf.skills or unit.skill != e.skill:
            out.append(f"GX-1 {e.file}: its unit and its skill disagree")
        if e.role not in ROLE_MARKS:
            out.append(f"GX-1 {e.file}: unknown role {e.role!r}")
        for name in e.names:
            if name in seen:
                out.append(f"GX-1 {name}: in {seen[name]} and in {e.file}")
            seen[name] = e.file
        path = entry_file(e)
        if not path.is_file():
            out.append(f"GX-1 {e.file}: no entry file")
            continue
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) > ENTRY_MAX_LINES:
            out.append(f"GX-3 {e.file}: over {ENTRY_MAX_LINES} lines")
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        out += [
            f"GX-1 {e.file}: does not name {n}" for n in sorted(set(e.names) - words)
        ]
    out += [f"GX-1 {n}: public, in no entry" for n in sorted(names - set(seen))]
    out += [f"GX-1 {n}: in {seen[n]}, not public" for n in sorted(set(seen) - names)]
    for key in shelf.units:
        path = card_file(key)
        if not path.is_file():
            out.append(f"GX-2 {key}: no card")
            continue
        card = path.read_text(encoding="utf-8").strip()
        if len(card.splitlines()) > CARD_MAX_LINES:
            out.append(f"GX-3 {key}: card over {CARD_MAX_LINES} lines")
        cited = set(re.findall(r"`gaussx\.([A-Za-z_][A-Za-z0-9_]*)", card))
        out += [f"GX-2 {key}: no public name {n}" for n in sorted(cited - names)]
    return out


def examples(root: Path = PLUGIN) -> list[tuple[Path, str]]:
    """Every fenced ``python`` block of the plugin's Markdown, with its file."""
    out = []
    for path in sorted(root.rglob("*.md")):
        for block in re.findall(r"```python\n(.*?)```", path.read_text("utf-8"), re.S):
            out.append((path, block))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    args = parser.parse_args(argv)
    shelf = load()
    if not args.check:
        for skill in shelf.skills:
            path = skill_file(skill)
            text = path.read_text(encoding="utf-8")
            new = render(shelf, skill, text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                print(f"wrote {path.relative_to(ROOT)}")
    found = violations(shelf, public_names())
    for message in found:
        print(message)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
