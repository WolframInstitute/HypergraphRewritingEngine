#!/usr/bin/env python3
"""Every option named in the shipped documentation is an option the paclet accepts.

Option names are strings on both sides of the boundary, so nothing checks them: a renamed
option leaves the old name in the prose, and the prose still reads as correct. That has already
happened here -- the sampling design was written against a `MatchRate` that ships as
`TransitionRate`, and against two options that do not exist at all.

The authoritative surface is the union of the WL `Options[...]` declarations (what the paclet
accepts) and the FFI option keys (what the engine reads off the wire). A name used as an option
in the documentation must appear in one of them.

Reads sources only; no build.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Documentation that ships. Untracked development notes are excluded deliberately: they record
# designs that were considered, so they name things that do not exist by construction.
DOCS = [
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CODEMAP.md",
    "docs/DATAFLOW.md",
    "docs/QUICKSTART.md",
]
DOC_GLOBS = ["paclet/Documentation/Source/*.md"]

# `"Name" -> value` is the option syntax. A rule is `{{...}} -> {{...}}` and a property is a
# positional argument, so neither matches.
USE_RE = re.compile(r'"([A-Za-z][A-Za-z0-9]*)"\s*->')


def accepted_names():
    """Option names the paclet accepts, from both sides of the boundary."""
    names = set()

    # WL: every string key inside an Options[...] = {...} declaration, plus the shared
    # initial-condition option list those declarations Join onto.
    wl = (ROOT / "paclet/Kernel/HypergraphRewriting.wl").read_text()
    for m in re.finditer(r"Options\[[A-Za-z$]+\]\s*=|(\$ICCommonOptions)\s*=", wl):
        # Take the balanced brace block that follows the assignment.
        i = wl.find("{", m.end())
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(wl)):
            if wl[j] == "{":
                depth += 1
            elif wl[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        for n in USE_RE.finditer(wl[i:j + 1]):
            names.add(n.group(1))

    # FFI: the keys the engine reads off the wire.
    ffi = (ROOT / "paclet_source/hypergraph_ffi.cpp").read_text()
    names.update(re.findall(r'option_key == "([A-Za-z][A-Za-z0-9]*)"', ffi))
    return names


def main():
    accepted = accepted_names()
    if len(accepted) < 20:
        print(f"FAIL: only {len(accepted)} option names parsed from the paclet sources; "
              f"the declarations this reads must have changed shape.")
        return 1

    findings = []
    files = [ROOT / d for d in DOCS]
    for g in DOC_GLOBS:
        files.extend(sorted(ROOT.glob(g)))

    for f in files:
        if not f.exists():
            continue
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            for m in USE_RE.finditer(line):
                name = m.group(1)
                if name not in accepted:
                    findings.append((f.relative_to(ROOT), lineno, name))

    if findings:
        print(f"{len(findings)} documented option name(s) the paclet does not accept:\n")
        for path, lineno, name in findings:
            print(f"  {path}:{lineno}: \"{name}\"")
        print(f"\nAccepted: {', '.join(sorted(accepted))}")
        return 1

    print(f"OK: every option named in {len(files)} shipped documents is one of the "
          f"{len(accepted)} the paclet accepts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
