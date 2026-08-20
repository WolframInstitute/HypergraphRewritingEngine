#!/usr/bin/env python3
"""What the paclet accepts, against what the documentation describes.

option_names_check.py answers the opposite question -- that every name the docs USE is real, and
appears in the built notebooks. This one asks what is real and NOT described: a property, option
or initial-condition type a caller can pass and find nothing written about.

Both directions are needed and neither implies the other. "GlobalEdges", "StateBitvectors" and
"All" were accepted properties named in no shipped page, and nothing reported it, because every
name the pages DID use was correct.

Each list is read from the place that DECIDES it, not from a copy: the properties from
propertyRequirementsBase, whose Keys computeRequiredData validates against before the engine is
called at all; the options from the Options[...] declarations; the initial-condition types from
the Switch that generates them.

Reads sources only; no build.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WL = ROOT / "paclet/Kernel/HypergraphRewriting.wl"
SRC = ROOT / "paclet/Documentation/Source"

# Stated once in the prose and true of every *Graph property, so the variants are not each
# expected to have their own entry. The convention itself must be present for that to hold.
STRUCTURE_RULE = "may take the suffix `Structure`"


def assoc_block(text: str, start: int) -> str:
    """The balanced <| ... |> association following an assignment."""
    i = text.find("<|", start)
    if i < 0:
        return ""
    depth = 0
    j = i
    while j < len(text) - 1:
        if text[j:j + 2] == "<|":
            depth += 1
            j += 2
            continue
        if text[j:j + 2] == "|>":
            depth -= 1
            j += 2
            if depth == 0:
                return text[i:j]
            continue
        j += 1
    return text[i:]


def brace_block(text: str, start: int) -> str:
    i = text.find("{", start)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return text[i:]


def main():
    wl = WL.read_text()
    prose = "\n".join(f.read_text() for f in sorted(SRC.glob("*.md")))
    described = lambda n: f'"{n}"' in prose

    findings = []

    # PROPERTIES -- the fourth argument. computeRequiredData rejects anything not a key here.
    m = re.search(r"propertyRequirementsBase\s*=", wl)
    if not m:
        print("FAIL: propertyRequirementsBase not found; this reads the wrong place now.")
        return 1
    props = re.findall(r'"([A-Za-z]+)"\s*->', assoc_block(wl, m.end()))
    if len(props) < 15:
        print(f"FAIL: only {len(props)} properties parsed; the declaration changed shape.")
        return 1
    convention = STRUCTURE_RULE in prose
    for p in props:
        if described(p):
            continue
        base = p[: -len("Structure")] if p.endswith("Structure") else None
        if base and convention and described(base):
            continue    # covered by the stated suffix convention
        findings.append(("property", p))

    # OPTIONS -- of the entry points a caller passes options to.
    opts = set()
    for fn in ("HGEvolve", "HGSessionStep", "HGSessionQuery"):
        m = re.search(r"Options\[" + fn + r"\]\s*=", wl)
        if m:
            opts |= set(re.findall(r'"([A-Za-z][A-Za-z0-9]*)"\s*->', brace_block(wl, m.end())))
    if len(opts) < 30:
        print(f"FAIL: only {len(opts)} options parsed; the declarations changed shape.")
        return 1
    findings += [("option", o) for o in sorted(opts) if not described(o)]

    # INITIAL-CONDITION TYPES -- alternative spellings of one type share an entry, so a type is
    # described when ANY of its spellings is.
    # Anchored on the Switch that dispatches on icType, not on the first "Grid" in the file --
    # that one is in an option declaration far above it.
    n = -1
    for m in re.finditer(r"Switch\[", wl):
        if "icType" in wl[m.end():m.end() + 80]:
            n = m.end()
            break
    if n < 0:
        print("FAIL: the initial-condition Switch was not found.")
        return 1
    groups = []
    for m in re.finditer(r'^\s+("(?:[A-Za-z]+)"(?:\s*\|\s*"[A-Za-z]+")*)\s*,\s*$',
                         wl[n:n + 12000], re.M):
        names = re.findall(r'"([A-Za-z]+)"', m.group(1))
        if names:
            groups.append(names)
    if len(groups) < 5:
        print(f"FAIL: only {len(groups)} initial-condition types parsed.")
        return 1
    findings += [("initial-condition type", " | ".join(g))
                 for g in groups if not any(described(x) for x in g)]

    if findings:
        print(f"{len(findings)} name(s) the paclet accepts and no shipped page mentions:\n")
        for kind, name in findings:
            print(f'  {kind:<24} "{name}"')
        print(f"\nDescribe them in {SRC.relative_to(ROOT)}/, then rebuild with ./build_docs.sh.")
        return 1

    print(f"OK: {len(props)} properties, {len(opts)} options and {len(groups)} "
          f"initial-condition types are all described in the shipped documentation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
