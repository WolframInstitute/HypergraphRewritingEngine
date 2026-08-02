#!/usr/bin/env python3
"""Audit the source map: unreferenced definitions, duplicate names, type coupling.

source_map.py answers "what does this definition reference". The questions an audit
asks are the reverse -- who references THIS, is this name defined more than once,
and which types are coupled to which -- so this builds the reverse index and reports
against it.

WHAT THIS CAN AND CANNOT CONCLUDE. A reference is a reference the COMPILER recorded,
so a name reported with zero referrers is unreferenced BY project code within the
translation units in the compile database. That is not the same as dead: an entry
point called from a test binary outside the database, a symbol used only through the
FFI, a template never instantiated in these TUs, and a virtual override all read as
unreferenced while being load-bearing. The report separates the cases it can rule out
mechanically from the ones a reader has to judge.
"""
import json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP = os.path.join(ROOT, "SOURCE_MAP.md")

DEF_RE = re.compile(r"^- \*\*(.+?)\*\* \*\((.+?)\)\* `:(\d+)`$")
REF_RE = re.compile(r"^  - references: (.+)$")
FILE_RE = re.compile(r"^## `(.+?)` — (\d+) definitions$")

# A definition in one of these is a consumer, not part of the shipped surface: a name
# referenced ONLY from here is not thereby load-bearing.
CONSUMER_PARTS = ("/tests/", "/test", "tools/", "benchmarks/", "benchmarking/",
                  "verification/")


def is_consumer(path):
    return any(p in path for p in CONSUMER_PARTS)


def area(path):
    """The architectural area a file belongs to -- the unit coupling is asked about."""
    for a in ("common/", "hypergraph/", "gpu/", "job_system/", "lockfree_deque/",
              "wxf/", "paclet_source/", "tools/", "testing/", "benchmarks/",
              "benchmarking/", "verification/"):
        if path.startswith(a):
            return a.rstrip("/")
    return path.split("/")[0]


def load():
    defs = []          # (path, line, kind, name, [refs])
    cur_file = None
    pending = None
    for raw in open(MAP):
        line = raw.rstrip("\n")
        m = FILE_RE.match(line)
        if m:
            if pending: defs.append(pending)
            pending = None
            cur_file = m.group(1)
            continue
        m = DEF_RE.match(line)
        if m:
            if pending: defs.append(pending)
            pending = [cur_file, int(m.group(3)), m.group(2), m.group(1), []]
            continue
        m = REF_RE.match(line)
        if m and pending:
            pending[4] = [r.strip("` ") for r in m.group(1).split(", ")]
    if pending: defs.append(pending)
    return defs


def main():
    if not os.path.exists(MAP):
        sys.exit(f"{MAP} not found -- run tools/dev/source_map.py first")
    defs = load()

    by_name = defaultdict(list)                 # name -> [(path, line, kind)]
    referrers = defaultdict(set)                # name -> {referring name}
    ref_sites = defaultdict(set)                # name -> {referring path}
    for path, line, kind, name, refs in defs:
        by_name[name].append((path, line, kind))
        for r in refs:
            referrers[r].add(name)
            ref_sites[r].add(path)

    out = []
    W = out.append
    W("# Source audit\n")
    W(f"From `SOURCE_MAP.md`: **{len(defs)} definitions**, **{len(by_name)} distinct "
      f"qualified names**, across **{len({d[0] for d in defs})} files**.\n")
    W("Reverse index built here; `source_map.py` records only the forward direction.\n")

    # ---- 1. duplicate names ------------------------------------------------------
    # Same qualified name defined at two or more DISTINCT (file, line) sites. Header
    # definitions seen by many TUs are merged by source_map, so a duplicate here is a
    # genuine second definition, not the same one counted twice.
    dupes = {}
    for name, sites in by_name.items():
        uniq = {(p, l) for p, l, _ in sites}
        if len(uniq) > 1:
            dupes[name] = sorted(uniq)
    W(f"\n## 1. Names defined in more than one place — {len(dupes)}\n")
    W("A second definition of one name is either an intentional device/host pair, an "
      "overload set, or a divergence waiting to happen.\n")
    # Two definitions in tools/ or tests/ are two standalone binaries' file-local
    # helpers -- `main`, `run`, `Workload` -- which share a name and nothing else. A
    # duplicate matters when at least one side is SHIPPED code, where both definitions
    # are linked into the same product and can drift apart.
    cross = {n: s for n, s in dupes.items()
             if len({area(p) for p, _ in s}) > 1
             and any(not is_consumer(p) for p, _ in s)}
    W(f"\n### Spanning areas, at least one side shipped — {len(cross)}\n")
    for name in sorted(cross, key=lambda n: (-len(cross[n]), n))[:60]:
        sites = cross[name]
        W(f"- **{name}** — {', '.join(f'`{p}:{l}`' for p, l in sites)}")

    # ---- 2. unreferenced definitions ---------------------------------------------
    # Zero referrers from any project definition. Split by whether the definition is
    # itself in a consumer directory, because unreferenced test code is expected.
    ship, consumer = [], []
    for path, line, kind, name, _ in defs:
        if referrers.get(name):
            continue
        (consumer if is_consumer(path) else ship).append((path, line, kind, name))
    W(f"\n## 2. Definitions no project code references — "
      f"{len(ship)} in shipped code, {len(consumer)} in tests/tools\n")
    W("Unreferenced BY PROJECT CODE in these translation units. Entry points called "
      "from outside the compile database, FFI exports, uninstantiated templates and "
      "virtual overrides all read as unreferenced while being load-bearing -- so this "
      "is a list to READ, not a delete list.\n")
    per_area = defaultdict(list)
    for path, line, kind, name in ship:
        per_area[area(path)].append((path, line, kind, name))
    for a in sorted(per_area, key=lambda a: -len(per_area[a])):
        items = sorted(per_area[a])
        W(f"\n### `{a}` — {len(items)}\n")
        for path, line, kind, name in items[:80]:
            W(f"- **{name}** *({kind})* `{path}:{line}`")
        if len(items) > 80:
            W(f"- … {len(items) - 80} more")

    # ---- 3. referenced ONLY by tests/tools ---------------------------------------
    only_consumer = []
    for path, line, kind, name, _ in defs:
        if is_consumer(path):
            continue
        who = ref_sites.get(name, set())
        if who and all(is_consumer(w) for w in who):
            only_consumer.append((path, line, kind, name, sorted(who)))
    W(f"\n## 3. Shipped definitions referenced ONLY from tests/tools — "
      f"{len(only_consumer)}\n")
    W("Each is either a deliberate test seam or a surface kept alive by its own test.\n")
    for path, line, kind, name, who in sorted(only_consumer)[:80]:
        W(f"- **{name}** *({kind})* `{path}:{line}` — used by {', '.join('`'+w+'`' for w in who[:3])}"
          + (f" +{len(who)-3}" if len(who) > 3 else ""))

    # ---- 4. coupling -------------------------------------------------------------
    # Fan-in: how many distinct definitions reference this name. A high fan-in type is
    # one the architecture is built around; changing it is expensive.
    W("\n## 4. Coupling\n")
    W("\n### Highest fan-in — the types the architecture rests on\n")
    W("| type | referring definitions | referring files | areas |")
    W("|---|---|---|---|")
    for name in sorted(referrers, key=lambda n: -len(referrers[n]))[:40]:
        sites = ref_sites[name]
        areas = sorted({area(s) for s in sites})
        W(f"| `{name}` | {len(referrers[name])} | {len(sites)} | {', '.join(areas)} |")

    # Fan-out: definitions that reference the most distinct project types.
    W("\n### Highest fan-out — the definitions that touch the most types\n")
    W("| definition | types referenced | file |")
    W("|---|---|---|")
    for path, line, kind, name, refs in sorted(defs, key=lambda d: -len(d[4]))[:40]:
        W(f"| `{name}` | {len(refs)} | `{path}:{line}` |")

    # Area-to-area coupling: the CPU/GPU/shared question, counted.
    W("\n### Area to area — who reaches into whom\n")
    W("Only names with ONE defining area are counted. A name defined in several areas "
      "(every tool's file-local `main`, `run`, `Workload`) cannot be attributed to one "
      "of them, and counting an edge to each would report the library reaching into "
      "`tools/` -- an artefact of name collision, not a dependency.\n")
    edge = defaultdict(int)
    for path, line, kind, name, refs in defs:
        a = area(path)
        for r in refs:
            areas = {area(p) for p, _, _ in by_name.get(r, [])}
            if len(areas) != 1:
                continue
            b = next(iter(areas))
            if a != b:
                edge[(a, b)] += 1
    W("| from | to | references |")
    W("|---|---|---|")
    for (a, b), n in sorted(edge.items(), key=lambda kv: -kv[1])[:30]:
        W(f"| `{a}` | `{b}` | {n} |")

    dest = os.path.join(ROOT, "SOURCE_AUDIT.md")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"{len(dupes)} duplicated names ({len(cross)} spanning areas), "
          f"{len(ship)} unreferenced in shipped code, "
          f"{len(only_consumer)} shipped-but-test-only -> {dest}")


if __name__ == "__main__":
    main()
