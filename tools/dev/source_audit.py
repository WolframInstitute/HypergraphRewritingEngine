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

# ---- body similarity ----------------------------------------------------------------
# Names catch a duplicate only when both copies were given the same name. The pair that
# cost the most here -- one canonicalizer as `IRCanonicalizer::compute_canonical_hash`
# and the other as `ir_canonical_hash` -- shares no name at all, and a name-based audit
# is blind to it by construction. So this compares BODIES.
#
# k-gram shingles over a normalised token stream, Jaccard over the shingle sets. Not a
# semantic equivalence check and not claimed as one: it finds bodies that are TEXTUALLY
# near-identical, which is what a copy-paste port produces, and misses a reimplementation
# that shares the algorithm and no phrasing. What it reports is a list to read.
SHINGLE_K = 9          # tokens per shingle
MIN_TOKENS = 60        # below this a body is too small for a match to mean anything
MAX_SHINGLE_DEFS = 24  # a shingle in more than this many bodies is boilerplate
MIN_SHARED = 4         # candidate pairs must share at least this many shingles
MIN_JACCARD = 0.45
MAX_MISALIGNED = 0.02  # above this share of misaligned sites the comparison is not run


def site_names_match(text, line, qualified_name):
    """Does the definition the map records at `line` still start there?

    The map's line is 1-based and points at the definition's first line; a signature that has
    since gained or lost a leading line moves it by one or two, so a small window is read rather
    than a single line. The unqualified name is what is looked for -- the qualified one appears
    in the map, not necessarily in the source.

    A MACRO-GENERATED DEFINITION NEVER SPELLS ITS OWN NAME. `BENCHMARK(foo, ...)` expands to
    definitions called `benchmark_foo` and `reg_foo`, and libclang rightly reports both at the
    macro's line, where neither string occurs. Requiring the literal name marked every such site
    stale -- 41% of the map on this tree, which is dense in BENCHMARK/TEST_F -- and a map scored
    stale suppresses the near-identical body comparison entirely. So a site also counts as aligned
    when the window holds a macro invocation one of whose arguments is a substring of the name:
    that is the token-pasting relationship, and it is checkable rather than assumed.

    Two further spellings differ between map and source and neither is drift. A template's constructor
    is recorded as `Pool<T>` where the source writes `Pool(`, so template arguments are stripped
    before matching. And a macro's argument is matched against the QUALIFIED name, because
    `TEST_F(Suite, Case)` generates `Suite_Case_Test::TestBody` -- the short name is `TestBody`,
    which relates to nothing in the line, while the qualified one carries `Suite`.
    """
    short = re.sub(r"<.*>", "", qualified_name.split("::")[-1]).strip()
    if not short:
        return False
    lines = text.split("\n")
    if line - 1 >= len(lines):
        return False
    lo = max(0, line - 2)
    window = "\n".join(lines[lo:line + 1])
    if short in window:
        return True
    for macro_args in MACRO_CALL_RE.findall(window):
        for arg in macro_args.split(","):
            arg = arg.strip()
            if arg and re.fullmatch(r"[A-Za-z_]\w*", arg) and arg in qualified_name:
                return True
    return False

# An ALL-CAPS identifier applied like a function: the shape of a definition-generating
# macro (BENCHMARK, TEST, TEST_F). The capture is its argument list.
MACRO_CALL_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\s*\(([^)]*)\)")

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|[^\s\w]")
COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def body_tokens(text, start_line):
    """(tokens, end_line) of the definition beginning at `start_line` (1-based), or None
    if it has no body -- a declaration, or one the brace scan cannot close. `end_line` is
    what lets a class and its own inline method be recognised as nested rather than
    duplicated."""
    lines = text.split("\n")
    if start_line < 1 or start_line > len(lines):
        return None
    rest = "\n".join(lines[start_line - 1:])
    rest = STRING_RE.sub('""', COMMENT_RE.sub(" ", rest))
    open_at = rest.find("{")
    semi = rest.find(";")
    if open_at < 0 or (0 <= semi < open_at):
        return None                      # declaration, not a definition
    depth, i = 0, open_at
    while i < len(rest):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                body = rest[open_at:i + 1]
                return TOKEN_RE.findall(body), start_line + body.count("\n")
        i += 1
    return None                          # unbalanced within the file


def shingles(tokens):
    out = set()
    for i in range(len(tokens) - SHINGLE_K + 1):
        out.add(hash(" ".join(tokens[i:i + SHINGLE_K])))
    return out

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

    # Same area, every site shipped: both definitions link into one product, so they can
    # drift apart with nothing to notice. This is the class the prime directive is about,
    # and it is the bulk of the count -- reporting only the cross-area subset above meant
    # printing 1 of 186.
    same = {n: s for n, s in dupes.items()
            if n not in cross and not any(is_consumer(p) for p, _ in s)}
    W(f"\n### Same area, all sites shipped — {len(same)}\n")
    W("An overload set looks identical to a second implementation from here: both are one "
      "name at two sites. What separates them is whether the two bodies DECIDE the same "
      "thing, which is a reading task -- so this is a list to read, not a defect count.\n")
    shown = sorted(same, key=lambda n: (-len(same[n]), n))
    for name in shown[:120]:
        W(f"- **{name}** — {', '.join(f'`{p}:{l}`' for p, l in same[name])}")
    if len(shown) > 120:
        W(f"\n({len(shown) - 120} further same-area duplicates not listed.)")

    # The remainder are consumer-side: two standalone binaries' file-local helpers.
    rest = len(dupes) - len(cross) - len(same)
    W(f"\n{rest} duplicates have at least one site in tests/ or tools/ and are not listed: "
      "two standalone binaries each defining `main` or `run` share a name and nothing else.\n")

    # ---- 1b. same UNQUALIFIED name across areas ------------------------------------
    # Everything above keys on the QUALIFIED name, so `hg_gpu::qc_emit` and the host's
    # `qc_emit` are two names and never collide -- which is exactly the shape a
    # host/device pair takes. Grouping by the last `::` component finds them.
    #
    # The container vocabulary (`size`, `clear`, `view`) matches across every area and
    # says nothing, so a name is only reported when it appears in FEWER areas than the
    # cutoff: a concept implemented on two sides is a pair, a method every container has
    # is not.
    MAX_AREAS = 2
    by_short = defaultdict(list)
    for name, sites in by_name.items():
        short = name.split("::")[-1]
        for p, l, _ in sites:
            by_short[short].append((name, p, l))
    pairs = {}
    for short, sites in by_short.items():
        qualified = {n for n, _, _ in sites}
        areas = {area(p) for _, p, _ in sites}
        if len(qualified) < 2 or len(areas) < 2 or len(areas) > MAX_AREAS:
            continue
        if all(is_consumer(p) for _, p, _ in sites):
            continue
        pairs[short] = sorted({(n, p, l) for n, p, l in sites})
    W(f"\n### Same unqualified name in {MAX_AREAS} areas under different qualified names "
      f"— {len(pairs)}\n")
    W("A concept implemented once per device. Each is either a shared body with two thin "
      "adapters (which is the target shape) or two bodies deciding the same thing (which "
      "is the prime directive's defect) -- the name alone does not say which; section 1c "
      "does.\n")
    for short in sorted(pairs, key=lambda s: (-len(pairs[s]), s))[:60]:
        W(f"- **{short}** — "
          + ", ".join(f"`{n}` @ `{p}:{l}`" for n, p, l in pairs[short]))

    # ---- 1c. near-identical bodies -------------------------------------------------
    # ONE ENTRY PER (path, line). A macro that expands to several definitions reports them
    # all at the macro's line -- TEST(X, Y) gives a class, its constructor, destructor,
    # TestBody and test_info_ at one line -- and reading the body from that line gives all
    # five the same tokens. Keyed by site, they are one body, which is what they are.
    # THE MAP IS A SNAPSHOT AND THE TREE IS NOT. Line numbers move with every edit, and this
    # pass reads the CURRENT file at a RECORDED line -- so against a stale map it tokenises
    # whatever happens to sit there now and compares bodies that are not the ones it names. The
    # numbers it printed then were not about this tree, and its misses were invisible: a
    # 0.59-Jaccard twin pair in parallel_evolution.cpp was absent from the report while the
    # threshold was 0.45, because the recorded lines pointed 40 lines above the definitions.
    #
    # So every site is checked before it is used: the name the map records must appear at the
    # line the map records. A site that fails is MISALIGNED and takes no part; if enough of them
    # fail, the section reports that instead of a similarity list, because a list assembled from
    # a mostly-misaligned map reads as a finding about the code and is a finding about the map.
    file_text = {}
    sig = {}                                # (path, line) -> (name, path, line, shingles, ntok)
    misaligned = 0
    considered = 0
    for path, line, kind, name, _refs in defs:
        if (path, line) in sig:
            continue
        full = os.path.join(ROOT, path)
        if path not in file_text:
            try:
                file_text[path] = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                file_text[path] = None
        text = file_text[path]
        if text is None:
            continue
        considered += 1
        if not site_names_match(text, line, name):
            misaligned += 1
            continue
        got = body_tokens(text, line)
        if not got or len(got[0]) < MIN_TOKENS:
            continue
        toks, end = got
        sh = shingles(toks)
        if sh:
            sig[(path, line)] = (name, path, line, sh, len(toks), end)

    stale_fraction = (misaligned / considered) if considered else 0.0
    map_is_stale = stale_fraction > MAX_MISALIGNED
    if map_is_stale:
        # The report still gets written -- returning here would leave the PREVIOUS report on
        # disk, which reads as current and is the same silent-staleness defect one level up.
        sig = {}
        print("[audit] SOURCE_MAP.md is stale: %d of %d sites misaligned (%.0f%%); the "
              "near-identical comparison is not run" % (misaligned, considered,
                                                        100 * stale_fraction), file=sys.stderr)

    index = defaultdict(list)
    for key, (_n, _p, _l, sh, _t, _e) in sig.items():
        for h in sh:
            index[h].append(key)
    shared = defaultdict(int)
    for h, idxs in index.items():
        if len(idxs) > MAX_SHINGLE_DEFS:
            continue                        # boilerplate, present everywhere
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                shared[(idxs[a], idxs[b])] += 1

    clones = []
    for (a, b), n in shared.items():
        if n < MIN_SHARED:
            continue
        # A class and one of its own inline methods share every token of that method, and
        # are one body, not two. Recognised by containment: same file, one's line inside
        # the other's brace span.
        (_na, pa, la, sa, _ta, ea) = sig[a]
        (_nb, pb, lb, sb, _tb, eb) = sig[b]
        if pa == pb and (la <= lb <= ea or lb <= la <= eb):
            continue
        j = len(sa & sb) / len(sa | sb)
        if j >= MIN_JACCARD:
            clones.append((j, a, b))
    # Both sides shipped is the class the prime directive is about: two bodies linked into
    # one product, free to drift. A repeated test body is a repeated test body.
    shipped = [c for c in clones if not is_consumer(c[1][0]) and not is_consumer(c[2][0])]
    shipped.sort(reverse=True)

    if map_is_stale:
        W("\n### Near-identical bodies — NOT RUN\n")
        W(f"`SOURCE_MAP.md` is stale against the tree: **{misaligned} of {considered}** recorded "
          f"definition sites ({stale_fraction:.0%}) do not carry the name the map records for "
          f"them. This comparison reads the CURRENT file at a RECORDED line, so against a stale "
          f"map it compares bodies that are not the ones it names -- which is how a 0.59-Jaccard "
          f"twin pair in `parallel_evolution.cpp` was absent from this list while the threshold "
          f"was {MIN_JACCARD}. Regenerate with `tools/dev/source_map.py` and re-run. No "
          f"similarity number is printed, because one computed from a stale map is a statement "
          f"about the map.\n")
    else:
        W(f"\n### Near-identical bodies, both sides shipped — {len(shipped)} pairs at "
          f"Jaccard >= {MIN_JACCARD}\n")
        W(f"{len(sig)} definition SITES of at least {MIN_TOKENS} tokens compared by "
          f"{SHINGLE_K}-gram shingles over a comment- and string-stripped token stream. This is "
          "the section that does not care what the two copies are CALLED, which is what makes it "
          "the one that catches a port: `IRCanonicalizer::compute_canonical_hash` and "
          "`ir_canonical_hash` share no name and every name-based check was blind to them.\n")
        W("It finds bodies that are TEXTUALLY near-identical -- what a copy-paste port produces "
          "-- and misses a reimplementation sharing the algorithm and no phrasing. A pair here "
          "is a list entry to READ: overload forwarding and a template's two instantiations look "
          "the same from here as a second implementation does.\n")
        for j, a, b in shipped[:80]:
            na, pa, la, _, ta, _ea = sig[a]
            nb, pb, lb, _, tb, _eb = sig[b]
            cross_area = "" if area(pa) == area(pb) else "  **[cross-area]**"
            W(f"- **{j:.2f}** — `{na}` @ `{pa}:{la}` ({ta} tok) vs `{nb}` @ `{pb}:{lb}` "
              f"({tb} tok){cross_area}")
        if len(shipped) > 80:
            W(f"\n({len(shipped) - 80} further shipped pairs not listed.)")

    if not map_is_stale:
        W(f"\n{len(clones) - len(shipped)} further pairs have at least one side in tests/ or "
      "tools/ and are not listed: a test that repeats its neighbour's shape is a test.\n")

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
    print(f"{len(dupes)} duplicated names ({len(cross)} spanning areas, {len(same)} same-area shipped), "
          f"{len(pairs)} unqualified-name pairs across areas, "
          f"{'near-identical bodies NOT RUN (stale map)' if map_is_stale else str(len(shipped)) + ' near-identical shipped bodies'}, "
          f"{len(ship)} unreferenced in shipped code, "
          f"{len(only_consumer)} shipped-but-test-only -> {dest}")


if __name__ == "__main__":
    main()
