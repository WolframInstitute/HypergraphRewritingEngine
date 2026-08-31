#!/usr/bin/env python3
"""Emit the host instruction profile table, and the per-unit costs behind it, from callgrind.

WHAT IT READS. `callgrind_annotate` output, which the attrib phase of tools/dev/remote_session.sh
produces as cg_wpp.txt from a single-threaded depth-6 run under callgrind. Instruction counts
are deterministic, so this needs no quiet machine and no repeats -- which is the whole reason to
attribute with callgrind rather than with wall time.

THE FLAT PROFILE ONLY. callgrind_annotate prints a flat per-function block and then per-file
source annotations for the same functions. Summing both double-counts: a first attempt at this
script attributed 886% of the program total. The flat block ends at the first separator line
after its header, and that is where parsing stops.

BUCKETS ARE MATCHED BY FUNCTION FIRST, FILE SECOND. A function's body is attributed to the
algorithm it belongs to even where the compiler placed part of it in a header it inlined from:
`concurrent_map.hpp:qc_add_producer` is the causal DP, not the map, so the reconstruction keys
come before the container keys. Anything unmatched is reported as Unattributed rather than
silently dropped, so the buckets and the residue always sum to the covered fraction, and a
large residue is a defect in this list, not a property of the engine.

PER-UNIT COSTS (--counts). The bench's own stats lines give the counts each phase's cost divides
by: canonicalization calls, reconstructed events, claim-set inserts, applied matches. Those
counts are engine outputs, deterministic for a workload, so the file may be produced on any
machine with a stats build; the provenance names it. The macros this writes are what the paper's
bounds paragraph transcludes.

CALLER ATTRIBUTION (--raw). libc's copies, fills and heap calls carry no algorithm of their own;
the raw callgrind file records every caller->callee edge with its inclusive cost, so the libc
bucket is attributed back to the engine functions that issued the calls.

Usage: instruction_profile.py <callgrind_annotate-output> [--counts bench-stdout]
       [--raw callgrind.out] [--out paper/tables] [--measured-on M]
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt   # noqa: E402

# Ordered: the first bucket whose key appears in the lowercased "file:function" entry wins, so
# the more specific (function-name) keys go first and the file-name fallbacks last.
BUCKETS = [
    ("Quotient reconstruction (replay, causal DP, TR)",
     ("quotient_replay_core", "quotient_causal_core", "qr_apply", "qc_", "register_quotient_transition",
      "tr_reduce", "transitive_reduction", "edgeorbittable", "qcappliedmatch", "for_each_reconstructed")),
    ("Canonicalization (IR and orbits)",
     ("ir_core.hpp", "ir_refine", "ir_canonical", "ir_hash_and_orbits", "compute_and_cache_state_orbits",
      "state_orbits", "slot_core", "irpartition", "wl_hash")),
    ("Match expansion, join and rewrite",
     ("submit_expand_task", "expand_task", "execute_rewrite_task", "join_core", "join_dfs",
      "pattern_matcher", "functionjob", "parallelevolutionengine", "index.hpp", "signature")),
    ("Concurrent sets and maps",
     ("concurrent_key_set", "concurrent_map", "lock_free_list")),
    ("Arena allocation",
     ("arena.hpp", "arena.cpp", "arena_worker_index", "scratch_alloc")),
    ("Hypergraph storage",
     ("segmented_array", "hypergraph.hpp", "hypergraph.cpp", "bitset", "types.cpp", "create_edge",
      "create_state")),
    ("libc copies, fills and heap",
     ("memcpy", "memmove", "memset", "malloc.c", "operator new", "operator delete", "libc.so",
      "libstdc++")),
    ("Job system and work stealing",
     ("job_system", "work_stealing", "deque", "job.hpp", "park")),
    ("Benchmark harness",
     ("corpus_gen", "corpus::corpus", "bench_cpu_evolve.cpp")),
]

LIBC_CALLEES = ("__memcpy_avx_unaligned_erms", "__memset_avx2_unaligned_erms", "malloc", "free",
                "_int_malloc", "_int_free")


def bucket_of(what):
    for name, keys in BUCKETS:
        if any(k in what for k in keys):
            return name
    return "Unattributed"


def parse(path):
    lines = open(path).read().split("\n")
    total = None
    for line in lines:
        m = re.match(r"^\s*([\d,]+) \(100\.0%\)\s+PROGRAM TOTALS", line)
        if m:
            total = int(m.group(1).replace(",", ""))
            break
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Ir") and "file:function" in line:
            start = i + 2
            break
    if total is None or start is None:
        raise SystemExit("%s is not callgrind_annotate output" % path)
    end = start
    while end < len(lines) and not lines[end].startswith("----"):
        end += 1

    buckets, covered = collections.Counter(), 0
    # Inside canonicalization, the refinement loop against everything around it: the per-call
    # setup and invariant hash, and the orbit computation the quotient needs. The bounds
    # paragraph divides the phase by these.
    canon_parts = collections.Counter()
    for line in lines[start:end]:
        m = re.match(r"^\s*([\d,]+) \(\s*[\d.]+%\)\s+(.*)$", line)
        if not m:
            continue
        n, what = int(m.group(1).replace(",", "")), m.group(2).lower()
        covered += n
        name = bucket_of(what)
        buckets[name] += n
        if name.startswith("Canonicalization"):
            if "ir_refine" in what:
                canon_parts["refine"] += n
            elif any(k in what for k in ("orbits", "slot_core")):
                canon_parts["orbits"] += n
            else:
                canon_parts["hash"] += n
    return total, covered, buckets, canon_parts


COUNT_PATTERNS = {
    "canonical": r"canonical=(\d+)",
    "raw": r"raw=(\d+)",
    "ir_calls": r"ir: calls=(\d+)",
    "ir_searched": r"ir: calls=\d+ searched=(\d+)",
    "events": r"replay: claims=\d+ events=(\d+)",
    "captured": r"captured=(\d+)",
    "set_inserts": r"keyset: inserts=(\d+)",
    "set_wins": r"keyset: inserts=\d+ wins=(\d+)",
    "causal_pairs": r"recon: causal_pairs=(\d+)",
}


def parse_counts(path):
    text = open(path).read()
    counts = {}
    for key, pat in COUNT_PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise SystemExit("%s: no '%s' line -- a stats build of bench_cpu_evolve prints it" % (path, key))
        counts[key] = int(m.group(1))
    return counts


def parse_callers(path):
    """Inclusive cost of every caller->callee edge, from the raw callgrind file.

    `fn=(id) name` opens a caller (the name appears on the id's first occurrence only), `cfn=(id)
    [name]` names the callee of the next `calls=` record, and the line after `calls=` carries the
    inclusive cost of those calls as its last number.
    """
    names, edges = {}, collections.defaultdict(collections.Counter)
    cur, callee, pending = None, None, False
    ref = re.compile(r"^\((\d+)\)(?: (.*))?$")
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("fn=") or line.startswith("cfn="):
                m = ref.match(line.split("=", 1)[1])
                if not m:
                    continue
                ident = int(m.group(1))
                if m.group(2):
                    names[ident] = m.group(2)
                if line.startswith("fn="):
                    cur = ident
                else:
                    callee = ident
                continue
            if line.startswith("calls="):
                pending = True
                continue
            if pending:
                pending = False
                parts = line.split()
                if len(parts) >= 2 and cur is not None and callee is not None:
                    edges[callee][cur] += int(parts[-1])
    return names, edges


def short(name):
    name = re.sub(r"\(.*$", "", name)          # drop the parameter list
    name = re.sub(r"<.*$", "", name)           # and template arguments
    return name.split("::")[-1] or name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("callgrind")
    ap.add_argument("--counts", help="bench_cpu_evolve stdout from a stats build of the same workload and depth")
    ap.add_argument("--raw", help="the raw callgrind.out behind the annotate output, for caller attribution")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--measured-on", default="")
    a = ap.parse_args()

    pt.OUT = a.out
    if a.measured_on:
        pt._MEASURED_ON = a.measured_on

    total, covered, buckets, canon_parts = parse(a.callgrind)
    b = [pt.provenance("tools/dev/remote_session.sh attrib phase, callgrind over "
                       "tools/bench_cpu_evolve"),
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Component & Instructions & Share \\", r"\midrule"]
    for name, n in buckets.most_common():
        if name == "Unattributed":
            continue
        b.append(r"%s & %s & %.1f\%% \\" % (name, "{:,}".format(n), 100.0 * n / total))
    b.append(r"\midrule")
    b.append(r"Unattributed & %s & %.1f\%% \\"
             % ("{:,}".format(buckets["Unattributed"]), 100.0 * buckets["Unattributed"] / total))
    b.append(r"\midrule")
    b.append(r"\textbf{Total} & %s & %.1f\%% \\"
             % ("{:,}".format(total), 100.0 * covered / total))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t15_instruction_profile.tex", "\n".join(b) + "\n")
    # The coverage figure the caption cites, from the same parse, so the sentence cannot
    # drift from the table it describes.
    pt.value("InstrFlatCoverage", "%.1f" % (100.0 * covered / total))
    pt.value("InstrUnattributedPct", "%.1f" % (100.0 * buckets["Unattributed"] / total))
    share = {
        "Recon": "Quotient reconstruction (replay, causal DP, TR)",
        "Canon": "Canonicalization (IR and orbits)",
        "Match": "Match expansion, join and rewrite",
        "Sets": "Concurrent sets and maps",
        "Arena": "Arena allocation",
        "Storage": "Hypergraph storage",
        "Libc": "libc copies, fills and heap",
    }
    for key, name in share.items():
        pt.value("InstrShare" + key, "%.1f" % (100.0 * buckets[name] / total))
    canon_total = sum(canon_parts.values()) or 1
    for part in ("refine", "hash", "orbits"):
        pt.value("InstrCanon" + part.capitalize() + "Pct", "%.0f" % (100.0 * canon_parts[part] / canon_total))
    print("total %s Ir, flat profile covers %.1f%%" % ("{:,}".format(total), 100.0 * covered / total))
    for name, n in buckets.most_common():
        print("  %-48s %5.2f%%" % (name, 100.0 * n / total))

    if a.counts:
        c = parse_counts(a.counts)
        # Instructions per unit of each phase's own output. The unit is the quantity the phase
        # cannot avoid producing once per item: one canonicalization call per state created, one
        # reconstructed event per replay claim, one claim-set insert per key offered, one applied
        # match per raw child state.
        per = {
            "InstrIrPerCanonCall": buckets[share["Canon"]] / c["ir_calls"],
            "InstrIrPerReconEvent": buckets[share["Recon"]] / c["events"],
            "InstrIrPerSetInsert": buckets[share["Sets"]] / c["set_inserts"],
            "InstrIrPerAppliedMatch": buckets[share["Match"]] / c["captured"],
        }
        for key, v in per.items():
            pt.value(key, "{:,}".format(int(round(v))))
        for key in ("canonical", "raw", "ir_calls", "ir_searched", "events", "captured", "set_inserts",
                    "causal_pairs"):
            pt.value("Instr" + "".join(p.capitalize() for p in key.split("_")), "{:,}".format(c[key]))
        pt.value("InstrSetRepeatPct", "%.1f" % (100.0 * (c["set_inserts"] - c["set_wins"]) / c["set_inserts"]))
        print("counts: " + ", ".join("%s=%d" % kv for kv in c.items()))
        for key, v in per.items():
            print("  %-28s %10.0f Ir" % (key, v))

    if a.raw:
        names, edges = parse_callers(a.raw)
        by_name = {n: i for i, n in names.items()}

        def resolved_callers(ident, depth=0):
            # A call into libc goes through a PLT stub, which callgrind records as an unnamed
            # function at an address; the engine function behind it is that stub's own caller.
            out = collections.Counter()
            for caller, n in edges[ident].items():
                if names.get(caller, "0x").startswith("0x") and depth < 2 and edges.get(caller):
                    for c2, n2 in resolved_callers(caller, depth + 1).items():
                        out[c2] += n2
                else:
                    out[caller] += n
            return out

        print("libc callers (inclusive Ir of the calls, top 3 per callee):")
        top_caller = None
        for callee in LIBC_CALLEES:
            ident = by_name.get(callee)
            if ident is None or not edges.get(ident):
                continue
            ranked = resolved_callers(ident).most_common(3)
            print("  %-30s " % callee + "; ".join("%s %s" % (short(names.get(i, "?")), "{:,}".format(n))
                                                   for i, n in ranked))
            if callee.startswith("__memcpy") and ranked:
                top_caller = (short(names.get(ranked[0][0], "?")), ranked[0][1])
        if top_caller:
            pt.value("InstrMemcpyTopCaller", pt.tex_escape(top_caller[0]))
            pt.value("InstrMemcpyTopCallerPct", "%.1f" % (100.0 * top_caller[1] / total))

    pt.write_values("values_instr.tex")


if __name__ == "__main__":
    main()
