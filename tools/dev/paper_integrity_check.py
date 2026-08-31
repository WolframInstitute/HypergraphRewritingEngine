#!/usr/bin/env python3
"""The paper's measured content percolates, or the build is red.

Every table and figure fragment the paper inputs is GENERATED from a measurement, and the
generation records where it came from (the provenance line: commit, machine, load, source).
Nothing between the instrument and the PDF re-checks that record, which is how a committed
paper carried a table whose verdict column read DIFFERS on every row (a stats-gated counter,
found by a reader, not a gate) and a device table measured on a kernel that had since been
rewritten. This checker is that gate. It fails when:

  MISSING     main.tex inputs a fragment that does not exist, or a fragment on disk is
              input by nothing (it can rot without anyone noticing).
  NOPROV      a fragment carries no provenance line, so staleness cannot be judged at all.
  VERDICT     a fragment body contains a failure token (DIFFERS, FAILED, NaN, ...). A verdict
              column is a gate, not a datum; its generator should have refused to write it.
  STALE       engine sources, or the fragment's own named instrument source, differ between
              the fragment's provenance commit and HEAD. The paper describes the system AS IT
              IS; a fragment measured on an engine that has since changed describes a system
              that no longer exists. The remedy is re-measurement, never a whitelist entry --
              except where the measurement needs a resource this tree cannot reach (a
              licensed Wolfram kernel), and then the entry names the reason and the pending
              re-run so the exception is visible in review.

Run from the repository root: python3 tools/dev/paper_integrity_check.py
  --tables-dir <dir>   check a different fragment directory (ground-truthing, box pulls)
  --no-git             skip the staleness check (no repository available)
"""

import argparse
import os
import re
import subprocess
import sys

# The directories whose diff makes a measurement stale: the engine the numbers describe.
ENGINE_DIRS = ["common", "hypergraph", "gpu", "job_system", "lockfree_deque"]

# Fragments whose re-measurement needs a resource this tree cannot reach. Each entry names
# the reason and what replaces it; anything else stale is a finding, not a candidate here.
STALE_WHITELIST = {}

VERDICT_RE = re.compile(r"\b(DIFFERS|FAILED|FAIL|NaN|nan|[-+]?inf)\b")
PROV_RE = re.compile(r"%\s*commit ([0-9a-f]{7,40})")
SOURCE_RE = re.compile(r"source:\s*([\w/.+~-]+(?:\s*\+\s*[\w/.+~-]+)*)")


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--main", default="paper/main.tex")
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()

    findings = []

    with open(a.main, encoding="utf-8") as f:
        tex = f.read()
    referenced = set(re.findall(r"\\input\{tables/([\w.]+?)(?:\.tex)?\}", tex))
    on_disk = {n[:-4] for n in os.listdir(a.tables_dir) if n.endswith(".tex")}

    for name in sorted(referenced - on_disk):
        findings.append("MISSING  tables/%s.tex is input by %s and does not exist"
                        % (name, a.main))
    for name in sorted(on_disk - referenced):
        findings.append("MISSING  %s/%s.tex is input by nothing; delete it or input it"
                        % (a.tables_dir, name))

    head_stale_cache = {}
    for name in sorted(on_disk):
        path = os.path.join(a.tables_dir, name + ".tex")
        with open(path, encoding="utf-8") as f:
            body = f.read()
        if not body.strip():
            findings.append("MISSING  %s is empty" % path)
            continue

        for line in body.splitlines():
            if line.lstrip().startswith("%"):
                continue
            m = VERDICT_RE.search(line)
            if m:
                findings.append("VERDICT  %s carries '%s': %s"
                                % (path, m.group(1), line.strip()[:100]))

        m = PROV_RE.search(body)
        if not m:
            findings.append("NOPROV   %s has no provenance line" % path)
            continue
        commit = m.group(1)

        if a.no_git:
            continue
        sources = []
        ms = SOURCE_RE.search(body)
        if ms:
            sources = [s.strip() for s in ms.group(1).split("+") if s.strip()]
        key = (commit, tuple(sources))
        if key not in head_stale_cache:
            if git("cat-file", "-e", commit + "^{commit}").returncode != 0:
                head_stale_cache[key] = ("its provenance commit %s is not in this "
                                         "repository" % commit)
            else:
                r = git("diff", "--name-only", commit, "HEAD", "--",
                        *ENGINE_DIRS, *sources)
                changed = [l for l in r.stdout.splitlines() if l.strip()]
                head_stale_cache[key] = (
                    "%d engine/instrument file(s) changed since %s (first: %s)"
                    % (len(changed), commit, changed[0]) if changed else None)
        why = head_stale_cache[key]
        if why:
            frag = name + ".tex"
            if frag in STALE_WHITELIST:
                print("allowed  %s stale: %s -- %s" % (path, why, STALE_WHITELIST[frag]))
            else:
                findings.append("STALE    %s: %s" % (path, why))

    for f_ in findings:
        print(f_)
    print("%d finding(s) over %d referenced fragments, %d on disk"
          % (len(findings), len(referenced), len(on_disk)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
