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
  --commit-msg <file>  check the message of the commit being made instead: a commit that
                       stages engine sources must declare Measurement-inert. This is the
                       commit-msg hook (tools/dev/commit_msg_gate.sh); pre-commit runs before the
                       message exists and cannot check it.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

# The directories whose diff makes a measurement stale: the engine the numbers describe.
ENGINE_DIRS = ["common", "hypergraph", "gpu", "job_system", "lockfree_deque"]

# Fragments whose re-measurement needs a resource this tree cannot reach. Each entry names
# the reason and what replaces it; anything else stale is a finding, not a candidate here.
STALE_WHITELIST = {}

# Commits that are measurement-inert but do not carry the "Measurement-inert:" line in their
# message (a pushed message cannot be amended). Hash-pinned, each with the proof; the walk
# treats them as declared.
COMMIT_ALLOWANCES = {
    "7a52bb6fab7a3f28c7bbc74986f337cde6e4c943":
        "bounds a steered continuation to the steps it asked for. continuation_ceiling_ is "
        "written only inside evolve_more, and set non-zero only when only_from is non-null; "
        "step_budget() answers max_steps_ while it is zero, and every gate it replaced read "
        "max_steps_. Every fragment in paper/tables is measured from evolve() runs -- no tool, "
        "benchmark or experiment in this tree calls evolve_more -- so on every measured path "
        "the ceiling is zero and each gate reads the value it read before. Its message states "
        "the proof as 'Measurement-inert for every caller that does not steer:', which the "
        "walk does not read as the declaration",
    "530a2c9503bfc157c9e6ac02eb95a94a63f9d281":
        "its job_system.hpp diff is six comment lines at ensure_default_cpu_order (git show "
        "confirms zero code lines); the comment records the stacked-vs-spill A/B and rode in "
        "through stale staging",
    "079654eb10d7c1aa44fa31796e27c251fa3f53f0":
        "emits VIZ_EMIT_MATCH_FOUND from store_match_for_state. The whole addition is inside "
        "#ifdef HYPERGRAPH_ENABLE_VISUALIZATION, and every fragment in paper/tables is "
        "measured from a build that does not define it -- the visualisation event sink is a "
        "renderer's seam, not part of any measured configuration. On those builds the "
        "preprocessor removes the block entirely, so store_match_for_state emits the same "
        "instructions it did before and no state, event, causal or branchial count, and no "
        "timing, can move",
    "ac96277d712dcc9f97a536fa979c2bb9f60428a0":
        "widens two parameters (Hypergraph::create_edge's arity, create_genesis_event's edge "
        "count) from uint8_t to size_t and drops the call-site casts, which were "
        "value-preserving for every count at or below 255. Both guards existed already and "
        "neither gains a comparison, so the executed work is unchanged; the added bound "
        "refuses a genesis edge count above 255, and every measured workload seeds far fewer "
        "edges than that and builds edges of arity at most MAX_ARITY (16). No guard outcome "
        "and no constructed object differs on any measured input, so no state, event, causal "
        "or branchial count can move",
    "118a0b8a856c7455dd40313c8c7fae6e82c0caed":
        "passes recycle_scratch=false on the enqueue branch that runs a job on the submitting "
        "thread when every queue is full; on a worker that branch already passed false, so the "
        "change applies only to a thread that is not a worker. The branch runs only when a "
        "submit finds the injector's 32,768 slots full. In a measured run the calling thread "
        "submits only the root seeds, one per initial state, into an empty injector, then parks "
        "in wait_for_completion, which pops no job; every later submit is made inside a job on "
        "a worker; serial mode takes its own branch at the top of enqueue, which this commit "
        "leaves unchanged; and no instrument named as a fragment source calls evolve_more, the "
        "one path that seeds from the calling thread during a run. The changed call therefore "
        "never executes on a measured path, so no state, event, causal or branchial count, and "
        "no timing, can move",
    "545ff7381b9a67855e289c14b1fdb6d42988483d":
        "sends the job-system branch that runs a job on the submitting thread, when there is "
        "no room to queue it, through run_inline_, which calls two hooks around the job; the "
        "engine registers them in its constructor to mark the calling thread's scratch arena "
        "before such a job and release to that mark after. Construction gains two "
        "std::function assignments per engine. The hooks run only on that overflow branch: "
        "the injector's 32,768 slots full in serial mode, or a worker's deque and the injector "
        "both full in threaded mode. Scratch holds only a job's temporaries -- the ordinary "
        "path resets it after every job -- so releasing a job's own scratch when it returns "
        "frees nothing another job reads, and no state, event, causal or branchial count can "
        "move. Callgrind on {{x,y},{y,z}}->{{x,y},{y,z},{z,w}} to depth 8 (46,234 states), "
        "threaded and serial, counts the same arena mark() and release() instructions at "
        "5b8a2efe and at this commit, so the overflow branch is not reached there",
}

VERDICT_RE = re.compile(r"\b(DIFFERS|FAILED|FAIL|NaN|nan|[-+]?inf)\b")
PROV_RE = re.compile(r"%\s*commit ([0-9a-f]{7,40})")
SOURCE_RE = re.compile(r"source:\s*([\w/.+~-]+(?:\s*\+\s*[\w/.+~-]+)*)")


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True)


def declares_inert(message):
    """Whether a commit message carries the declaration the staleness walk accepts."""
    return "Measurement-inert:" in message


def check_commit_msg(path):
    staged = [l for l in git("diff", "--cached", "--name-only", "--", *ENGINE_DIRS)
              .stdout.splitlines() if l.strip()]
    if not staged:
        return 0
    # Lines starting with '#' are git's template, removed from the message it records.
    with open(path, encoding="utf-8") as f:
        message = "".join(l for l in f if not l.startswith("#"))
    if declares_inert(message):
        return 0
    print("commit-msg: this commit changes %d engine file(s) (first: %s) and its message has no "
          "'Measurement-inert:' line. Every paper fragment goes stale and CI's paper job fails "
          "after the push. State the proof that no measured number moves on a line starting "
          "'Measurement-inert:', or re-measure the fragments." % (len(staged), staged[0]))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--main", default="paper/main.tex")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--commit-msg")
    a = ap.parse_args()
    if a.commit_msg:
        return check_commit_msg(a.commit_msg)

    findings = []

    with open(a.main, encoding="utf-8") as f:
        tex = f.read()
    # Sections are \input from their own files; a fragment referenced there is referenced.
    for sec in sorted(glob.glob(os.path.join(os.path.dirname(a.main), "sections", "*.tex"))):
        with open(sec, encoding="utf-8") as f:
            tex += f.read()
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
                why = None
                if changed:
                    # A commit may declare itself measurement-inert -- a change whose engine
                    # diff provably cannot move any measured number (a branch that never
                    # executes on the measuring machine, a comment). The declaration is the
                    # line "Measurement-inert:" in the commit message, carrying the proof, so
                    # the claim is reviewable where the change is. The fragment is allowed
                    # only when EVERY engine/instrument commit since its stamp declares it.
                    log = git("log", "--format=%H", commit + "..HEAD", "--",
                              *ENGINE_DIRS, *sources)
                    shas = [l.strip() for l in log.stdout.splitlines() if l.strip()]
                    inert = []
                    for sha in shas:
                        body = git("show", "-s", "--format=%B", sha).stdout
                        if not declares_inert(body) and sha not in COMMIT_ALLOWANCES:
                            inert = None
                            break
                        subject = body.splitlines()[0][:70]
                        inert.append("%s %s" % (sha[:8], subject))
                    if inert is not None:
                        print("allowed  %s: every engine/instrument commit since %s declares "
                              "Measurement-inert: %s" % (name + ".tex", commit, "; ".join(inert)))
                    else:
                        why = ("%d engine/instrument file(s) changed since %s (first: %s)"
                               % (len(changed), commit, changed[0]))
                head_stale_cache[key] = why
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
