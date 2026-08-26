#!/usr/bin/env python3
"""Generate the paper's measurement tables from the instruments that produce them.

Each table is written to paper/tables/<name>.tex and \\input by paper/main.tex, so a
number in the paper is a file the tools wrote and not a value someone typed. Every
table carries the commit, the machine and the tool it came from; regenerating on a
different machine rewrites those lines with it.

Run:  python3 tools/dev/paper_tables.py [--gpu] [--build-dir build_linux]

--gpu adds the device tables, which need a CUDA build and a visible GPU.
"""

import argparse
import os

import procstat
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "tables")


def run(cmd, timeout=1800):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit("FAILED (%d): %s\n%s" % (p.returncode, " ".join(cmd), p.stderr[-2000:]))
    return p.stdout


# The host the MEASURED BINARIES ran on, which is not always the host generating the table.
# A Windows .exe launched from WSL executes natively on Windows -- native CUDA driver, no
# /dev/dxg paravirtualisation and no Hyper-V scheduling between the benchmark and the hardware --
# so a number taken that way must not be stamped with the generator's Linux uname.
_MEASURED_ON = None

# Load on this machine BEFORE any measurement starts. Sampling it afterwards would report the
# benchmark's own load and say nothing; sampled at the start it reports what else was running,
# which is the difference between a number that can be published and one that cannot. The
# fifteen-minute figure is the one that shows sustained background work rather than a spike.
_BASELINE_LOAD = None

# The CPU set the workers were bound to, empty when placement was left to the operating system.
_PINNED_CPUS = ""


def _pinned_note(pinned=False):
    """The pin stamp, and ONLY for a table whose measurement actually received the CPU set.

    provenance() used to add this to every table because _PINNED_CPUS was set once from the
    command line, while only scaling() forwards the set to the binary it runs. Seven of the
    eight tables therefore carried "workers pinned to cpus ..." over numbers measured with
    placement left to the operating system. A provenance line that describes a different run
    than the one that produced the row is worse than no provenance line.
    """
    if not pinned or not _PINNED_CPUS:
        return ""
    note = " | workers pinned to cpus %s" % _PINNED_CPUS
    if procstat.topology_is_virtualised():
        note += " (VIRTUAL cpus: this kernel is on a hypervisor, so physical core placement is " \
                "the host's and these cores are NOT known to be identical)"
    return note


def _load_note():
    if _BASELINE_LOAD is None:
        return ""
    one, _five, fifteen = _BASELINE_LOAD
    # A quiet box sits near zero. Anything above a core's worth of sustained background work
    # means the timings below were taken in competition with it, and the table has to say so
    # rather than leave a reader to assume otherwise.
    # Relative to the machine, for the reason remote_session.sh's gate is: 0.7 on a 32-thread
    # box is two per cent utilisation, so every table taken on a large host was stamped
    # CONTENDED for background noise, and the marker stopped meaning anything.
    limit = max(0.7, (os.cpu_count() or 4) * 0.1)
    # JUDGED ON THE ONE-MINUTE FIGURE, which is the same quantity remote_session.sh's wait_quiet
    # gates on. Judging on max(1m, 15m) made the gate and the stamp disagree about one run: the
    # session measured because the box was quiet (1.78 against a limit of 3.2) and then stamped
    # the result CONTENDED because the fifteen-minute average was 3.29 -- and that average is the
    # SESSION'S OWN -j32 build decaying, not anything competing with the measurement. Eight
    # fragments carried the marker from a box that was 5.6% utilised.
    #
    # Both figures are still printed. The fifteen-minute number is context a reader may want; it
    # is not evidence about the minute the timings were taken in.
    flag = "" if one < limit else "  *** CONTENDED"
    return " | load at start %.2f/%.2f (1m/15m)%s" % (one, fifteen, flag)


def binary(build, name):
    """Resolve a tool in `build`, taking the native name or its .exe.

    A Windows build directory holds bench_cpu_evolve.exe where a Linux one holds
    bench_cpu_evolve, and the caller should not have to know which it pointed at.
    """
    global _MEASURED_ON
    plain = os.path.join(build, name)
    if os.path.exists(plain):
        return plain
    win = plain + ".exe"
    if os.path.exists(win):
        _MEASURED_ON = _windows_host()
        return win
    raise SystemExit("no %s (or %s.exe) in %s -- build it there first" % (name, name, build))


def _windows_host():
    """CPU and RAM are this machine either way; only the operating system differs."""
    ver = ""
    try:
        ver = subprocess.run(["cmd.exe", "/c", "ver"], capture_output=True, text=True,
                             timeout=30).stdout.strip().splitlines()
        ver = " ".join(v.strip() for v in ver if v.strip())
    except Exception:
        ver = ""
    return "%s, %s GB RAM, %s" % (procstat.cpu_name(), procstat.ram_gb(),
                                  ver or "Windows (native)")


def commit():
    return run(["git", "rev-parse", "--short", "HEAD"]).strip()


def machine():
    """The provenance line's machine string. procstat answers it on both platforms."""
    return procstat.machine()


def provenance(tool, pinned=False):
    # _MEASURED_ON is set by binary() and is therefore only known AFTER the tools have been
    # resolved, which every table does before it formats its provenance line.
    where = _MEASURED_ON or machine()
    generated_on = ""
    if _MEASURED_ON and _MEASURED_ON != machine():
        generated_on = " | table generated on %s" % machine()
    return ("%% GENERATED by tools/dev/paper_tables.py -- do not edit.\n"
            "%% commit %s | measured on %s%s%s%s | source: %s\n"
            % (commit(), where, generated_on, _pinned_note(pinned), _load_note(), tool))


def tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# A TABLE THAT DOES NOT FIT THE TEXT BLOCK IS SHRUNK, AND ONLY THEN.
#
# These tables are generated from measurements, so their width is decided by the data: a run
# that produces longer case names or larger numbers silently pushes the tabular past the margin,
# and LaTeX reports that as an "Overfull \hbox" warning it is very easy to not read. Five of
# these tables were overflowing, the worst by 248pt -- about 8.7cm of table hanging off the page.
#
# \resizebox with the \ifdim guard scales ONLY when the natural width exceeds the line, and
# leaves a table that already fits completely untouched, so nothing that currently typesets
# correctly changes shape. Wrapping at the single point every table is written means a future
# measurement cannot reintroduce the problem in one table and not another.
# SHRINK THE TYPE FIRST, THE GEOMETRY ONLY IF THAT IS NOT ENOUGH. \resizebox alone scales a wide
# table by whatever factor it takes, and the factor is decided by the column count: a ten-column
# table rendered at roughly half the size of the six-column table directly beneath it, smaller
# than its own caption. Setting \footnotesize and a tighter \tabcolsep first reduces the natural
# width by about a fifth, so most tables then fit with no scaling at all and the ones that still
# need it are scaled far less. The result is a type size chosen on purpose rather than one that
# falls out of how many columns the data happened to have.
def _fit(inner):
    """Wrap a tabular so it shrinks to the line width only when it would otherwise overflow."""
    return ("{\\footnotesize\\setlength{\\tabcolsep}{4pt}%\n"
            "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
            + inner + "\n}}\n")


# Values the PROSE cites. A number quoted in a sentence drifts from the table it came from as
# soon as the table is regenerated, and nothing detects it, so the sentence cites a macro and the
# macro is written here. Accumulated across generators and emitted once, since several contribute.
VALUES = {}


def value(name, latex):
    VALUES[name] = latex


def write_values(name="values_tables.tex"):
    """Emit the accumulated macros. Each script writes its OWN file: they run as separate
    processes, so a shared name would mean whichever ran second erased the other's macros.

    Macros already in the file whose section did NOT run this invocation are CARRIED FORWARD
    under their own marker rather than dropped: the prose cites them, so dropping them breaks
    the build silently -- a --gpu-only regeneration removed the four authority-ratio macros
    and the paper stopped compiling until the next `make`. A carried macro keeps the value of
    the run that measured it; regenerating its section replaces it and the marker."""
    if not VALUES:
        return
    carried = {}
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        with open(path) as f:
            for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$",
                                 f.read(), re.MULTILINE):
                if m.group(1) not in VALUES:
                    carried[m.group(1)] = m.group(2)
    body = [provenance("the generators in this file")]
    body += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in sorted(VALUES.items())]
    if carried:
        body.append("% carried forward from the previous generation of this file: their")
        body.append("% measurement did not run in this invocation, and the prose cites them.")
        body += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in sorted(carried.items())]
    write_raw(name, "\n".join(body) + "\n")


def write_raw(name, body):
    """Emit a fragment verbatim. Figure data is pgfplots commands, not a tabular, so it must not
    be wrapped in the resizebox write() applies."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(body)
    print("wrote %s" % os.path.relpath(path, ROOT))


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    # The provenance comment lines stay OUTSIDE the box: they are comments, and a \resizebox
    # argument must be typesettable material.
    head = [ln for ln in body.splitlines() if ln.startswith("%")]
    rest = [ln for ln in body.splitlines() if not ln.startswith("%")]
    boxed = "\n".join(head) + "\n" + _fit("\n".join(rest).strip())
    with open(path, "w") as f:
        f.write(boxed)
    print("wrote %s" % os.path.relpath(path, ROOT))


# --- T1: exactness and deterministic memory, one row per corpus workload ------------


# ONE cost_matrix RUN, SHARED BY EVERY TABLE THAT READS IT.
#
# T1 and T3 both report per-workload columns from this tool and each used to invoke it
# separately. That is two measurements presented as one: the tables carry the same provenance
# line, name the same tool at the same commit, and disagreed on `multi-rule` -- 672608 arena
# bytes and 59 heap allocations against 672692 and 60.
#
# The divergence is real and it is in the tool, not in the plumbing. Measured on the evaluation
# box, two consecutive runs of the SAME binary differ on that one workload out of seventeen, and
# the heap figure is bimodal between exactly 1,536,060 and 2,060,428 bytes -- a difference of
# 524,368, which is one arena block. Whether a worker needs that block depends on scheduling.
# It is not ASLR (identical under `setarch -R`) and not thread count (`--serial` varies too,
# by 96 arena bytes, though its allocation count is stable).
#
# Sharing one run cannot make the tool deterministic, but it makes the PAPER consistent: every
# table quoting cost_matrix quotes the same run of it, which is what their shared provenance
# line already claims.
_COST_MATRIX_OUT = None


def cost_matrix_out(build):
    global _COST_MATRIX_OUT
    if _COST_MATRIX_OUT is None:
        _COST_MATRIX_OUT = run([binary(build, "cost_matrix")])
    return _COST_MATRIX_OUT


def t1(build):
    out = cost_matrix_out(build)
    rows = []
    total = ""
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s+(\S+)\s+(EXACT|INEXACT|\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                     r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            rows.append(m.groups())
        elif line.startswith("arena bytes, all artifacts"):
            total = line.strip()
    if not rows:
        raise SystemExit("cost_matrix produced no parseable rows")

    b = [provenance("tools/cost_matrix.cpp"),
         r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
         r"Workload & Class & Canon.\ states & Events & Causal & Branchial & "
         r"Arena B & Heap B & Heap allocs & Exactness \\", r"\midrule"]
    # cost_matrix's first numeric column is `raw` -- the raw state count, ahead of `canon`. Every
    # column here is named against that header, so the unpacking names it and drops it rather
    # than letting the values slide one place to the left.
    for (case, cls, exact, _raw, canon, ev, ca, br, arena, heapb, heapa) in rows:
        b.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            tex_escape(case), tex_escape(cls), canon, ev, ca, br, arena, heapb, heapa,
            exact.lower()))
    b += [r"\bottomrule", r"\end{tabular}", "", "%% " + total]
    write("t1_exactness.tex", "\n".join(b) + "\n")
    return len(rows)


def cr_ratio(build, low_steps):
    """C/R per corpus case at two depths, for Proposition 1's ceiling.

    C/R is cost_matrix's `wlceil` column: the fraction of IR invocations a Weisfeiler--Leman
    pre-filter could elide, which the proposition bounds above by the ratio of canonical classes
    to raw states. Two runs, because the claim is that the ratio FALLS with depth, and one depth
    cannot show a direction. The lower sample is taken with an explicit step override; the higher
    is each case's own measure_steps, which is what every other table here reports.

    The eight cases are named rather than taken from the whole corpus: the figure plots one bar
    pair each and the surrounding text refers to them individually.
    """
    row_re = (r"^(\S+)\s+(\S+)\s+(?:EXACT|INEXACT|\S+)\s+\d+\s+\d+\s+(\d+)"
              r"\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+([\d.]+)%")

    def sample(args):
        out = run([binary(build, "cost_matrix")] + args, timeout=1800)
        got = {}
        for line in out.splitlines():
            m = re.match(row_re, line)
            if m:
                got[m.group(1)] = (m.group(2), int(m.group(3)), float(m.group(4)))
        if not got:
            raise SystemExit("cost_matrix produced no parseable C/R rows:\n%s" % out[-1500:])
        return got

    high = sample([])
    low = sample([str(low_steps)])

    # (case in cost_matrix, short name on the figure's x axis)
    cases = [("cycle4-automorphic", "cycle4"), ("star4-automorphic", "star4"),
             ("disconnected-lhs", "disconn"), ("binary-growth", "binary"),
             ("multi-rule", "multi"), ("triangle", "triangle"),
             ("arity3-growth", "arity3"), ("self-loop", "self-loop")]
    rows = [(c, s, high[c], low.get(c)) for (c, s) in cases if c in high]
    rows.sort(key=lambda r: r[2][2])

    b = [provenance("tools/cost_matrix.cpp"),
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Case & Rule type & Events & $C/R$ & $C/R$ at lower depth \\", r"\midrule"]
    for (case, _short, h, l) in rows:
        b.append("\\texttt{%s} & %s & %s & %s\\%% & %s \\\\" % (
            tex_escape(case), tex_escape(h[0]), "{:,}".format(h[1]).replace(",", "{,}"),
            ("%g" % h[2]), ("%g\\%%" % l[2]) if l else "--"))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t_crratio.tex", "\n".join(b) + "\n")

    coords = lambda idx: "".join("(%s,%g)" % (r[1], r[idx][2]) for r in rows if r[idx])
    # The x axis is symbolic and its coordinate list must be declared before the axis begins,
    # while the plots go inside it. Two fragments, because one file cannot be input in both
    # places, and the order of the names has to match the order of the bars.
    syms = ",".join(r[1] for r in rows)
    write_raw("f_crratio_syms.tex",
              provenance("tools/cost_matrix.cpp") + "\n"
              + r"\pgfplotsset{crsymbolic/.style={symbolic x coords={%s}}}" % syms + "\n")
    f = [provenance("tools/cost_matrix.cpp"),
         r"\addplot[fill=gray!55, draw=black] coordinates {%s};" % coords(3),
         r"\addlegendentry{lower depth}",
         r"\addplot[fill=white, draw=black] coordinates {%s};" % coords(2),
         r"\addlegendentry{higher depth}"]
    write_raw("f_crratio.tex", "\n".join(f) + "\n")
    return len(rows)


# --- T8: thread scaling, and the single-thread column T2 compares against -----------

def scaling(build, tool, name, steps, iters, caption_tool, cpus="", threads=""):
    # argv[3] is the thread sweep and argv[5] the CPU set, so naming a set means naming a sweep
    # too -- a curve is only homogeneous while the thread count stays inside the cores it pins to.
    argv = [binary(build, tool), str(steps), str(iters)]
    if cpus:
        argv += [threads or "", "wpp", cpus]
    out = run(argv, timeout=3600)
    rows = []
    for line in out.splitlines():
        m = re.search(r"threads=(\d+)\s+steps=(\d+)\s+canonical=(\d+)\s+raw=(\d+)\s+"
                      r"median_ms=([\d.]+)\s+min_ms=([\d.]+)", line)
        if m:
            rows.append(m.groups())
    if not rows:
        raise SystemExit("%s produced no parseable rows:\n%s" % (tool, out[-2000:]))
    base = float(rows[0][4])
    b = [provenance(caption_tool, pinned=bool(cpus)),
         r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"Threads & Steps & Canonical states & Raw states & Median ms & Speedup vs.\ 1 thread \\",
         r"\midrule"]
    for (th, st, canon, raw, med, _mn) in rows:
        b.append("%s & %s & %s & %s & %s & %.2f \\\\" % (th, st, canon, raw, med, base / float(med)))
    b += [r"\bottomrule", r"\end{tabular}"]
    write(name, "\n".join(b) + "\n")

    # The conclusion states the best speedup and the thread count it occurs at. Emitting them as
    # macros is what keeps that sentence from drifting when this table is regenerated: the prose
    # cites the value rather than repeating it.
    best_t, best_s = max(((int(r[0]), base / float(r[4])) for r in rows), key=lambda p: p[1])
    value("MaxHostSpeedup", "%.1f" % best_s)
    value("MaxHostSpeedupThreads", "%d" % best_t)
    return len(rows)


def t7(build, gpu_build, steps_list, iters):
    """GPU against CPU on ONE workload, at increasing depth.

    bench_cpu_evolve and bench_gpu_evolve run the same WPP rule, the same two-edge initial
    condition, Full canonicalization and quotient exploration, so the medians are comparable.
    Both are medians of the same iteration count taken in the same session, because wall clock
    on this class of machine drifts more between sessions than the effects being reported.
    """
    rows = []
    for st in steps_list:
        gout = run([binary(gpu_build, "bench_gpu_evolve"), str(st), str(iters)], timeout=3600)
        m = re.search(r"steps=(\d+) states=(\d+) events=(\d+) \| evolve\(\)_median_ms=([\d.]+) \| "
                      r"PersistentEvolver_median_ms=([\d.]+)", gout)
        if not m:
            raise SystemExit("bench_gpu_evolve produced no parseable row:\n%s" % gout[-2000:])
        _, states, events, gpu_call, gpu_persist = m.groups()

        cout = run([binary(build, "bench_cpu_evolve"), str(st), str(iters)], timeout=3600)
        cpu = {}
        for line in cout.splitlines():
            c = re.search(r"threads=(\d+)\s+steps=\d+\s+canonical=\d+\s+raw=(\d+)\s+median_ms=([\d.]+)", line)
            if c:
                cpu[int(c.group(1))] = float(c.group(3))
        if 1 not in cpu or 8 not in cpu:
            raise SystemExit("bench_cpu_evolve gave no 1- and 8-thread rows:\n%s" % cout[-2000:])
        rows.append((st, states, events, cpu[1], cpu[8], float(gpu_persist), float(gpu_call)))

    b = [provenance("tools/bench_cpu_evolve.cpp + tools/bench_gpu_evolve.cpp"),
         r"\begin{tabular}{rrrrrrrr}", r"\toprule",
         r"Steps & Raw states & Events & CPU 1t ms & CPU 8t ms & GPU ms & vs.\ CPU 1t & vs.\ CPU 8t \\",
         r"\midrule"]
    for (st, states, events, c1, c8, gpu, _call) in rows:
        b.append("%d & %s & %s & %.1f & %.1f & %.1f & %.2f & %.2f \\\\" % (
            st, states, events, c1, c8, gpu, c1 / gpu, c8 / gpu))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t7_gpu.tex", "\n".join(b) + "\n")

    # The conclusion states where the device wins and by how much at the deepest depth measured.
    deepest = rows[-1]
    value("GpuDeepestDepth", "%d" % deepest[0])
    value("GpuSpeedupAtDeepest", "%.1f" % (deepest[4] / deepest[5]))

    # The left panel of the GPU figure plots the CPU 8-thread and GPU columns of this table.
    cpu_pts = "".join("(%d,%.1f)" % (r[0], r[4]) for r in rows)
    gpu_pts = "".join("(%d,%.1f)" % (r[0], r[5]) for r in rows)
    f = [provenance("tools/bench_cpu_evolve.cpp + tools/bench_gpu_evolve.cpp"),
         r"\addplot[mark=*, black] coordinates {%s};" % cpu_pts,
         r"\addlegendentry{CPU, 8 threads}",
         r"\addplot[mark=square, black, dashed] coordinates {%s};" % gpu_pts,
         r"\addlegendentry{GPU}"]
    write_raw("f_gpu_depth.tex", "\n".join(f) + "\n")
    return len(rows)


def t6(build, maxd):
    """Quotient exploration against full capture, and what exactness costs on top.

    One workload (wolfram-2to4) at increasing depth. The events and causal columns are the
    point: quotient alone reports fewer, quotient plus reconstruction reports the full-capture
    numbers exactly, at the quotient's cost.
    """
    out = run([binary(build, "quotient_reconstruction_cost_probe"), str(maxd)], timeout=3600)
    rows = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+) \|\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+) \|"
                     r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+) \|"
                     r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\w+)", line)
        if m:
            rows.append(m.groups())
    if not rows:
        raise SystemExit("quotient probe produced no parseable rows:\n%s" % out[-2000:])
    b = [provenance("tools/quotient_reconstruction_cost_probe.cpp"),
         r"\begin{tabular}{rrrrrrrrl}", r"\toprule",
         r"Depth & Events (full) & Causal (full) & ms (full) & Events (quot.) & ms (quot.) & "
         r"Events (quot.+recon) & ms (quot.+recon) & Exact \\", r"\midrule"]
    for (d, _fst, fev, fca, fms, _qst, qev, _qca, qms, _rst, rev, _rca, rms, exact) in rows:
        b.append("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            d, fev, fca, fms, qev, qms, rev, rms, exact))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t6_quotient.tex", "\n".join(b) + "\n")
    return len(rows)


def wolframscript():
    """The kernel driver, native if present, else the Windows install this box runs under WSL."""
    from shutil import which
    exe = os.environ.get("HG_WOLFRAMSCRIPT") or which("wolframscript")
    if exe:
        return [exe], False
    import glob
    cands = sorted(glob.glob("/mnt/c/Program Files/Wolfram*/*/*/wolframscript.exe"))
    if cands:
        return [cands[-1]], True
    return None, False


def t2(build, maxd, reps):
    """The engine against the authority, on one workload, at increasing depth.

    THE COMPARISON BASIS IS THE WHOLE PROBLEM HERE, and getting it wrong inflated an earlier
    version of this table by two orders of magnitude. Wolfram/Multicomputation's MultiwaySystem
    exposes NO property that returns the state set as data -- every one of them is graph-valued
    (StatesGraph, CausalGraph, EvolutionGraph, ...) -- so its time necessarily includes building
    Wolfram Graph objects and no subtraction from outside can remove that. Timing OUR side at the
    C++ API, which does no marshalling and builds no graph, and dividing, compares two different
    quantities.

    So the table reports two rows per depth and takes its ratio from the LIKE-FOR-LIKE pair:

      authority   MultiwaySystem, timed the only way its API allows.
      engine      this engine reached the way a user reaches it -- through the paclet, over the
                  wire to the worker process, building the same kind of Wolfram structures.

    The engine's C++ core time is carried in its own column, clearly labelled, because it answers
    a different question (what the engine costs) and must never be divided into the authority's.
    """
    ws, windows = wolframscript()
    if not ws:
        print("T2 skipped: no wolframscript on this machine")
        return 0
    script = os.path.join(ROOT, "reference", "bench_authority.wls")
    if windows:
        script = subprocess.run(["wslpath", "-w", script], capture_output=True, text=True).stdout.strip()
    out = run(ws + ["-file", script, str(maxd), str(reps)], timeout=7200)
    auth, hgev, ref = {}, {}, {}
    for line in out.splitlines():
        m = re.search(r"AUTH d=(\d+) states=(\d+) causal=(\d+) ms=([\d.]+)", line)
        if m:
            auth[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        m = re.search(r"HGEV d=(\d+) states=(\d+) causal=(\d+) ms=([\d.]+)", line)
        if m:
            hgev[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
        # OUR OWN reference implementation, the second thing the engine is compared with.
        m = re.search(r"REF\s+d=(\d+) states=(\d+) causal=(\d+) ms=([\d.]+)", line)
        if m:
            ref[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    if not auth or not hgev:
        raise SystemExit("bench_authority.wls printed no comparable rows:\n%s" % out[-2000:])

    eng = run([binary(build, "quotient_reconstruction_cost_probe"), str(max(auth))],
              timeout=3600)
    core = {}
    for line in eng.splitlines():
        m = re.match(r"\s*(\d+) \|\s*(\d+)\s+\d+\s+\d+\s+([\d.]+) \|", line)
        if m:
            core[int(m.group(1))] = float(m.group(3))

    b = [provenance("reference/bench_authority.wls + tools/quotient_reconstruction_cost_probe.cpp"),
         r"\begin{tabular}{rrrrrrrrl}", r"\toprule",
         r"Depth & States & Authority ms & Reference ms & Engine ms (paclet) & "
         r"Engine ms (C++ core) & Speedup vs authority & Speedup vs reference & "
         r"Counts agree \\", r"\midrule"]
    for d in sorted(auth):
        if d not in hgev:
            continue
        a_states, a_causal, a_ms = auth[d]
        h_states, h_causal, h_ms = hgev[d]
        r_states, r_causal, r_ms = ref.get(d, (None, None, None))
        # Every implementation in the row must have computed the same thing, or the times are
        # not comparable. The reference is included in that test, not exempt from it.
        same = (a_states, a_causal) == (h_states, h_causal)
        if r_ms is not None:
            same = same and (r_states, r_causal) == (a_states, a_causal)
        agree = "yes" if same else "NO"
        c = core.get(d)
        # The engine's C++ core against the reference: both compute DATA, so neither is charged
        # for building Wolfram Graph objects and the ratio is not a graph-construction artefact.
        vs_ref = ("%.0f$\\times$" % (r_ms / c)) if (r_ms is not None and c) else "--"
        # The engine's C++ core against the authority. Both sides are stated in the row beside
        # it -- the authority's time includes building Wolfram Graph objects, because its
        # MultiwaySystem exposes no property returning the state set as data, while the core
        # builds none. That asymmetry is the reason the paclet column sits here too: it is the
        # same-basis comparison, and this column is the engine's own speed with nothing
        # marshalled.
        vs_auth = ("%.0f$\\times$" % (a_ms / c)) if c else "--"
        b.append("%d & %d & %.1f & %s & %.1f & %s & %s & %s & %s \\\\" % (
            d, a_states, a_ms, ("%.1f" % r_ms) if r_ms is not None else "--", h_ms,
            ("%.1f" % c) if c else "--", vs_auth, vs_ref, agree))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t2_speedup.tex", "\n".join(b) + "\n")

    # The figure beside this table plots the same two series. Emitting them here is what keeps
    # the figure and the table from drifting: one run produces both, so a regenerated table
    # cannot leave the plot showing the previous run's numbers.
    # The abstract, the figure caption and the conclusion each quote the ratio against the
    # authority at the shallowest and deepest measured depth, so both are emitted as macros.
    ratios = [(d, auth[d][2] / core[d]) for d in sorted(auth) if d in hgev and core.get(d)]
    # Depth 2 carries the authority's one-time initialisation, and the engine's time there is at
    # the timer's resolution, so the quoted range starts one depth in.
    quoted = [p for p in ratios if p[0] >= 4] or ratios
    if quoted:
        value("SpeedupLowDepth", "%d" % quoted[0][0])
        value("SpeedupAtLowDepth", "{:,}".format(int(round(quoted[0][1]))).replace(",", "{,}"))
        value("SpeedupHighDepth", "%d" % quoted[-1][0])
        value("SpeedupAtHighDepth", "{:,}".format(int(round(quoted[-1][1]))).replace(",", "{,}"))

    depths = [d for d in sorted(auth) if d in hgev and core.get(d)]
    auth_pts = "".join("(%d,%.1f)" % (d, auth[d][2]) for d in depths)
    core_pts = "".join("(%d,%.1f)" % (d, core[d]) for d in depths)
    f = [provenance("reference/bench_authority.wls + tools/quotient_reconstruction_cost_probe.cpp"),
         r"\addplot[mark=square*, black, dashed] coordinates {%s};" % auth_pts,
         r"\addlegendentry{\texttt{Wolfram/\allowbreak Multicomputation}}",
         r"\addplot[mark=*, black] coordinates {%s};" % core_pts,
         r"\addlegendentry{this engine}"]
    write_raw("f_speedup.tex", "\n".join(f) + "\n")
    return len(auth)


def t9(build, iters):
    """Per-contribution ablation, from the switches the shipped binary still has.

    There are no compiled-out ablation builds to run: when a replacement lands here the
    replaced path is deleted in the same commit, so an "off" arm exists only where the choice
    is still a live switch. Those are match forwarding (which the rule set decides) and the
    state-canonicalization mode. Each row is a median over the same iteration count, and the
    state and event counts are printed so a row that changed the ANSWER is visible rather than
    read as a speedup.
    """
    def smoke(arm, rule, edges, steps, canon):
        best = None
        for _ in range(max(3, iters // 3)):
            out = run([binary(build, "sampling_cost_smoke"), arm, rule, str(edges),
                       str(steps), "4", "4", canon], timeout=1800)
            m = re.search(r"done ([\d.]+) ms\s+states=(\d+).*events=(\d+)", out)
            if not m:
                raise SystemExit("sampling_cost_smoke gave no row:\n%s" % out[-1000:])
            ms, st, ev = float(m.group(1)), m.group(2), m.group(3)
            if best is None or ms < best[0]:
                best = (ms, st, ev)
        return best

    rows = []
    a = smoke("off", "growth", 1, 8, "full")       # forwarding decided: off on a one-edge LHS
    b = smoke("fwdon", "growth", 1, 8, "full")     # forced on
    rows.append(("Match forwarding, one-edge LHS", "on", b, "decided off", a))
    a = smoke("off", "pair", 4, 6, "full")         # forwarding decided: on for a two-edge LHS
    b = smoke("fwdoff", "pair", 4, 6, "full")      # forced off
    rows.append(("Match forwarding, two-edge LHS", "off", b, "decided on", a))
    a = smoke("off", "pair", 4, 6, "automatic")
    b = smoke("off", "pair", 4, 6, "full")
    rows.append(("State canonicalization", "Full (exact)", b, "Automatic (hash)", a))

    b_ = [provenance("tools/sampling_cost_smoke.cpp"),
          r"\begin{tabular}{lllrlrl}", r"\toprule",
          r"Contribution & Arm A & A ms & Arm B & B ms & Ratio & Same answer \\", r"\midrule"]
    for (name, an, av, bn, bv) in rows:
        same = "yes" if (av[1], av[2]) == (bv[1], bv[2]) else "NO"
        b_.append("%s & %s & %.1f & %s & %.1f & %.2f & %s \\\\" % (
            name, an, av[0], bn, bv[0], av[0] / bv[0], same))
    b_ += [r"\bottomrule", r"\end{tabular}"]
    write("t9_ablation.tex", "\n".join(b_) + "\n")
    return len(rows)


def t3(build):
    """Is the global allocator on the concurrent path?

    The de-heap, the causal-closure rework and the copy-on-write states were scaffolded in the
    paper as three before/after tables. Their "before" columns are not reproducible as a
    comparison: each replaced path was deleted in the commit that replaced it (there is no build
    flag that restores it), the work spans dozens of commits on one branch, and the instrument
    that measures arena bytes today did not exist at the far end -- so a before/after pair would
    be two different instruments reporting one column.

    What IS checkable, and is the claim those three items were for: allocation count does not
    grow with the work. cost_matrix counts every global operator new during the measured
    evolution, so the table below puts that count next to the size of the run it served.
    """
    out = cost_matrix_out(build)
    rows = []
    for line in out.splitlines():
        m = re.match(r"^(\S+)\s+(\S+)\s+(EXACT|INEXACT|\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                     r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            g = m.groups()
            # Against cost_matrix's header: g[3] is raw, g[4] canon, g[5] events, g[6] causal,
            # g[7] branch, g[8] arenaB, g[9] heapB, g[10] heapAllocs. This table wants the
            # ALLOCATION count beside the size of the run, so it takes g[10] and not the byte
            # totals either side of it.
            rows.append((g[0], int(g[4]), int(g[5]), int(g[8]), int(g[10])))
    if not rows:
        raise SystemExit("cost_matrix produced no parseable rows")
    rows.sort(key=lambda r: r[2])   # by events: the work, which is what allocation must not track

    b = [provenance("tools/cost_matrix.cpp"),
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Workload & Canon.\ states & Events & Arena B & Heap allocs & Allocs per event \\",
         r"\midrule"]
    for (name, canon, ev, arena, allocs) in rows:
        per = ("%.4f" % (float(allocs) / ev)) if ev else "--"
        b.append("%s & %d & %d & %d & %d & %s \\\\" % (
            tex_escape(name), canon, ev, arena, allocs, per))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t3_heap.tex", "\n".join(b) + "\n")
    return len(rows)


def t10(build):
    """What each event-identity convention identifies, on the same runs.

    The three conventions are refinements of one another: ByEndpointStates calls two rewrites the
    same event when they run between the same states, ByConsumedProducedEdges when they also move
    the same edges, DistinctApplications never. Reading the CANONICAL EVENT COUNT across them on
    a fixed workload is what shows they are distinct points rather than one convention wearing
    three names -- which inspection cannot show, because a duplicated convention still equals
    itself.

    The counts are taken under Full state canonicalization, one thread. No timing column: the
    conventions differ in what they IDENTIFY, and the cost of the identity is the added IR pass,
    which T9's canonicalization row already measures.
    """
    out = run([binary(build, "mode_matrix_probe")], timeout=1800)
    rows = []
    workload = None
    for line in out.splitlines():
        m = re.match(r"^(\S+) \(steps=(\d+)\)", line)
        if m:
            workload, steps = m.group(1), m.group(2)
            continue
        m = re.match(r"^\s+Full\s+(\d+)/(\d+)/(\d+)/(\d+)\s+(\d+)/(\d+)/(\d+)/(\d+)\s+(\d+)/(\d+)/(\d+)/(\d+)", line)
        if m and workload:
            g = [int(x) for x in m.groups()]
            rows.append((workload, steps, g[0], g[1], g[5], g[9]))
            workload = None
    if not rows:
        raise SystemExit("mode_matrix_probe produced no parseable Full rows:\n%s" % out[-2000:])

    b = [provenance("tools/mode_matrix_probe.cpp"),
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Workload & Steps & Canon.\ states & By endpoint states & By consumed/produced edges & "
         r"Distinct applications \\", r"\midrule"]
    for (name, steps, canon, e1, e2, e3) in rows:
        b.append("%s & %s & %d & %d & %d & %d \\\\" % (tex_escape(name), steps, canon, e1, e2, e3))
    b += [r"\bottomrule", r"\end{tabular}"]
    write("t10_event_canon.tex", "\n".join(b) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_linux")
    ap.add_argument("--gpu-build-dir", default="build_gpu")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--quotient-depth", type=int, default=7)
    ap.add_argument("--authority-depth", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cr-low-steps", type=int, default=2,
                    help="step override for the lower C/R depth sample")
    ap.add_argument("--wolfram", action="store_true",
                    help="add T2, which needs a Wolfram kernel")
    ap.add_argument("--cpus", default="",
                    help="logical CPUs to pin workers to, e.g. 0,2,4,6,8,10,12,14. A speedup "
                         "column across cores of different speeds divides by a quantity that is "
                         "not compute, so a published curve names a homogeneous set.")
    ap.add_argument("--thread-sweep", default="",
                    help="thread counts for the scaling table; must not exceed --cpus")
    a = ap.parse_args()

    global _BASELINE_LOAD, _PINNED_CPUS
    _PINNED_CPUS = a.cpus
    try:
        _BASELINE_LOAD = os.getloadavg()
    except (OSError, AttributeError):
        _BASELINE_LOAD = None

    n = t1(a.build_dir)
    print("T1: %d workloads" % n)
    n = cr_ratio(a.build_dir, a.cr_low_steps)
    print("C/R: %d cases" % n)
    n = scaling(a.build_dir, "bench_cpu_evolve", "t8_scaling.tex", a.steps, a.iters,
                "tools/bench_cpu_evolve.cpp", a.cpus, a.thread_sweep)
    print("T8: %d thread counts" % n)
    if a.wolfram:
        n = t2(a.build_dir, a.authority_depth, a.reps)
        print("T2: %d depths" % n)
    n = t10(a.build_dir)
    print("T10: %d workloads" % n)
    n = t3(a.build_dir)
    print("T3: %d workloads" % n)
    n = t9(a.build_dir, a.iters)
    print("T9: %d contributions" % n)
    n = t6(a.build_dir, a.quotient_depth)
    print("T6: %d depths" % n)
    if a.gpu:
        # DEPTHS 5 TO 7, WHICH IS WHAT THIS TABLE CAN AFFORD. Extending it to 8 and 9 was
        # tried and abandoned: the CPU arm at depth 8 is 262,144 states and had not finished
        # a 15-iteration thread sweep after 40 minutes, and depth 9 doubles the states with
        # superlinear cost per state. The persistent-evolver crossover that motivated the
        # extension is a GPU-only comparison and is reported by t_persistent_depth instead,
        # which needs no CPU arm and therefore reaches depth 9 in minutes.
        n = t7(a.build_dir, a.gpu_build_dir, [5, 6, 7], a.iters)
        print("T7: %d depths" % n)
    write_values()
    return 0


if __name__ == "__main__":
    sys.exit(main())
