#!/usr/bin/env python3
"""Scaling, not a ratio: how the engine responds to more cores, more blocks, and more work.

WHY THIS EXISTS. A speedup quoted at one thread count and one depth says nothing about whether
the design scales -- it is one point on a curve that could be flat, rising or already falling.
The three questions this answers are the ones a ratio cannot:

  STRONG SCALING   fix the work, add threads. The column that matters is EFFICIENCY
                   (speedup / threads): where it falls away is where the parallelism stops
                   paying, and stopping the sweep before that point hides it.
  INPUT DEPENDENCE run the same sweep at several depths. A design can scale on a big frontier
                   and not on a small one, and reporting only the big one is a choice of input.
  DEVICE SATURATION the persistent kernel's grid is a knob (HG_GPU_PERSISTENT_BLOCKS). Sweeping
                   it answers "does the device saturate, and where" behaviourally: time stops
                   improving when it does. SM utilisation is sampled alongside, coarsely -- the
                   driver reports a 1 Hz average, so it bounds the answer rather than settling
                   it, and this file says so rather than printing it as an occupancy figure.

Each section is independent and is selected by name, because they do not cost the same: the
rule-shape sweep is a minute and the CPU curve is an hour, so binding them together would mean
paying for the second to re-run the first.

  WHERE IT GOES   a falling efficiency curve does not name its cause. user vs system time,
                   resident set and page faults, read from getrusage on the same runs, separate
                   "the workers are contending" from "the run is paying for memory per worker".

Run:  python3 tools/dev/scaling_sweep.py --sections shapes[,cpu,memory,gpu] [--steps 5,6,7]
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "dev"))
import paper_tables as pt  # provenance, tex_escape, write -- one implementation of each
import procstat            # machine facts and child resource usage, one implementation of each

NVIDIA_SMI = procstat.nvidia_smi_path() or "/usr/lib/wsl/lib/nvidia-smi"



def run(cmd, env=None, timeout=3600):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=e)
    if p.returncode != 0:
        raise SystemExit("FAILED (%d): %s\n%s" % (p.returncode, " ".join(cmd), p.stderr[-2000:]))
    return p.stdout


class UtilisationSampler:
    """SM utilisation while something else runs, sampled from the driver at ~5 Hz.

    This is the driver's own coarse average, not a per-kernel occupancy figure: it says whether
    the device is busy, not how many warps are resident. Treated as a bound, never as occupancy.

    IT ONLY MEANS ANYTHING IF THE SAMPLED RUN IS LONG. One evolution here is tens of
    milliseconds and the driver updates on the order of tens of milliseconds too, so a handful
    of iterations gives a mean dominated by the gaps between them, not by the kernel. The caller
    therefore runs enough iterations to make the busy window seconds long, and the sample COUNT
    is reported next to the mean so a reader can see whether the window was long enough.
    """

    def __init__(self, path=NVIDIA_SMI):
        self.path = path
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def available(self):
        return os.path.exists(self.path)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [self.path, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    self.samples.append(int(out.split("\n")[0]))
            except (subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(0.05)

    def __enter__(self):
        if self.available():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        busy = [s for s in self.samples if s > 0]
        if not busy:
            return None
        return max(busy), sum(busy) / len(busy), len(busy)


ROW_RE = re.compile(r"threads=(\d+)\s+steps=(\d+)\s+canonical=(\d+)\s+raw=(\d+)\s+"
                    r"median_ms=([\d.]+)\s+min_ms=([\d.]+)\s+speedup=([\d.]+)\s+"
                    r"efficiency=([\d.]+)")


def cpu_scaling(build, steps_list, iters, cpus="", sweep=""):
    """One strong-scaling curve per depth, so the input axis is visible next to the thread axis.

    PINNED WHEN A CPU SET IS GIVEN, because this is the table the paper renders. It writes
    t8_scaling.tex, the same name paper_tables.py's own scaling() writes, and it runs second --
    so the pinned six-column table that generator produced was being replaced by this one, and
    the paper's strong-scaling figure was the UNPINNED measurement. A speedup column taken
    across cores of different speeds divides by a quantity that is not compute, which is the
    whole reason the pin set exists.
    """
    per_depth = {}
    for steps in steps_list:
        argv = [os.path.join(build, "bench_cpu_evolve"), str(steps), str(iters)]
        if cpus:
            # bench_cpu_evolve is positional: steps iters sweep workload cpus.
            argv += [sweep or "", "wpp", cpus]
        out = run(argv, timeout=7200)
        rows = [m.groups() for m in (ROW_RE.search(l) for l in out.splitlines()) if m]
        if not rows:
            raise SystemExit("bench_cpu_evolve produced no rows:\n%s" % out[-2000:])
        per_depth[steps] = rows
        print("  depth %d: %s" % (steps, " ".join("%st=%.2fx" % (r[0], float(r[6])) for r in rows)))

    threads = [r[0] for r in per_depth[steps_list[0]]]
    b = [pt.provenance("tools/bench_cpu_evolve.cpp", pinned=bool(cpus)),
         r"\begin{tabular}{r" + "rr" * len(steps_list) + "}", r"\toprule",
         "Threads & " + " & ".join(r"\multicolumn{2}{c}{depth %d}" % s for s in steps_list) + r" \\",
         " & " + " & ".join("ms & eff." for _ in steps_list) + r" \\", r"\midrule"]
    for i, t in enumerate(threads):
        cells = []
        for s in steps_list:
            r = per_depth[s][i]
            cells.append("%.1f & %.2f" % (float(r[4]), float(r[7])))
        b.append("%s & %s \\\\" % (t, " & ".join(cells)))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t8_scaling.tex", "\n".join(b) + "\n")
    return per_depth


def gpu_saturation(gpu_build, steps, iters, sm_count):
    """Wall time against the persistent grid: where it stops falling is where the device is full."""
    rows = []
    for mult in (1, 2, 4, 8, 16):
        blocks = sm_count * mult
        with UtilisationSampler() as sampler:
            out = run([os.path.join(gpu_build, "bench_gpu_evolve"), str(steps), str(iters), "2"],
                      env={"HG_GPU_PERSISTENT_BLOCKS": str(blocks)}, timeout=3600)
            util = sampler.summary()
        m = re.search(r"^\s*(\S+)\s+([\d.]+)\s+(\d+)\s+(\d+)", out, re.M)
        if not m:
            raise SystemExit("bench_gpu_evolve mode 2 gave no row:\n%s" % out[-2000:])
        ms = float(m.group(2))
        rows.append((mult, blocks, ms, util))
        print("  %d blocks/SM (%d): %.2f ms%s" % (
            mult, blocks, ms, "" if not util else "  util peak %d%% mean %.0f%%" % (util[0], util[1])))

    base = rows[0][2]
    b = [pt.provenance("tools/bench_gpu_evolve.cpp (mode 2) + driver utilisation sampling"),
         r"\begin{tabular}{rrrrl}", r"\toprule",
         r"Blocks per SM & Blocks & Median ms & vs.\ 1 per SM & Peak SM-busy \\",
         r"\midrule"]
    for (mult, blocks, ms, util) in rows:
        # PEAK ONLY. The mean over a run window falls as the run gets FASTER -- the same busy
        # kernel occupies a smaller share of the window -- so it tracks duration, not residency,
        # and reporting it would invite the reading it cannot support.
        u = "--" if not util else "%d\\%%" % util[0]
        b.append("%d & %d & %.2f & %.2f$\\times$ & %s \\\\" % (mult, blocks, ms, base / ms, u))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t11_gpu_saturation.tex", "\n".join(b) + "\n")
    # The conclusion states the grid size beyond which time stops improving; it is the smallest
    # multiplier whose median is within one percent of the best, which is what "ceases to
    # improve" means when the remaining differences are smaller than the run-to-run spread.
    best = min(ms for (_m, _b, ms, _u) in rows)
    pt.value("SaturationBlocks", "%d" % next(m for (m, _b, ms, _u) in rows if ms <= best * 1.01))

    # The right panel of the GPU figure plots the median against the grid size.
    pts = "".join("(%d,%.2f)" % (mult, ms) for (mult, _blocks, ms, _u) in rows)
    pt.write_raw("f_gpu_saturation.tex",
                 pt.provenance("tools/bench_gpu_evolve.cpp (mode 2)") + "\n"
                 + r"\addplot[mark=*, black] coordinates {%s};" % pts + "\n")
    return rows


def rule_shape_scaling(build, shapes, iters):
    """Does the scaling hold across rule SHAPES, or only on the one the headline uses?

    A rule set decides how much parallel work exists: a single-edge LHS re-matches by scanning an
    index and every state's expansion is small, while a two-edge LHS joins and produces a wider
    frontier. Reporting the shape that scales best is a choice of input, so both are swept at
    every thread count and the table carries EFFICIENCY, which is where the difference shows.

    SIZED SO THE ANSWER IS NOT FIXED COST. A sweep whose serial point is tens of milliseconds
    cannot distinguish "the parallelism stopped paying" from "thread setup and the final drain
    are now the whole run". The depths here put the serial point in the seconds-to-minutes range,
    and the same shape of curve was confirmed at a size 100x smaller, so the ceiling this reports
    is a property of the engine at both sizes rather than of one problem.

    Uses tools/sampling_cost_smoke, which already takes (arm, rule, edges, steps, threads, k,
    canon) -- no new instrument for a question an existing one answers.
    """
    rows = {}
    for (rule, edges, steps) in shapes:
        for threads in (1, 2, 4, 8, 16, 24, 32):
            # MIN over repeats, not mean. Contention on a shared box can only make a run
            # SLOWER, so the minimum is the estimate that interference cannot inflate; a mean
            # would report this machine's other tenants as this engine's scaling.
            #
            # REPEATS ARE SPENT WHERE THEY BUY SOMETHING. Scheduling noise here is on the order
            # of milliseconds, so it is a large share of a 70 ms point and none of a 90 s one.
            # A point that already ran for longer than LONG_RUN_S is therefore taken once, which
            # is what makes a serial point of minutes affordable in the same sweep.
            LONG_RUN_S = 10.0
            best = None
            for _ in range(max(3, iters // 2)):
                out = run([os.path.join(build, "sampling_cost_smoke"), "off", rule, str(edges),
                           str(steps), str(threads), "4", "full"], timeout=7200)
                m = re.search(r"done ([\d.]+) ms\s+states=(\d+).*events=(\d+)", out)
                if not m:
                    raise SystemExit("sampling_cost_smoke gave no row:\n%s" % out[-1000:])
                ms = float(m.group(1))
                if best is None or ms < best[0]:
                    best = (ms, m.group(2), m.group(3))
                if ms > LONG_RUN_S * 1000.0:
                    break
            rows.setdefault(rule, []).append((threads, best[0], best[1], best[2]))
            base = rows[rule][0][1]
            print("  %-7s %2dt: %9.1f ms  %5.2fx  eff %.2f" % (
                rule, threads, best[0], base / best[0], (base / best[0]) / threads))

    b = [pt.provenance("tools/sampling_cost_smoke.cpp"),
         r"\begin{tabular}{r" + "rr" * len(rows) + "}", r"\toprule",
         "Threads & " + " & ".join(r"\multicolumn{2}{c}{%s LHS}" % ("one-edge" if r == "growth"
                                                                   else "two-edge")
                                   for r in rows) + r" \\",
         " & " + " & ".join("ms & eff." for _ in rows) + r" \\", r"\midrule"]
    order = list(rows)
    for i in range(len(rows[order[0]])):
        cells = []
        for r in order:
            t, ms, _st, _ev = rows[r][i]
            base = rows[r][0][1]
            cells.append("%.1f & %.2f" % (ms, (base / ms) / t))
        b.append("%d & %s \\\\" % (rows[order[0]][i][0], " & ".join(cells)))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t12_rule_shapes.tex", "\n".join(b) + "\n")

    # The conclusion quotes the efficiency floor at eight and sixteen workers across rule shapes.
    # The floor rather than either shape's own figure, because the sentence says "depending on
    # the rule's shape" and the weaker shape is what that qualifies.
    # Spelled out, because a LaTeX control sequence cannot contain a digit.
    for at, word in ((8, "Eight"), (16, "Sixteen")):
        effs = []
        for r in rows:
            base = rows[r][0][1]
            effs += [(base / ms) / t for (t, ms, _s, _e) in rows[r] if t == at]
        if effs:
            pt.value("EffAt" + word, "%.2f" % min(effs))

    # NO f_efficiency FRAGMENT. This block used to draw a two-curve efficiency panel of the
    # same axis Figure~\ref{fig:lhs-scaling} now sweeps at four sizes with a stated method, and
    # the paper stopped inputting it. Two live generators for one measurement is how the table
    # and the figure drifted into stating opposite trends in the first place, so the second one
    # is deleted rather than left writing a fragment nothing reads. The macros above
    # (EffAtEight, EffAtSixteen) are still emitted -- the conclusion cites them.
    return rows


def thread_memory_cost(build, shape, iters):
    """WHERE the efficiency goes when it stops paying, split into user and kernel time.

    A falling efficiency curve does not say what is consuming the threads, and the two candidate
    answers demand opposite fixes: if the workers are contending, they burn USER time doing
    redundant work; if the run is paying for memory it did not need, they burn SYSTEM time
    faulting pages in. Those are distinguishable with no new instrument -- getrusage already
    reports both, plus the fault count and the peak resident set -- so this reports all four
    against thread count and lets the numbers choose.

    Read the columns together: user time roughly flat while system time and minor faults climb
    with the resident set means the ceiling is the cost of memory the run acquires per worker,
    not threads fighting over a data structure.
    """
    rows = []
    rule, edges, steps = shape
    for threads in (1, 8, 16, 24, 32):
        best = None
        for _ in range(max(2, iters // 4)):
            m = procstat.measure(
                [os.path.join(build, "sampling_cost_smoke"), "off", rule,
                 str(edges), str(steps), str(threads), "4", "full"],
                cwd=ROOT, timeout=7200)
            if m.returncode != 0:
                raise SystemExit("sampling_cost_smoke failed:\n%s" % m.stderr[-2000:])
            u = m.usage
            # The fault column is a DIFFERENT quantity per platform -- Linux counts minor faults,
            # Windows counts all of them -- so the column is labelled from the measurement rather
            # than from a literal, and a table says which it holds.
            fault_kind = u.fault_kind
            row = (threads, u.wall, u.user, u.system, u.peak_rss_mb, float(u.faults))
            if best is None or row[1] < best[1]:
                best = row
        rows.append(best)
        print("  %2dt: wall %6.2f s  user %6.2f s  sys %6.2f s  RSS %7.0f MB  %s %9.0f"
              % (best[0], best[1], best[2], best[3], best[4], fault_kind.lower(), best[5]))

    b = [pt.provenance("tools/sampling_cost_smoke.cpp under procstat.measure"),
         r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"Threads & Wall s & User s & System s & Peak RSS MB & %s \\" % fault_kind,
         r"\midrule"]
    for (t, wall, u, s, rss, mf) in rows:
        b.append("%d & %.2f & %.2f & %.2f & %.0f & %s \\\\"
                 % (t, wall, u, s, rss, "{:,}".format(int(mf)).replace(",", "\\,")))
    b += [r"\bottomrule", r"\end{tabular}"]
    pt.write("t13_thread_memory.tex", "\n".join(b) + "\n")

    # The conclusion contrasts how user and system time grow across the thread sweep. Both are
    # ratios of the last row to the first, which is the comparison the sentence makes.
    if len(rows) > 1 and rows[0][2] and rows[0][3]:
        pt.value("UserTimeGrowth", "%.1f" % (rows[-1][2] / rows[0][2]))
        pt.value("SysTimeGrowth", "%.0f" % (rows[-1][3] / rows[0][3]))
        pt.value("ThreadSweepFactor", "%d" % (rows[-1][0] // rows[0][0]))

    # Right panel of the scaling figure: user against system time, from the same getrusage rows.
    user_pts = "".join("(%d,%.2f)" % (t, u) for (t, _w, u, _s, _r, _m) in rows)
    sys_pts = "".join("(%d,%.2f)" % (t, s) for (t, _w, _u, s, _r, _m) in rows)
    pt.write_raw("f_thread_memory.tex",
                 pt.provenance("tools/sampling_cost_smoke.cpp under procstat.measure") + "\n"
                 + r"\addplot[mark=*, black] coordinates {%s};" % user_pts + "\n"
                 + r"\addlegendentry{user}" + "\n"
                 + r"\addplot[mark=triangle*, black, dashed] coordinates {%s};" % sys_pts + "\n"
                 + r"\addlegendentry{system}" + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_linux")
    ap.add_argument("--gpu-build-dir", default="build_gpu")
    ap.add_argument("--steps", default="5,6,7")
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--sm-count", type=int, default=128)   # RTX 4090
    ap.add_argument("--cpus", default="",
                    help="logical CPUs to pin workers to; the table says so in its stamp")
    ap.add_argument("--thread-sweep", default="",
                    help="thread counts; must not exceed --cpus")
    ap.add_argument("--shape-depths", default="growth:1:9,pair:4:7",
                    help="rule:edges:steps triples for the rule-shape sweep")
    ap.add_argument("--sections", default="cpu",
                    help="comma-separated subset of: cpu, shapes, memory, gpu")
    a = ap.parse_args()

    # SEED THE STAMP THIS GENERATOR BORROWS. Its tables go through paper_tables.provenance(),
    # whose CONTENDED marker reads paper_tables._BASELINE_LOAD -- a module global that only
    # paper_tables' own main() was setting. Unset, _load_note() returns "" and every table
    # scaling_sweep writes (t8, t11, t12, t13) claimed nothing about the load it was taken
    # under, while remote_session.sh advertised that every table is stamped.
    try:
        pt._BASELINE_LOAD = os.getloadavg()
    except (OSError, AttributeError):
        pt._BASELINE_LOAD = None
    pt._PINNED_CPUS = a.cpus

    sections = [x.strip() for x in a.sections.split(",") if x.strip()]
    unknown = [x for x in sections if x not in ("cpu", "shapes", "memory", "gpu")]
    if unknown:
        raise SystemExit("unknown section(s): %s" % ", ".join(unknown))

    a.shape_depths = [(r, int(e), int(d))
                      for r, e, d in (x.split(":") for x in a.shape_depths.split(","))]
    steps_list = [int(s) for s in a.steps.split(",") if s.strip()]
    if "cpu" in sections:
        print("CPU strong scaling, depths %s" % steps_list)
        cpu_scaling(a.build_dir, steps_list, a.iters, a.cpus, a.thread_sweep)
    if "shapes" in sections:
        print("Rule-shape scaling")
        rule_shape_scaling(a.build_dir, a.shape_depths, a.iters)
    if "memory" in sections:
        print("Where the efficiency goes: user vs kernel time, resident set, page faults")
        thread_memory_cost(a.build_dir, a.shape_depths[0], a.iters)
    if "gpu" in sections:
        print("GPU saturation, depth %d" % steps_list[-1])
        gpu_saturation(a.gpu_build_dir, steps_list[-1], a.iters, a.sm_count)
    pt.write_values("values_sweep.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
