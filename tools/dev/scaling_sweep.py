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

Run:  python3 tools/dev/scaling_sweep.py [--gpu] [--steps 5,6,7] [--iters 9]
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

NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


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


def cpu_scaling(build, steps_list, iters):
    """One strong-scaling curve per depth, so the input axis is visible next to the thread axis."""
    per_depth = {}
    for steps in steps_list:
        out = run([os.path.join(build, "bench_cpu_evolve"), str(steps), str(iters)], timeout=7200)
        rows = [m.groups() for m in (ROW_RE.search(l) for l in out.splitlines()) if m]
        if not rows:
            raise SystemExit("bench_cpu_evolve produced no rows:\n%s" % out[-2000:])
        per_depth[steps] = rows
        print("  depth %d: %s" % (steps, " ".join("%st=%.2fx" % (r[0], float(r[6])) for r in rows)))

    threads = [r[0] for r in per_depth[steps_list[0]]]
    b = [pt.provenance("tools/bench_cpu_evolve.cpp"),
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
    return rows


def rule_shape_scaling(build, shapes, iters):
    """Does the scaling hold across rule SHAPES, or only on the one the headline uses?

    A rule set decides how much parallel work exists: a single-edge LHS re-matches by scanning an
    index and every state's expansion is small, while a two-edge LHS joins and produces a wider
    frontier. Reporting the shape that scales best is a choice of input, so both are swept at
    every thread count and the table carries EFFICIENCY, which is where the difference shows.

    Uses tools/sampling_cost_smoke, which already takes (arm, rule, edges, steps, threads, k,
    canon) -- no new instrument for a question an existing one answers.
    """
    rows = {}
    for (rule, edges, steps) in shapes:
        for threads in (1, 2, 4, 8, 16, 24, 32):
            best = None
            for _ in range(max(2, iters // 4)):
                out = run([os.path.join(build, "sampling_cost_smoke"), "off", rule, str(edges),
                           str(steps), str(threads), "4", "full"], timeout=3600)
                m = re.search(r"done ([\d.]+) ms\s+states=(\d+).*events=(\d+)", out)
                if not m:
                    raise SystemExit("sampling_cost_smoke gave no row:\n%s" % out[-1000:])
                ms = float(m.group(1))
                if best is None or ms < best[0]:
                    best = (ms, m.group(2), m.group(3))
            rows.setdefault(rule, []).append((threads, best[0], best[1], best[2]))
            print("  %-7s %2dt: %8.1f ms" % (rule, threads, best[0]))

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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_linux")
    ap.add_argument("--gpu-build-dir", default="build_gpu")
    ap.add_argument("--steps", default="5,6,7")
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--sm-count", type=int, default=128)   # RTX 4090
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--shapes", action="store_true",
                    help="add the rule-shape sweep (T12)")
    a = ap.parse_args()

    steps_list = [int(s) for s in a.steps.split(",") if s.strip()]
    print("CPU strong scaling, depths %s" % steps_list)
    cpu_scaling(a.build_dir, steps_list, a.iters)
    if a.shapes:
        print("Rule-shape scaling")
        rule_shape_scaling(a.build_dir, [("growth", 1, 8), ("pair", 4, 6)], a.iters)
    if a.gpu:
        print("GPU saturation, depth %d" % steps_list[-1])
        gpu_saturation(a.gpu_build_dir, steps_list[-1], a.iters, a.sm_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
