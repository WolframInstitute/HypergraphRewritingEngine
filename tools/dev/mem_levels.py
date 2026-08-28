#!/usr/bin/env python3
"""Bytes moved at each level of the memory hierarchy, and where they came from.

    tools/dev/mem_levels.py -- ./build_linux/bench_cpu_evolve <args...>

WHY FILL SOURCES RATHER THAN MISS COUNTS. A miss count says a level did not have the line; it does
not say who supplied it, and on a chiplet part that is the whole question -- a line served by this
core's own L3 and a line dragged across the fabric from another CCX cost differently and are
different problems. Zen counts the SOURCE of every fill, so one run separates own-L2, this CCX's
L3, a neighbouring CCX, a remote socket and DRAM, near and far. Each fill is one 64-byte line, so
the counts convert to bytes directly.

WHY NOT CACHEGRIND. It models two levels (D1 and LL) and a fixed machine, so it cannot answer the
cross-CCX question at all, and its LL is not this part's L3. It stays the right tool for
attributing misses to SOURCE LINES deterministically; this one measures what the hardware actually
moved.

WHERE IT RUNS. Zen only, and not under WSL -- that kernel ships no usable perf. Both are checked
and reported rather than degrading to a table of zeros, because a bandwidth number that is
silently zero is worse than no number.

perf_event_paranoid must be <= 2 to measure your own process; the rented EPYC has it at 1.
"""
import argparse
import shutil
import subprocess
import time
import sys

LINE_BYTES = 64

# Ordered nearest-first: the table reads as a distance ladder, which is the point.
LEVELS = [
    ("ls_any_fills_from_sys.local_l2", "own L2"),
    ("ls_any_fills_from_sys.local_ccx", "L3, this CCX"),
    ("ls_any_fills_from_sys.near_cache", "cache, near CCX"),
    ("ls_any_fills_from_sys.far_cache", "cache, far CCX"),
    ("ls_any_fills_from_sys.remote_cache", "cache, remote socket"),
    ("ls_any_fills_from_sys.dram_io_near", "DRAM, near"),
    ("ls_any_fills_from_sys.dram_io_far", "DRAM, far"),
]
CORE = [
    ("cycles", "cycles"),
    ("instructions", "instructions"),
    ("L1-dcache-loads", "L1 loads"),
    ("L1-dcache-load-misses", "L1 load misses"),
]


def check_host():
    if shutil.which("perf") is None:
        sys.exit("mem_levels: no perf on this host (WSL ships none usable); run on the bare-metal box")
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            if int(f.read().strip()) > 2:
                sys.exit("mem_levels: perf_event_paranoid > 2, cannot count this process")
    except FileNotFoundError:
        pass


def _one_group(events, cmd, repeat):
    """One perf invocation per counter group.

    THE GROUPS ARE NOT COSMETIC. This part has six general-purpose counters; asking for eleven
    events at once makes perf refuse the surplus and report them as <not supported>, which reads
    identically to an event the silicon does not have. Measured: cycles, instructions and the
    L1 events all came back "unsupported" beside seven fill-source events that counted fine.
    Two runs of the same command, each inside the counter budget, and nothing is multiplexed --
    so every number here is a full count rather than a scaled estimate.
    """
    argv = ["perf", "stat", "-x,", "-r", str(repeat), "-e", ",".join(events), "--"] + cmd
    t0 = time.perf_counter()
    p = subprocess.run(argv, capture_output=True, text=True)
    wall = (time.perf_counter() - t0) / max(repeat, 1)
    counts, unsupported = {}, []
    for line in p.stderr.splitlines():
        f = line.split(",")
        if len(f) < 3:
            continue
        value, event = f[0], f[2]
        if value.startswith("<not"):
            unsupported.append(event)
            continue
        try:
            counts[event] = float(value)
        except ValueError:
            pass
    return counts, unsupported, wall, p.returncode


def run(cmd, repeat):
    counts, unsupported, wall, rc = _one_group([e for e, _ in LEVELS], cmd, repeat)
    c2, u2, _, rc2 = _one_group([e for e, _ in CORE], cmd, repeat)
    counts.update(c2)
    unsupported += u2
    # Wall clock is taken around perf itself rather than parsed from it: the -x, format does not
    # carry the elapsed line, and a GB/s computed from a missing denominator is worse than none.
    return counts, unsupported, wall, (rc or rc2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--repeat", type=int, default=1)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("mem_levels: give a command after --")

    check_host()
    counts, unsupported, elapsed, rc = run(cmd, a.repeat)
    if unsupported:
        # Named rather than dropped: a level silently missing from the table would read as zero
        # traffic at that level, which is a different and much more attractive claim.
        print(f"UNSUPPORTED on this part, omitted: {', '.join(unsupported)}\n", file=sys.stderr)

    total = sum(counts.get(e, 0.0) for e, _ in LEVELS)
    print(f"{'source':<24} {'fills':>14} {'bytes':>16} {'share':>7}"
          + (f" {'GB/s':>9}" if elapsed else ""))
    for event, label in LEVELS:
        if event not in counts:
            continue
        n = counts[event]
        b = n * LINE_BYTES
        row = f"{label:<24} {n:>14,.0f} {b:>16,.0f} {(100 * n / total if total else 0):>6.1f}%"
        if elapsed:
            row += f" {b / elapsed / 1e9:>9.2f}"
        print(row)
    print(f"{'-' * 24} {'-' * 14} {'-' * 16} {'-' * 7}")
    tb = total * LINE_BYTES
    row = f"{'all fills':<24} {total:>14,.0f} {tb:>16,.0f} {100.0:>6.1f}%"
    if elapsed:
        row += f" {tb / elapsed / 1e9:>9.2f}"
    print(row)

    print()
    for event, label in CORE:
        if event in counts:
            print(f"{label:<24} {counts[event]:>14,.0f}")
    if elapsed:
        print(f"{'elapsed (s)':<24} {elapsed:>14.4f}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
