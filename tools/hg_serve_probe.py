#!/usr/bin/env python3
"""Probe the engine binary's --serve (worker) mode and its per-job latency.

Sends N length-prefixed WXF jobs to `<binary> --serve` and reads the
length-prefixed results, reporting the time for each. Confirms the framing and,
for the GPU binary, that the CUDA context is created once (first job) and
amortised across the rest.

Frame (both directions): [8-byte little-endian length][payload]. A zero-length
reply frame means that job errored (detail on the binary's stderr).

Usage:
    tools/hg_serve_probe.py <binary> <job.wxf> [num_jobs]

The <job.wxf> is a BinarySerialize'd HGEvolve input association (as the paclet
sends); generate one from Wolfram, e.g.
    BinaryWrite[OpenWrite["job.wxf", BinaryFormat -> True],
      BinarySerialize[<|"InitialStates" -> {init}, "Rules" -> rulesAssoc,
                        "Steps" -> n, "Options" -> <||>|>]]
"""
import struct
import subprocess
import sys
import time


def read_exact(f, k):
    buf = b""
    while len(buf) < k:
        chunk = f.read(k - len(buf))
        if not chunk:
            raise EOFError("short read (worker exited?)")
        buf += chunk
    return buf


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exe, job_file = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with open(job_file, "rb") as f:
        job = f.read()

    p = subprocess.Popen([exe, "--serve"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    times, sizes = [], []
    for i in range(n):
        t0 = time.time()
        p.stdin.write(struct.pack("<Q", len(job)))
        p.stdin.write(job)
        p.stdin.flush()
        rlen = struct.unpack("<Q", read_exact(p.stdout, 8))[0]
        _ = read_exact(p.stdout, rlen) if rlen else b""
        times.append((time.time() - t0) * 1000.0)
        sizes.append(rlen)
        print(f"job {i}: {times[-1]:8.1f} ms   result {rlen} bytes")
    p.stdin.close()
    p.wait()

    if len(set(sizes)) != 1:
        print(f"WARN: result sizes differ across jobs: {sorted(set(sizes))}")
    if n > 1:
        print(f"\nfirst job: {times[0]:.1f} ms   "
              f"mean of rest: {sum(times[1:]) / (n - 1):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
