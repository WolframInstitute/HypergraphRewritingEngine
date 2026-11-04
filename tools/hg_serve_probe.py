#!/usr/bin/env python3
"""Probe the engine binary's --serve (worker) mode and its per-job latency.

Sends N length-prefixed WXF jobs to `<binary> --serve` and reads the
length-prefixed results, reporting the time for each. Confirms the framing and,
for the GPU binary, that the CUDA context is created once (first job) and
amortised across the rest.

Frame (both directions): [8-byte little-endian length][payload]. On the worker's
output, the top three bits of the length mark progress, error and log frames
(paclet_source/frame_writer.hpp); the probe skips progress and log frames and
times each job to its reply frame or its error frame.

Usage:
    tools/hg_serve_probe.py <binary> <job.wxf> [num_jobs] [--log]

--log prints each progress and log frame the worker sends.

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


# paclet_source/frame_writer.hpp: bit 63 marks a progress frame, bit 62 an error frame and bit 61
# a log frame; a frame with none of the three is the job's reply.
PROGRESS_BIT, ERROR_BIT, LOG_BIT = 1 << 63, 1 << 62, 1 << 61
KIND_BITS = PROGRESS_BIT | ERROR_BIT | LOG_BIT


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
    show_log = "--log" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--log"]
    exe, job_file = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 8
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
        while True:
            head = struct.unpack("<Q", read_exact(p.stdout, 8))[0]
            kind, rlen = head & KIND_BITS, head & ~KIND_BITS
            payload = read_exact(p.stdout, rlen) if rlen else b""
            if kind & ERROR_BIT:
                print(f"job {i}: error: {payload.decode('utf-8', 'replace')[:200]}")
                break
            if kind == 0:
                break
            if show_log:
                print(f"  log: {payload.decode('utf-8', 'replace').rstrip()[:300]}")
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
