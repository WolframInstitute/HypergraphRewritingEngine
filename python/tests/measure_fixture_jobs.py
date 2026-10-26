#!/usr/bin/env python3
"""Measure the fixture jobs generate_fixtures.wls captured, with hg_evolve under a memory cap,
before any of them runs in the paclet's engine (which no cap reaches on Windows).

    python3 measure_fixture_jobs.py HG_EVOLVE DIR [CAP_MB] [TIMEOUT_S]

Runs DIR/job<i>.wxf one at a time, each under an address-space cap of CAP_MB (default 2048) and
TIMEOUT_S seconds (default 60), and writes DIR/measured.json: per job its exit code, peak
resident memory (KB) and reply bytes. Prints one line per job as it finishes."""
import json
import os
import re
import resource
import subprocess
import sys


def main():
    binary, d = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    jobs = sorted((int(m.group(1)), f) for f in os.listdir(d)
                  if (m := re.fullmatch(r"job(\d+)\.wxf", f)))

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (cap << 20, cap << 20))

    out = {}
    for i, f in jobs:
        with open(os.path.join(d, f), "rb") as job:
            try:
                p = subprocess.run(["/usr/bin/time", "-f", "PEAK %M", binary], stdin=job,
                                   capture_output=True, timeout=timeout, preexec_fn=limit)
                code = p.returncode
                m = re.search(rb"PEAK (\d+)", p.stderr)
                peak = int(m.group(1)) if m else None
                size = len(p.stdout)
            except subprocess.TimeoutExpired:
                code, peak, size = "timeout", None, None
        out[i] = {"exit": code, "peak_kb": peak, "reply_bytes": size}
        print(i, code, peak, size, flush=True)
    with open(os.path.join(d, "measured.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
