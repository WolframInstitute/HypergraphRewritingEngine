"""Child process of test_memory_limit.py: hg.evolve calls one after another in one process, as an
engine worker that imports the module once runs them. A 6-step job, the growth job of
memory_limit_child.py under "MemoryLimit" -> 64 MB three times, and the 6-step job again. Prints
each call's state count, whether it was truncated, and VmRSS after it, as JSON."""
import json
import sys
import warnings

import hgevolve as hg

LIMIT = 64 << 20


def rss_kb():
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))


calls = []
for steps, limited in ((6, False), (12, True), (12, True), (12, True), (6, False)):
    options = {"tuning": {"MemoryLimit": LIMIT}} if limited else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = hg.evolve(([[1, 2]], [[1, 2], [2, 3]]), [[1, 2], [2, 3]], steps, threads=2,
                      data=["num_states"], **options)
    calls.append({"states": r.num_states, "rss_kb": rss_kb(),
                  "truncated": any("TRUNCATED" in str(w.message) for w in caught)})
print(json.dumps(calls), flush=True)
sys.exit(0)
