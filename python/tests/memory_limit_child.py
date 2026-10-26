"""Child process of test_memory_limit.py: a growing multiway evolution under "MemoryLimit" -> 64 MB
in "Tuning", with the address space capped 1 GB above the process's size, so a run the limit does
not stop ends at the cap. Prints the counts, the device's limit and the warnings as JSON."""
import json
import resource
import sys
import warnings

import hgevolve as hg

LIMIT = 64 << 20
with open("/proc/self/status") as f:
    vm_kb = next(int(line.split()[1]) for line in f if line.startswith("VmSize:"))
cap = (vm_kb + (1 << 20)) * 1024
resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = hg.evolve(([[1, 2]], [[1, 2], [2, 3]]), [[1, 2], [2, 3]], 12, threads=2,
                  tuning={"MemoryLimit": LIMIT}, data=["num_states", "num_events", "device"])
print(json.dumps({"num_states": r.num_states, "num_events": r.num_events,
                  "memory_limit": r.device.get("MemoryLimit"),
                  "warnings": [str(w.message) for w in caught]}), flush=True)
sys.exit(0)
