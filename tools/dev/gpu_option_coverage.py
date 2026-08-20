#!/usr/bin/env python3
"""An option the device does not implement must say so.

The GPU backend implements a subset of the engine's options. Where it does not, the FFI pushes
an OptionSkipped warning, because silently returning a different answer per device is the
divergence class the differential suite exists to catch -- a caller who asked for a cap and did
not get one is told, rather than handed an uncapped result that reads as the real system.

Nothing enforced that. "MatchesPerStateRule" was parsed into ParsedJob, applied to the CPU
engine, and never mentioned again: not carried to the device, not warned about. A GPU run with
it set returned the uncapped state set in silence, while the documentation names that option as
the reproducible alternative to the two caps that DO warn.

An option is covered when run_gpu_job either reads the ParsedJob field it writes -- carrying it
to the device or testing it -- or names the option in a warning. Reads sources; no build.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FFI = ROOT / "paclet_source/hypergraph_ffi.cpp"

# Options whose effect is not a ParsedJob field the device could carry or skip. Each is listed
# with the reason it cannot be checked mechanically, so the exemption is auditable.
EXEMPT = {
    # Selects which properties the reply carries. The device path builds its reply from the
    # include_* flags this sets, which are checked individually as GpuJob fields.
    "RequestedData",
    "GraphProperties",
}


def block_of(text: str, start: int) -> str:
    """The brace-balanced body following the `if (...)` at `start`."""
    i = text.find("{", start)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return text[i:]


def main():
    ffi = FFI.read_text()

    # Every option the parser reads, and the ParsedJob fields its block writes.
    opts = {}
    for m in re.finditer(r'option_key == "([A-Za-z0-9]+)"\s*\)', ffi):
        name = m.group(1)
        body = block_of(ffi, m.end())
        opts[name] = set(re.findall(r"\breq\.([a-z_0-9]+)\s*=", body))

    if len(opts) < 15:
        print(f"FAIL: only {len(opts)} option keys parsed from {FFI.name}; the parser this "
              f"reads must have changed shape.")
        return 1

    start = ffi.index("static std::vector<uint8_t> run_gpu_job(")
    end = ffi.index("\nstatic ", start + 10)
    gpu = ffi[start:end]

    uncovered = []
    for name, fields in sorted(opts.items()):
        if name in EXEMPT:
            continue
        carried = any(re.search(r"\breq\." + re.escape(f) + r"\b", gpu) for f in fields)
        warned = f"'{name}'" in gpu
        if not carried and not warned:
            uncovered.append((name, fields))

    if uncovered:
        print(f"{len(uncovered)} option(s) reach the CPU engine but neither the device nor an "
              f"OptionSkipped warning:\n")
        for name, fields in uncovered:
            fs = ", ".join(sorted(fields)) or "(no ParsedJob field)"
            print(f'  "{name}"  writes req.{{{fs}}}')
        print("\nEither carry it to the GpuJob, or push an OptionSkipped warning naming it.")
        return 1

    print(f"OK: all {len(opts) - len(EXEMPT & opts.keys())} device-relevant options are "
          f"carried to the GPU job or warned about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
