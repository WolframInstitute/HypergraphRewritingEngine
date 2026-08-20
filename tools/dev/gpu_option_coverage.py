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

The same gap exists on the OUTPUT surface. "BranchialStateEdges", "BranchialStateEdgesAllSiblings",
"GlobalEdges" and "StateBitvectors" are built by the FFI for the host path and appear nowhere in
the GPU backend, so a device reply carried no such key and the caller got a shorter association
than the identical CPU call returns, again with nothing said.

An option is covered when run_gpu_job either reads the ParsedJob field it writes -- carrying it
to the device or testing it -- or names the option in a warning. A requested-data component is
covered when the GPU backend emits its key, or run_gpu_job names it in a warning. Reads sources;
no build, and no device.
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

BACKEND = ROOT / "paclet_source/hg_gpu_backend.cpp"
# A component emitted by the SHARED marshaller is emitted by both engines by construction, so
# its key is looked for there as well as in the backend -- that is the whole point of the rule
# living there rather than being spelled once per engine.
SHARED = ROOT / "paclet_source/graph_marshal.hpp"

# A field CARRIED into the GpuJob is not thereby honoured: "ShowGenesisEvents" was packed into
# every device job and read by nothing, so the device answered a request to show genesis events
# with a graph that has none. Coverage therefore requires the BACKEND to read the field.

# ParsedJob field -> the GpuJob field it is carried as, where the two names differ.
FIELD_ALIASES = {
    "causal_transitive_reduction": "transitive_reduction",
    "event_signature_keys": "event_canon_mode",
    "positional_event_identity": "event_canon_mode",
}

# Fields whose effect is applied by the FFI itself rather than by the backend, with the reason.
EXEMPT_FIELDS = {
    # run_gpu_job calls core_progress before and after run_gpu_evolution; the device does not
    # report progress of its own, so there is nothing for the backend to read.
    "show_progress",
    # These two RESOLVE INTO max_states_per_step rather than reaching the device under their own
    # names: effective_max_states_per_step() answers "which of the two step caps won", and both
    # the host engine setup and the GpuJob ask it. The GpuJob therefore carries the resolved
    # value, and searching run_gpu_job for these field names cannot find them -- the read is one
    # call away, which is the point of resolving it once.
    "uniform_random",
    "matches_per_step",
}

# Components whose key is not the name the caller asks under, so a literal search for the name
# in the backend cannot find them. Each is listed with the key it is actually served as.
EXEMPT_COMPONENTS = {
    # Served under the "Events" key: the backend emits the same association with the two edge
    # lists omitted, gated on GpuJob::include_events_minimal.
    "EventsMinimal",
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

    backend = BACKEND.read_text()

    uncovered = []
    for name, fields in sorted(opts.items()):
        if name in EXEMPT:
            continue
        warned = f"'{name}'" in gpu
        honoured = False
        for f in fields:
            if f in EXEMPT_FIELDS:
                honoured = True
                break
            for cand in {f, FIELD_ALIASES.get(f, f)}:
                if re.search(r"\bjob\." + re.escape(cand) + r"\b", backend):
                    honoured = True
                    break
            if honoured:
                break
        if not honoured and not warned:
            uncovered.append((name, fields))

    # The output surface: every RequestedData component the FFI parses.
    components = sorted(set(re.findall(r'comp == "([A-Za-z0-9]+)"', ffi)))
    if len(components) < 10:
        print(f"FAIL: only {len(components)} requested-data components parsed from {FFI.name}.")
        return 1

    shared = SHARED.read_text()
    unserved = [c for c in components
                if c not in EXEMPT_COMPONENTS
                and f'"{c}"' not in backend
                and f'"{c}"' not in shared
                and f"'{c}'" not in gpu]

    if uncovered or unserved:
        if uncovered:
            print(f"{len(uncovered)} option(s) reach the CPU engine but neither the device nor "
                  f"an OptionSkipped warning:\n")
            for name, fields in uncovered:
                fs = ", ".join(sorted(fields)) or "(no ParsedJob field)"
                print(f'  "{name}"  writes req.{{{fs}}}')
            print("\nEither have the backend READ the GpuJob field, or push an OptionSkipped "
                  "warning naming the option.")
        if unserved:
            print(f"\n{len(unserved)} requested-data component(s) the GPU backend does not emit "
                  f"and does not warn about:\n")
            for c in unserved:
                print(f'  "{c}"')
            print("\nEither emit the key from hg_gpu_backend.cpp or the shared marshaller, "
                  "or push an OptionSkipped warning naming it.")
        return 1

    print(f"OK: all {len(opts) - len(EXEMPT & opts.keys())} device-relevant options and "
          f"{len(components) - len(EXEMPT_COMPONENTS)} requested-data components are carried to "
          f"the device or warned about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
