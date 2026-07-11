# Process-isolation binary (WXF over stdio)

The deployment path for the engine is a **standalone executable**, not a
LibraryLink DLL. The notebook serializes the job to WXF, spawns the executable,
writes the WXF to its stdin, and reads the WXF result from its stdout. This
mirrors the approach already proven in
`symbolic_dynamics/code/NeuralLearnability` (its `sweep` / `sweep_gpu` binaries).

## Why

- **Abort is a process kill.** The notebook holds the child's `ProcessObject`;
  aborting an evolution is `KillProcess[proc]`. No cooperative `AbortQ` polling,
  no `should_abort` flag threaded through the engine. All of that plumbing is
  removable once the DLL is retired.
- **Crash isolation.** A segfault or OOM in the engine kills the child, not the
  Wolfram kernel / notebook front end. The parent sees a non-zero exit code and
  whatever the child wrote to stderr.
- **CUDA is trivial to link.** `nvcc.exe` builds an `.exe` directly, so one
  binary carries both the CPU and GPU backends selected by the `TargetDevice`
  option — no CUDA-in-a-LibraryLink-DLL cross-compile problem to solve.

## Protocol

| Channel | Direction | Content |
|---------|-----------|---------|
| stdin   | WL → exe  | WXF `BinarySerialize` of the input association (`InitialEdges`/`Roots`, `Rules`, `Steps`, `Options`). |
| stdout  | exe → WL  | WXF of the result association (`States`, `Events`, `CausalEdges`, `BranchialEdges`, analysis sections, warnings). Raw bytes, binary mode. |
| stderr  | exe → WL  | Progress lines (one per line), and a `HGEvolve fatal: ...` line on a caught exception. |
| exit    | exe → WL  | 0 = success, 1 = fatal error (message on stderr). |

The input/output WXF associations are **byte-identical** to what the LibraryLink
`performRewriting` consumed/produced — the marshaling is shared code
(`run_rewriting_core`), so the two front ends cannot drift.

## Code structure

- `paclet_source/hg_core.hpp` — `HostBridge` (progress + abort callbacks, either
  may be empty) and the declaration of
  `run_rewriting_core(const std::vector<uint8_t>& wxf_in, const HostBridge&) -> std::vector<uint8_t>`.
- `paclet_source/hypergraph_ffi.cpp` — defines `run_rewriting_core` (the parse →
  evolve → marshal body, host-agnostic). The LibraryLink entry points
  (`performRewriting` and the blackhole analyses) are a thin adapter compiled
  only when `HG_STANDALONE_BINARY` is **not** defined; they build a `HostBridge`
  that routes progress to the notebook (`Print` over WSTP) and abort to
  `libData->AbortQ()`.
- `paclet_source/hg_evolve_main.cpp` — the binary's `main`: read stdin, build a
  `HostBridge` that writes progress to stderr and supplies no abort callback
  (the parent kills us), call `run_rewriting_core`, write stdout. Compiled with
  `-DHG_STANDALONE_BINARY`, so it pulls in **no** Wolfram SDK headers.

## WL invocation (NeuralLearnability pattern)

- **Blocking, no progress:** `RunProcess[{exe}, All, wxfBytes, ProcessEnvironment -> <||>]`;
  recover raw stdout bytes with `ToCharacterCode[proc["StandardOutput"], "ISO8859-1"]`
  then `BinaryDeserialize`.
- **Streaming progress + abort:** `StartProcess[{exe}]`, `BinaryWrite[proc, wxfBytes]`
  then close stdin, read progress lines from stderr, read the result from stdout,
  `KillProcess[proc]` to abort.

## GPU backend

`TargetDevice -> "GPU"` selects a second binary, `hg_evolve_gpu`, built with the
CUDA backend (`HG_GPU_BACKEND`, links `hg_gpu`). It reads/writes the SAME WXF as
the CPU binary; the WL front end picks it when present, else falls back to the
CPU binary (message `HGEvolve::gpudev`). Two binaries rather than one keep CUDA
out of the CPU-only deployment (mirrors NeuralLearnability's `sweep`/`sweep_gpu`).

- `paclet_source/hg_gpu_backend.cpp` (`run_gpu_evolution`) builds a
  `hg_gpu::EvolveInput` from the parsed job, runs `hg_gpu::evolve`, and marshals
  the `EvolveResult` into the CPU-identical WXF association.
- The GPU result is a raw per-provenance space; the CPU FFI emits a
  canonical-class space. The marshaler bridges them host-side: recompute the
  host IR canonical hash (`hypergraph::IRCanonicalizer`) per GPU state, group
  into classes and emit one `States` entry per class; keep `Events` raw (so
  multiplicity and counts match the CPU); dedup `CausalEdges` by `(from,to)`;
  leave `BranchialEdges` undeduped; derive `Step`/`IsInitial` from the events.
- Validated: the golden corpus (`reference/golden_corpus.wl`,
  `reference/verify_paclet_gpu.wls`) matches the CPU golden exactly through the
  Linux GPU binary.

## Build

Native `nvcc`/`g++` for the Linux GPU binary (same ABI, links `hg_gpu` directly);
the MinGW/clang cross toolchains for the CPU-only targets. The **Windows** GPU
binary needs the whole stack built with MSVC `cl.exe` + `nvcc.exe` (the CUDA lib
is MSVC-ABI and cannot link into a MinGW binary) via WSL2 interop with
`wslpath -w`, as `NeuralLearnability/build_gpu.sh` does for its single-file exe —
still to be wired for this multi-target project.
