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

## Build

`nvcc.exe` (Windows CUDA toolkit) + MSVC `cl.exe` via WSL2 interop for the
Windows GPU exe; native `nvcc`/`g++` for Linux; the existing MinGW/clang cross
toolchains for CPU-only targets. Path translation with `wslpath -w`, exactly as
`NeuralLearnability/build_gpu.sh` does.
