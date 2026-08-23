# Changelog

## v1.0.0 (unreleased)

User-visible semantic changes since v0.0.1-alpha.6, carried here so the release notes state them:

- `ExplorationProbability` samples **per canonical state**, not per transition.
- `QuotientInitialStates` defaults to keeping every root, matching the reference `MultiwaySystem`.
- Quotient exploration expands each canonical state once, **at its shortest depth**.

---

## v0.0.1-alpha.6

_Changes since **v0.0.1-alpha.5** (2026-01-11) — 202 commits._

A large release focused on making the engine dramatically faster, adding a working GPU backend,
isolating evolution in a standalone process, and shipping a proper cross-platform paclet with
markdown-sourced documentation.

## Highlights

- **Zero-waste performance overhaul** — the hot path is now essentially malloc-free (arena-backed
  jobs, maps, IR, and causal closures), with a rewritten matcher, IR canonicaliser, and causal
  reachability. Substantially faster and lower-memory across the board.
- **GPU backend (`TargetDevice -> "GPU"`)** — CUDA engine with full multiway support, now honoring
  `CanonicalizeStates -> None | Automatic | Full`, multiple initial states, quotient exploration,
  and graceful partial results on device-memory limits. A native **Windows CUDA binary** builds and
  runs, in addition to Linux.
- **Process isolation** — evolution runs in a standalone `hg_evolve` binary over a socket/stdio
  transport, so a crash or abort never takes down the notebook. A persistent worker amortises
  per-process (and GPU context) setup for 6–12× on interactive runs.
- **Cross-platform paclet** — one command (`./build_paclet.sh`) produces a paclet with libraries for
  all six platforms (Linux x86-64/ARM64, Windows x86-64/ARM64, macOS x86-64/ARM64), evaluated
  documentation notebooks, and the `.paclet` archive.
- **Documentation** — reference/tutorial pages are authored in markdown and built to notebooks via
  `MarkdownToNotebook`, with a comprehensive, fully-evaluated `HGEvolve` reference.

## Performance

- Malloc-free hot path: per-worker bump-cursor arena; every `ConcurrentMap`, the task/job path, and
  the causal `Desc/Anc` closures are de-heaped onto it.
- Causal graph: `O(N²)` descendant closure replaced by an id-pruned reachability walk; closure
  arena footprint cut ~28.5% (key-only sets, `Anc` dropped); per-event sets start tiny.
- Matcher: `MatchRecord` forwarded by reference through a shared immutable `MatchCore`; wasted work
  removed from the hot loop; pattern signatures read through the rule (no per-session copy).
- IR canonicalisation: sorted `lower_bound`/precomputed edge indices instead of per-child hash maps;
  degree-signature and vertex-set grouping via sort rather than `std::set`/`std::map`.
- States: copy-on-write derived states share immutable parent chunks; `Event` shed 132 bytes;
  edges with arity ≤ 2 inline their vertices.
- Streaming, single-pass WXF read/serialisation and single-`Join` socket reassembly.

## GPU

- `TargetDevice -> "GPU"` routes to the CUDA engine via the standalone binary.
- Honors `CanonicalizeStates -> None | Automatic | Full` (state counts match the CPU exactly in
  every mode); `Automatic` uses a content-ordered hash, `Full` uses exact IR.
- **Full property parity with the CPU**: every graph property (`StatesGraph`, `CausalGraph`,
  `BranchialGraph`, the `Evolution*` graphs, and their `Structure` variants) is built on the GPU
  path through a single shared marshaller, so CPU and GPU return identical graphs (verified by
  vertex/edge count and degree sequence across all properties).
- **Fully static-linked** GPU binary: static CUDA runtime (`libcudart_static`) and static C/C++
  runtime (`/MT`), so `hg_evolve_gpu.exe` imports only `KERNEL32`/`WS2_32` — no `cudart` DLL and
  no VC++ redistributable. The only runtime dependency is the NVIDIA driver (`nvcuda.dll`), which
  is loaded on demand and present wherever a usable GPU is.
- Multiple initial states (multiway with several roots); quotient exploration.
- User-settable device-memory cap with **graceful partial results** and a notebook warning on
  overflow (never throws); OOM-safe grow-and-retry.
- `PersistentEvolver` keeps the device engine across calls (amortises the ~0.7 s CUDA context).
- WL and IR hashing share a single `hgcommon` core with the CPU (verified bit-identical), and a
  CPU↔GPU differential test asserts state/event/causal/branchial equivalence up to isomorphism.
- **Native Windows CUDA binary** builds via `./build_windows_gpu.sh` (MSVC + nvcc); Linux too.

## Engine correctness

- Fixed a `num_canonical_states()` undercount in the default (`None`) mode: the id-0 state collided
  with the concurrent map's empty-slot sentinel and was silently uncounted.
- Quotient exploration expands each canonical state once at its shortest depth; completeness fix for
  truncated budgets under multithreading; `quotient_initial_states` option (default keeps all roots,
  matching the reference `MultiwaySystem`).
- `exploration_probability` samples per canonical state rather than per transition.
- Matcher no longer drops matches when the signature cache overflows; correct 64-bit byte swap on
  the big-endian WXF path; rule matching data finalised in `add_rule`.

## Transport & isolation

- Evolution runs in a standalone `hg_evolve` process on **every platform and device** — the process
  binary is shipped for all six platforms (and `hg_evolve_gpu` on the CUDA platforms), so a crash or
  abort kills the process, never the notebook. The in-engine abort mechanism was removed in favour of
  this; the LibraryLink library remains only as a last-resort fallback and for the standalone
  analysis functions.
- `HGEvolve` communicates over a persistent socket worker (`--serve-socket`), falling back to
  one-shot WXF-over-stdio. Abort = process kill.
- GPU capacity-overflow warnings surface to the notebook.

## Paclet & Wolfram Language

- `SyntaxInformation` supplies argument-count colouring and the option-name dropdown for the paclet
  symbols.
- Comprehensive `HGEvolve` reference documenting every option (evolution, output, and the
  dimension/geodesic/topological/curvature/entropy/Hilbert/branchial/multispace analysis families
  and the initial-condition generators), with worked examples.
- Markdown-sourced documentation pipeline (`./build_docs.sh`): pages authored in markdown, built and
  evaluated to notebooks; incremental rebuild keyed on markdown + engine hash.

## Build & platforms

- `./build_paclet.sh` — one command: six-platform libraries → docs → `.paclet` archive.
- Every platform build produces **both** the LibraryLink library and the `hg_evolve` process binary
  (Linux x86-64/ARM64, Windows x86-64/ARM64, macOS x86-64/ARM64), verified per platform.
- **Self-contained binaries**: the process binaries and fallback DLL fold in the C/C++ runtime
  (`-static` on mingw folds libwinpthread; `-static-libstdc++`/`-static-libgcc` on Linux), so they
  load on a clean machine with no mingw/libstdc++ runtime on its search path. A `clean` flag on
  `build_all_platforms.sh` / `build_paclet.sh` forces a fresh configure after a toolchain change.
- Native Windows CUDA build (`./build_windows_gpu.sh`, MSVC + nvcc) targets the toolkit's own VS
  integration so it works without the CUDA installer's VS-integration copy; the broad shippable arch
  set (Turing→Hopper) is compiled with `nvcc --threads` across cores.
- Host-aware, fault-tolerant multi-platform build; Linux/WSL cross-compiles all six targets.
- macOS build portability shim (`atomic_ref`) and MinGW-safe thread-exit guard; dropped the
  vestigial WSTP SDK path.

## CI

- Linux correctness gate (GitHub Actions); scaffolds for a free Wolfram-Engine paclet + golden
  gate and a CUDA-compile gate.

---

### Verification (this build)

- All 6 platform libraries **and** all 6 `hg_evolve` process binaries built (+ `hg_evolve_gpu` on the
  two CUDA platforms); `.paclet` archive produced, `DocumentationBuild` 24/24.
- The assembled `.paclet` was installed and exercised via wolframscript: `HGEvolve` runs through the
  `hg_evolve` / `hg_evolve_gpu` **processes** (isolation confirmed), CPU results correct across
  `None`/`Automatic`/`Full`, and **GPU results match CPU `CanonicalizeStates -> Full`** with no
  device fallback.
- CPU test suite green (190 tests); CPU↔GPU differential green (states/events/causal/branchial
  equivalent up to isomorphism, plus per-mode `NumStates`).
- `HGEvolve` example pages evaluate cleanly against the local engine.
