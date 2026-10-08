# Changelog

## Unreleased

User-visible changes since v1.0.0-rc1:

- The `"Step"` of an event identity is the step of the state the event produces. `Automatic`
  and key lists with `"Step"` count an event that repeats at a later step as a new event.

- In a `"CanonicalizeEvents"` key list, `"ConsumedEdges"` and `"ProducedEdges"` are the input and
  output states with the consumed and produced edges marked, up to isomorphism. Such a list
  counts the same events with and without `"ExploreFromCanonicalStatesOnly"` and at any thread
  count, and runs on the CPU engine under `TargetDevice -> "GPU"`.

- `"StepStatistics"`: for each step, the number of states, the isomorphism classes, the entropy
  of the states over the classes and summaries of per-state invariants, on both devices.
- Every option of `Graph` is an option of `HGEvolve` and `HGSessionOpen` and goes to the graph a
  graph property returns.
- `"MultiedgeStyle" -> "Merged"` draws the events between two states as one edge.
- `"EventClasses"`: for each event under `"CanonicalizeEvents"`, the ids of the rule applications
  it stands for.
- Styled graphs draw their state pictures in the kernel; they no longer call the
  `WolframModelPlot` and `WolframPhysicsProjectStyleData` resource functions.
- Under quotient exploration, `"NumEvents"` and `"NumBranchialEdges"` asked for alone are
  computed from class multiplicities, so much deeper runs are possible. Counts above 2^63 - 1
  are reported as 2^63 - 1 with the warning `CountSaturated`, given only for a requested
  count and naming it.
- Malformed input issues one message and gives `$Failed`: `HGEvolve::badrule`,
  `HGEvolve::badinit` and `HGEvolve::steps`. A negative vertex in an initial state is refused.
- An unrecognised `"CanonicalizeEvents"` or `"CanonicalizeStates"` value is reported through
  `HGEvolve::warn` and the default is used. `Positional` must be written as the string
  `"Positional"`.
- The GPU reports the same warnings as the CPU. Only warnings that mean a partial result are
  reported as `HGEvolve::overflow`; the others are `HGEvolve::warn`.
- Seven options that changed nothing are removed: `"InitialCondition"`, `"Topology"`,
  `"MajorRadius"`, `"MinorRadius"`, `"IncludeStateContents"`, `"IncludeEventContents"` and
  `"QuotientInitialStates"`.
- Under quotient exploration, `"Events"` lists every rule application, as under full
  exploration, and the GPU draws its graphs over them.
- The surface initial conditions (`"Torus"`, `"Sphere"`, `"Cylinder"`, `"Klein"`, `"Mobius"`)
  take `"RandomSeed"`.
- Under `"CanonicalizeStates" -> Full`, `"States"` is keyed by each class's canonical state id.
  `"ContentStateId"` is the lowest id of the states with the same edge list.
- A symbolic initial state is renumbered the way rule variables are. An integer
  `"TransitionRate"` or `"ExplorationProbability"` is accepted.
- Sessions: `HGSessionOpen` takes the input forms and `"TargetDevice"` handling of `HGEvolve`;
  `HGSessionStep` takes `"From"` without a property; a step with `"From"` advances by the steps
  it asks for; `"BranchialStep" -> Automatic` is resolved for each verb's properties.
- `"ShowProgress"` messages reach the kernel through the persistent worker.
- Fixed: the match drain considered only the first 64 rules; the GPU reported
  `NumBranchialEdges` 0 under quotient exploration; the sampling spine chose among a subset of a
  state's transitions, so seeded samples differ from rc1's; the GPU transitive reduction was
  inexact past its local arrays; GPU quotient exploration did not capture states above a size
  limit.
- `"MaxSuccessorStatesPerParent"`, `"MaxStatesPerStep"` and `"UniformRandom"` with
  `"MatchesPerStep"` keep the lowest-ranked transitions, ranked from each transition's identity
  and `"RandomSeed"`. A transition not kept is not taken. The kept set is the same at any worker
  count and on both devices.
- Transitions tied with the last one kept at a cap's cut are kept too, so a cap can keep more
  than its value: `"MaxStatesPerStep" -> 3` keeps 4 states at some steps of the documented
  example. The kept set does not depend on the schedule, on either device.
- `"ExplorationProbability"` keeps the same states on both devices.
- Under `"CanonicalizeStates" -> Full`, `"ExplorationProbability"` draws one coin per
  isomorphism class, and a class that holds an initial state is always expanded. Full capture and
  quotient exploration sample the same classes.
- `"RandomSeed" -> Automatic` is the seed 0 for the sampling draws, so a sampled run without a
  seed gives the same result every time. A generated initial condition still draws a new seed.
- Packed arrays are accepted as initial states and as `"RuleWeights"`.
- A refused session verb reports the engine's reason in `HGSessionOpen::refused`.
- Session handles from the CPU and GPU workers, or from a restarted worker, are distinct.
- Under `"Delivery" -> "Delta"`, a branchial graph at the final step is the graph of the current
  final step.
- `HGEvolve::overflow` states the reason for a partial result and is issued on either device.
  A capture dropped during reconstruction is reported as `CapturesDropped`.
- An error inside a worker is reported as an error of the run on the CPU; it could end the
  process. After an error the run stops promptly.
- An empty rule set gives the initial states and no events.
- An edge with no vertices (`{}`) in a rule or an initial state is refused with
  `HGEvolve::badrule` or `HGEvolve::badinit`.
- An edge of more than 16 vertices, a rule side of more than 16 edges and an empty left-hand side
  are refused with `HGEvolve::enginemsg` on both devices, before a device is chosen.
- A generated initial condition with no edges issues `HGEvolve::emptyic`, naming the condition.
- A negative `"MaxStatesPerStep"`, `"MaxSuccessorStatesPerParent"`, `"MatchesPerStateRule"` or
  `"MatchesPerStep"` is ignored with a warning and the run is uncapped.
- A NaN or infinite `"ExplorationProbability"`, `"TransitionRate"` or `"RuleWeights"` entry is
  ignored with a warning and the default is used, on both devices.
- A warning or error that quotes bytes from the request which are not valid UTF-8 shows each
  such byte as `\xNN`.
- `"ExploreFromCanonicalStatesOnly" -> True` is applied only under
  `"CanonicalizeStates" -> Full`. Otherwise every state is expanded, and the `QuotientNeedsFull`
  warning names only the option the call set.
- Setting up a rule whose left-hand edges have many distinct vertices takes milliseconds: a rule
  with a 14-vertex left-hand edge sets up in 0.005 s on the CPU, against 1.06 s in rc1. The GPU
  accepts left-hand edges of up to 16 distinct vertices.
- Under `"CanonicalizeStates" -> Full`, a state's `"Step"` is the least step at which its class
  occurs and is the same on every run. On the GPU this holds under full capture; under quotient
  exploration the GPU's `"Step"` is not yet the class's least step.
- The empty state has one `"CanonicalHash"` in every state mode, on both devices.
- Under `"CanonicalizeStates" -> Full` with `"CanonicalizeEvents" -> Automatic`, `"Events"` lists
  the rule applications that the causal and branchial records name.
- Under `"ShowGenesisEvents" -> True`, `"States"` does not list the genesis event's empty input
  state. `"NumEvents"` counts one genesis event per initial state, and the causal edges include
  the genesis pairs, under quotient exploration as under full capture and on both devices. A
  genesis event has `"RuleIndex"` 65535 on both devices.
- The branchial graph keeps every branchial pair, so its edge count equals
  `"NumBranchialEdges"`.
- `"StepStatistics"` histogram keys round halves to even, as `Round` does.
- Very large `"Steps"` cost only the depth the evolution reaches: a run with `"Steps"` 10^9 that
  stops growing after a few steps finishes in milliseconds.
- A run that reaches about 2^30 states no longer stops with an internal error in the branchial
  index.
- A reply larger than the available memory gives an error instead of ending the worker process.
  A failure to start a worker thread is reported as an error.
- Sessions: a steered `HGSessionStep` expands every frontier state its id stands for; after a
  failed session verb the next `"Delta"` request is sent as `"Full"`; a verb on an invalidated
  session says it was invalidated, on both binaries.
- GPU: a session verb is served under the settings the session was opened with; a session
  opened after a larger run fits its engine; a continuation matches through every edge of the
  state; a continuation reports only its own warnings.
- GPU: `"Delivery" -> "Delta"` is reported as not served, with an `OptionSkipped` warning, and the
  whole graph is sent.
- GPU: `"States"` lists every state outside `"CanonicalizeStates" -> Full`, and causal and
  branchial endpoints name canonical events under an event identity, as on the CPU.
- GPU: a run sized past the device returns partial work, and a partial result holds no record
  that a failed rewrite left unwritten. A full causal or branchial map and a full index at upload
  are reported as `HGEvolve::overflow`.
- GPU: the transitive reduction is exact whichever thread runs it. `"MatchesPerStateRule"` keeps
  the same transitions as the CPU when ranks are equal.
- GPU: an initial state of 32,768 edges run for three steps is sized in 64 bits and runs; it
  returned an empty result.
- GPU: device output printed during a long run no longer corrupts the reply.
- GPU: `"MaxStatesPerStep"` selection reads only the candidates since the previous step: 100,000
  steps with `"MaxStatesPerStep" -> 1` take 4.17 s, against 28.7 s.
- GPU: the device stack reservation is sized from the kernels: 1,522 MB on an RTX 4090, against
  2,304 MB.

---

## v1.0.0-rc1 (2026-09-01)

User-visible semantic changes since v0.0.1-alpha.6, carried here so the release notes state them:

- A steered session `Step` (`"From" -> {ids}`) works under `TargetDevice -> "GPU"`; the device
  previously refused it. Both devices report `"Frontier"` in every session reply, resolve the
  selection against it, and put unselected entries back at the depths they were stranded at.
- Fixed a rare high-contention nondeterminism: a concurrent set's membership query could miss a
  settled key while a table growth carried it, dropping one causal edge from a full-capture run
  (seen once on a 4-core ARM64 CI machine at 16 threads). Model-checked exhaustively after the
  fix.
- Parallel evolution is faster and uses far less memory at high worker counts. Two concurrent
  structures built a replacement hash table before the exchange that installs it, so every worker
  but one abandoned a full table on each growth; one worker is now elected per crossing and the
  others carry on without waiting. On a 409k-state run at sixteen threads this is 841 ms to 630 ms
  and 4.3 GB of arena to 1.8 GB, with 2.36 GB of abandoned tables gone entirely. Separately, the
  pointer every operation reads no longer shares a cache line with the counter every insert
  writes. Output is unchanged at every worker count.
- Fixed a device-only defect in quotient reconstruction: a canonical class published its frame
  owner and its step as two separate map insertions, so a thread that lost the first read the
  second before it existed and signed its events with its own depth instead of the class's,
  making the two signature sets disjoint. Both halves now publish in one exchange. This affected
  `TargetDevice -> "GPU"` runs with quotient exploration under contention.
- Device capacity-overflow warnings: the `count` field now reports THAT a kind of overflow
  occurred, not how many times. It never was a count of missing capacity -- it counted inner-loop
  iterations -- and recording each one serialised the whole device on a single counter precisely
  when a run was already degraded. The retry path doubles the configuration field the KIND names
  and never read the number.

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
