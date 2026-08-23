# GPU engine — architecture

The device engine is a second, first-class implementation of the multiway evolution defined in
`docs/SPEC.md`. It computes the same observables as the host engine — states, events, causal and
branchial relations, quotient reconstruction — under the same option surface, and the equality is
gated, not aspired to: `gpu_differential_tests` compares the two engines' own hashes and
relations per workload, `gpu_ffi_tests` asks the device the same serialization questions the host
answers (including the `RelationCoherence` sweep over state mode × event mode × quotient × TR),
and the golden corpus requires `CPU == GPU` exactly.

## 1. One rule, two engines

Every decision that defines the semantics — matching, rewriting, canonicalization, event
identity, causal and quotient rules — has exactly one implementation, in `common/include/hgcommon/`,
compiled for host and device (`match_core`, `join_core`, `rewrite_core`, `wl_core`, `ir_core`,
`event_core`, `slot_core`, `signature_core`, `sampling_core`, `quotient_causal_core`,
`quotient_replay_core`, `tr_reduce`). Shared code is allocation-free and synchronisation-free by
construction; anything that allocates or synchronises is orchestration and belongs to one side.
What this file describes is the DEVICE'S orchestration: storage, scheduling, capacity, readback.

## 2. Execution model: one launch per evolution

`Engine::run` executes a whole evolution as one persistent kernel launch
(`run_persistent_evolve`, `gpu/src/persistent.cu`): worker blocks pull work items from a
device-resident MPMC ring, process them — match, rewrite, canonicalize, deduplicate — and push
the successor items back, until a termination detector observes a stable empty system.

- **A work item is a (state, rule) pair carrying its own `step`.** Depth rides on the item, so a
  step budget is a predicate on the item rather than a loop bound — the same way the host
  carries depth on its tasks. There are no phases and no per-step barrier: the observable
  contract is schedule-independent because states and events are keyed by canonical identity,
  not because production is synchronised.
- **Plain grid-stride persistent blocks, not cooperative launch.** A cooperative grid exists to
  provide a device-wide barrier, and a barrier is the thing this model removes; cooperative
  launch also caps the grid at simultaneous residency for no gain here.
- **Termination is decided on device.** A worker finding its queue empty cannot conclude the run
  is finished — other workers may still be producing — so `TerminationDetector` counts pushes
  and completions and requires a stable observation window before exit.
- **The kernel cannot hang the machine.** A run watchdog and per-worker spin budgets turn a
  stalled kernel into a recorded warning with partial work; `max_blocks_per_launch` bounds a
  single launch below the display driver's timeout where one applies.

`run_persistent_match` (one role, seed-once queue) and `run_persistent_match_rewrite` (two roles
feeding each other) are the stages the shipping scheduler is built from, kept as gates so a
failure lands in the stage that introduced its ingredient.

## 3. The boundary: nothing crosses host↔device during evolution

Uploads and queue seeding happen before the launch; results are read after it; in between the
device decides everything. A host round trip per step is the thing this design removes, and a
round trip for any other reason is the same defect in a different place. Three consequences:

- **Capacities are sized up front, from `EngineConfig`.** Pools are bump allocators
  (`atomic_pool.hpp`): `claim()` past capacity returns `kInvalid`, the counter is not a count of
  valid entries (`size()` clamps), and every exhaustion records its `ErrorKind`.
- **Overflow returns partial work, never throws.** A device-resident loop cannot grow-and-retry
  mid-run, so a full pool ends the run early with the error recorded and everything produced so
  far returned — the project-wide contract, load-bearing here. The host-side
  `PersistentEvolver` wrapper is where grow-and-retry lives: it reads the recorded kind,
  enlarges the config, and re-runs.
- **Exact canonicalization has no per-state ceiling and no approximation fallback.** IR scratch
  is claimed from a device-side arena (`device_arena.hpp`), sized per state from its own counts;
  a block reuses its slot and re-claims only for a larger state. Arena exhaustion is
  `kScratchOverflow` — recorded, partial work — never a silent switch to a coarser hash.

## 4. Storage

- **Pools** (`atomic_pool.hpp`): append-only within a run, reset between runs.
- **Hash tables** (`hash_table.hpp`): open-addressing concurrent maps with the EMPTY/LOCKED key
  discipline; a key equal to a sentinel is rejected rather than silently lost.
- **DeviceArena** (`device_arena.hpp`): bump allocation for variable-size per-state data.
- **State and event records** live in structure-of-arrays form on `EngineState`
  (`engine_state.hpp`); the device stack is a small constant plus a bounded
  reconstruction-nesting term, requested per run as the minimum of the budget and the depth.

## 5. Identity on device

All three state modes (`None`, `Automatic`, `Full`) and all event identities (`None`, `Full`,
`Automatic`, `Positional`, custom key sets) run on device with the host's definitions
(`SPEC.md` §4): `state_key_device` computes exactly what the mode identifies states by; the
exact IR hash is a second per-state quantity filled when an event identity reads it; event
signatures come from the shared `event_core` with ranks resolved on device, and a rank that is
unavailable substitutes the raw edge id and is counted (`kEventSigRawFallback`).

## 6. Quotient exploration and reconstruction

The device defaults to quotient exploration for bounded state growth. Its capture and replay are
the host's, through the shared cores: each class retains its representative's expansion as
slot-named matches, instances are replayed forward, and the reconstructed relations are read
back as raw application-id pairs alongside the schedule-stable content-triple pairs a
cross-engine set comparison keys on (`reconstructed_pairs_host`, `gpu/src/quotient.cu`), with
per-event signatures so a caller can build the graph whose vertex set the count describes.
`materialize_relations` gates the pair expansion: counts are device counters and cost nothing;
the pairs are an expansion of the applied lists and are built only when the reply serves them.

## 7. Reply assembly

`hg_evolve_gpu` translates jobs and marshals results through the same WXF path and the same
graph marshaller (`paclet_source/graph_marshal.hpp`) as the host, so the two devices emit
identical reply shapes. The relation observables follow the one-relation rule of `SPEC.md`
§5.2, gated on this engine by `gpu_ffi_tests`.

## 8. Performance characteristics, measured (RTX 4090)

- **The per-call floor is ~3.3 ms**, independent of workload — readback (≈23%) plus setup
  (≈3.2%) of a small run — and it is what bounds interactive and small-workload use: a workload
  under ~10 ms of CPU time is floor-dominated by construction. Within-run scaling is what the
  device is for: wpp depth 7 runs 45,317 states in 47 ms against 216 ms on 8 host threads.
- **The hardware-utilisation ceiling is the algorithm, not the implementation.** Every avenue
  was measured and excluded: DRAM 0.71% of peak, L1 3.28%, L2 5.01%, atomics with 20× headroom;
  registers 255→128 via `-maxrregcount` left occupancy unchanged; doubling occupancy via the
  grid made the kernel 6.5% slower; warp-cooperative refinement is refuted on the data (states
  average ~10 vertices, where five shuffle steps plus a barrier cost more than ten sequential
  operations). Individualization–refinement is a dependent chain, it is 51–77% of block cycles,
  and no quantity of resident parallelism executes a dependent chain faster.
- **Class-collapsed workloads lose by width, not by floor**: under quotient the device's
  parallel width is the class count while the independent work is the instance count (1,705×
  apart on cycle4), and depth does not close it. Report such cells as width-bound, not as
  device losses.
- **Wall-clock measurement requires a warm device.** Idle, the GPU sits at 210 MHz against a
  3,150 MHz maximum, and the ramp shows up as a 10× spread between back-to-back identical runs;
  `bench_gpu_evolve` warms for a bounded 400 ms before timing, and locking the clock is better
  where root is available.

## 9. Hardware baseline and differential testing

Baseline is sm_75 (Turing) or newer; the shipping build compiles real SASS for the configured
architectures and device LTO is deliberately off (measured within noise of LTO on, and the
multi-architecture LTO link is what made release builds unbuildable on the development box).
Every kernel-level behaviour is differential-tested against the host engine; the suites and the
golden corpus run in CI's GPU lane.
