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
- **Per-run clears** (`clear_batch.hpp`): `EngineState::clear`, `QeState::clear` and the
  persistent launch's setup each collect their regions in one `ClearBatch` and clear them with one
  kernel launch. Per-state, per-edge and per-event arrays are cleared up to the previous run's
  counter, and in full on the first clear.
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

The device defaults to quotient exploration for bounded state growth. A canonical state is
expanded once, at the shortest depth any path reaches it by, through the host's rules
(`hgcommon/explore_depth_core.hpp`, `gpu/include/hg_gpu/explore_depth.hpp`): each arrival
registers the canonical child under its parent and lowers the child's depth to one past the
parent's current depth, and a lowered state is admitted and its descendants lowered in turn by a
depth-first walk on the block's thread 0. The expansion claim is separate from the dedup identity
and from the depth, so a class first reached past the budget is still expanded when a shorter
path arrives. An admitted state is appended to an expand log (`gpu/include/hg_gpu/work_log.hpp`)
that blocks consume at the top of the persistent loop, pushing one (state, rule) item per rule
and matching an item on the block when the ring is full; roots and a session's resumed frontier
enter the same log. The detector counts expand entries with the records.

Its capture and replay are
the host's, through the shared cores: each class retains its representative's expansion as
slot-named matches, instances are replayed forward, and the reconstructed relations are read
back as raw application-id pairs alongside the schedule-stable content-triple pairs a
cross-engine set comparison keys on (`reconstructed_pairs_host`, `gpu/src/quotient.cu`), with
per-event signatures so a caller can build the graph whose vertex set the count describes.
`materialize_relations` gates the pair expansion: counts are device counters and cost nothing;
the pairs are an expansion of the applied lists and are built only when the reply serves them.

The replay is the one place the device schedules differently from the host. The host drives a
class's instance cascade on the thread that produced it. On the device one thread runs the same
code about 60 times slower than a host thread, and a warp of 32 lanes is the unit of issue. So
each (instance, match) application is a task in an append-only log (`QeView::tasks`): a block's
lane 0 claims up to 32 consecutive tasks with one compare-exchange on the task cursor, each lane
runs one, and the tasks an application produces are appended to the same log. An append claims
its slot, writes the task and sets its published flag with release; a lane waits for the flag of
the task it claimed. The termination detector counts the claimed slots as produced and
`tasks_done` as consumed, and every append happens inside a unit that has not yet been booked
consumed (the record being rewritten, or the task being run). The claim per (instance, match)
makes the order in which tasks run irrelevant to the result; it is the host's rule
(`hgcommon::qr_claim_words`/`qr_claim_bits`): a bit in the instance's claim words for a match
whose class index is below the words' capacity, the shared `applied` map otherwise. A class and depth's instance list is
split over sixteen buckets, one chosen by the pushing lane, because concurrent pushes onto one
list head serialized the replay; the keyed lists' bucket counts grow with the event budget,
because a walk visits every node of its bucket. The redundancy search's overflow scratch is one
slice per lane, claimed from the expansion arena on first need (`QeView::lane_reach`).

## 7. Reply assembly

`hg_evolve_gpu` translates jobs and marshals results through the same WXF path and the same
graph marshaller (`paclet_source/graph_marshal.hpp`) as the host, so the two devices emit
identical reply shapes. The relation observables follow the one-relation rule of `SPEC.md`
§5.2, gated on this engine by `gpu_ffi_tests`.

## 8. Performance characteristics, measured (RTX 4090)

- **The per-call floor is ~0.82 ms** (`bench_gpu_evolve 2 30 1 wpp`, a 5-state run). About
  0.37 ms is kernel time, most of it device IR on one small state per step (72 us for a
  3-vertex state, one thread, 12,971 instructions at 13 cycles each). The rest is one device
  synchronization, the per-run clears (95 us of device time) and 7 host reads, each of which
  feeds the next host step. A persistent launch's ring, dedup maps and termination detector
  live with the engine; host-to-device uploads are asynchronous; the readback is one batch into
  a pinned buffer. A workload under a few ms of CPU time is floor-dominated. Within-run scaling is what the
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

Baseline is sm_75 (Turing). The shipped Linux and Windows binaries carry real SASS for sm_75, 80,
86, 89 and 90 (Turing to Hopper) and no PTX, so a card outside that list cannot run them. A build
from source uses the CMake default `HG_GPU_ARCHS`, which adds sm_100 and sm_120 (Blackwell). The
CUDA 13 runtime it links statically needs NVIDIA driver 580 or newer. Device LTO is deliberately off (measured within noise of LTO on, and the
multi-architecture LTO link is what made release builds unbuildable on the development box).
Every kernel-level behaviour is differential-tested against the host engine; the suites and the
golden corpus run in CI's GPU lane.
