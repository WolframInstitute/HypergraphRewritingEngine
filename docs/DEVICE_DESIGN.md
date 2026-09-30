# Device engine design

The target implementation of the CUDA engine: every lane doing useful work, no device-wide
barriers, work proportional to what changed, and the same results as the host. This document
states the target, the reasons for it, and the order of work. The current implementation is
described where the target replaces it, with file:line references to the code as of 8be18bb8.

## 1. The hardware and the rules it sets

The target part is the RTX 4090 (AD102, sm_89). The design uses only properties shared by every
NVIDIA part since Volta, so it carries to other parts; the numbers below are the 4090's.

- 128 SMs, 4 warp schedulers each. Up to 48 resident warps per SM (6,144 on the device), 64K
  32-bit registers and up to 100 KB of shared memory per SM.
- 72 MB of L2, about 1 TB/s of DRAM bandwidth. A warp load of 32 consecutive 4-byte words is one
  128-byte transaction; 32 scattered words are 32.
- Latency is hidden by other resident warps. A warp waiting on memory costs nothing while
  another warp is ready; with one or two resident warps per scheduler, the scheduler idles.
- Atomics resolve at L2. Atomics on different addresses proceed in parallel; atomics on one
  address serialise. One atomic per warp in place of one per lane divides the traffic by 32.
- Memory model (PTX): an acquire load at gpu scope invalidates the SM's L1 (SASS CCTL.IVALL).
  Loads that bypass L1 (`ld.cg`, `__ldcg`) see L2 without invalidating anything. A release
  store or `fence.acq_rel.gpu` orders prior writes. Publication therefore costs one fence per
  batch of writes, and a reader of shared mutable data reads from L2 rather than acquiring.
- Divergence: lanes of a warp execute one path at a time. Work of different kinds in one warp
  runs serially; work of one kind in one warp runs together.

Rules that follow, applied everywhere below:

1. Every lane has an item. Work is handed to a warp in batches of 32 items of one kind.
2. No device-wide barrier. Work moves between stages through queues; a warp takes whatever
   stage has work.
3. One atomic per warp per shared counter: allocations and queue appends are aggregated with
   `__ballot_sync` and a warp prefix sum.
4. No acquire loads in polling or probing. Readers use L2 loads; writers publish a batch with one
   release.
5. Work proportional to change: a child state differs from its parent in a few edges, and its
   matching, storage and replay cost are proportional to that difference.
6. The host and the device run the same algorithms through shared `hgcommon` cores, so a change
   here is a change to both engines, and the differential tests keep them equal.

## 2. What the current implementation does, measured

| Finding | Evidence |
|---|---|
| A worker is one warp, and 8 are resident per SM: 256 of 1,536 thread slots | `match.hpp:101`, `persistent.cu:1161` |
| 60.5% of warp instructions run with one active lane (wolftri d6) | RC1 campaign O5, ncu |
| Rewrite, identity claims, event stamping, exploration bookkeeping and every queue operation run on thread 0 | `persistent.cu:753-1106`, `rewrite.cu:240-546` |
| One warp per state averages 3.5 active lanes on small states | RC1 campaign, parallel design |
| Matching: each lane runs a whole DFS subtree; below the root every level scans the full slice serially and tests membership by binary search | `match.cu:171-215, 320-353, 85-102` |
| IR: under the warp policy the refinement, splitting, leaf evaluation and emits run on the leader lane | `ir_core.hpp:396-614, 851-1035` |
| One tiny state's IR is 12,971 instructions of straight-line code at CPI 13, 72 us | RC1 campaign D3, ncu |
| A replay instance stores a producer word per frame slot and copies it per child: bigpath 256 at depth 3 needs about 17 GB | `quotient_expansion.hpp:1291-1306` |
| A replay application does about 8 single-address atomics and 5 `__threadfence`s | replay map, `quotient_expansion.hpp:930-1310` |
| Branchial pairs cost O(K^2) list walks per instance, in buckets shared with other instances | `quotient_expansion.hpp:1255-1288` |
| During few-class replays the grid is 98% idle; one lane drives a class's captures and cascade | RC1 campaign D2 |
| Hash maps: keys and values in separate arrays, modulo start slot, acquire load per probe | `hash_table.hpp:190-275, 417-418` |
| A call clears about 75 MB and spends 293 us spinning up and terminating an empty grid | nsys 2026-09-29, RC1 campaign D3 |

The design below removes each row.

## 3. Execution model

### 3.1 One persistent kernel, many resident warps

- Block size 128 (four warps), as many blocks as fit: target 32 to 48 resident warps per SM. The
  block is a scheduling unit only; warps do not synchronise with each other.
- Registers are bounded with `__launch_bounds__` so that the occupancy target holds; the large
  per-lane arrays (join state, IR scratch) move to shared memory or to per-warp global scratch.
- There is no detector block. Termination is described in 3.4.

### 3.2 Stages and queues

Work is a set of item kinds. Each kind has its own queues, and an item of one kind is processed
the same way whatever produced it:

| Kind | Item | Produces |
|---|---|---|
| EXPAND | (state, rule) | MATCH items, or matches directly for small states |
| MATCH | a match record | REWRITE items |
| REWRITE | a match to apply | a child state, an event; CANON item |
| CANON | a state to canonicalise | CLAIM item |
| CLAIM | a state with its form | EXPAND items for a new class; CAPTURE item |
| CAPTURE | a class's newly captured match | REPLAY items |
| REPLAY | an instance, or a (match, instance range) | child instances; relations |

Items are sorted into size classes on the way in: a state with at most 32 edges goes to the
small queue of its kind, up to 1,024 to the medium queue, and larger to the large queue. The
granularity follows the size class:

- small: one item per lane, 32 items per warp (state-per-lane IR with the serial policy, one
  match per lane in the rewrite);
- medium: one item per warp, lanes over the item's edges or candidates;
- large: one item per block of four warps, cooperating through shared memory.

This is the thread / warp / block assignment used by GPU graph frameworks for irregular
frontiers, applied per stage.

### 3.3 Queues without barriers

- Each queue is sharded per SM (`%smid`), with work stealing from other shards when the local
  shard is empty.
- Append: the warp ballots the lanes with an item, lane 0 reserves `popc` slots with one
  `atomicAdd` on the shard tail, lanes write their items, and one `__threadfence` plus a release
  store of a per-chunk ready count publishes the batch.
- Take: a warp claims up to 32 items with one `atomicAdd` on the shard head, then reads them with
  L2 loads once the chunk's ready count covers them.
- A warp polls queues in a fixed priority that drains downstream work first (REPLAY and CLAIM
  before EXPAND), which bounds the memory held by queued intermediate results.

### 3.4 Termination

Every queue shard keeps a produced and a consumed count, updated once per batch. A warp that
finds every shard of every queue empty reads the sums twice, a fixed interval apart. If the sums
are equal and unchanged in both reads, it sets the exit flag. The check is the shared
`hgcommon::term_detect` body (termination_core.hpp) run by whichever warp is idle, not by a
reserved block. Idle warps back off with `__nanosleep`, so a nearly finished run does not load
L2 with polls.

### 3.5 Memory model use

- Items and records are written with plain stores, then one release per batch.
- Readers of data other warps write (maps, lists, queues, state slices) use `__ldcg`.
- Acquire loads remain only where a value must order later reads and no batch fence exists.
  Each such site is listed in the code with its reason.

## 4. Stage designs

### 4.1 States: shared chunks

A child state today copies its parent's edge list and appends the produced edges
(`rewrite.cu:316-344`), so a state costs O(|S|) memory and bandwidth to create.

Target: a state is an array of pointers to fixed chunks of 32 edge ids, sorted by edge id. A
child shares every chunk of its parent that no consumed edge touches, rewrites the chunks that
lose an edge, and appends chunks for produced edges. Creation cost is O(chunks changed), and a
chunk is read by one coalesced warp load. Readers (matching, IR flatten, content hash) iterate
chunks in order; the iteration is one shared function, as `DeviceContentCursor` is now.

### 4.2 Matching: drain-time inheritance and delta joins

Today the device matches every state from scratch for every rule. The host inherits (2bfb5198):
a state's matching drains exactly once, and at the drain the parent hands each registered child
the stored matches that use none of the child's consumed edges; a child registered after the
drain takes them at registration (rv::ChildInheritance, MatchJoin::drained / inherited). The
child then matches only what its produced edges anchor. No state reads above its parent.

Target, for the device: the same inheritance, through the same rendezvous rule, with the parent's
list filtered by the lanes of a warp (one match per lane) and appended per warp.

Target, for both engines, judged by time and instruction counts against the current inheritance:
the inherited list is stored in shared chunks (4.1). The child keeps references to the parent's
chunks that hold no match using a consumed edge and rewrites only the chunks that do, found
through an edge -> match index of the parent; the per-child cost is then O(chunks affected) in
place of a filtered copy of the parent's list. The drain handoff is unchanged: the parent's list
is complete when the child takes it.

The host's remaining ancestor walk is candidate lookup (`ancestry.hpp:30-41`, chain length x
produced edges per query). A state's vertex -> edge index in shared chunks reads one level; it is
measured against the walk.

For the anchored part and for roots:
- Each state keeps, per chunk, its edges ordered by edge signature, so the candidates for a
  pattern edge are a contiguous range found by binary search on the signature, not a scan.
- Joins run level by level within a warp: lanes hold partial matches, each level expands them
  against candidate ranges, survivors are compacted with ballot, and the frontier of partial
  matches lives in shared memory. Lanes stay busy regardless of subtree sizes, and the global
  atomic per emitted match becomes one per warp.
- Membership tests read the state's edge bitmap, a word per 32 edges held beside the chunks,
  in place of a binary search.

### 4.2a Keyed rewrites

A produced edge's token is (rewrite id, RHS index), the rewrite id interned exactly from the
rule and the consumed edges' tokens (`hgcommon/token_core.hpp`). The same rewrite applied
in two states gives its edges the same tokens, so two raw states with the same token set are
isomorphic through the tokens: the later one takes the earlier one's class, ranks and orbits
and runs no IR. A twin's ranks are the earlier state's labelling, which differs from its own
on a state with automorphisms, so a run that compares rank tuples (edge-keyed event identity,
the transition draw) does not key (`keyed_rewrites_apply`). A state's token sum (parent's sum
minus consumed terms plus produced terms) selects the candidate; the token sets decide it, in
O(edges). Nothing is interned before a run's first inherited match, which is its first repeated
rewrite; older tokens are computed on demand from the edge's creator event. A run that makes
1,024 twin claims finding no twin, published or not, stops (`keyed_note_claim`). Built on the host
first; the device takes the same rule, with one lane per state in the twin check. The device
canonicalises about a thousand children at once, so a twin is often claimed and not yet
published: a child of more than 32 edges waits on the twin's follower stack, and the twin, once
published, hands its followers to a ready ring from which blocks complete them as single records
(no IR); a smaller child runs its own IR, which costs less than the wait.

### 4.3 Rewrite

A block claims one record, then up to seven more in one exchange when the first child has at
most 32 edges, 64 records are readable and the block found work in its last two iterations;
a short burst, or one of large children, spreads over the grid one record per block, where a
batch would hold records for a tile IR while other blocks wait. Each four-lane tile applies
one match, eight per warp. Every allocation (state id, event id, edges, vertices, slice) is
one atomic per warp (`coalesced_add`, `coalesced_bounded_claim`). The per-record phase-timing
atomics (`rewrite.cu:536-542`) are removed; stats go to per-warp counters flushed at exit.

### 4.4 Canonicalisation

- States of every size: IR on tiles whose width is a run-time value (`IrTile`), as many states
  per warp as fit the block's scratch region (a 130-edge cycle needs about 13.5K words of the
  65,536, so four). A batch holds up to eight states on four-lane tiles; a block busy for 16
  consecutive iterations claims up to 32, run one per lane. The refinement is leader-serial on a
  cycle-like state, so a large state on the warp leaves most lanes idle, and four in flight per
  warp cut bigpath n128 by 16%. One state per lane costs 3.2-7.4x the warp's per-state latency
  for 4-10x its throughput per warp, so it runs only when every block is busy: cycle4 -12%,
  multirule -15%, disc2x2 -42% against four-lane tiles throughout, the latency-bound runs
  level.
- Medium and large states: the refinement is made warp-parallel. A refinement round is a
  segmented sort of (cell, signature) keys and a scan for cell boundaries, both warp or block
  primitives; the leader-only splitter pop, gather and split (`ir_core.hpp:434-608`) become one
  data-parallel round per splitter batch. Search nodes of the individualisation tree are
  independent after the first refinement and are handed to different warps.
- The form is written by the lanes that own the edges, and hashed once.
- `ir_core.hpp` keeps one body. The policy parameter already separates serial from warp
  execution; the data-parallel refinement is added under that parameter, and the host uses the
  serial policy.

### 4.5 Identity claims

- Hash tables store key and value in one 16-byte slot. The capacity is a power of two and the
  start slot is a mask.
- A probe is warp-cooperative: the warp reads 32 consecutive slots in one pass (four 128-byte
  lines), compares keys in every lane, and resolves the first match or first empty slot by
  ballot. A single claim costs one or two memory round trips in place of one per slot.
- Batched claims: a warp claims 32 states at once, one per lane, each probing independently.
- Records (canonical forms) are allocated per warp and compared by the lanes of the warp.

### 4.6 Replay

The replay is the largest cost in quotient runs, and its per-instance storage is what fills the
device on large states. Three changes, all in `hgcommon` so the host does the same:

1. **No producer arrays.** An instance stores its parent instance, the match that created it and
   the event id: 12 bytes. The producer of slot s is found by walking the ancestry: if s is a
   produced slot of the creating match, the producer is the creating event; otherwise map s to
   the parent's slot through the match's survivor map and repeat at the parent. The walk is at
   most the depth, and depth is the step count. This takes bigpath 256 at depth 3 from about
   17 GB to about 200 MB.
2. **Branchial pairs from the class.** Two applications on one instance overlap exactly when
   their matches' consumed frame slots overlap, which is a property of the two matches and the
   class, not of the instance. The class's overlap list is built once, incrementally as matches
   are captured (a new match is tested against the class's existing ones). An instance's
   branchial pairs are that list with the instance's event ids substituted; there is no
   per-instance O(K^2) test and no shared-bucket walk.
3. **Instances as batches.** A REPLAY item is an instance with all its class's matches: a warp
   applies up to 32 matches at once, one per lane, reading the class's match records coalesced.
   A match captured after instances exist becomes items over ranges of those instances, lanes
   over instances. Child instances, event ids and relation entries are allocated per warp.
   Few-class workloads, where one class has most of the instances, spread over every warp,
   because the unit is (instance, match range) rather than the class.

The exactly-once claim of an (instance, match) pair keeps its bit per match; with the batch as
the unit, the owner of the batch applies the matches it holds, and the bit resolves only the
race with a late capture.

The multiplicity counts become warp-parallel over a class's matches, and the cascade is a queue
of (class, depth) points processed by any warp.

The online transitive reduction keeps its per-event kept sets and search (`reach_core.hpp`); the
search runs one event per lane, and its scratch comes from per-warp slices.

### 4.7 Per-call cost

- Every buffer is allocated once per engine and grows only.
- Pools and queues reset by writing their counters. Hash tables are cleared per run, in one
  kernel with the used prefixes of per-state and per-edge arrays. Epoch tags were built and checked
  and are not used: the tables they could serve are the claim maps, at most 12.6 MB per call at the
  default size, and they would shorten claim keys to 56 bits; the large per-call clears are maps
  keyed by exact 64-bit values, which cannot carry an epoch.
- The run's counters and results are read back in one batch, as now.

## 5. Expected effect, by the rules they come from

| Change | What it removes |
|---|---|
| 128-thread blocks, 32-48 resident warps per SM | idle schedulers: 8 one-warp blocks per SM today |
| Batches of 32 items of one kind | 60.5% single-lane instructions; 3.5 of 32 lanes on small states |
| Warp-aggregated allocation and appends | one atomic per item on single counters, about 8 per replay application |
| L2 loads and batch releases | CCTL.IVALL per acquire; about 5 fences per application |
| Delta matching | O(|S|^k) scans per state per rule |
| Shared chunks | O(|S|) copy per child state |
| Replay without producer arrays | O(nslots) memory per instance: 17 GB for bigpath 256 d3 |
| Branchial from the class overlap list | O(K^2) tests per instance |
| Interleaved power-of-two tables, warp-cooperative probes | two memory round trips per hit, a modulo per probe step, one slot per thread |

Each change is judged by time and instruction counts: warp instructions and instructions per
active lane (ncu, one launch, the GPU otherwise idle), host instructions (callgrind), device
memory, and times taken in a quiet window. The differential tests hold the results equal.

## 6. Order of work

Each step lands with its gates: `hg_gpu_tests`, `gpu_differential_tests`, `gpu_ffi_tests`, the
host suite for shared cores, and the collision tests.

0. **Keyed rewrites** on the host, then the device (4.2a).
1. **Containers and queues.** Slot-interleaved, power-of-two, epoch-tagged hash tables with
   warp-cooperative probes; per-SM sharded queues with warp-aggregated append and take;
   warp-aggregated pool allocation. Everything else is built on these.
2. **Execution model.** Four-warp blocks, stage queues with size classes, termination by idle
   warps, no reserved detector.
3. **Rewrite and claims per tile.** One match per four-lane tile in the rewrite; one atomic per
   counter per warp. Built.
4. **Replay.** Ancestry producers, class overlap lists and instance batches, in `hgcommon` for
   both engines.
5. **Shared chunks.** The chunked copy-on-write structure in `hgcommon`; child states on the
   device, inherited match lists and vertex -> edge indices on both engines, each kept only if
   it wins on time and instruction counts against what it replaces.
6. **Matching.** Drain-time inheritance on the device; delta joins; signature-ordered chunks and
   level-wise warp joins.
7. **Canonicalisation.** State-per-tile batches for small states (built); data-parallel
   refinement for large ones.
8. **Per-call cost.** One clear kernel over used ranges; nothing else remains to clear.

Steps 4 and 5 change the host too, and are measured on both engines.

## 7. Decisions

- **Shared chunks (4.1, 4.2): agreed 2026-09-29.** One chunked copy-on-write structure in
  `hgcommon` for a state's edges (device), its inherited match list and its vertex -> edge index
  (both engines). The drain-time inheritance stays; the chunks change what it transfers.
- **Replay producers (4.6.1): decided by time and instruction counts.** Two candidates, both
  measured on the corpus and the large-state axis: the ancestry walk (12 bytes per instance,
  O(depth) per consumed slot) and a producer map keyed by edge identity shared between instances
  of one lineage. Consecutive instances usually belong to different classes, whose frame slots
  are permuted by the survivor map, so slot-indexed arrays cannot be shared; the second candidate
  avoids slots. The one that wins on both counts is kept, and the other is not built further.
- **The arbiter for every step is time and instruction counts.** Instruction counts are taken as
  work lands; times are taken in a quiet window.
