# v1.0.0 — everything that remains

One list. The task board (MCP) is unavailable, so this file and `V1_FINAL_SWEEPS.md` are the
tracker. Clock expired 2026-08-13T04:47Z; this is written 2026-08-18.

Each item states what closes it. Nothing here is "investigate" — the investigations are done and
recorded in `V1_FINAL_SWEEPS.md`; what is left is decisions, code, and one rebuild.

---

## A. BLOCKING — code that must be written

### A1 — #159: RECLASSIFIED. The leak is real, bounded to capacity exhaustion, and has no
### observable effect otherwise.

`apply_one_match` is DEVICE code (`gpu/src/rewrite.cu:247`), not host — the earlier framing sent
me reading `rewriter.cpp`. It uses a **preflight reservation**: claim every capacity-bounded
resource before any mutation, so a failure aborts without a half-initialised state. Its own
comment records what that replaced — a "claim, then silently early-return mid-kernel" pattern
that left the new state's bitset uninitialised and produced spurious OOBs downstream.

The claims are sequential, so a later failure does leak the earlier ones, and `Pool::claim` is a
bump allocator with no free. That is the defect as filed.

**But every early return is gated on exhaustion.** Checked all of them: `cur >= ds.max_states`,
`my_event == Pool<DeviceEvent>::kInvalid`, `first_eid == Pool<Edge>::kInvalid`,
`first_vert_off == Pool<VertexId>::kInvalid`, `vid_base + num_new_vars > ...num_keys` — and each
records a `*PoolFull` error. There is no non-exhaustion path out of the function.

So slots leak only once a pool is already full, at which point the run is truncating and emitting
a capacity warning regardless, and *which* states got in is already race-dependent (#161).

**Measured, clean run with no capacity warning:** wolfram24 depth 5 — GPU 815 states / 814 events,
host raw 815. Exact. Consistent with the depth-6 corpus sweep, where all eight workloads matched
the host exactly on `causal_pairs` and `reduced_pairs`, and with `disc-l3a2g2r2` depth 3 agreeing
on 1,662,528 / 971,040.

**Verdict:** not a v1.0.0 blocker. A true atomic multi-pool reserve would add real complexity to
the hottest device function to reclaim a handful of slots at the moment a run has already failed
its capacity budget. The earlier "raw=838860 vs 838861" divergence that opened this is explained:
it was measured at the edge ceiling, i.e. exactly at exhaustion.

**Closes when:** this reclassification is accepted, or a non-exhaustion path out of
`apply_one_match` is exhibited.

### A2 — #12b: GPU sessions (D9)
`V1_PLAN.md:426` calls this the largest remaining item, and Richard put it IN v1.0. The CPU half
is shaped for it (`EngineHolder::extend` is virtual, `SessionSlot` names no device, the FFI uses
a checked `dynamic_cast`). Missing: the device *extend-the-frontier* entry point. `PersistentEvolver`
already retains allocation between calls; what does not persist is the explored graph, because
`engine_->run(in)` evolves from `in.initial_states` every call. A throw must invalidate the session
handle rather than silently continue against a fresh engine (`evolve.cu:838`).
**Closes when:** a device session extends a frontier across calls, an overflow invalidates the
handle, and `verify_sessions.wls` covers the device path.

---

## B. BLOCKING — one decision from Richard, then code

### B1 — S1: how the branchial relation is returned
Measured: the quotient scan is output-optimal (163.4M visits for 133.4M pairs, 81.6% survive), so
there is no algorithmic waste. But the pair FORM is a **137x expansion** of the per-instance
applied lists, which are already stored and already carry the consumed slots that decide overlap
— 971,064 entries against 133,351,476 pairs on `disc-l3a2g2r2` d3.
**The decision:** does a caller asking for branchial structure receive the pairs, or the grouped
form they are derived from? One decision covers host and device — on the device the full relation
needs ~2 GB of pair records (#164), which is why it truncates there today.
**Closes when:** decided, and if grouped output ships, the pair count derivable from it and
verified against full capture.

---

## C. BLOCKING — hygiene the release checklist depends on

### C1 — S4: `tools/` is 55 files nothing builds, and some no longer compile
66 `.cpp`, 11 registered. A syntax-check sample of 8 found **3 that fail to compile**
(`branchial_nondeterminism_probe`, `canonical_causal_oracle`, `causal_tr_determinism_probe`), so
this is not dead weight but unknown state. Two were registered today and both immediately paid:
`ir_core_phase_probe` and `ir_vs_wl` are what closed S2 and found the 1.09% IR win — a real cost
survived precisely because its instrument was unregistered.
**Closes when:** every file under `tools/` is registered in CMake or deleted, and the tree builds.

### C2 — WITHDRAWN. There is no layering violation.
Reported on a grep for `hypergraph/` in `gpu/`, which matched PATH STRINGS INSIDE COMMENTS —
each of the four files names a host file in prose to say what it mirrors
(`edge_signature.hpp:14`, `quotient_expansion.hpp:6`, `quotient_causal.hpp:6`,
`hash_table.hpp:18`). Checked against actual include directives:

    grep -rn '^\s*#\s*include\s*[<"]hypergraph/' gpu/include gpu/src   -> none
    grep -rn '^\s*#\s*include\s*[<"]hg_gpu/' hypergraph/include hypergraph/src -> none

Both directions are clean and always were. The layering property holds without work.

### C3 — WITHDRAWN. The paths are live; the BENCH CORPUS has a coverage gap.
Reported three matcher paths as executing zero times. They are reachable and one of them fires —
the instrument was fine and the corpus was too narrow.

Candidate generation has THREE branches (`pattern_matcher.hpp` ~205-235):
1. no bound vars, no repeated variable in the seed edge -> scan `state_edges` directly, filter by
   arity. Its comment records that this REPLACED drawing from the signature index, which walked
   whole-evolution history filtered by a state bitset.
2. no bound vars, **repeated variable in the seed edge** (`{{x,x}}`) -> `for_each_candidate_cached`
3. bound vars -> `for_each_edge_containing_all`

Measured with three counters in one build, so a zero can be told from a broken instrument:

    all 8 bench workloads      cached=0  sig=0  containing_all=fires on the 6 with multi-edge LHS
    oracle corpus, self-loop   cached=6  sig=6  <- fires exactly there and nowhere else

**The real finding is the coverage gap:** none of the eight bench-corpus workloads has a repeated
variable within a single LHS edge, so a whole matcher branch is unexercised by the corpus that
`bench_cpu_evolve` and the GPU sweeps run. The ORACLE corpus does cover it (`self-loop`), which is
why `all_tests` and `cost_matrix` stayed green.

**Folded into S4/#158** as a corpus-coverage item rather than a deletion: `corpus_gen.hpp` should
emit at least one repeated-variable seed rule.

### C4 — The release checklist's numbers are stale
It records `hg_gpu_tests 99/99`; measured today is **88/88**. `all_tests` is 276 not 275, and
`gpu_differential_tests` 37 not 36 — those two are tests added today and are accounted for. The
99→88 gap is not.
**Closes when:** every gate number in `RELEASE_CHECKLIST.md` matches a measured run, with the
discrepancy explained.

---

## D. BLOCKING — the architectural item Richard scoped IN

### D1 — S5: verdict and containment are separate
One design defect, three instances, all found and fixed on 2026-08-18 as separate bugs:
`SegmentedArray` (`fd78a806`), `ConcurrentMap` (stranded entries), `ConcurrentKeySet`
(`f694c062`). In each, correctness derives from a VERDICT while output comes from ENUMERATION,
kept in step by hand. Every fix re-synchronised one site; none removed the split.
**Direction:** make the verdict and the containment the same act. For `ConcurrentKeySet`: the
winning claim places the key in the head table and that is the whole operation; readers walk the
chain (bounded by growth count); `MIGRATED`/`drained` become advisory compaction that may fail
freely; `count_unique()` becomes the definition and any counter is a hint.
**Closes when:** no state exists in which a thing is claimed-but-unreachable in any of the three,
each carries a GenMC completeness harness (two exist: `lock_free_list_completeness` `33244e82`,
`arena_worker_index_exclusive` `11c8cd2b`), and 20 consecutive suite runs are green.

---

## E. LAST, after everything above

### E1 — Artifact rebuild
`RELEASE_CHECKLIST.md`'s one remaining action, deliberately last because every commit invalidates
all fourteen stamped artifacts:
```
HG_REQUIRE_GPU=1 ./build_all_platforms.sh
python3 tools/dev/artifact_stamp_check.py --require-clean
HG_REQUIRE_ORACLE=1 ./build_linux/all_tests
```
**Runs once, when Richard says the tree is final.** Not a loop condition.

### E2 — Two lines that are not engineering
The Wolfram documentation example pages (needs the doc toolchain), and the doc-accuracy
judgement (no checker decides whether a sentence is true).

---

## NOT BLOCKING — closed or refuted, recorded so they are not reopened

| item | outcome |
|---|---|
| **De-header (#20)** | CLOSED `4614f18`. Moving remaining bodies measured at +0.21% instructions, 0.00 s compile. The lever was the include closure, done (`types.hpp` 757→154 ms). A header:source ratio is not evidence. |
| **S2 IR "1100x pathology"** | REFUTED. Search is flat at leaves=2 nodes=2 depth=1 from C_6 to C_384. Worst real ratio 32.33x, and IR is CHEAPER than WL at engine level (59.0M vs 68.0M instructions) as well as exact. |
| **S2b WL as a pre-filter** | REFUTED. `wlceil` ceiling is 0.0% on cycle4-automorphic and 0.3% on star4 — useless exactly where IR is expensive. Tiered scheme was built anyway: +28%, reverted. |
| **S6 chunk locality memo** | REFUTED. +8.30% instructions; hit rate 1.18% (13.5M hits, 1.13B misses). Access pattern is measured non-local. Do not retry. |
| **S7 linear chunk scan** | REFUTED on the instrument that matters. −37.5% instructions, **0% wall clock**, LL misses and branch mispredicts identical. **The engine is latency-bound**, so instruction-count optimisation is dead work here. |
| **S8 the "15% serial fraction"** | REFUTED. It was exhausted parallelism: 6.80x at depth 6 becomes **14.59x at depth 7** on 32 cores. No serial bottleneck exists. |
| **Performance generally** | Not a blocker. Latency-bound, overlapped to 14.59x on 32 cores, 46% efficiency on an irregular workload. S2b-ii and S3 remain as ATTRIBUTION questions, not known losses. |
