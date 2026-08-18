# v1.0.0 — everything that remains

One list. The task board (MCP) is unavailable, so this file and `V1_FINAL_SWEEPS.md` are the
tracker. Clock expired 2026-08-13T04:47Z; this is written 2026-08-18.

Each item states what closes it. Nothing here is "investigate" — the investigations are done and
recorded in `V1_FINAL_SWEEPS.md`; what is left is decisions, code, and one rebuild.

---

## A. BLOCKING — code that must be written

### A1 — #159: a failed `apply_one_match` leaks state and event slots
Pre-existing. A rewrite that fails after claiming ids leaves them claimed. Reorder was attempted
and reverted; the count divergence was never explained after three refuted mechanisms.
**Closes when:** the leak is reproduced in a test, fixed, and the claim/publish counts agree
across 20 suite runs.

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

### C3 — Three matcher paths that execute zero times
`SignatureIndex::for_each_edge_with_signature` (index.hpp:79), `InvertedVertexIndex::for_each_edge`
(index.hpp:248) and `pattern_matcher.hpp:509` were instrumented and ran **zero times** on
wolfram24, wpp and multirule — matching goes through the shared join core.
**Closes when:** each is shown live on some workload, or deleted.

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
