# v1.0.0 final sweeps

Five items, all v1.0.0 scope (Richard, 2026-08-18: "they are v1.0.0 work"). Each states the
evidence that opened it, what would close it, and its status. The task board is unavailable, so
this file IS the tracker; keep it current in the same commit as the work.

---

## S1 — Quotient-side branchial fan-out: is there a quadratic, and is it removable?

**Status: MEASURED. Verdict: the algorithm is output-optimal; the REPRESENTATION is a 137x
expansion. No implementation defect; a representation decision is owed.**

**Why it is open.** `4587254a` refuted the clique hypothesis on the DIRECT path
(`CausalGraph::record_branchial`): max clique over the whole oracle corpus is 5, and on the only
large case (`cycle4-automorphic`, 68,184 edges) it is 2 — 68,184 buckets against 136,368 members,
so the clique form is BIGGER than the pair form. No quadratic to remove there.

But the number that started this, **133,351,476 branchial pairs at depth 3** on
`disc-l3a2g2r2`, was measured under QUOTIENT reconstruction, which mints pairs somewhere else
entirely: `record_branchial_pair` in `hgcommon/quotient_replay_core.hpp` (~line 156), pairing
SIBLINGS OF ONE INSTANCE whose consumed slots overlap. Different structure, different fan-out.
The quadratic claim was made about a path that was never measured.

Growth on the host: 276 pairs (d1) → 162,996 (d2) → 133,351,476 (d3), ~800x per step.

**Close it by:** measuring the applied-matches-per-instance distribution and the
(instance, slot) bucket sizes, exactly as `HG_CLIQUE_STATS` did for the direct path. Then either
(a) the fan-out is wide and a clique/group representation is a real linearisation, or (b) it is
narrow like the direct path and the pair count tracks the bucket count, in which case the
relation is simply that large and the honest answer is a representation question for the caller,
not an optimisation.

### Measured (host, `disc-l3a2g2r2`, branchial only, 8 threads)

Counters added to the host replay Ctx: one scan per application, one visit per element the scan
touches. `visits/scans` is the mean applications per instance, m.

| depth | scans (applications) | visits | mean m | branchial pairs |
|------:|---------------------:|-------:|-------:|----------------:|
| 1 | 24 | 300 | 12.5 | 276 |
| 2 | 2,904 | 175,117 | 60.3 | 162,996 |
| 3 | 971,064 | 163,391,222 | 168.3 | 133,351,476 |

**The scan is output-optimal, so there is no algorithmic quadratic to remove.** 163,391,222
visits produce 133,351,476 pairs: pairs are **81.6%** of visits, so the overlap test rejects
under a fifth of what it looks at. Nearly every sibling pair of an instance genuinely IS
branchially related. Emitting N pairs cannot cost less than N, so the scan is within 1.23x of
its own output, and an inverted index keyed by (instance, slot) — the structure the DIRECT path
uses — could recover at most that 19%. It is not a linearisation and should not be built as one.

**But the output form is a 137x expansion of a structure the engine already has.** m = 168 at
depth 3, against a maximum clique of 2–5 on the direct path, so the two paths are not comparable
and the earlier refutation (`4587254a`) does not transfer. The per-instance applied list is
**971,064 entries against 133,351,476 pairs — 137.3x** — and it already carries the consumed
slots, which are exactly what decides overlap. So the applied lists ARE the branchial relation in
compressed form, and the pair set is their expansion.

### What is owed

A representation decision, not an optimisation: whether a caller asking for branchial structure
should receive 133M pairs or the ~1M-entry grouped form it is derived from. The engine computes
the compressed form either way. This is the same question `#164` raised on the device, where
holding the full relation needs ~2 GB of pair records.

**Gate:** the decision recorded with these numbers; if grouped output ships, the pair count
derivable from it unchanged and the verified against full capture.

---

## S2 — IR high-symmetry pathology (up to 1100x)

**Status: OPEN.**

**Why it is open.** `project_ir_vs_wl_verdict`: IR is exact and subsumes WL in correctness (WL
collides on all 5 1-WL-hard pairs), but blows up on high-automorphism states — cycles measured up
to **1100x slower** — because orbit pruning is weak. `BACKLOG.md` names the mechanism: full
partition copy plus fresh scratch per search node.

Both automorphic corpus cases (`cycle4-automorphic`, `star4-automorphic`) sit at 68,185 raw
states, which is where this bites.

**Close it by:** profiling one pathological state to attribute the 1100x, then either landing an
improvement with the ratio re-measured, or recording a refutation with numbers. Candidates named
in BACKLOG: avoid the per-node partition copy; strengthen orbit pruning.

**Gate:** the 1100x ratio re-measured after any change, on the same state; `all_tests` and
`cost_matrix` 17/17 EXACT unchanged; if refuted, the numbers are recorded and the item closes.

---

## S3 — Optimality numbers, per phase, against bounds

**Status: PARTIAL.**

**What exists.** Device roofline is in `GPU_REENGINEERING_PLAN.md` §1: no phase is
bandwidth-bound or issue-bound (corpus max 6.82% DRAM, 11.53% SM), occupancy pinned at 16.67%,
and two distinct binding resources — divergence (4.0% lane efficiency) where the device wins,
parallel width where it loses (instances 68,185 over 40 classes on cycle4, a 1,705x deficit).

**What is missing.** The HOST has no equivalent. There is no per-phase statement of which
resource binds and what the algorithmic lower bound is for matching, canonicalization,
rewrite/dedup on CPU. `hgcommon::PhaseTimer` exists and `bench_cpu_evolve` reports phase cycles,
so the instrument is there and the analysis is not.

**Close it by:** per-phase host attribution across the corpus with the binding resource named and
the lower bound stated, recorded beside the device table.

**Gate:** every phase either within 2x of a stated bound or its residual attributed to a named,
evidenced cause.

---

## S4 — Neatness sweep

**Status: OPEN.**

**Evidence.**
- `tools/` is 12,155 lines and 57 of 66 files are built by nothing (project CLAUDE.md), one of
  which already fails to compile. That is more than half the size of the host engine, unbuilt.
- FOUR device headers include `hypergraph/` — `edge_signature.hpp`, `quotient_expansion.hpp`,
  `quotient_causal.hpp`, `hash_table.hpp` — a device→host layering violation. The reverse
  direction is clean (0 files), so the property is otherwise true and worth restoring.
- `RELEASE_CHECKLIST.md` cites `hg_gpu_tests 99/99`; measured today is **88/88**. Either tests
  were consolidated and the checklist is stale, or some are gone. Unexplained either way.

**NOT in scope, and recorded so it is not re-opened:** de-header (#20) is CLOSED (`4614f18`).
Moving the remaining bodies was measured at **+0.21% instructions and 0.00 s compile time**; the
lever was the include closure, which was done (`types.hpp` 757 → 154 ms). A raw header:source
ratio is not evidence of a defect here.

**Gate:** every file under `tools/` either registered in CMake or deleted; zero
`gpu/ -> hypergraph/` includes; the checklist's gate numbers match a measured run.

---

## S5 — The verdict/containment design defect

**Status: OPEN. Richard: v1.0.0 scope, not post-release.**

**The defect.** A claim counter maintained independently of the structure that must contain the
thing claimed. Three instances, all found and fixed on 2026-08-18, all the same shape:

| structure | verdict says | reality |
|---|---|---|
| `SegmentedArray` | `count_` publishes index i | the segment holding i may not exist (`fd78a806`) |
| `ConcurrentMap` | insert returns "you won" | the entry may be stranded in a superseded table |
| `ConcurrentKeySet` | `count_` counts the claim | `migrate_into` may leave the key in neither table (`f694c062`) |

Correctness is derived from a VERDICT while output comes from ENUMERATION, and the two are kept
in step by hand. Each fix re-synchronised one more site; none removed the split.

**The direction.** Make the verdict and the containment the same act. For `ConcurrentKeySet`: the
winning claim places the key in the current head table and that is the whole operation; readers
walk the chain (bounded by growth count, logarithmic); `MIGRATED`/`drained` become advisory
compaction that may fail freely because nothing depends on it; `count_unique()` — the enumeration
— becomes the definition and any counter is a hint.

**Gate:** no state in which a thing is claimed-but-unreachable, in any of the three structures; a
GenMC harness stating the completeness property for each (two now exist —
`lock_free_list_completeness` `33244e82`, `arena_worker_index_exclusive` `11c8cd2b`); all gates
green across 20 consecutive suite runs.
