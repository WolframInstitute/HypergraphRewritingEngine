# v1.0.0 execution queue

**This file is the authority for WHAT IS NEXT.** It is tracked, so it survives a fresh clone,
a wiped machine, and a lost conversation. Everything else about v1.0 — rationale, scoping
arguments, per-defect history — lives in the untracked working notes and is NOT needed to
continue.

## Cold pickup, in three commands

```
cat docs/V1_PLAN.md          # this file: what is next, and the gate that closes it
git log --oneline -25        # what actually landed; every message carries its evidence
grep -n "IN PROGRESS" docs/V1_PLAN.md
```

Every commit message names the item id (`P1.2`, `P3.1`). A commit whose message does not carry
the measurement that proves it is not finished.

## Rules this queue is executed under

1. **One item at a time, in order.** Do not start P2 while P1 is open. If an item is blocked,
   say so in its row and move to the next *unblocked* item — do not open three.
2. **One concern per commit, with its gate in the message.** The gate is a command and its
   output, not an adjective.
3. **Never a second implementation of a rule.** Extract to `common/include/hgcommon/`, point
   both callers at it, assert the two agree. This is why P1 is first.
4. **The replaced thing dies in the same commit.** No flag selecting old vs new.
5. **Every new file is registered in CMake in the same commit** or it is not created.
6. **Never commit red.** Fix or revert.

---

## P1 — Delete the duplication that makes everything else cost twice

The matcher is one algorithm written twice: `hypergraph/include/hypergraph/pattern_matcher.hpp`
(619 lines) and `gpu/src/match.cu` (616 lines), sharing only the 20-line
`hgcommon::bind_pattern_edge`. That duplication is the direct cause of the shipping event-count
divergence pinned by `CanonicalEventCount.ReconstructionGapIsStillOpen` (CPU 21, GPU 23). Every
item in P2 costs double until this lands.

| id | what | gate | status |
|---|---|---|---|
| P1.1 | `hgcommon/join_core.hpp`: the join as ONE `HG_HD` body — pattern-edge order, binding, edge-injectivity, recursion, emit. Templated on candidate enumeration only. | compiles host + `nvcc -arch=sm_89`; no engine wired yet | **IN PROGRESS** |
| P1.2 | CPU adapter: `pattern_matcher.hpp` supplies an inverted-index candidate iterator and calls `join_core`. Its own DFS is deleted in the same commit. | `all_tests` green; match counts bit-identical on the oracle corpus before/after | |
| P1.3 | GPU adapter: `match.cu` supplies a CSR-slice strider and calls `join_core`. Its own DFS is deleted in the same commit. | `hg_gpu_tests` + `gpu_differential_tests` green; device match counts unchanged | |
| P1.4 | Delta matching (`find_delta_matches`) folded into the same body — it is the same join anchored at a produced edge. | `test_match_completeness` rate unchanged | |

**Done-line for P1:** one join implementation in the tree. `grep -c "expand_match\|DFS"` finds
one body, not two.

---

## P2 — Close the shipping CPU/GPU divergence (board #81 + slice of #32)

`HGEvolve` returns `hg.observable_num_events()`. Under quotient or `EVENT_SIG_AUTOMATIC` that is
the reconstruction's count. The device has the causal DP (`quotient_causal.hpp`) but no expansion
replay, so it answers with per-state ranks: CPU 21 / GPU 23, and CPU 144 / GPU 15 under quotient
+ mode None. Step 1 landed unwired in `e6d1cb5`.

| id | what | gate | status |
|---|---|---|---|
| P2.1 | Wire `qe_capture_expansion` into the rewrite path beside `qc_register_transition`. | a device probe shows captured matches per class == host's `for_each_expansion_match` count | |
| P2.2 | Device instance pool — `QcInstance`'s twin: per-class instances carrying the producing event per slot, plus alignment of an instance's edges onto the frame. | instance count matches the host's on the oracle corpus | |
| P2.3 | Replay: the `(instance, match)` claim rendezvous, minting one raw event per application. The host's two-sided drive and seq_cst fences must survive the port — an application is not idempotent. | device raw-event count == CPU 144 on `4-cycle + quotient + mode None` | |
| P2.4 | Run-signature dedup: distinct identities under the selected mode. | device event count == CPU 21 on `two_rules_overlap` | |
| P2.5 | Flip `run_cpu` in the differential harness to `observable_num_events()`; replace `ReconstructionGapIsStillOpen` with an equality between devices. | `gpu_differential_tests` green with the equality asserted | |

**Done-line for P2:** the differential suite compares the number `HGEvolve` actually returns,
and the two devices agree on it.

---

## P3 — Correctness closure. Nothing ships with a known nondeterminism.

| id | what | gate | status |
|---|---|---|---|
| P3.1 | #33 stage 2: orbit-keyed producer-set rendezvous for quotient causal attribution. Stage 1 (growth determinism) landed. | `quotient_determinism_rate_probe` 0 mismatches over 100 reps | |
| P3.2 | #65: determinism gate fails ~1/30. Races excluded by TSAN; the ordering class is open. Minimise to a failing case first. | 6000-run sweep, 0 failures | |
| P3.3 | #32: sweep the FFI and GPU paths for the six defect classes the engine audit found. | each class either absent or fixed, listed in the commit | |

---

## P4 — The walk-away surface. What a user needs when Richard steps back.

Strict order: each unblocks the next.

| id | what | gate | status |
|---|---|---|---|
| P4.1 | #6 RecordSet gating — causal/TR/branchial are computed even when unrequested. Also the precondition for any later `Θ(M)` work. | requesting only States does no causal work; measured drop in a cost probe | |
| P4.2 | #10 quiescence signal (per-depth completion). Required by the sampler and the session model. | a depth reports complete exactly once, verified at 1/4/16 threads | |
| P4.3 | #11 serial engine wiring — `job_system` serial mode landed in `9f3fb3b`; the engine never asks for it. | identical output serial vs 1 worker, on the oracle corpus | |
| P4.4 | #12 session model + `Result` — evolve returns a resumable handle, not a one-shot. | continue a run and get the same graph as running it in one call | |
| P4.5 | #16 WASM: a linkable engine library for consumers (hypergraph_viz's WebGPU path). No exported-evolve binary. | `libhypergraph.a` links into a wasm32 consumer and runs the oracle corpus | |

---

## P5 — Cleanup. Delete, do not archive.

| id | what | gate | status |
|---|---|---|---|
| P5.1 | `tools/` triage: 66 files, **57 built by nothing**, 10,365 lines, `ir_incremental_probe.cpp` already broken. Every survivor is registered in CMake; everything whose question is settled is deleted (its finding lives in the commit that answered it). | `ls tools/*.cpp tools/*.cu` count == CMake-registered count | |
| P5.2 | Dead code from the audit: `EdgeCausalInfo` (`hypergraph/include/hypergraph/types.hpp:490`) is referenced by nothing and is listed in CODEMAP as if it exists. | `tools/dev/source_audit.py` reports zero unreferenced *types* in shipped code | |
| P5.3 | Fold the three untracked planning docs (`V1_ROADMAP` 163, `V1_EXECUTION` 249, `V1_SCOPING_REGISTER` 753) — one authority, not three. **Needs Richard's go: they are untracked, so deletion is irreversible.** | one planning doc remains | |
| P5.4 | Regenerate `docs/CODEMAP.md` from `tools/dev/source_map.py` instead of maintaining it by hand, or delete it. It has already drifted. | CODEMAP is generated, or gone | |

---

## P6 — Architecture. Design document to Richard BEFORE any code moves.

| id | what | gate | status |
|---|---|---|---|
| P6.1 | #41 design doc: split shared rewrite semantics from hardware orchestration. Measured input: `gpu → hypergraph` is only **27** references, so the GPU duplicates rather than depends. | Richard has read and approved it | |
| P6.2 | #41 execution, in the order the doc fixes. | each step green on both devices | |
| P6.3 | #20 de-header / restructure, as one dedicated phase, alone. | full suite green; no behaviour change in any gate | |

---

## P7 — Evidence

| id | what | gate | status |
|---|---|---|---|
| P7.1 | #42 rule static analysis: critical pairs (finite, decidable, complete — every parallel-dependent match pair is an instance of one) + GYO acyclicity and `ρ*` for per-rule join classification. | predicted branching matches the observed multiway graph on every shipped rule set | |
| P7.2 | #24 paper + reproducible measurements. | every number in it regenerable by one command | |

---

## Explicitly NOT in v1.0

- The unfolding programme (occurrence net + isomorphism cutoffs). Justified as a v2 headline;
  the argument and its measurements are in the untracked `docs/UNFOLDING_PROGRAMME.md`.
- Match-set representation change (arms B/C refuted with derivations; D/E bounded by `Θ(N·e)`).
- #58 finer sampling knobs — `TransitionRate` is the general sampler and ~70 options already
  exceed the proven surface.
- #30 cache layout — no measurement says it is the bottleneck.
- #77 forwarding-site waste — changes no output; may vanish with P7.1.
- #78 GPU incremental matching — large, and unmeasured as the device bottleneck.

## Richard's one-time items

- paclet-golden CI secret (`WOLFRAMSCRIPT_ENTITLEMENTID`).
- `tools/install_hooks.sh` on the laptop clone.
