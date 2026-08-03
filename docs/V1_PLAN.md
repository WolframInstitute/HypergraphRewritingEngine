# v1.0.0 execution queue

**This file is the authority for WHAT IS NEXT.** It is tracked, so it survives a fresh clone,
a wiped machine, and a lost conversation. Everything else about v1.0 — rationale, scoping
arguments, per-defect history — lives in the untracked working notes and is NOT needed to
continue.

## Cold pickup, in three commands

```
cat docs/V1_PLAN.md          # this file: what is next, and the gate that closes it
git log --oneline -25        # what actually landed; every message carries its evidence
grep -n "IN PROGRESS\|BLOCKED" docs/V1_PLAN.md
```

Two sections below carry the state a lost conversation would otherwise take with it:
**Landed ledger** (what shipped and the number that proves it) and **Live defect register**
(what is broken now, with the command that shows it).

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
| P1.1 | `hgcommon/join_core.hpp`: the join as ONE `HG_HD` body — pattern-edge order, binding, edge-injectivity, recursion, emit. Templated on candidate enumeration only. | compiles host + `nvcc -arch=sm_89`; no engine wired yet | **DONE** |
| P1.2 | CPU adapter: `pattern_matcher.hpp` supplies an inverted-index candidate iterator and calls `join_core`. Its own DFS is deleted in the same commit. Both host schedulers (recursive and task-based) select the next pattern position with `join_next_position`. | `all_tests` 228/228 incl. `OracleCorpus`/`GoldenMatrix`/`ReferenceOracle`/`MatchCompleteness`; cost +0.26–0.34% instructions, D1 flat | **DONE** |
| P1.3 | GPU adapter: `match.cu` supplies a CSR-slice strider and calls `join_core`. Its own DFS is deleted in the same commit. | `hg_gpu_tests` 97/97 + `gpu_differential_tests` 30/30; 45317 states / 45316 events unchanged; device timing neutral (median 46.8 vs 48.5 ms, spread ~10%) | **DONE** |
| P1.4 | Delta matching (`find_delta_matches`) folded into the same body — it is the same join anchored at a produced edge. | Delivered by P1.2: `scan_pattern_from_edge` IS `hgcommon::join_seed`. Rate unchanged across the extraction — 2b624c8 gave 5,2,3,5,2 failing runs of 204, 59b3cd8 gave 4,1,6,4,1, all forwarding-attributed. The DEVICE has no delta matching at all (full scan per state per rule); that is a missing feature, not a duplicated body. | **DONE** |

| P1.5 | One causal attribution mechanism, not two. The raw-edge rendezvous in `rewriter.cpp` and the quotient reconstruction both compute the causal graph; `configure_identity_and_quotient` picks between them by event-identity mode. | **BLOCKED on four measured gaps — see below. Do NOT delete the rendezvous.** | **BLOCKED** |

**Done-line for P1:** one join implementation in the tree. `grep -c "expand_match\|DFS"` finds
one body, not two. **REACHED for the join** (P1.1–P1.4). P1.5 is a different rule (causal
attribution) and is blocked.

### P1.5 — why it is blocked, measured not argued

Setting `qc = true` in `configure_identity_and_quotient`, so the reconstruction serves every
mode and the rendezvous never runs, builds clean and gives **193/198**. The five failures are
the work list, each with its reproducer:

1. **CRASH on an emptying rule** under the non-Automatic identity modes. FIXED in `3b724c9`
   (two hash functions disagreed on the empty state; one returned 0, the ConcurrentMap EMPTY
   sentinel). `Unified_CanonicalHash.EmptyingRuleEvolvesWithoutError` now crosses all four
   identity modes and both exploration strategies.
2. ~~**ONLINE TR IS NOT EXACT over the reconstructed relation.**~~ **CLOSED** in `65740dc`.
   Was 68 kept against a minimal 48; now minimal at every thread count, and the guard that
   refused to serve it is deleted. Three causes, all measured: the oracle's id-order prune
   (canonical ids are not monotonic), test-once-on-arrival, and retraction being one-shot
   under concurrency — closed by reducing on READ where the arrival discipline cannot hold.
3. **POSITIONAL identity.** `EventIdentityAuthority.PositionalPreservedAndForcesFullCapture`:
   87 events against full capture's 86. Positional reads ranks from each raw state's own
   canonical labelling; the reconstruction does not materialise raw presentations.
4. **CROSS-THREAD DETERMINISM.** `OracleCorpus.CausalBranchialCountsDeterministicAcrossThreads`
   fails once the reconstruction covers the whole corpus rather than the Automatic slice.
   Note `quotient_determinism_rate_probe` measures 0/650 on the CURRENT routing, so widening
   the routing widens the exposure.

(`GoldenMatrix.EveryIdentityCellMatchesItsCachedExpectation` also fails, downstream of 2–4.)

Gate for P1.5 when the four close: `tools/quotient_reconstruction_observables_probe` (exit code
is the disagreement count; **already 0 over 80 configurations** — 4 identity modes x 10
workloads x 2 thread counts, five observables each) plus those five tests.

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
| P3.1 | #33 stage 2: orbit-keyed producer-set rendezvous for quotient causal attribution. Stage 1 (growth determinism) landed. | **NOT REPRODUCIBLE at HEAD**: `quotient_determinism_rate_probe` 0/200 on WPP, and 0/450 across WPP+mixed1+mixed2 with `--load 6`; total 0/650, each sweep threads {1,2,8} x seeds {fixed, random}. Left open: the original firing was 1 in ~90, so 650 clean runs bounds the rate below roughly 0.5% without establishing absence. Re-run after any change to quotient attribution. | **UNREPRODUCIBLE** |
| P3.2 | #65: determinism gate fails ~1/30. Races excluded by TSAN; the ordering class is open. Minimise to a failing case first. | 6000-run sweep, 0 failures | |
| P3.3 | #32: sweep the FFI and GPU paths for the six defect classes the engine audit found. | each class either absent or fixed, listed in the commit | |
| P3.4 | **GPU ignores the sampling caps.** `MaxStatesPerStep` / `MaxSuccessorStatesPerParent` are applied to the CPU engine (`hypergraph_ffi.cpp:425-426`) and have no GPU equivalent in `gpu/include/hg_gpu/evolve.hpp`, with no warning on the GPU path — so the same call returns a different state set per device. Either implement on device or emit `OptionSkipped`. | a capped run agrees CPU vs GPU, or warns | **DONE** `warns`, see below |
| P3.5 | **A dropped frontier state records no error.** `gpu/src/evolve.cu:183` is `pos = atomicAdd(out_count,1); if (pos < out_cap) out_ids[pos] = sid;` — past capacity the state is discarded silently. `kFrontierCapFull` exists and drives grow-and-retry at `:656`, but is not recorded at the drop site. | overflowing the frontier warns and returns partial, per the overflow contract | **DONE** `28c5cc2` |
| P3.6 | **TR under quotient.** The item's premise ("the cited blocker is already closed") was FALSE — the reduction over the reconstructed relation really was non-minimal. So the guard was not lifted; the defect was fixed and the guard then had nothing to do and was DELETED. Three causes: the oracle's id-order prune, test-once-on-arrival, and retraction being one-shot under concurrency. | Automatic arm EXACT at th 1/2/4/8 (512/30/37); full capture ALL EXACT at 1/2/4/8/16; `quotient_determinism_rate_probe` 0/360 at `--load 4`; `all_tests` 229/229 | **DONE** `65740dc` |

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
| P4.6 | #82 residual: the COUNTS route through `observable_*` (`hypergraph_ffi.cpp:1279`), but `CausalGraph`/`BranchialGraph` graph STRUCTURE is still built from materialised raw-event records. Under an identity mode the vertices should be identity classes. | the returned graph's vertices are identity classes under Automatic | |

---

## P5 — Cleanup. Delete, do not archive.

| id | what | gate | status |
|---|---|---|---|
| P5.1 | `tools/` triage: 66 files, **57 built by nothing**, 10,365 lines, `ir_incremental_probe.cpp` already broken. Every survivor is registered in CMake; everything whose question is settled is deleted (its finding lives in the commit that answered it). | `ls tools/*.cpp tools/*.cu` count == CMake-registered count | |
| P5.2 | Dead code from the audit: `EdgeCausalInfo` (`hypergraph/include/hypergraph/types.hpp:490`) is referenced by nothing and is listed in CODEMAP as if it exists. | Deleted; CODEMAP no longer lists it. The causal rendezvous it described is `CausalGraph`'s `get_or_create_edge_producers`/`_consumers`, keyed by `CanonicalEdgeKey`. | **DONE** |
| P5.3 | Fold the three untracked planning docs (`V1_ROADMAP` 163, `V1_EXECUTION` 249, `V1_SCOPING_REGISTER` 753) — one authority, not three. **Needs Richard's go: they are untracked, so deletion is irreversible.** | one planning doc remains | |
| P5.4 | Regenerate `docs/CODEMAP.md` from `tools/dev/source_map.py` instead of maintaining it by hand, or delete it. It has already drifted. | CODEMAP is generated, or gone | |
| P5.7 | **One CUDA error check, not seventeen.** Eight file-local copies in `.cu` plus nine private statics in device containers, byte-identical apart from a hand-written module name. | `HG_CUDA_CHECK` in `gpu/include/hg_gpu/cuda_check.hpp`; 165 call sites; `hg_gpu_tests` 97/97, `gpu_differential_tests` 30/30, error path ground-truthed with a deliberate failing `cudaMalloc` | **DONE** `498185a` |
| P5.8 | **One rule for ConcurrentMap keys built from ids.** Seven maps packed ids raw and collided with EMPTY when an id was 0; the causal nudge additionally ALIASED pair (0,0) with (0,1) and undercounted. | `hypergraph::id_key(a[, b])`, offset-by-one and injective, at every site; `all_tests` 229/229 | **DONE** `3b724c9` |
| P5.5 | `IR_VERIFICATION_NOTES.md` is **tracked** and states the CPU `HashStrategy` enum "still exposes" WL/UT options — they do not exist. Untrack it; move the IR-vs-WL correctness argument to `reference/CANONICALIZATION.md`. | no tracked doc states a non-existent option | |
| P5.6 | The option list is duplicated in three places — `HypergraphRewriting.wl`, `hypergraph_ffi.cpp`'s parser, `HGEvolve.md`. Generate the reference from one definition. | one option definition, reference generated | |

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

---

---

# Landed ledger (append-only; newest block last)

Each row is a commit whose message carries the full evidence. This table is the index, not the
record — `git show <hash>` is the record.

## 2026-08-02

| commit | what | the number that proves it |
|---|---|---|
| `2b624c8` | Restored the grid sweep `3d26343` over-deleted, into `bench_gpu_evolve` mode 2 | Occupancy-bound, not contention-bound: 7 steps, grid 32→3072 gives 338→61 ms, monotone, plateau from ~8x SM. First version reported a FLAT curve — the grid caches in a function-local static, so every in-process row ran at the warmup's grid |
| `9263712` | **P1.2** host matcher on `hgcommon/join_core.hpp`; its DFS and 3 copies of the completion block deleted; `execute_expand_task`'s 4th copy of the next-position rule replaced | `all_tests` 229/229. +0.34%/+0.26%/+0.26% instructions, D1 flat. The adapter itself is +3,428 instr (+0.006%); the rest is a force-inline that costs +0.33% on the unmodified baseline too |
| `59b3cd8` | **P1.3** device matcher on the shared join; `PartialMatch` and its test deleted (−521 lines) | `hg_gpu_tests` 97/97, `gpu_differential_tests` 30/30, 45317 states / 45316 events unchanged, timing neutral (46.8 vs 48.5 ms) |
| `01199bf` | The completeness gate's header named EAGER as the shipping default; `batched_matching_{true}` is | eager arm 2/204, batched arm 0/51 |
| `a99b59c` | **P1.5 step 1**: crossed the S3 gate with event identity — the axis P1.5 turns on | 0 disagreements over 80 configurations (4 identity modes x 10 workloads x 2 thread counts). The instrument was wrong first and its own data caught it: 66 disagreements, ALL in the one column where the reconstruction already ships |
| `bccc5f9` | Emptying-rule test crossed with all four identity modes | green on all four |
| `9a23521` | **P5.2** deleted `EdgeCausalInfo`; CODEMAP listed a type nothing uses | `all_tests` 229/229 |
| `12dedd7` | The source audit counted 186 duplicate names and **printed one** | now prints the 58 shipped ones; first entry led to `498185a` |
| `498185a` | **P5.7** one CUDA error check, not seventeen | 165 call sites; 97/97 + 30/30; throw path ground-truthed with a deliberate failing `cudaMalloc` |
| `9d1a96a` | The source map dropped **every call made from a template body** (dependent `OVERLOADED_DECL_REF`) | 159 dependent calls resolved; false "unreferenced in shipped code" 1176 → 1085. Caught because the audit called `join_dfs`/`join_seed` test-only one commit after both engines were wired to them |
| `3b724c9` | **Quotient exploration threw on any rule that empties the state** — reachable through the public API | `compute_and_cache_state_orbits` started an empty state at hash 0 while its sibling used `EMPTY_STATE_CANONICAL_HASH`; 0 is the ConcurrentMap EMPTY sentinel and that hash keys every quotient map. Also **P5.8** `id_key`. `all_tests` 229/229 |
| `114c903` | **`was_inserted` came from comparing values, not from the exchange** — ConcurrentMap told two callers they inserted one key | fuzz 0/400, was 2/150 and 5/300. NOT a split rendezvous: 720 instrumented runs gave 2 double-claims and ZERO with distinct winning values |
| `50d0408` | Crossed the TR exactness probe with event identity — it swept rule/workload/threads while holding identity at the default, so every EXACT it printed was about the rendezvous | Found the defect below. Its order-replay model does NOT reproduce the engine (714/87/59 vs 654/43/50), so the probe reports three numbers and names NO cause |
| `6895746` | The reduction's id-order assumption was asserted as a fact of the engine; it is a fact of the ID ASSIGNMENT, and the reconstruction violates it | Canonical ids are not monotonic: measured on chain6, producer 9 -> consumer 8, 36 -> 5/10/14, 70 -> 10. `set_ids_are_topological` parameterises it; chain6 43 -> 32 (minimal 30), others unmoved, so it was one cause of two |
| `d15e1b1` | **Retract a kept edge when a later edge supersedes it** — the reduction tested each pair once on arrival and never again, exact only when the consumer's ancestry is already complete | th=1 now EXACT on all three: wolfram5 654 -> 512, chain6 32 -> 30, tri4 50 -> 37. Full capture unchanged, ALL EXACT at 1/2/4/8/16. Cost +0.25%/+0.31%/+0.27%. A test asserting the defect as the spec ("all 3 edges should be stored") now asserts the minimal answer |
| `65740dc` | **Reduce on READ where the arrival discipline cannot hold; guard DELETED** | Retraction is still one-shot under concurrency, so th>=2 stayed non-minimal and varying (chain6 36/38/37). The stored relation is a set and a DAG's reduction is unique, so reducing on read is minimal at ANY thread count by construction. Automatic arm 512/30/37 at th 1/2/4/8; full capture ALL EXACT at 1/2/4/8/16; determinism 0/360 at --load 4; cost +0.24%/+0.29%/+0.26% |
| `84d7c07` | **Automatic event identity returned a REDUCED, non-minimal causal graph.** The TR guard tested the exploration strategy; what makes the reduction wrong is the RECONSTRUCTION serving the graph, which Automatic turns on under full capture too | Before: wolfram5 654 vs minimal 512, chain6 43 vs 30, tri4 50 vs 37. After: un-reduced returned, with a user-visible warning. Rendezvous arm EXACT at 1/2/4/8/16 threads throughout |

### What `114c903` says about the verification, and what was done about it

`concurrent_map_double_growth_2t` offers 100 from W1 and 200 from W2 — always DISTINCT — so
"the stored value equals mine" and "my exchange won" agree in every execution it can enumerate.
It passed **exhaustively at 130,897 executions while the defect shipped**. A harness that cannot
separate a property from its wrong implementation is not evidence.

`verification/genmc/concurrent_map_repeated_offer.cpp` now holds that property and is
**calibrated**: restore the comparison and it reports the safety violation after 1,452
executions; with the fix it exhausts clean at 3,755. Separate harness because folding the
repeated offer into the double-growth one did not finish an exhaustive run in 560s, and an
enumeration that cannot finish proves nothing. `_2t` is unchanged and still exhausts at 130,897.

**The rule this establishes:** a GenMC harness in which the correct rule and the plausible wrong
rule agree on every input is not a check. When adding one, state which wrong implementation it
would catch, and calibrate by breaking the property.

---

# Live defect register

Open, reproducible, with the command. Anything here that is closed moves to the ledger.

| what | reproducer | rate / size |
|---|---|---|
| **Positional identity cannot run through the reconstruction** | `EventIdentityAuthority.PositionalPreservedAndForcesFullCapture`, same forcing | 87 events vs 86 |
| **Cross-thread causal determinism breaks when the reconstruction covers the whole corpus** | `OracleCorpus.CausalBranchialCountsDeterministicAcrossThreads`, same forcing | fails; note the probe is 0/650 on the CURRENT routing |
| **Forwarding loses matches under EAGER submission** | `MatchCompleteness.ForwardedPlusDeltaFindsEveryMatch` | 1–6 of 204 runs, always `fwd=1 delta=0`, on the non-default path. Batched arm asserts 0 and holds. Unchanged across the join extraction (2b624c8: 5,2,3,5,2; 59b3cd8: 4,1,6,4,1) |
| **`tools/` is 66 files with 57 built by nothing** | `ls tools/*.cpp tools/*.cu` against the CMake list | P5.1, 10,365 lines, `ir_incremental_probe.cpp` already broken |

Each row above is also a task in the session task list, so the two cannot drift:
"Online TR is not exact over the reconstructed causal relation",
"Positional event identity cannot run through the quotient reconstruction",
"Cross-thread causal determinism breaks when the reconstruction covers the whole corpus",
"Forwarding loses matches under EAGER submission, 1-6 of 204 runs",
"P5.1 tools/ triage: 57 of 66 files are built by nothing".

The task list is per-session and does not survive a fresh clone; THIS FILE does. When they
disagree, this file is right and the list is rebuilt from it.

---

# Salvaged from the working notes (folded 2026-08-02)

`V1_ROADMAP.md`, `V1_EXECUTION.md` and `V1_SCOPING_REGISTER.md` were read in full (1,165 lines)
and every actionable claim was re-checked against the code. What is below is what survived.

## The bar

> A MAXIMALLY EFFICIENT MULTIWAY HYPERGRAPH REWRITING ENGINE WITH CAUSAL AND BRANCHIAL EDGES,
> OPTIONAL TRANSITIVE REDUCTION OVER THE CAUSAL EDGES, ALL CANONICALISATION MODES FOR STATES AND
> EVENTS, THE QUOTIENTING OPTIMISATION AND ANY OTHERS WE CAN THINK OF, WL HASHING WITH IR
> FALLBACK ONLY WHEN NECESSARY, ALL DATA STRUCTURES OPTIMAL, ALL ALGORITHMIC COMPLEXITY IN EVERY
> CORNER OPTIMAL, THE MOST EFFICIENT MAPPING OF A MODEL OF FUNDAMENTAL PHYSICS TO TODAY'S
> HARDWARE, CPU AND GPU, WORKING WITHIN AND MAXIMISING EFFICIENCY UNDER THEIR PHYSICAL
> CONSTRAINTS.

The paper is **"Rewriting the universe"** and comes last, after the code is done.

## What v1.0.0 has to be

**v1.0.0 must land such that there is no need for a v2.0.0.** That is a claim about **interfaces
and invariants**, not performance — performance can improve forever without a major version.
What would force a v2.0.0: an option surface we regret, a session model that has to change
shape, a spec that contradicts the implementation, a semantic default we have to flip. Those are
what to get right.

**Walk-away-complete** is the organising principle: useful without its author (performance,
utility, no half-finished surface that lures the next person into a trap), legible without its
author (spec, review, one implementation per intersection), self-guarding without its author (CI
that runs what it claims, docs that cannot go stale).

**Every axis gets an explicit DONE-LINE, not "optimal".** The failure mode is asymptotic
optimisation with no stopping condition. When an axis hits its line it is done — stop.

**Constraints are hypotheses, not axioms.** Lock-freedom, no-phases, incremental-everywhere hold
only insofar as they hold the performance bar. The done-line for a constraint is a benchmark,
not a principle.

## Definition of done

- A specification exists and the implementation is reviewed against it; no undocumented corner.
- Quotient is default-capable: observably identical to full capture, on the full corpus
  including cycles, single and multi thread, CPU and GPU.
- No silent correctness degradation anywhere.
- CI runs it all (Linux, GPU, macOS, Windows, paclet golden) on push and PR.
- Every advertised utility works end to end.
- Performance measured to its frontier and ablated; docs generated and doctested.
- Each axis has hit its done-line — nothing is "still being optimised".

## The product is a Pareto FAMILY, not one number

Ship the frontier as coherent operating points, each proven bit-exact: full-capture materialised
(fastest, O(raw)) · **quotient lazy-reconstructed (~16× at depth 7, O(canonical) — the
headline)** · WL-fast canon (approximate, IR-verified) · IR-exact canon · sampled/pruned
(bounded). The claim is ownership of the (speed × memory × exactness) frontier.

## Method rules that keep paying

- **Gate before fix.** A defect is not workable until it has a reproducing gate. Write the gate,
  watch it fail, then fix.
- **Prefer an INVARIANCE check to an EQUALITY check.** An equality between two engines is
  satisfied by two engines wrong in the same way; an invariance cannot be satisfied by an engine
  that is presentation-dependent at all. This is what located #66 after four refuted hypotheses.
- **Fault-inject every new gate.** Two injections against the completeness gate did not fire —
  that is how the structurally-unreachable validator was found.
- **Verify, do not recall.** A fluent unverified claim is the costliest defect class here.
- **Do not fit a bound on a racy quantity to one sample.** `kKnownEagerRaceRuns` was fitted to
  one observation and went red at 7/204; observed spread is 1–7.
- **PROVENANCE: every number carries (object, instrument, commit, date).** A number measured
  before a change that targeted the thing it measures is not evidence about the present.

## Stale claims in the old notes — CHECKED AGAINST CODE, do not re-open

| old claim | verdict |
|---|---|
| GPU silently falls back to 1-WL on oversized states, "no error recorded" | **FALSE now.** `gpu/include/hg_gpu/overflow.hpp:62` states there is no 1-WL fallback; the state is left un-canonicalised and `kIRArenaExhausted`/`kIRDepthExceeded` are recorded. |
| "Two matching engines on CPU — the worst offender" | **STALE.** `parallel_evolution.cpp:1755,1788` call `find_matches`; the task scheduler drives the one matcher. The real duplication is CPU-vs-GPU (P1). |
| `TargetDevice -> "GPU"` silently falls back to CPU on Windows | **FALSE.** `HypergraphRewriting.wl:23` documents fallback **with a message**. |
| `ConcurrentMap` never reclaims superseded resize tables (~2× memory) | **MISLEADING.** The destructor frees the whole chain (`concurrent_map.hpp:136`) and arena-backed tables are bulk-reclaimed. Retention *during* a run is by design — lookups walk the chain. |

## Negative results — DO NOT RE-ATTEMPT

- Tiered canon (WL bucket → IR on collision): **+28%**. "Do not refine the equivalence; refine
  the granularity."
- Incremental WL hashing: built bit-identical, then dropped.
- CUDA-graph capture alone: targets the smallest slice; the real cost was per-call allocation.
- Chain-decomposition / 2-hop for causal reachability: causal DAG width is Θ(N) ⇒ grows memory.
- GPU presentation-order sort for rank stability; raising the device generator budget for #66.
- Sharded ring buffer for GPU occupancy: +30–50% at every grid × shard count.
- Warp-wide canon drain and fully-fused lane burst: 8.0 → 10.9 ms and 10.3 ms.
- "Eager under-pushes, and that is the race" — checked and wrong; eager's low push count is by
  design (push covers existing children, pull covers the rest).

## Measurement facts worth preserving (each with its instrument)

154 µs/step GPU barrier floor · heapAllocs **1567 → 69** · a naive single arena caused **6×**
contention · per-child canonicalisation **≈50%** of instructions (refine ≈26%) · **1100×** IR
blowup on high symmetry, and canonicalisation vs symmetry_groups 1→6 is 8.95 → 37,954 µs
(**~4240×**) · quotient **4.1× @depth6, 16× @depth7** (45k vs 290k raw states) · online TR
1868→1332, 872→872, 201→135 · GPU match kernel 190→4 ms, e2e 834→252 ms, depth 8 373 s→5.5 s ·
binary isolation CPU ~7 ms, GPU 700 ms→28 ms warm · orbit-keying necessity proven on **3000
random hypergraphs** (83 with Aut fusing distinct edge contents) · `full_capture_overhead` ≈
zero (215.33 vs 211.98 µs) · GPU persistent 11.0 → 8.0 ms (6 steps), 65.8 → 37.3 ms at 8× grid.

**STALE, re-measure before any paper use:** the vs-`MultiwaySystem` speedups (3.5× → 94.9× at
steps 1→7) and the thread-scaling curve were measured at `commit-b6c7372`, Oct 2025, on a build
~4–5× slower than today's. The *shape* (speedup grows with depth) is plausibly robust; the
magnitudes are not current.

## Hard-won constraints (expensive to re-discover)

- **WL `StartProcess` cannot carry binary over pipes** — `BinaryWrite` delivers nothing,
  `WriteString` truncates at the first NUL. Hence loopback TCP + atomically-renamed portfile.
- **TDR probing forbidden** on the WSL dev box (physical-reboot risk). **nsys cannot trace GPU
  kernels under WSL2 — use ncu.**
- Absolute GPU numbers need a headless A100/H100, not the 4090 (correctness and relative
  before/after only).
- `macos-13` Intel runner retired Dec 2025 ⇒ x86_64 macOS needs Rosetta or a self-hosted Mac.
- Multi-arch CUDA: `build_linux_gpu` (arch 75–90) blows a 12 GB cicc ceiling on `match.cu`;
  single-arch `build_gpu` is <1 GB.

## Where the future performance lives (direction, not scheduled work)

- **The 1100× IR blowup is not a theoretical wall** — it is IR failing to exploit a structured
  automorphism group (weak orbit pruning). Fix: wreath/imprimitive-aware orbit pruning.
- **The granularity ladder:** (1) state isomorphism *(where we are)* → (2) **match/event orbit**
  — collapse matches under `Aut(state) × Aut(rule)` *before* applying, where "orbit size =
  multiplicity" unifies the causal/branchial bookkeeping → (3) partial-match pruning inside the
  matcher → (4) cross-step/global.
- **Decision criterion:** a quotient invariant avoids looping back iff it is *locally computable*
  — true for automorphisms, false for the global semigroup, so rule algebras / Krohn–Rhodes are
  a verification lens, not a computational shortcut.
- **Scope limit:** random sprinklings have trivial `Aut` generically, so exact symmetry methods
  give **zero** there; that regime needs locality/sparsity/streaming.
- **Cost law:** `work = Θ(#distinct states × per-state canonicalisation)`.

## Open questions register (append-only; DECIDED requires evidence + provenance)

- **Q1 — task grain.** One edge per expand step, or a fraction of the state's edges? Coarser
  amortises canon setup and improves locality; finer exposes more parallelism — and evolution
  scaling is *frontier-width* bound, so they pull opposite ways. Likely answer: adaptive to
  `frontier_width / thread_count`. Blocked on a fresh scaling baseline. **OPEN.**
- **Q2 — is the join worst-case-optimal?** Answered statically: optimal at 1–2 LHS edges (every
  rule in the corpus), Ω(N²) vs the AGM bound N^1.5 on a 3-edge cyclic LHS. No ordering work can
  close it; it needs variable-at-a-time enumeration and an index that supports ordered seek.
  **DECIDED (not a v1.0 blocker); trigger to revisit is a 3+-edge cyclic LHS workload.**
- **Q3 — can pull-forwarding be dropped?** All reclamation of `state_matches_` depends on push
  completeness provably covering the pull walk under the race. **OPEN.**
- **Q4 — is the relaxation look-back bounded?** If a late short path can only relax depth by ≤ B,
  the quiescence signal is a clean depth window at `frontier − B`. **OPEN, measurable.**
- **Q5 — WL prefilter under quotient.** The +28% verdict was measured on a duplicate-heavy
  workload; a singleton WL bucket needs no IR at all. **OPEN (old verdict is the prior to beat).**
- **Q6 — incremental IR / WL.** Two separable claims: warm-start cannot be exact (fundamental);
  no parent→child locality under work-stealing (contingent — child-batching would create it).
  **OPEN (research).**
- **Q7 — is the online TR optimal, and is the concurrent variant novel?** Needs a literature
  review before any novelty claim. **OPEN.**
- **Q8 — reservoir sampling: is depth-scoped completion enough**, or is a barrier-free
  approximate scheme preferable? **OPEN.**
- **Q9 — should unknown options hard-error?** Today they are silently skipped, which is how a
  documented-but-nonexistent option produced no diagnostic. **OPEN — recommend hard-error.**
- **Q11 — concurrent sessions?** **DECIDED: exactly one**, revisitable.
- **Q12 — which Pareto operating points actually ship** as the advertised family? **OPEN.**

## Four shared primitives (build once, several deliverables each)

| primitive | unlocks |
|---|---|
| quiescence (per-depth completion) | mid-run reclamation · reservoir sampling without lockstep · arena tier reset |
| RecordSet / artifact gating | the largest perf win · the session's lazy `Query` · any later `Θ(M)` work |
| serial execution mode | WASM · deterministic debugging · a reference path for differentials |
| session model (`Result`) | continuation · lazy query · GPU device-resident session |

## Also recorded

- **GPU device-resident session** is NOT the persistent kernel. `PersistentEvolver::run()` calls
  `reset()` → `EngineState::clear()`, wiping states/events/causal/frontier — it persists
  allocations only. A session needs the graph to survive across calls.
- **Reclamation hazard:** forward-by-reference `MatchCore`s live *across* generations, so a
  per-generation arena reset is unsafe for match data. Reclamation must split per-state list
  nodes (freeable at expansion quiescence) from `MatchCore`s.
- **Structural debt for P6:** `hypergraph_ffi.cpp` ~4,700 lines and `HypergraphRewriting.wl`
  ~3,800 both need splitting; `types.hpp` is a grab-bag of ~12 unrelated structs; namespaces are
  inconsistent (`hypergraph`, `hgcommon`, `hgmarshal`, `hg_gpu`, `wxf`, `job_system`,
  `lockfree_deque`). **Naming decisions are Richard's** — the convention doc comes before any
  mechanical rename. De-header tension: header-only is implicit inlining, so moving hot code
  into a TU can cost performance; the done-line is a build-time target *with benchmark parity*.
- **Physics asks from Stephen Wolfram, never run:** flat-space/Minkowski equilibrium with closed
  topology (flagged must-have); geodesic bundle divergence; bundle_width=1 classical limit;
  lensing quantification; vertex-lineage tracking.
- **Deferred to v1.x with intent:** full rulial evolution (2-spans acting on spans — the span
  formulation in the spec keeps it reachable); distributed evolution; cooperative single-run
  abort (process-kill is the documented boundary).
- **Needs one line from Richard:** what "removing `std::`" means — hot-path allocation removal
  (already done, 1567→69), freestanding for WASM, or replacing std containers for performance
  (unmeasured).
