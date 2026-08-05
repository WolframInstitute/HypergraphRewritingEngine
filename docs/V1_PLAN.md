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
tools/dev/evidence.sh        # regenerate every headline number; skips say why they skipped
```

Two sections below carry the state a lost conversation would otherwise take with it:
**Landed ledger** (what shipped and the number that proves it) and **Live defect register**
(what is broken now, with the command that shows it).

## WHAT v1.0 IS WAITING ON

**Read `Explicitly NOT in v1.0` below before opening anything.** Four board items (#30, #58, #77,
#78) are named there as out of scope, and it is easy to spend a session on one of them believing
it is milestone work. It has happened.

Of the P1–P7 rows, every one that is still open needs a decision from Richard. None is waiting on
effort.

| row | what is needed | why it cannot be decided here |
|---|---|---|
| ~~**P6.1**~~ | ~~read and approve `docs/P6_ARCHITECTURE_SPLIT.md`~~ **DISCHARGED — its done-line is met, so there is nothing left to approve.** The doc's own §3 names ONE rule still written twice, the quotient reconstruction, and its §4 done-line asks for it in `hgcommon` with host and device supplying policies only. That is what S3/#41 landed: `quotient_causal_core.hpp` (8 `HG_HD` bodies) and `quotient_replay_core.hpp` (4), against `Hypergraph::QcCtx`/`QrCtx` and `hg_gpu::DeviceQcCtx`/`DeviceQrCtx` — storage only, both engines' own bodies deleted. The proposal expected ONE `quotient_core.hpp`; the reconstruction turned out to be TWO rules (the causal DP and the per-instance replay), which is a refinement found by doing it. **Both drifts §3 names are closed, and the first structurally:** the reachability walk is now one body, `hgcommon::qc_reach`, whose dedup lives inside it at `:135`, so a device-only missing visited set can no longer exist; and the device stack is re-measured (`kDeviceStackBytesPerDepth = 8704`) with the recursion bounded. P6.2/P6.3 are no longer sequenced behind an approval |
| **P7.2** | the paper | the machine that regenerates every number in it is done (`tools/dev/evidence.sh`); the writing is Richard's |
| **P4.4** | ~~whether the FFI transport may hold one in-flight job across calls~~ **ANSWERED: it already does.** `hg_evolve --serve` streams length-prefixed WXF jobs, and `--serve-socket` serves them over a loopback TCP port published race-free through a portfile; `hgWorkerStart`/`$hgWorkerProc`/`$hgWorkerSock` in `HypergraphRewriting.wl` start it, connect, frame jobs and fall back to `RunProcess` on any transport failure. A socket rather than the pipe because `StartProcess` drops `BinaryWrite` to stdin and truncates `WriteString` at NUL. **So the session model does NOT need a transport redesign.** What it needs is SESSION SEMANTICS on top of the transport that exists: a handle naming retained engine state, and the four verbs over it. The worker keeps the CUDA context and warm caches across jobs but treats each job as independent — that, not the pipe, is the gap |

**AND THE P1–P7 ROWS ARE NOT THE WHOLE OF v1.0.** `docs/RELEASE_CHECKLIST.md` is the acceptance
gate a release must pass, and it carries requirements this file does not:

| from the checklist | state |
|---|---|
| **Native Windows MSVC+nvcc whole-stack config** — the checklist calls this a **v1.0 blocker** in as many words: until it lands, `TargetDevice -> "GPU"` SILENTLY FALLS BACK TO CPU on Windows | not started, and not tracked anywhere else. This is a silent-degradation blocker of exactly the kind P3 exists to eliminate |
| All 6 platform libraries and 6 `hg_evolve` binaries (Linux/Windows/macOS × x86-64/ARM64); `hg_evolve_gpu` on both CUDA platforms; the `.paclet` archive; `DocumentationBuild` (which evaluates every example cell, so it is also the docs-can't-rot gate); the static-link contract | none verifiable on this machine — they need the release matrix |

Neither file should be read alone. This one orders the WORK; the checklist gates the RELEASE.

`P3.1` is UNREPRODUCIBLE at 0/1100 sweeps and is left open by its own instruction to re-run
after any change to quotient attribution — it blocks nothing.

Everything else is DONE: P1 (all five), P2 (all ten), P3 (all six), P4 except the FFI half of
P4.4, P5 (all eight), and P7.1 in full. P7.2's gate is met by the evidence machine; its paper is
the only deliverable outstanding there.

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

| P1.5 | One causal attribution mechanism, not two. The raw-edge rendezvous in `rewriter.cpp` and the quotient reconstruction both compute the causal graph; `configure_identity_and_quotient` picks between them by event-identity mode. | Gate green on both halves: `quotient_reconstruction_observables_probe` reports **0 disagreements over 80 configurations**, and the suite is **239/239 with the routing forced wide** (`wants_qc = !positional_event_identity()`), up from 193/198. The rendezvous STAYS — see the restatement below. | **DONE** |

**Done-line for P1:** one join implementation in the tree. `grep -c "expand_match\|DFS"` finds
one body, not two. **REACHED for the join** (P1.1–P1.4). P1.5 is a different rule (causal
attribution) and is closed on the restated done-line below.

### P1.5 — the four gaps, measured not argued

Setting `qc = true` in `configure_identity_and_quotient`, so the reconstruction serves every
mode and the rendezvous never runs, gave **193/198** when the item was opened. It now gives
**239/239**. The five failures were the work list, each with its reproducer:

1. **CRASH on an emptying rule** under the non-Automatic identity modes. FIXED in `3b724c9`
   (two hash functions disagreed on the empty state; one returned 0, the ConcurrentMap EMPTY
   sentinel). `Unified_CanonicalHash.EmptyingRuleEvolvesWithoutError` now crosses all four
   identity modes and both exploration strategies.
2. ~~**ONLINE TR IS NOT EXACT over the reconstructed relation.**~~ **CLOSED** in `65740dc`.
   Was 68 kept against a minimal 48; now minimal at every thread count, and the guard that
   refused to serve it is deleted. Three causes, all measured: the oracle's id-order prune
   (canonical ids are not monotonic), test-once-on-arrival, and retraction being one-shot
   under concurrency — closed by reducing on READ where the arrival discipline cannot hold.
3. **POSITIONAL identity — NOT a gap in code sharing, and no work here.** `event_signature`
   (`hgcommon/event_core.hpp:62`) is `HG_HD` and BOTH engines already call it; the key lattice
   is one definition. What differs is the RANK SOURCE: the host under Automatic feeds
   CLASS-FRAME SLOTS from the reconstruction (21 events), the device always feeds ranks in each
   RAW state's own labelling (23). Per-raw-state ranks ARE the Positional convention, so the
   device already computes Positional correctly and merely labels it Automatic; what it lacks
   is the class-frame convention, which needs the expansion replay — **that is P2, not this.**
   On the host, `positional_event_identity_` never touches the rank source at all: it forces
   full capture and excludes the run from `qc`, nothing more.
4. ~~**CROSS-THREAD DETERMINISM.**~~ **CLOSED.** It WAS the TR schedule dependence; closed by
   `65740dc`. `OracleCorpus.CausalBranchialCountsDeterministicAcrossThreads` now passes with the
   routing forced wide, which went 193/198 -> 196/198.

(`GoldenMatrix.EveryIdentityCellMatchesItsCachedExpectation` also failed, downstream of 2–4.)

5. **The five above all pass with the routing forced wide.** Re-running that experiment on top
   of P2 and P3 surfaced two more, both fixed and gated:
   - `RecordSetSkipsOnlyWhatItWasNotAskedFor` reported 191 causal pairs for a run that asked to
     record none, because it read `causal_graph()` — full capture's store — on a run the
     reconstruction was serving. The three corpus gates now read the served relation (`7062996`).
   - A continued run's reconstruction stood at the depth the first `evolve` stopped on.
     `evolve_more` raised the engine's `max_steps_` and never the replay's `qc_max_steps_`
     (`3898da5`).

**RESTATEMENT.** "One causal attribution mechanism" is the wrong done-line. The two mechanisms
consume DIFFERENT DATA — a raw state's own labelling versus a canonical class's frame — and
which is available is decided by the requested identity and the state canonicalization mode. The
correct done-line is **one mechanism per available presentation, with the routing explicit and
checked**, which `2d40f9f` established (and which caught a live defect: any state mode but Full
with Automatic identity returned ZERO events). P2 gave the device the class-frame convention;
CPU and GPU agree at 21.

The rendezvous therefore STAYS, and not as a fallback: it is the only mechanism for the two
presentations the reconstruction cannot read — Positional identity, which needs raw
presentations, and any state canonicalization but `Full`, which computes no edge orbits
(`qc = wants_qc && full_states`). Deleting it would delete those modes.

Gate, both halves green: `tools/quotient_reconstruction_observables_probe` (exit code is the
disagreement count; **0 over 80 configurations** — 4 identity modes x 10 workloads x 2 thread
counts, five observables each) plus the whole host suite with `wants_qc` forced to
`!positional_event_identity()`: **239 passed, 0 failed**, where the item opened at 193/198.

What that measures is CORRECTNESS, and only that. Whether the reconstruction should also become
the DEFAULT for the non-Automatic modes it can now serve is a cost question, and the cost of
running it where full capture currently runs is unmeasured — the "cheaper than full capture"
result (board #4) was measured on the quotient route alone.

---

## P2 — Close the shipping CPU/GPU divergence (board #81 + slice of #32)

`HGEvolve` returns `hg.observable_num_events()`. Under quotient or `EVENT_SIG_AUTOMATIC` that is
the reconstruction's count. The device has the causal DP (`quotient_causal.hpp`) but no expansion
replay, so it answers with per-state ranks: CPU 21 / GPU 23, and CPU 144 / GPU 15 under quotient
+ mode None. Step 1 landed unwired in `e6d1cb5`.

| id | what | gate | status |
|---|---|---|---|
| P2.1 | Wire `qe_capture_expansion` into the rewrite path beside `qc_register_transition`. | a device probe shows captured matches per class == host's `for_each_expansion_match` count | **DONE** `165d15b` |
| P2.2 | Device instance pool — `QcInstance`'s twin: per-class instances carrying the producing event per slot, plus alignment of an instance's edges onto the frame. | instance count matches the host's on the oracle corpus | **DONE** `b0a26b8` |
| P2.3 | Replay: the `(instance, match)` claim rendezvous, minting one raw event per application. The host's two-sided drive and seq_cst fences must survive the port — an application is not idempotent. | device raw-event count == CPU 144 on `4-cycle + quotient + mode None` | **DONE** `b292a03` |
| P2.4 | Run-signature dedup: distinct identities under the selected mode. | device event count == CPU 21 on `two_rules_overlap` | **DONE** `da2d08c` |
| P2.5 | Flip `run_cpu` in the differential harness to `observable_num_events()`; replace `ReconstructionGapIsStillOpen` with an equality between devices. | `gpu_differential_tests` green with the equality asserted | **DONE** `a8aaa1e` |

| P2.6 | Device replay records the reconstructed CAUSAL relation. The device's qc_* DP computes producer SETS; the relation the host serves comes from the per-instance replay. | device pair and edge counts == host's | **DONE** `8e582f2` |
| P2.7 | Device replay records the reconstructed BRANCHIAL relation — the per-instance applied list, publish-then-scan, pair claimed exactly once. | device branchial count == host's | **DONE** `b27cd9e` |
| P2.8 | The differential compares both relations as SETS on the quotient route; `run_cpu` reads the served relation rather than full capture's residue. | cpu set == gpu set on every reconstruction workload | **DONE** `eaf7112` |
| P2.9 | The device serves the REDUCED relation: inline tagging against the kept predecessor adjacency, pruned on ids that are topological for this assignment. | device reduced count AND set == host's; `quotient_wolfram_steps5` gives the reference's 1332 | **DONE** `cbb43e3` |
| P2.10 | `NumCausalEdges` / `NumBranchialEdges` serve the reconstruction on the device, as `hypergraph_ffi.cpp:1319` does on the host. | the differential compares the observable accessors on both routes | **DONE** `9ebe53b` |

**Done-line for P2:** the differential suite compares the number `HGEvolve` actually returns,
and the two devices agree on it.

---

## P3 — Correctness closure. Nothing ships with a known nondeterminism.

| id | what | gate | status |
|---|---|---|---|
| P3.1 | #33 stage 2: orbit-keyed producer-set rendezvous for quotient causal attribution. Stage 1 (growth determinism) landed. | **NOT REPRODUCIBLE**: `quotient_determinism_rate_probe`, each sweep threads {1,2,8} x seeds {fixed, random} over WPP+mixed1+mixed2. 0/650 when the item was opened, and 0/450 again at `f4c478c` with `--load 6` after this session's four changes to quotient attribution — served baselines unchanged (WPP 19772/30063, mixed1 19118/24078, mixed2 10630/5316), which is what the fixes predict, since they touch continuation and this probe does not exercise it. Total 0/1100. Left open: the original firing was 1 in ~90, so this bounds the rate below roughly 0.1% without establishing absence. Re-run after any change to quotient attribution. | **UNREPRODUCIBLE** |
| P3.2 | #65: determinism gate fails ~1/30. Races excluded by TSAN; the ordering class is open. Minimise to a failing case first. | **0 of 6000 sweeps** (8 x 750, load 0/4/8/12; a sweep is threads {1,2,8} x seeds {fixed, random} x 3 workloads = 6 engine runs, so 36,000 runs) on the instrument that reported 3/3 before `0a1141b`. The minimisation found a real defect first: the pair walks reported every endpoint as event id + 1 | **DONE** `0a1141b` |
| P3.3 | #32: sweep the FFI and GPU paths for the six defect classes the engine audit found. | Two present and fixed, three absent with the evidence: **sentinel keys** — a canonical hash of 0 was keyed on, merging every uncomputed state into one dedup slot (`6e86db7`); **empty-state hash** — the devices reserved different values, 0 against `0x9E3779B97F4A7C15` (`99e6856`); **id-order election** — absent, the device pairs both causal and branchial by push-then-scan with pair dedup, and the TR oracle's `ids > p` prune holds on its full-capture route because a producer's event is created before its consumer's; **error swallowing** — absent, every launch chain in `gpu/src` ends in an `HG_CUDA_CHECK`'d `cudaDeviceSynchronize`, and only `cudaFree` returns are discarded; **FFI reads a live graph** — absent, `evolve()` calls `wait_for_completion()` (`parallel_evolution.cpp:181`) before returning and the FFI serializes after it | **DONE** |
| P3.4 | **GPU ignores the sampling caps.** `MaxStatesPerStep` / `MaxSuccessorStatesPerParent` are applied to the CPU engine (`hypergraph_ffi.cpp:425-426`) and have no GPU equivalent in `gpu/include/hg_gpu/evolve.hpp`, with no warning on the GPU path — so the same call returns a different state set per device. Either implement on device or emit `OptionSkipped`. | a capped run agrees CPU vs GPU, or warns | **DONE** `warns`, see below |
| P3.5 | **A dropped frontier state records no error.** `gpu/src/evolve.cu:183` is `pos = atomicAdd(out_count,1); if (pos < out_cap) out_ids[pos] = sid;` — past capacity the state is discarded silently. `kFrontierCapFull` exists and drives grow-and-retry at `:656`, but is not recorded at the drop site. | overflowing the frontier warns and returns partial, per the overflow contract | **DONE** `28c5cc2` |
| P3.6 | **TR under quotient.** The item's premise ("the cited blocker is already closed") was FALSE — the reduction over the reconstructed relation really was non-minimal. So the guard was not lifted; the defect was fixed and the guard then had nothing to do and was DELETED. Three causes: the oracle's id-order prune, test-once-on-arrival, and retraction being one-shot under concurrency. | Automatic arm EXACT at th 1/2/4/8 (512/30/37); full capture ALL EXACT at 1/2/4/8/16; `quotient_determinism_rate_probe` 0/360 at `--load 4`; `all_tests` 229/229 | **DONE** `65740dc` |

---

## P4 — The walk-away surface. What a user needs when Richard steps back.

Strict order: each unblocks the next.

| id | what | gate | status |
|---|---|---|---|
| P4.1 | #6 RecordSet gating — causal/TR/branchial are computed even when unrequested. Also the precondition for any later `Θ(M)` work. | `RecordSet` in `hgcommon`, read by BOTH engines at the four sites that build the relations; both FFIs derive it from what the call returns, sharing `graph_property_needs` with the marshaller. Measured on the axis that matters: where the relations fit in LL, dropping them is 9.5% of DRAM traffic; where they do not, **62%** (353.6 MB → 134.4 MB on `cycle4-automorphic`) | **DONE** `ca3e324` `364ebaf` `68e3456` `b0284dd` `70ae738` `023e680` |
| P4.2 | #10 quiescence signal (per-depth completion). Required by the sampler and the session model. | Derived from LIVE WORK per depth, no barrier: fired once per depth, in depth order, every state at it drained, zero late arrivals, at 1/4/16 threads. REFUSED under quotient, where a child is submitted at its parent's live minimum and a relaxation can put work at a settled depth | **DONE** `ef1f452` |
| P4.3 | #11 serial engine wiring — `job_system` serial mode landed in `9f3fb3b`; the engine never asks for it. | `ExecutionMode::Serial` reaches it; serial vs 4 threads agree on every corpus workload — state and event counts, causal pair and branchial counts, the canonical state hash multiset, and the causal relation as a SET | **DONE** `413cf70` |
| P4.4 | #12 session model + `Result` — evolve returns a resumable handle, not a one-shot. | ENGINE HALF DONE `9c96d41` `3898da5` `f4c478c`: `evolve_more(steps)` resumes from the frontier; N-1 then +1 equals N in one call on every corpus workload. "The frontier" is THREE different sets and each had to be closed separately — full capture's deferred tasks, the reconstruction's depth bound, and quotient exploration's relaxation frontier — so the gate runs all three legs x 17 workloads, comparing states, events, applications, the unreduced causal base, the reduced relation, branchial, the state-hash multiset and the causal relation as a SET. The FFI half (four verbs, memoized artifact graph, splitting `run_rewriting_core`) stays open — but NOT, as P4.4 above used to say, on the transport: the persistent socket worker already ships. It is open on SESSION SEMANTICS over that worker (a handle naming retained engine state; today each job is independent) | **PARTIAL** |
| P4.5 | #16 WASM: a linkable engine library for consumers (hypergraph_viz's WebGPU path). No exported-evolve binary. | `libhypergraph.a` (734 KB) links into `cost_matrix` on wasm32 and runs the FULL oracle corpus under node in 15.9 s, **ALL EXACT**, with counts identical to the native build. A CI leg runs it | **DONE** `7c4ff80` |
| P4.6 | #82 residual: the COUNTS route through `observable_*`, but `CausalGraph`/`BranchialGraph` STRUCTURE is built from materialised raw-event records. | The Source adapter presents the reconstruction as the event set, so the marshaller is untouched: "raw event" is one of the replay's applications, `effective_event_id` is a dense identity, endpoints are the class FRAMES. The pinned 24-against-25 is replaced by the equality | **DONE** `98ade0b` |

---

## P5 — Cleanup. Delete, do not archive.

| id | what | gate | status |
|---|---|---|---|
| P5.1 | `tools/` triage. The headline was stale: a glob gives every non-GPU tool a target. | **65 tracked sources = 62 (host glob) + 3 (gpu CMake)**, `make host_tools` compiles all 62 with zero errors. 21 hand-build recipes removed — that line is what let them rot. The one skipped name is one clone's untracked scratch, not a repo tool. Deleting settled probes is not what the gate asks and is Richard's call | **DONE** `365273d` |
| P5.2 | Dead code from the audit: `EdgeCausalInfo` (`hypergraph/include/hypergraph/types.hpp:490`) is referenced by nothing and is listed in CODEMAP as if it exists. | Deleted; CODEMAP no longer lists it. The causal rendezvous it described is `CausalGraph`'s `get_or_create_edge_producers`/`_consumers`, keyed by `CanonicalEdgeKey`. | **DONE** |
| P5.3 | Fold the three untracked planning docs (`V1_ROADMAP` 163, `V1_EXECUTION` 249, `V1_SCOPING_REGISTER` 753) — one authority, not three. | **THE FOLD IS DONE; only a file deletion remains, and it is not a v1.0 blocker.** All 1,165 lines were read and every actionable claim re-checked against the code — see *Salvaged from the working notes* below, which is the result. THIS FILE is the single authority and has been since that fold: the three are superseded personal notes, not rival plans. They survive only on this clone, excluded through `.git/info/exclude` rather than `.gitignore`, which is where per-clone personal scratch belongs — so removing them is Richard's housekeeping on his own machine, irreversible, and changes nothing about what the release contains. **A fresh clone already sees exactly one planning doc, which is the gate.** | **DONE (gate met)** |
| P5.4 | Regenerate `docs/CODEMAP.md` from `tools/dev/source_map.py` instead of maintaining it by hand, or delete it. It has already drifted. | **NEITHER, and the item's premise was wrong.** `source_map.py` emits a per-definition reference index; CODEMAP is prose saying what a directory is FOR, which no index derives — and deleting it removes the tree's only orientation document. What rots is the INVENTORY, so that is what is now checked: `tools/dev/codemap_check.py` reports MISSING (a source file in a documented directory that CODEMAP does not name) and STALE (a file or backticked identifier CODEMAP names that is in no tracked file), with a CI leg. Ground-truthed before any number: the first run said 80 and the instrument was wrong three ways; 80 → 12 after fixing it, each of the 12 verified by hand, and planting `EdgeCausalInfo` reproduces exactly the P5.2 defect. The 12 were 5 names the tree does not have (three `CausalGraph` methods, an `evolve.cu` kernel, a benchmark) and 7 files present and undocumented. **0 findings** over 13 directories, 443 identifiers, 237 sources | **DONE** `4c5b982` |
| P5.7 | **One CUDA error check, not seventeen.** Eight file-local copies in `.cu` plus nine private statics in device containers, byte-identical apart from a hand-written module name. | `HG_CUDA_CHECK` in `gpu/include/hg_gpu/cuda_check.hpp`; 165 call sites; `hg_gpu_tests` 97/97, `gpu_differential_tests` 30/30, error path ground-truthed with a deliberate failing `cudaMalloc` | **DONE** `498185a` |
| P5.8 | **One rule for ConcurrentMap keys built from ids.** Seven maps packed ids raw and collided with EMPTY when an id was 0; the causal nudge additionally ALIASED pair (0,0) with (0,1) and undercounted. | `hypergraph::id_key(a[, b])`, offset-by-one and injective, at every site; `all_tests` 229/229 | **DONE** `3b724c9` |
| P5.5 | `IR_VERIFICATION_NOTES.md` and the option surface. | **PREMISE FALSE**: the file is NOT tracked — it is in `.git/info/exclude`, so the stated fix is already the state and its `HashStrategy` claim is in one clone's scratch. The gate was tested instead by diffing `Options[HGEvolve]` against the options the FFI parses, and it found the opposite defect: an option the code IGNORES. **`RandomSeed` never reached the sampler it documents** — fixed and calibrated in `d8eea70` | **DONE** `d8eea70` |
| P5.6 | The option list is duplicated — and in FOUR places, not three: declared, SENT by the wrapper, parsed, documented. | A standing gate reconciles all four by reading the sources (`OptionSurface.*`): 18 sent all parsed, 10 documented all accepted. Calibrated by unparsing `RandomSeed`. Generating the reference from one definition would have left the WL declaration and the C++ parser as two hand-written copies — which is where both defects were | **DONE** `1dd8282` |

---

## P6 — Architecture. Design document to Richard BEFORE any code moves.

| id | what | gate | status |
|---|---|---|---|
| P6.1 | #41 design doc: split shared rewrite semantics from hardware orchestration. | **WRITTEN: `docs/P6_ARCHITECTURE_SPLIT.md`.** It revises the item's own premise: `gpu → hypergraph` is **0** library references, not 27 — no GPU library file includes `hypergraph/`, only two test files do, so there is no dependency to break. Every device semantic file already routes through an `hgcommon` core except ONE rule: the quotient reconstruction, 421 host lines against 1,358 device lines. Both copies had already drifted and no gate caught either — the device walk has no visited set where the host does, and the device cascades faulted on a 7-step run (`00e21ee`). Done-line is one body over a container policy, on the `join_core` precedent; schedulers, memory strategy and overflow policy stay per-device by design | **AWAITING RICHARD** |
| P6.2 | #41 execution, in the order the doc fixes. | each step green on both devices | |
| P6.3 | #20 de-header / restructure, as one dedicated phase, alone. | full suite green; no behaviour change in any gate | |

---

## P7 — Evidence

| id | what | gate | status |
|---|---|---|---|
| P7.1 | #42 rule static analysis: critical pairs (finite, decidable, complete — every parallel-dependent match pair is an instance of one) + GYO acyclicity and `ρ*` for per-rule join classification. | GATE: *predicted branching matches the observed multiway graph on every shipped rule set* — **MET**. **The critical-pair half is DONE and WIRED** (`a0a1947`, `4408f57`). `hypergraph/rule_analysis.hpp` decides edge delta, vertex creation rate, LHS shape and `can_branch` — whether two distinct matches can share a CONSUMED edge, which is the branchial relation's own condition. Sound in the FALSE direction only; the gate is asymmetric to match, so a predicted-no that branches FAILS and a predicted-yes that does not is reported. **8 of 17 corpus workloads PROVED branchial-free, all 8 observed exactly 0, no false negatives**; 3 of the remaining 9 are the over-approximation. The engine acts on the false and skips building the relation. **GYO acyclicity and the edge-cover bound are also DONE** (`f98c46a`): `lhs_is_acyclic` runs the GYO ear reduction and `lhs_edge_cover` returns the integral cover, so `N^cover` bounds the matches one state of N edges yields (the AGM bound, integral rather than fractional and so deliberately weaker). The corpus CANNOT test either — every corpus LHS is a path or a pair and all are acyclic — so the gate uses the standard separating examples: triangle and 4-cycle cyclic, 3-path and single edge acyclic. That control caught a first version returning FALSE unconditionally, which the whole corpus accepted. Corpus classification: **all 17 acyclic, covers 1 or 2** — it contains no cyclic join. **`ρ*` is answered without an LP** (`2b31f75`): an alpha-acyclic hypergraph has the integrality property, so its fractional cover LP attains an integral optimum and `ρ* == lhs_edge_cover` EXACTLY. The two separate only on cyclic queries (triangle: cover 2, `ρ*` 3/2; k-cycle: `ρ*` k/2), and the corpus has none, so every bound the header produces is already tight. `edge_cover_tight` reports which case a rule is in, and the closed forms are written down as the test a future LP must reproduce — building a solver now would give it no input and no way to be checked. **The remaining half is not an implementation task.** Of the six adaptations the item lists, one is done and five are blocked on a decision or by measurement: *orbit representatives under high LHS automorphism* and *"always use exact IR" when non-growing* both CHANGE WHAT THE ENGINE OUTPUTS (match multiplicity is observable in the event count; the state mode fixes event identity) and are **Richard's semantics calls, not optimisations**; *map sizing under dedup pressure* is REFUTED in its blanket form (see the do-not-retry table); *denser layout under bounded vertices* is a representation change; and *the WL/IR threshold* needs the cost sweep first — and its premise needs re-checking, since Full/IR already beats None/WL end-to-end (59,028,301 vs 68,004,127). **Those adaptations are board item #42's text, NOT this row's**: the row asks for critical pairs, GYO and `ρ*` for join classification, and all three are delivered and gated. Driving the matcher from the classification is a further step, and two of its five parts change what the engine outputs | **DONE** |
| P7.2 | #24 paper + reproducible measurements. | GATE: *every number in it regenerable by one command* — **MET**. **The MACHINE is done** (`f73f557`): `tools/dev/evidence.sh` regenerates every headline number in one command — corpus exactness, static analysis, reconstruction vs full capture over 80 configurations, the quotient determinism rate, cost matrix, the callgrind route comparison, CODEMAP drift, the option surface, and both device suites. It REPORTS rather than asserts (the gates assert), uses no wall clock anywhere, and a leg that cannot run says SKIPPED with the guard that failed rather than being omitted. It earned its keep on the first run: the cited route figures had drifted ~7% and nothing would have said so (`ee69c66`). **The PAPER is not written** | **PARTIAL** |

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
| `165d15b` | **P2.1** the device captures the class-frame expansion, and routes the reconstruction by the same condition as the host | `qc_route` was quotient-only while the host also routes on Automatic identity; both now read `Full && (quotient \|\| AUTOMATIC)`. Captured matches per class == host's `for_each_expansion_match` |
| `b0a26b8` | **P2.2** device instance store — `QcInstance`'s twin: one instance per raw occurrence of a class at a depth, carrying the producing event PER FRAME SLOT | Slots, not edge ids: an instance built from any raw state of the class replays the frame's captured matches without knowing which raw edges that state has. Roots seeded with `kQeNoProducer` throughout, under the same frame claim the capture uses, so root and capture share one labelling by construction |
| `b292a03` | **P2.3** the `(instance, match)` claim rendezvous and its raw events; both sides publish-then-scan with `__threadfence()` between | device == host: `two_rules_overlap_automatic` 24 frame matches / 58 raw events, `quotient_4cycle_none` 15 / **144** — the count the plan names. 31/31 + 97/97 |
| `15acbdb` | **A class's slots are in the class FRAME, not in whichever state recorded them.** The device had ONE map where the host has two: which state DEFINES a class's expansion, and whose labelling those slots are IN | 65 slots on `4-cycle + quotient` land somewhere other than the recording state's own slot, so 65 were written into a labelling the replay does not read. Aligned by canonical position (`ds.state_edge_rank`), which is what the host's Full correspondence computes; 0 alignment failures |
| `da2d08c` | **P2.4** the run's event identity over the reconstruction | **21 identities from 58 applications** on `two_rules_overlap` under Automatic — the CPU's count. The step signed is the canonical OUTPUT state's, not the replay depth |
| `e36a3fb` | **The determinism rate probe fingerprinted a relation the run does not serve.** It sets quotient exploration, which routes the RECONSTRUCTION, then hashed full capture's `causal_graph()` | served vs hashed pairs: WPP 19772/6755, mixed1 19118/**193**, mixed2 10630/351 — on mixed1 it covered 1% of the relation. Every rate it had reported (0/200, 0/650, 0/360) bounds a relation no caller receives. Also registered in CMake: it was built by nothing and carried a `// Build: g++` line while the plan quoted its numbers |
| `8948d0d` | The probe now carries the five columns its gate compares — states, causal and branchial as SEPARATE fingerprints, plus the instance count and `--dump` | 6000 sweeps had reported 0/6000 while every sweep held six distinct branchial relations |
| `0a1141b` | **The reconstructed pair walks reported every endpoint as event id + 1.** `id_key(a,b)` packs `((a+1)<<32)\|(b+1)`; both readers unpacked with a raw shift, so the maps were right, every COUNT was right, and every count-based gate passed while the relation was wrong | Two runs of ONE configuration (th=1, fixed seed) in separate processes: 3898 of 5316 branchial lines differ, 374 vs 409 distinct pairs, while events (10632 triples), states (52) and causal (10630 pairs) are byte-identical. Identical under `setarch -R`; memcheck 0 errors. After: **0 of 6000 sweeps**, `all_tests` 229/229 |
| `8e582f2` | **P2.6** the replay records the reconstructed causal relation; `id_key`/`id_from_key`/`id_pair_from_key` move to `hgcommon` as `HG_HD` | device == host: 56 pairs / 56 edges on `two_rules_overlap`, 152 / 216 on the 4-cycle |
| `b27cd9e` | **P2.7** the replay records the reconstructed branchial relation. The device buckets the applied list, so each record carries its instance id — without that filter a shared bucket pairs applications of two different occurrences of one class, whose slots are coordinates in the same frame | device == host: 29 and 144 pairs |
| `eaf7112` | **P2.8** the relations are compared as SETS, not counts. The harness had skipped them under quotient on a premise that stopped being true when the reconstruction was routed, and `run_cpu` read full capture's residue there — the same instrument defect as `e36a3fb` and `5bc817d`, in the cross-device gate itself | cpu == gpu on eight workloads, up to `quotient_wolfram_steps6` at 26332 causal and 30063 branchial pairs |
| `cbb43e3` | **The device returned the UN-REDUCED relation where the host serves a minimal one** — it mirrored a host guard `65740dc` deleted, and `165d15b` had widened that to every Automatic run | Inline tagging ported: reduced counts AND sets agree — 56/56, 152/136, 1868/1332, 26332/19772. `quotient_wolfram_steps5`'s 1332 is the reference's causal(TR) |
| `9ebe53b` | **`NumCausalEdges` / `NumBranchialEdges` counted the device's materialised edges** where the CPU FFI takes the reconstruction branch first | The differential now compares the `observable_*` accessors on BOTH routes, so the gated number is the shipped number |
| `98ade0b` | **P4.6** the causal graph's vertices are the events the count reports. The graph scanned materialised events canonicalized per raw state while the count came from the reconstruction's class frames | Needed two things first: a reconstructed event keeping its COMPONENTS (a hash identifies an event and cannot describe one), and the identity set yielding (dense id, raw id, content). The payload carries identity, rule and the endpoint class frames — not consumed/produced edge lists, which the replay never materialises and inventing would be worse than omitting |
| `25a4700` | **The continuation frontier cost 30.1 MB on EVERY run** — P4.4 landed it unmeasured. The delta context stored per boundary state buys nothing there (a frontier state had no scan, so the full scan IS the resume): −17.6 MB. The rest was paid whether or not anyone continued: `set_continuable`, off by default | 324,910,140 (none) / 355,032,864 (with contexts) / 337,423,096 (slim) / 324,910,220 (opt-in). `evolve_more` THROWS rather than returning the graph it already had, which reads as converged. **This commit also carries P4.6 step 1** (`QcEventContent`: a reconstructed event keeps its components, not a hash of them) — two concerns in one commit, from staging a whole directory |
| `1dd8282` | **P5.6** the option surface is four hand-written copies with nothing linking them, so an option can be declared and never sent, sent and never parsed, or documented after it stops existing — each reading as a working option that does nothing | The gate that found `RandomSeed`, made standing. Each half asserts its own regex still matched, because a pattern that quietly stops matching turns a reconciliation into a tautology |
| `d8eea70` | **`RandomSeed` never reached the sampler it is documented to control.** The paclet consumed it in the initial-condition generators and never sent it; the FFI never called `set_random_seed`. So a sampled evolution with a fixed seed returned whatever the engine's default produced | Calibrated by reverting the forward: every seed then gives 132 states on the pinned workload. Found by the P5.5 audit, which was looking for a doc naming a missing option and found an option the code ignores |
| `7c4ff80` | **P4.5** the engine RUNS on wasm32. The compile leg (`04e7738`) said nothing about that: it compiled while the first evolution trapped in `JobSystem::start()` with "thread constructor failed", reached from the oracle helper's hard-coded threaded engine | The Emscripten heap and stack settings were tried FIRST and did not fix it, which is how the thread constructor was reached as the cause rather than assumed away. Full corpus ALL EXACT, `cycle4-automorphic` 68184 events / 109992 causal / 167 MB arena, identical to native. CI runs it, because a compile-only gate is what let a non-running library look finished |
| `9c96d41` | **P4.4 engine half**: a run can be continued. Both budget refusals are kept as the frontier — the match tasks AND the rewrites, because forwarding stores a match on a boundary child and the child's own matching will not re-offer what it already holds | Without the rewrite half the continuation reached EVERY canonical state and still lost transitions: binary-growth 733 events against 873, cycle4-automorphic 52275 against 68184. States matching while events did not is what named it |
| `413cf70` | **P4.3** the engine can ask for serial execution. Serial is not "one worker": one worker spawns a thread and routes through the work-stealing deques, serial spawns nothing and runs every job inline through the injector FIFO — which is what a threadless target needs and what makes the run deterministic by construction | Serial == 4 threads on every corpus workload, compared on the graph rather than on ids |
| `ef1f452` | **P4.2** per-depth completion. Counting STATES was refuted by its own check: match forwarding submits a REWRITE booked against no state's join, and 4 states arrived at a settled depth 4, identically at 1/4/16 threads. Counting TASKS with guards in the EXECUTORS was refuted too — an executor is reachable inline as well as from its job, so chunked rewrites decremented without incrementing: 349 late arrivals | Guards at the job boundary, one per push. Fired once, in order, every state drained, zero late arrivals. `depth_late_arrivals()` is kept, because the signal's claim is that it cannot happen |
| `ca3e324` | **P4.1** a run records what it was asked for. The requested properties gated SERIALIZATION only — the relations were built in full and dropped at the output | 35.4% of the corpus arena is the two relations; `cycle4-automorphic` 179.7 MB → 106.2 MB |
| `364ebaf` | **Two artifacts were recorded through one entry point**, so gating it turned off both: the per-state event LIST (read only by an all-siblings branchial state view) and the branchial PAIR index | Split; the corpus gate proves the three artifacts independent. The FFI derives the record set, and the marshaller's source now THROWS rather than serve an empty graph for a property whose need the name test missed |
| `b0284dd` | `cost_matrix --case NAME --record all\|none`, so a profiler can attribute one case under one record set | Arena bytes overstates one workload and understates another: it said 25% and 41% where DRAM traffic is **9.5% and 62%**. The LL misses that go away are READS, 2.59M → 0.32M — the relation is read-heavy because the reduction walks it |
| `68e3456` | **P4.1 device**: `EvolveInput` carries the record set, `DeviceState` the flags, and all four device recording sites are gated | `RecordSet` moved to `hgcommon`; gate covers both routes. Its first version keyed the comparison on device event ids and reported the causal relation as changed on every run — ids are handed out in arrival order |
| `70ae738` | **One content index**, where two output sections each built the same map with their own pass over every state — the second's comment already read "if not already computed" | Built lazily and shared |
| `023e680` | **A state's serialized edges were re-derived by every incident event** — S + 2E calls over S distinct states, each a full IR canonicalization under Full | 55,860,606 → 47,232,004 instructions (−15.4%) on a StatesGraph request under Full. The item's "IR canonicalization up to 3x per state" was in the wrong place and understated: the two IR sites in the States section are mutually exclusive |
| `99e6856` | **P3.3 the two devices reserved different canonical hashes for the empty state** — host `0x9E3779B97F4A7C15`, device 0 | `CanonicalHash.DeviceHashEqualsHostHash` on `{{x,y}} -> {}`: device `{0, 0, ...}` against host `{..., 11400714819323198485, 11400714819323198485}`. 0 is also the device's "not computed yet", so an empty state was indistinguishable from an unhashed one. The constant now lives once in `hgcommon/core.hpp`; no workload in either corpus reached the empty state, which is why the comparison never saw it |
| `6e86db7` | **P3.3 a canonical hash of 0 was keyed on**, so every uncomputed state shared one dedup slot and all but the first were deduplicated away — a subtree never explored | Such a state is kept and `kUncomputedStateHash` recorded. Ground-truthed: eight states through the predicate with hash 0, all survive, warning raised; calibrated by restoring the nudge, which reports states 1..7 deduplicated away. `state_survives_dedup` moved into the header its own contract demands — a device function in one `.cu` is unreachable from another target's device link |
| `5bc817d` | **The determinism gate fingerprinted two relations the quotient route does not serve.** Its causal component split on `quotient_reconstruction()`; its events and branchial components did not | Calibrated by breaking the property: with the id+1 unpack reintroduced, `QuotientStatesEventsBranchialDeterministic` fails 3 of 3; before the change it passed while that defect shipped. At HEAD 0 failures in 30 runs |
| `a8aaa1e` | **P2.5** both devices serve the reconstructed count; the harness compared a number no caller receives (`num_events()` where the FFI serves `observable_num_events()`) | Every mode x exploration cell agrees, cpu vs gpu: None 144/144, Automatic 15/144, Full 3/144 under full capture; 144/15, 15/15, 3/15 under quotient. `ReconstructionGapIsStillOpen` replaced by an equality that keeps the literal 21 |
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

# Measured do-not-retry

Routes closed by measurement, with the number that closed them. Each is here because the idea is
plausible enough to be had again.

| idea | measurement | verdict |
|---|---|---|
| Raise `ConcurrentMap::DEFAULT_INITIAL_CAPACITY` above 1024 | 1024 → 8192 costs **+3.97%** of the end-to-end corpus total (3,126,320,794 → 3,250,520,730): most maps in a run are small and each pays 8192-entry construction | blanket increase is wrong; only per-map sizing from a real estimate could pay, and the estimate does not exist yet |
| Memoize `MatchRecord::hash()` in `MatchCore` | saves **0.003%** (106,203 instructions) against an 8-byte growth per core and a seal-before-hash invariant that is silent when violated | not worth it at that size |
| Key-only sets for the claim maps | a real **0.53%** saving, then 4 of 12 invocations fail — `ConcurrentIdSet` double-claims across a resize | reopen only with that defect fixed AND a fuzz-rate gate; see the register |
| `VariableBinding`'s 128-byte `0xFF` fill is waste | it is the contract `rewrite_core.hpp` depends on — it reads `bindings[var]` directly and takes `INVALID_ID` to mean unbound | not waste; removing it reads uninitialised memory |
| Tiered canon (WL bucket, IR on collision) | +28% pessimisation, recorded earlier | stands, and is stronger now that WL is known not to be the cheap path |

---

## 2026-08-03

| commit | what | the number that proves it |
|---|---|---|
| `7062996` | Three corpus gates read `causal_graph()` — full capture's store — on runs the RECONSTRUCTION serves | RecordSet reported 191 causal pairs for a run asked to record none. One shared `served_causal_pairs()` picks the walk the run actually serves |
| `3898da5` | **P4.4** a continued run's reconstruction stood at the depth the first `evolve` stopped on: `evolve_more` raised `max_steps_` and never `qc_max_steps_` | Five gates read that bound; raising it alone is not enough, because each decides at PUBLISH time, so `raise_quotient_max_steps` re-drives the frontier over a new enumerable reach list. `disconnected-lhs` replayed 254 applications against 126 before the bound was restricted to `[old, new)` |
| `f4c478c` | **P4.4** quotient exploration DROPPED its continuation frontier instead of deferring it — a third frontier, distinct from deferred match and rewrite tasks | `binary-growth` s=20 vs 36, `wolfram-2to4` s=10 vs 45. The claim is deliberately not taken over budget, so a shorter path can still relax below it. `evolve_more`'s early return on an empty frontier also went: with ONE canonical class the exploration settles at depth 0 while the replay has every depth to run |
| `4c5b982` | **P5.4** CODEMAP cannot drift unnoticed — checker + CI leg | 0 findings over 13 directories, 443 identifiers, 237 sources. The instrument was wrong three ways first (80 → 12); all 12 hand-verified; `EdgeCausalInfo` planted as the positive control |
| `00e21ee` | **GPU RELEASE BLOCKER**: a 7-step quotient run faulted the device and returned NOTHING, poisoning the context so 11 unrelated workloads also returned nothing | Two mutually recursive cascades descend once per depth against a stack sized for the matcher. Cost per level MEASURED: 32 KB faults entering depth 7, 64 KB entering depth 13 → 32768/6 == 65536/12 == 5461 bytes. Stack now sized from the depth AND both cascades bounded, because stack is per resident thread and cannot simply be enlarged |
| `4f93149` | The device reachability walk expanded a node once per PATH, not once; the host's has carried a visited set all along | No speedup measurable: the walk is over KEPT predecessors, and reduction leaves most events with one, so paths and nodes stay close. It was the BOUND that was wrong, not a cost being paid |
| `ec16f78` | `match_epoch_` and `child_epoch_` were WRITE-ONLY — never loaded anywhere — while the header described them as the live pull/push retry protocol | 9 mentions, all declarations, comments or the `fetch_add` itself. Deleted: two cache lines, one RMW per match store and per child registration, an 8-byte field per `ChildInfo`, and 14 lines of comment asserting a protocol that does not exist |
| `aa2fd41` → `1aae458` | **#95 is not a defect.** Eager forwarding was tracked as losing matches at 3%, then 0.1%. Both were biased proxies | `contains_match` — the validator's own test, probing the whole chain and comparing the RECORD — says **ZERO lost over 4,080 validated runs**. Positive control: disabling `push_match_to_children` makes it report 10 lost in 7 runs. Gate is now exact, with no tolerance |
| `a256c0e` `a1a7a00` | **#40** three instruments invited inferences their own data contradicts | `ir_vs_wl`'s 5–38x per-call ratio reads as "WL is the fast path"; end-to-end, None/WL costs 68,004,127 instructions against Full/IR's 59,028,301. Scope now travels with the number |
| `a0a1947` `4408f57` | **P7.1** critical pairs: `can_branch` decides whether two matches can share a CONSUMED edge — the branchial relation's own condition | **8 of 17 corpus workloads PROVED branchial-free, all 8 observed exactly 0, no false negatives.** Sound in the FALSE direction only, and the gate is asymmetric to match. The engine acts on the proof and skips building the relation |
| `f98c46a` `2b31f75` | **P7.1** GYO acyclicity, the integral edge cover, and `ρ*` | The corpus CANNOT test GYO — every corpus LHS is acyclic — and my first version returned FALSE unconditionally with the corpus accepting it. Gated on the triangle and 4-cycle instead. `ρ*` needs no LP: acyclic hypergraphs have the integrality property, so `ρ* == cover` exactly, and the corpus has no cyclic rule |
| `f73f557` `4d8a2e7` | **P7.2** the evidence machine: one command regenerates every headline number | Reproduces to **2 ppm** (corpus total 3,125,670,179 → 3,125,675,709 across runs) against wall clock's >10%. It earned its keep immediately: the route figures cited in three files were ~7% stale (`ee69c66`) |
| `e4c735d` | `MatchRecord::hash()` tested `is_bound` on all 32 slots; the mask it reads is the same one | 461,946 instructions, 0.015%. Small, and stated as small |
| `5382367` → `f48d227` | **REVERTED.** Key-only sets for seven claim maps measured a real 0.53% saving, then failed | `ConcurrentIdSet` double-claims across a resize: 4 of 12 invocations, 0 with the maps. Two defects in a class with ZERO users, so no gate could reach either. The green suite that shipped it was luck |
| `c60fabe` | Attributed a +0.036% drift rather than leaving it | Bisected with callgrind to `1aae458`, on a path where the validator is off and the changed map is never touched — a layout artifact, not work |
| `fc6d24a` | **RELEASE BLOCKER.** `./build_all_platforms.sh` built **2 of 6** platforms. The Windows `.def` exported three functions the visualisation split deleted | A Windows linker resolves every name under `EXPORTS`; an ELF `.so` links with undefined symbols and reports nothing, and every gate here runs on Linux. Windows-x86-64 and Windows-ARM64 both build after |
| `29c9345` | **RELEASE BLOCKER.** macOS could not compile `park.hpp` — both its wait primitives were out of reach | OSXCross carries MacOSX12.3.sdk, and `os_sync_wait_on_address` needs 14.4; the `std::atomic::wait` fallthrough is unavailable below macOS 11 while `macos-cross.cmake:34` deploys to 10.15. A condvar backend, following `atomic_compat.hpp`'s precedent for the same SDK rather than raising the deployment target, which drops 10.15 users and is not mine to decide. **6 of 6 platforms build**; `all_tests` 213/213 |
| `a1a1181` | Both breaks were invisible for ONE structural reason, so both got a gate | Every gate builds on Linux, and the on-push Windows/macOS legs configure `-DBUILD_WOLFRAM_LANGUAGE_PACLET=OFF`, so neither consumed the broken artifact. `def_exports_check.py` checks the export list against the sources with no linker — **ground-truthed: 3 findings on the `.def` at `fc6d24a^`, naming exactly the three deleted functions; 0 now.** macOS gains a compile-only configure at the SHIPPED `-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15` |
| `958c83c` | `CreatePacletArchive` shipped a gitignored doc-build intermediate | Archiving `paclet/` gave **96 MB from a 599 MB tree, 512 MB of it `Documentation/Source/generated/`**, against 6.4 MB of built docs. Staging only what `PacletInfo.wl` declares gives **30 MB, 41 entries, a library for each of the six SystemIDs**, and the script fails if a `Documentation/Source` entry appears |
| `e130dae` | The golden verifier loaded `paclet/` from disk, which says nothing about the artifact a user receives | The archive is a staged SUBSET, so a dropped file or a library unfindable by `$SystemID` shows up only on an install. `--archive` installs `dist/*.paclet` and runs the corpus against it: **12/12, Failed NONE**, CPU == GPU == `{5,33,32,43}`, on the archive's **Windows-x86-64** library |

# v1.0.0 sprint — ordered by dependency, status live

The release CHECKLIST is 20/20 (`docs/RELEASE_CHECKLIST.md`); what follows is the ENGINEERING
work v1.0.0 needs beyond shipping mechanics. Ordered so nothing is built on something that is
about to be replaced. De-header is last and alone, by standing decision.

## What actually remains — FIVE items

Read this before the tables. The task list carries many open entries; most are not blockers, and
counting them as such has misread the state more than once.

| # | what | why it is not done | needs |
|---|---|---|---|
| #41 | device half of the per-instance replay | WRITTEN, reverted, **now diagnosed by measurement** (`33e7cc0`). The fault was never a data-size problem: the replay's own new data is 64 bytes (`producers[MAX_PATTERN_EDGES]`), and the cost is TWO ADDITIONAL ABI FRAMES in the recursion cycle — `qe_apply`, reduced to building the Ctx and forwarding, and `DeviceQrCtx::descend`, called once at `qr_apply`'s tail. **`__forceinline__` IS REFUTED** (2026-08-05): nvcc ignored it on both, because both sit in a recursive cycle where it cannot bound the expansion. Measured after the build — all six frames still present, depot sum **3168**, byte-identical to the failing build — and the gate came back `gpu_differential_tests` **8/36** with `PastTheStackDepthItRecordsRatherThanFaults` failing and `gpu=0` on every endpoint count, which is a fault aborting the run. Reverted; the extraction is saved at `scratchpad/41_extraction_RESOLVED.patch`. So the constant DOES have to move, and that is now a measurement rather than an inflation: 6 frames against 4. | **DONE** `b300af0`. The device drives `hgcommon::qr_apply` through a storage-only `DeviceQrCtx`; its own replay is deleted. `kDeviceStackBytesPerDepth` moved 5632 → **8704**, DERIVED from two depot measurements (4-frame cycle 2000 bytes, 6-frame 3168) plus the ABI term the original bisection pins ((5461−2000)/4 = 865/frame), so 3168 + 6×865 = 8360 → 8704 at the next 512. `__forceinline__` was tried first and is REFUTED — nvcc ignores it inside a recursive cycle, frames unchanged at 3168, gate 8/36. Gates: `gpu_differential_tests` 36/36 incl. the stack test, `hg_gpu_tests` 98/98 |
| #119 | one CSR lookup instead of three | WRITTEN, gate-ready, parked unbuilt. Patch saved. | the same CUDA build |
| #114 | intra-state IR parallelism on device — decide | **CLOSED, DO NOT BUILD IT** (`3a35bde`). Needed no GPU: the IR body is shared since S1, so the counts are the device's too, and the only differing input (generator budget, 512 host / 32 device) is a parameter. The search tree is **1–4 leaves at depth ≤1** — and `C_6` through `C_384`, the worst case symmetry can present, all report **leaves=2, nodes=2, depth=1**, so it does not grow with state size. Per-state cost is flat: the heaviest 1% of states carry ~1.5% of all leaves in both populations. Nothing to fan out, no straggler tail. What remains unsettled is per-NODE work (refinement, 31% of the pipeline; and the O(generators × vertices) orbit union-find) — that is where a large cycle's cost actually is | **DONE** `3a35bde` |
| #12 | FFI session model | **DESIGN CLOSED 2026-08-05 (D7-D11, `FFI_INTERFACE_DESIGN` 3.4b).** Three of its four parts were already done and had to be MEASURED to find that out: the transport ships (`--serve-socket`), the artifact set is honoured in the rewrite hot path, and canonicalization runs at **1.00 per state** (the floor). What remains is session semantics: an opaque `uint64` handle, a worker-side handle->engine map, the four verbs, a lifetime rule, and **raw vertex labels preserved across `Open`** (D10). One session at a time (D7), `Query` returns one blob with chunking deferred (D8). **D12 is discharged (D15): #12 no longer waits on #41.** D12 deferred it so a CPU-only design could not get a device constraint wrong; the two it protected against are now on record — D13 (the gap is retained **exploration**, not retained allocation: `PersistentEvolver` already reuses its engine when the config is unchanged, `evolve.cu:811`, but `run(in)` re-evolves from `in.initial_states` every call) and D14 (a throw discards the engine by design, `evolve.cu:838`, so an overflow must **invalidate the handle** rather than silently continue against a fresh empty engine). Both were read out of the device source, neither needed #41. The surviving rule is narrower: the envelope and handle space are **designed once for both devices**, and a CPU-shaped holder that cannot carry a device engine is what must not be built | **STEPS 1–2 LANDED.** `68f8e55` — the envelope has optional `Op`/`Session`; no `Op` means `Evolve`, so an existing caller's bytes take the path they took before, and a named-but-unserved verb is **refused rather than ignored** (a silently ignored verb has the caller read a one-shot result as a session's, every field looking right). `1ac2f22` — `paclet_source/session.hpp`: `EngineHolder` interface + `SessionSlot`, **device-agnostic by construction**, 7 tests, encoding D7 (a second `Open` is an error, not an eviction), D11 (handles never reused after `Close`), D13 (`extend` advances the *current* frontier), D14 (an invalidated handle stays addressable so the next verb reports the history is gone). `d7e8d1d` — `Open` retains the engine and returns its handle, `Close` releases it. `4269df3` — **`Step` and `Query` serve a held session, and steps 1–3 are done.** The op-boundary split was NOT the serializer extraction this row predicted: the boundary is **where the engine comes from**, not where serialization begins. `run_rewriting_core` already binds `hg` and `engine` as references into a holder, so binding them from the session's holder instead of a fresh one makes the existing 900-line serializer serve all four verbs unchanged — no request struct, no second output path. What is gated is the fresh-run configuration plus the evolution, which is exactly what a held engine already has. Two semantics decisions came out of it: **D16** a held verb takes its identity (`CanonicalizeStates`/`Events`, positional, genesis, `Steps`) from the SESSION, read back off the engine — reading the envelope's copy would report `Full`-mode canonical forms as tree-mode ones in fields that all still parse — and rules on a held verb are refused; **D17** an unrecorded artifact WARNS rather than errors, against §2's table, because `include_num_causal_edges`/`branchial` default to TRUE and an error would reject the ordinary default `Query`. The GPU binary now refuses every session op naming `TargetDevice`; before, an `Open` there returned a result with no `Session` key, which reads as a session in every field a caller can see. Step 4 is GPU wiring (#121). Gate: `all_tests` **256/256** including `StepContinuesTheHeldExplorationAndQueryOnlyReportsIt` (1 step then 2 equals 3 steps on four counts, with a non-vacuity guard that found the `value_bytes` defect fixed in `48bc0bf`); only step 4 needs a CUDA window |
| **#124** | **CLOSED `5ded42a` — the device reported every reconstructed event identity wrong** | `gpu/include/hg_gpu/quotient_expansion.hpp` seeded FNV with **1469598103934665603** — the 64-bit FNV-1a basis `14695981039346656037` **with its last digit missing** (`14695981039346656037 // 10` is exactly it). Same prime, same field order, same inputs; only the starting constant differed, and FNV mixing is invertible, so it is a BIJECTION: the partition of events into identity classes survives while every value changes. Measured signature — `gpu_differential_tests` 26/36, 10 of 28 corpus cases, exactly 30 assertions (10 causal + 10 reduced + 10 branchial); counts, event multisets, `NumCausalEdges`/`NumBranchialEdges` and the device's own canonical hashes all AGREE; only the relation identities differ, with equal distinct-identity counts and ZERO overlap. **PROVEN arithmetically without a build** (`tools/dev/invert_content_triple.py`): on `deep_cone_reduction_d6`, which has exactly one distinct identity per engine so the pairing is forced, the correct basis reproduces the CPU's `8836476779998324405` and the digit-dropped basis reproduces the GPU's `11458423332341610703`, both 64-bit exact. **How it arose:** before `213c710` the HOST open-coded the same wrong constant, so the two agreed — both wrong. `213c710` pointed the host at `hgcommon::qr_content_hash` (correct basis) and left the device's third copy; unifying two of three copies turned a latent duplication into a live divergence, and no gate caught it because the device gates had not run since 2026-08-03 | **DONE** `5ded42a`. The fix DELETES the spelling rather than correcting the digit — the device calls `hgcommon::qr_content_hash`, the same body the host reaches through `QcEventContent::triple_hash`; a corrected literal would have left a third copy to drift again. Gate: `gpu_differential_tests` **36/36** (was 26/36; the corpus leg 18/28 → 28/28), `hg_gpu_tests` 98/98 |
| #12b | **GPU sessions** (D9) | Richard put these IN v1.0, and they are the largest remaining item. **The mechanism was recorded wrong and is now read from the code** (`gpu/src/evolve.cu:788`): `PersistentEvolver` does *not* reset between successful calls — it rebuilds only when the config changed (`memcmp(&cfg, &cfg_, ...)`), so retained **allocation** already works and needs nothing. What does not persist is the **explored graph**: `engine_->run(in)` takes a whole `EvolveInput` and evolves from `in.initial_states` every call. The new device API is therefore *extend the existing frontier*, not *keep buffers warm*. Second constraint (`evolve.cu:838`): a throw discards the engine deliberately, so an overflow must **invalidate the session handle** rather than silently continue against a fresh empty engine — which would serve a graph that lost its history and pass every internal check. Both are D13/D14 | **UNBLOCKED.** #41 (`b300af0`) and #119 (`73a0d60`) are closed, and the CPU half is shaped for this: `EngineHolder` is an interface with a virtual `extend`, `SessionSlot` never names a device, and the FFI's holder acquisition uses a **checked** `dynamic_cast` so a device-opened handle cannot be read as a host one. What remains is the device extend-the-frontier entry point D13 names. Needs a CUDA window |
| #20 | de-header + redesign | standing decision: LAST, alone, once the native code is locked. #41 is its real prerequisite — de-headering two copies of a rule bakes the divergence in. **SCOPE MEASURED 2026-08-05:** 68% of the host engine and 56% of the device engine live in headers — `hypergraph/include` 22 files / 9650 lines against `hypergraph/src` 4499, `gpu/include/hg_gpu` 26 files / 4876 against `gpu/src` 3905. `common/include/hgcommon` (13 files, 2414 lines) is EXCLUDED by design: its bodies are `HG_HD` and nvcc compiles them into device code, so a `.cu` cannot link them from a host `.cpp`. **De-headering will NOT fix the device build cost, and it is worth not believing it would:** `gpu/src/persistent.cu` needs more than **8 GB of ptxas RSS** at `-j1` (killed at an 8 GB group-RSS ceiling with 13 GB free and no competing tenant), yet it is not the largest translation unit — preprocessed it is 122,434 lines against `evolve.cu`'s 137,699 and `match.cu`'s 126,530, both of which compile without trouble. The consumer is **`cicc`**, the LLVM-based CUDA frontend — it dies with `LLVM ERROR: out of memory`, and the earlier attribution to ptxas came from `ptxas died due to signal 15` in a *killed* build, which names the process holding the signal rather than the memory. What `persistent.cu` uniquely contains is the recursive replay cycle inside one large kernel (nvlink: `stack size for entry function k_persistent_evolve cannot be statically determined`). **`--threads` is refuted as the cause** (`0d0cf39`): capped at 10 GB, `--threads=1` peaked at 8,490,308 KB and `--threads=0` at 8,773,304 KB — 3% apart — because it parallelises per-*architecture* codegen and `cicc` runs once per *virtual* architecture, of which there is one. Tracked as #123 | **MEASURED 2026-08-05; the premise does not survive.** #41 is closed (`b300af0`) so this is unblocked, and the first thing it bought was a measurement that changes what the item IS. **What can move:** `hypergraph/include` has 404 movable bodies over 2100 lines of 9656 -- median 3 lines, largest 51, 291 of 404 under 6 lines, and the four over 40 lines are `MatchRecord::hash`, `allocate_local`, `InvertedVertexIndex::add_edge` and `SignatureIndex::add_edge`, all hot path. 825 lines are template and cannot move at all. **Where the cost is:** a TU including the three main engine headers takes 1198 ms; the 27 standard headers they reach take 928 ms of it, so the engine's own headers are **270 ms, 22.5%**. Preprocessed volume across 15 host TUs is 1,674,093 lines of which **3.7%** is this project. **And the costs are superadditive:** dropping `<sstream>` from the engine's std set saves 11 ms and `<random>` saves 51, but dropping BOTH saves 196 -- they share a sub-closure that only goes when the last consumer does, so a per-file commit quoting `<sstream>`'s 617 ms in isolation would be wrong by ~56x. **The lever is the include closure, as a SET, not the bodies.** Instruments in `078a781` (`source_map.py --pin/extent`, `deheader_shape.py`). Landed: `f9d537d` -- `pattern.hpp` used `std::vector` without including it and compiled only because every includer supplied it first, so all 22 host headers now compile alone. `455e35f` -- **`<sstream>` and `<random>` leave the closure: 1582 ms -> 1440 ms, -142 ms, -9.0%** on the same TU under the flags the build actually uses (`-O3 -DNDEBUG -std=c++20`, conda gcc 15.2.0, the compiler in `CMakeCache.txt`), min of 8 interleaved runs. `455e35f`'s own message quotes -216 ms / -18.8%, which was measured at `-std=c++17 -O2` -- flags this project does not build with. **Measure a build cost with the build's flags**: c++20 raises the whole TU from 1149 ms to 1582 ms, so the same edit is a smaller fraction of a larger number. Three bodies pinned them and each moved: `CanonicalForm::to_string` was a TEST-ONLY formatter in a shipping header (its only two callers are assertion messages) and moves to the tests; `sampling_rng` was a member whose declaration spelled `std::mt19937` in the header while its only use of `this` was to read two fields, so it is now file-local in the `.cpp`; and the work-stealing victim draw ran `std::mt19937` -- 2.5 KB of state per worker to compute `rng() % n` -- now `hgcommon::splitmix64`, the project's existing mixer. `job_system.hpp` was the binding constraint: `parallel_evolution.hpp` includes it, so dropping `<random>` from one alone changes nothing. `afc1976` -- `pattern.hpp` throws in **10** places and declared none of it, compiling only because `types.hpp` carried `<stdexcept>` for it; `types.hpp` throws nothing and is included by every engine header, so it handed the exception hierarchy to the whole tree for one consumer. **A `types.hpp`-only TU: 825 ms -> 667 ms, -158 ms, -19%.** The full engine closure is UNCHANGED and the commit claims nothing there -- `concurrent_map.hpp` and `job_system.hpp` throw and include it themselves, verified by preprocessing (0 occurrences in a `types.hpp`-only TU, 3 in the engine closure). Both halves in one commit: removing without adding is the regression that reverted the first attempt. **Structural half STARTED**: `bea2d9c` -- `types.hpp` held 19 types while consumers use one to four (`index.hpp` needs `GlobalCounters`, `signature.hpp` needs `Edge`, `causal_graph.hpp` four), and it was two concerns in one file. The quotient identity cluster (`QcEventContent`, `CanonicalEdgeKey`, `EdgeRankTable`, `EdgeOrbitTable`, `CanonicalTransition`, `SlotMatch`) moves to `quotient_types.hpp` with the replay-core include it alone needs; 765 -> 610 lines. **Nothing renamed** -- every type keeps its name and namespace, `types.hpp` includes the new header, so no consumer is edited and no build-time win is claimed. The trap: the include must go ABOVE `namespace hypergraph {`; placed where the block sat it nests the included file and `hgcommon::qr_apply_key` becomes `hypergraph::hgcommon::qr_apply_key` -- compiles, and is then a different function from the device's. Caught by the compile-each-header-alone check. `5a8a4ce` -- **`SubtreeBloomFilter` deleted: nothing ever allocated it.** 41 lines of hashing reached only through `VertexHashCache::subtree_filters`, a pointer set to `nullptr` in the constructor and assigned a non-null value nowhere in the tree, so both readers are behind guards that are always false; its two methods have zero callers, and `lookup_with_subtree` was a LINEAR scan over the array `lookup()` binary-searches. `types.hpp` 610 -> 543. CODEMAP named it as live -- #88's class exactly -- and `codemap_check` caught that the moment the type went. `6719cf7` -- **`<algorithm>` leaves the `types.hpp` closure**, by two edits neither of which pays alone: `VertexHashCache::lookup` held the one `std::lower_bound` and moves to `wl_hash.hpp`, its ONLY consumer (which already includes `<algorithm>`), and `bitset.hpp` included it while naming nothing from it. A `types.hpp`-only TU: 242 -> 154 ms, -88 ms; **across the day's four commits on this file, 757 -> 154 ms** and 765 -> 503 lines. The full engine closure still reaches it 8 times from elsewhere and the commit claims nothing there. **`types.hpp` IS FINISHED AND THE FLOOR IS MEASURED:** 170 ms against a 24 ms empty TU, and `<atomic>` alone is 124 ms of that. It cannot go: `bitset.hpp` uses atomics 15 times (it IS the concurrent bitset) and `types.hpp` includes it, so the last consumer never leaves. **And the general result, established three times over: a narrow header does not help THIS build.** Every engine `.cpp` includes `hypergraph.hpp` or `parallel_evolution.hpp` and so reaches the full closure regardless; a narrow header helps a DOWNSTREAM consumer that wants a few types. Further carving of `types.hpp` into `Edge`/`Event`/`State` files is legibility with no measured build effect, so it is not done on build-time grounds. `1cb7acd` + `4fd452c` -- **five shadowed names**, found by compiling with `-Wshadow`, which the build does not enable. `engine` re-bound to the object it already named; `key` meaning the OPTION name outside and a signature key inside, in the parse that decides event identity; `Token token` re-peeked twice in `wxf.hpp` where the outer one is live and equal; the index callback's `Context*` named for the `Context` the same call passes it; and a structured binding shadowing the SEED edges 40 lines above it. **The first sweep reported ZERO and was wrong** -- it grepped each `.cpp`'s basename, so header warnings were invisible, and three of the five are in headers. Making it a build flag is what exposed the instrument. The flag itself is NOT enabled: unscoped it reaches nvcc and invalidates all 8 `.cu` objects including `persistent.cu`, and the failed build's stale binaries then reported 36/36 -- see #128. `23b8f65` -- a wolframscript LAUNCH failure (`accept4 failed 110`) reddened the suite while its sibling test retried the same transient; same rule now applies to both |

**ALL FIVE GATES GREEN AT HEAD (`1513ebd`), re-run rather than inherited:** `all_tests`
256/256, `gpu_differential_tests` **36/36** including
`QuotientReconstruction.PastTheStackDepthItRecordsRatherThanFaults` (#41's stack test, 31.1 s)
and the 28-case `InitialCorpus/DifferentialEvolution` sweep, `hg_gpu_tests` **98/98**, and
`codemap_check` / `def_exports_check` / `doc_symbols_check` 0 findings. The GPU legs needed
**no nvcc**: `make -n` reported 0 nvcc invocations and 6 host compiles, because no `.cu`
includes any header this session changed — so the device gates were re-run against the
host changes without a CUDA window, on a box with 10 GB of another tenant resident.

**ORDERING, after D9.** #41 and #119 are no longer just "two parked device items" — they are now
the prerequisite for the largest remaining feature. The single CUDA window they are waiting on is
the critical path for #12b as well as for #114 and #20. Everything else in v1.0 can proceed on the
host while the box is busy; that window cannot.

**NOT blockers, and marked so they stop reading as such:** #58 and #30 are out of v1.0 by this
plan; #24 (paper) is deferred by standing instruction; #77 is DIAGNOSED (`da97c1b`) and its fix is
a design change to match forwarding, not a v1.0 obligation; #14 is CLOSED for v1.0 with an
on-demand knob as residue. **Needing Richard, not me:** #109 (reproduces on his laptop) and #116
(a semantics decision about what `Automatic` means).

| | item | why it is where it is | status |
|---|---|---|---|
| **S2** | Gate the graph-property marshalling surface | The count/list surface is heavily gated and the marshalling surface was not, which is how a regression making `HGEvolve`'s default call return nothing lived for hours. Coverage must exist BEFORE the things under it are rebuilt. | **DONE** `877a77b` — 54 cases, found #116 on run one |
| **S1** | One IR: delete `IRCanonicalizer`, make the core escalate | IR was the only `hgcommon` core with a live second implementation — 1,771 lines of one algorithm, already divergent in generator discipline. De-headering two bodies doubles the work and bakes the divergence in. | **DONE** `535279b` `4036937` `86cae18` `4313f3c` `e6b4b4c` `8c70b61` `4bf4a9a` `c0baca6` `5635279` `99b440b` `652d967` `2913309` — one body host and device; the search pipeline deleted (−598 lines); the device 1-WL fallback deleted and `ir_depth` made a retryable config field; the probe now gates against a BRUTE-FORCE DEFINITION, which found the orbit numbering reading the union-find root and the false "lexicographically smallest" claim. all_tests 247/247, hg_gpu_tests 98/98, gpu_differential_tests 36/36. #112 |
| **S3** | One quotient reconstruction (P6.2, P6.2a rewritten) | The other duplicated rule: the causal DP and the per-instance replay, each implemented on both sides. Both copies have already drifted, neither drift caught by a gate. | **CAUSAL DP DONE** `98d185d` `2ad5b29` — `hgcommon/quotient_causal_core.hpp` holds `qc_reach`/`qc_process_transition`/`qc_add_producer` and the three key spaces; host and device each supply a storage Ctx and their own bodies are deleted (169 lines). **REPLAY: HOST DONE** `39b8d5a` — `hgcommon/quotient_replay_core.hpp` holds `qr_apply` (eight decisions that were made twice: the exactly-once claim, the from_slots drop, minting, the content triple, the run signature with out_step from the FRAME, causal in descending producer order, branchial by publish-then-scan, the child instance); `Hypergraph::QrCtx` supplies storage. **DEVICE DONE** `b300af0` — `hg_gpu::qe_apply` drives `qr_apply` through a storage-only `DeviceQrCtx` and the device's own replay is deleted, so S3's BOTH duplicated rules are closed. It cost a measured constant move: the extraction takes the recursion cycle from 4 frames to 6, so `kDeviceStackBytesPerDepth` went 5632 → 8704, derived from the two depot sums (2000 and 3168) and the ABI term the original fault bisection pins at 865 bytes/frame. `__forceinline__` on the two added frames is REFUTED — nvcc declines it inside a recursive cycle. Gates: `gpu_differential_tests` 36/36 including the stack test, `hg_gpu_tests` 98/98. The body-similarity audit missed this pair (names differ `qc_`/`qe_`, phrasing under Jaccard 0.45) — its documented caveat, found by reading. #41 |

| **S4** | Fix the duplication audit's two blind spots | It reported ONE cross-area duplicate (`main`) and missed both real ones — it compared names, so same-algorithm-different-name was invisible, and it compared QUALIFIED names, so host/device pairs in different namespaces were too. | **DONE** `45873bd` `323f6a1` — body-shingle similarity (Jaccard over 9-grams) plus unqualified-name grouping; ground-truthed on a positive control (three copies of one insertion sort at 1.00, under three names — fixed) and a negative (the join, correctly absent). 41 shipped pairs to read. #113 |
| **S5** | Decide intra-state IR parallelism on device | One thread per state (`k_exact_hash_range`, `gpu/src/ir_canon.cu:264`, a grid-stride loop over states), so the premise was that the GPU cannot accelerate a single hard canonicalisation — the high-symmetry case where quotienting pays. **MEASURED, AND THE PREMISE DOES NOT HOLD: there is no single hard canonicalisation.** The search is 1–4 leaves at depth ≤1, `C_6`–`C_384` all give leaves=2/nodes=2/depth=1, and the heaviest 1% of states carry ~1.5% of all leaves. It needed no GPU — S1 left one shared IR body, so the counts are the device's, and the generator budget is a parameter. The high-symmetry cost is real but lives INSIDE a node (refinement; the O(generators × vertices) orbit walk), not across leaves | **DONE** `3a35bde`. **THE DECISION, which this row is named for: DO NOT BUILD branch-level intra-state parallelism.** The search tree is 1–4 leaves at depth ≤1 and does not grow with state size even on a 384-cycle, so there is nothing to fan out across threads; and the per-state distribution is flat — the heaviest 1% carry ~1.5% of the work in both populations — so the one-thread-per-state mapping has no straggler tail either. **What it does NOT settle:** leaf count is not total IR cost. The remaining time is per-NODE — refinement (31% of the pipeline) and the O(generators × vertices) orbit walk — so if intra-state parallelism is ever built it must target REFINEMENT, and this measurement says nothing for or against that. Ground-truthed on a 6-cycle first, because a population of rigid states exercises none of the counters, and the first population measured had 99.4% of states never searching, which made both generator budgets agree trivially and proved nothing |
| **S6** | Re-read the incrementalisation probes | Their recorded verdict is what justified dropping incremental WL and pointing the lever at IR. Read what they measured before repeating it. | **DONE** — the verdict had two legs and ONE IS STALE. (a) warm-starting `refine` cannot be exact (the canonical order IS the refinement trajectory) — STANDS, and it is a correctness argument. (b) "no parent→child locality, each match is its own work-stealing task" — FALSE at HEAD: `dispatch_expansion` runs up to `kExpandChunkSize` of one parent's children inline on the discovering thread. The unlock path the refutation itself named has landed. What it blocked — reusing the child's occurrence CSR from the parent's, patched by the delta — is exact and now has its hook. Recorded in `docs/BACKLOG.md` §2; a fresh profile comes before any build, since `build_adjacency` no longer exists. #115 |

| — | De-header + redesign | Standing decision: last, alone, once the native code is locked. | pending, #20 |

**THE DEVICE QUEUE.** Every item here touches a `gpu/include/hg_gpu/` header that most `.cu`
files include, so each one costs a near-full recompile — batch them. Build `-j1`, ONE at a time,
with nothing else started while it runs (`CLAUDE.md`, first hard rule: `-j2` drove this box into
swap twice, and a single CUDA TU at `-j1` exceeds the 600 s foreground cap, so background exactly
one and wait).

| item | what | gate |
|---|---|---|
| #41 | **DIAGNOSED BY MEASUREMENT, fix written, awaiting one build.** Pointing `hg_gpu::qe_apply` at `hgcommon/quotient_replay_core.hpp` inserts TWO frames (`qr_apply`, `DeviceQrCtx::descend`) into the replay's recursion cycle. `EngineState::kDeviceStackBytesPerDepth = 5632` is a MEASURED property of that cycle, so the guard fired AFTER the frame that faults and `QuotientReconstruction.PastTheStackDepthItRecordsRatherThanFaults` reported an illegal memory access — the test exists to assert "bounded partial result, NEVER a fault". `tools/dev/ptx_frame_sizes.py` (`33e7cc0`) reads each frame's `.local` depot out of the already-built PTX and settles what grew: `qe_drive_instance` 56, `qe_for_each_match_from` 8, its match lambda 1104, **`qe_apply` 64**, `qr_apply` 832, **`DeviceQrCtx::descend` 1104** = 3168 over 6 frames, with `qe_add_instance` inlined away entirely and absent from the PTX. The replay's own new data is 64 bytes; the rest is two ABI frames of pure plumbing. So the fix is to DELETE THE FRAMES, not inflate the constant: both are `__forceinline__` in `scratchpad/qe_replay_core_v2.patch`, each with a comment recording that the marker is load-bearing. Legal despite the recursion — the cycle still closes through real calls to `qe_drive_instance` and `qr_apply`. | `git apply` the v2 patch, ONE `-j1` build, then `ptx_frame_sizes.py --cycle` must show 4 frames with both forwarders inlined away. `5632` carries 171 bytes of margin over the measured 5461, so it moves only if the depot grew past that, and then by the measured delta. Gate: `hg_gpu_tests` + `gpu_differential_tests` |

| #114 | **CLOSED** (`3a35bde`): the distribution is flat and the search is 1–4 leaves at depth ≤1, so branch-level intra-state parallelism has nothing to divide. Counted rather than timed, because the counts are exact and are the device's numbers too | delivered: two populations plus a `C_6`–`C_384` sweep, both generator budgets, ground-truthed on a 6-cycle |

# Live defect register

Open, reproducible, with the command. Anything here that is closed moves to the ledger.

| what | reproducer | rate / size |
|---|---|---|
| **Host and device collect automorphisms differently; the device drops past 32 silently** | read `ir_canonicalization.cpp:517` against `ir_core.hpp:659`; `ir_canon.cu` passes `IR_DEVICE_GENERATORS` | The host appends every non-identity automorphism to a GROWABLE vector with no cap. The shared core appends into a fixed `gens[gen_cap * n]` guarded by `n_gens < gen_cap`, and past the cap the automorphism is dropped with **no status, no flag, no counter** — while depth exhaustion has both `IR_NEED_DEPTH` and a caller retry. Orbits are fused over the generator set (`for gi < n_gens`), so a short set yields **orbits that are too fine**, and those orbits are what `EdgeOrbitTable::slot` — hence the quotient reconstruction's instance identification — slots on. `IR_HOST_GENERATORS = 512` is referenced only by a test; production host never takes the capped path, so the defect is bounded-vs-unbounded, not 512-vs-32. Reachable: one generator per same-canonical leaf, so \|Aut\| = k gives up to k−1, and five interchangeable components is 120. **The measurement that decides latent vs live**: max `n_gens` per workload across the differential corpus. `gpu_differential_tests` 36/36 today either means nothing crosses 32, or that the difference does not surface in what it compares. Board #110 |
| **`Automatic` merges at DISPLAY time only, so `NumStates` and `StatesGraph` disagree** | `GraphPropertySurface.GraphVerticesAgreeWithTheCountsReportedBesideThem` with `Automatic` added to `kModesWhereEstablished` | `hypergraph_ffi.cpp:294` sets the ENGINE to `None` when the user asks for `Automatic`, and groups by content hash only at serialization. So `NumStates` (= `num_canonical_states()`, which in `None` keys on `sid+1`) is the PRE-merge count while the graph's vertices are POST-merge: 19 against 17 at 3 steps, 75 against 47 at 4, deterministic over 5 runs. The engine's own `Automatic` is self-consistent — measured directly, 17 canonical and 17 distinct content hashes, gap 0, at 1 and 4 threads — the FFI simply never selects it. `HGEvolve.md` tells the user `Automatic` will "merge states by isomorphism class", and `NumStates` does not. Needs a decision: report the class count (consistent, but changes a number and the `None`-for-evolution choice exists to match MultiwaySystem) or document that the merge is display-only. Either way one of the two meanings of `Automatic` should stop existing. Board #116 |
| **The IR equivalence probe cannot see the divergence that matters** | rewrite `IRCanonicalizer` as an adapter over `hgcommon::ir_canonical_hash`; the probe passes and `all_tests` loses 18 | `ir_core_equivalence_probe` compares the canonical HASH and, since `4313f3c`, the canonical FORM — 4063 states, 0 mismatches on both, for an adapter that makes `GoldenMatrix.EveryIdentityCellIsIndependentOfWorkerCount` report "the event count depends on the worker count (2 at 1 worker, 5 at 8)". So the two implementations agree on identity and representative and disagree on something else: the per-edge CLASS, RANK or ORBIT, which are what event identity and the quotient reconstruction key on and which nothing compares. The probe must gain those arrays before any swap is attempted — the swap looked safe precisely because the gate could not ask the question. Board #112 |
| **`ConcurrentIdSet` cannot back an exactly-once claim** | convert a claim map to it and run `all_tests --gtest_filter=CausalDeterminism.*` repeatedly | **4 of 12 invocations fail; 0 of 12 with the map.** Two defects relative to `ConcurrentMap`, both found by trying to use it, neither reachable by any gate while it had no users. (1) `for_each` emitted a key once per table it occupied, because `resize` COPIES and leaves the old table reachable — caught deterministically at THREADS=1, 79215 edges from 30063 claimed pairs, 2.64x. (2) `insert` reads the superseded chain then inserts into the current table, so across a resize a key can be claimed in two tables — the same defect `ConcurrentMap` was hardened against (#79, #91, ~1.5% of fuzz runs then). Reverted in `f48d227`. What it costs to leave: seven maps at 16 B/entry where 8 would do (`qc_applied_` is one entry per application, 68,184 on cycle4) and a measured **16,548,695 instructions, 0.53%** of the end-to-end corpus total. Reopening needs (2) fixed and gated by a FUZZ RATE, not one green suite run — the run that shipped it was green |
| `e2f6f75` `f6c1012` | **RELEASE BLOCKER, and a regression from `a0a1947` in this same session.** `HGEvolve[rules, init, steps]` — the first example in `HGEvolve.md`, and every call that omits the property — returned `$Failed` | The default property is `EvolutionCausalBranchialGraph`, and `record_set().branchial` answered two questions at once. It meant "the caller asked for this"; `can_branch` added a second writer that CLEARS it when the rules prove no two matches can share a consumed edge, and the FFI read the cleared flag as "the caller did not ask" and threw. A single-edge LHS is exactly the provable case, so the simplest rule in the documentation hit it. **Every gate stayed green**: the branchial-free proof is gated on the branchial COUNT being 0, which it still is, and the golden corpus asks for `BranchialEdges`, a non-graph property that never reaches the guard. Guards now ask `gneeds` — the caller's request — so a requested-but-provably-empty relation is served as the empty graph the proof says it is. Found by `verify_doc_examples.wls`, which took `HGEvolve.md` from 6 failing blocks to **26/26**; pinned by `WxfSerializationPin.ProvablyBranchialFreeRulesStillServeTheBranchialGraph`, ground-truthed FAIL at `e2f6f75^` / PASS at HEAD, no wolframscript needed. Suite 213 → **214** |
| `a935f9c` | The example runner counted SUPPRESSED messages as failures | It wrapped each block in `Quiet` and counted every message its handler saw, but `Quiet` does not stop the handler — an off message arrives wrapped in `$Off[...]`. Two bare assignments were reported as failing. **8 of 33 "failures", 6 of them spurious**; recording the message NAME is what separated the real defect from the noise |

| **The documentation build does not rebuild when the KERNEL changes** | change `paclet/Kernel/HypergraphRewriting.wl` in a way that alters rendering, then `./build_docs.sh` | It reports `placed 0 rebuilt notebook(s)`. The incremental manifest keys on the MARKDOWN SOURCE only, so a Kernel change that alters what the examples render leaves every notebook stale and the build says it succeeded. Found by nearly reporting a byte-identical comparison as evidence that a Kernel change was behaviour-preserving — the notebooks had simply not been re-rendered. Workaround today is deleting `.doc_manifest.wl`; the fix is to include the Kernel's hash in the manifest key |
| **Device reconstruction depth is capped near 40** | `QuotientReconstruction.PastTheStackDepthItRecordsRatherThanFaults` | Both device cascades recurse once per depth against a per-thread stack, measured at 5461 bytes/level. `00e21ee` sizes the stack from the depth and bounds the recursion, so a deeper run returns partial work and records `kScratchOverflow` instead of faulting. Removing the recursion (explicit work list) makes depth cost O(1) stack and is the real fix |

Two rows were removed as closed, not forgotten: **Positional identity cannot run through the
reconstruction** and **cross-thread causal determinism under the widened routing** both pass now
— the full host suite is 239/239 with `wants_qc` forced to `!positional_event_identity()`, where
the item opened at 193/198 (P1.5, `bb1ccab`). **`tools/` built by nothing** closed as P5.1
(`365273d`): a glob gives every tracked tool a target.

Each row above is also a task in the session task list, so the two cannot drift:
"Forwarding loses matches under EAGER submission, 1-6 of 204 runs",
"CLOSED: deep quotient run faulted the device; stack sized from depth + recursion bounded".

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
- **Structural debt for P6, RE-MEASURED 2026-08-05 (the recorded numbers were stale by 2.5x
  and 1.8x):** `hypergraph_ffi.cpp` is **1,849** lines, not ~4,700, and `HypergraphRewriting.wl`
  is **2,142**, not ~3,800. The file size is not the problem in either case. What is: ONE
  function, `run_rewriting_core`, is **1,664 of those 1,849 lines** -- every other definition in
  the file is under 40 -- and the WL file carries 98 top-level definitions. So the cut is a
  function decomposition, not a file split, and the op-boundary work (#12) already found where
  its first seam is. **FOUR CUTS LANDED: `run_rewriting_core` is 1664 -> 1129 lines.** `4f66efc` `parse_job` (256), `d4d9ef9` `run_gpu_job` (80), `7d6a929` `configure_and_evolve` (182), `8ef5254` `read_back_session_identity` (8, and it is D16 getting a name rather than a paragraph). Each is a phase whose dependency runs ONE WAY, and after being wrong twice about which ones those are, each was established by LISTING what the block reads from its enclosing scope -- `configure_and_evolve` reads `req` 20, `engine` 18, `hg` 8, `host` 4, `record` 1 and nothing else. No field is renamed; a field's initialiser is
  now the option's DEFAULT in one place. The compiler proved the rename complete (the locals
  are gone, so a miss stops resolving) and found the one thing an eye does not: the parse
  reports a malformed option through `core_progress(host, ...)`, so `host` is a real input. `types.hpp` **is done** (757 -> 170 ms, 765 -> 503 lines, floor measured);
  namespaces are
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
