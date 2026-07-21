# Paper revision plan — "Rewriting the Universe"

Scope: how to take the recovered draft (`RECOVERED_paper/paper/main.tex`, 870 lines)
from its December-2025 state to a submittable paper that matches the engine as it
exists today. Authority for "what is true now": `docs/ARCHITECTURE.md`,
`docs/OBJECTIVE.md`, `docs/MEMORY_ARCHITECTURE_DESIGN.md`, `docs/BACKLOG.md`,
`reference/CANONICALIZATION.md`, `docs/PAPER_RESULTS.md`, and the current git log.

The draft predates the two largest shifts in the project:

1. **Canonicalization was re-founded.** Uniqueness trees (UT) are **removed**. The
   reference exact canonicalizer is **McKay-style individualization–refinement (IR)**;
   the fast hot-path hash is **Weisfeiler–Leman (WL)** with IR fallback on collision.
   The draft still presents UT as the production/preferred canonicalizer with an
   "O(V^7) polynomial" guarantee and a runtime "hash-strategy selector." All of that
   is now wrong. (`reference/CANONICALIZATION.md`, `docs/ARCHITECTURE.md` lines 55–58,
   86–91.)
2. **A memory/allocator overhaul happened** (de-heap to a per-worker-cursor arena;
   causal-closure reduction; copy-on-write states; quotient exploration; exact offline
   reconstruction). These are genuine new contributions and are **absent** from the
   draft. (`docs/MEMORY_ARCHITECTURE_DESIGN.md`, `docs/PAPER_RESULTS.md`, git log.)

Every benchmark number in the draft is a **projection or an explicit placeholder** and
must be replaced with measured numbers (§3 below). The pgfplots comment on
`main.tex:587` literally reads `% Wolfram Language (placeholder data)`.

---

## 1. Section-by-section status

| # | Section (main.tex lines) | Verdict | Reason |
|---|---|---|---|
| — | Abstract (105–110) | **REWRITE** | Leads on UT "O(V^7)" and "configurable hash strategies"; headline speedups (50–200× CPU, 5–20× GPU) are projections. Re-anchor on IR/WL + the memory/quotient contributions. |
| 1 | Introduction (116–135) | **REVISE** | Contribution list item (1) is UT-centric (126); item (4) sells a runtime hash-strategy selector (132). Replace with IR-reference/WL-fast, and add the allocator + quotient contributions. |
| 2 | Background & Related Work (141–184) | **KEEP (light revise)** | Hypergraph/multiway/GI definitions are sound. Fix the Related Work UT sentence (180) and add nauty/IR framing; add multiway-systems references. |
| 3 | Graph Canonicalization (190–302) | **REWRITE** | Entire section is built on UT as primary. §3.1 UT hashing (195–232), §3.2 incremental UT via Bloom filters (234–260), §3.4 hash-strategy selection (289–302) do not describe the current engine. §3.3 WL (262–287) is keepable but must be reframed as the hot-path hash, not "an alternative." |
| 4 | Pattern Matching (308–382) | **REVISE** | Signature indexing, vertex index, task-based SCAN/EXPAND matching, and match-forwarding are all real and current. Update to reflect the WCO-join framing (`pattern_matcher.hpp`), the signature-index demotion (repeated-var edges only), and mutate/undo DFS. Match-forwarding proposition (378–380) is correct in principle; note the engine now forwards by reference, not deep copy. |
| 5 | System Architecture (388–465) | **REVISE + EXPAND** | Unified edge pool (391–422) and lock-free structures (424–434) are real. The "Memory Model" subsection (436–451) is about *memory ordering* and should be renamed; the actual **allocation architecture** (per-worker-cursor arena, de-heap) is the missing contribution and belongs here (§4 below). Event-canonicalization subsection (453–465) uses stale names (`ByState`/`ByStateAndEdges`) — replace with the `EventSignatureKeys` bitflag model and the Positional-vs-Canonical distinction. |
| 6 | GPU Acceleration (471–550) | **REVISE** | Reframe: the engine's GPU path is a **host-driven level-synchronised BFS step loop** (four bounded kernel phases/step) with a `PersistentEvolver`, not a single persistent "megakernel." §4.2 "GPU Uniqueness Trees" (507–530) must become GPU WL/IR. Real GPU results exist (`PAPER_RESULTS.md` §4) and replace the fabricated throughput figure. |
| 7 | Experimental Evaluation (556–735) | **REWRITE** | Every figure/table is placeholder or projected data (587 says so explicitly). Hardware (564) says RTX 3090; real runs are RTX 4090. Rebuild around real cost_matrix + timing + ablation tables (§3). |
| 8 | Interactive Visualization (741–766) | **KEEP (verify)** | Vulkan/ImGui 3D viewer is real; features list is plausible. Two caveats: replace the screenshot placeholder (748–755) with a real figure, and confirm the viz currently **builds** before claiming it (per `BACKLOG.md` §5 it is not wired into CI and may not build). Soften/verify the export-format and layout claims. |
| 9 | Conclusion & Future Work (772–802) | **REVISE** | Contribution recap (777–782) repeats the UT claim (778). Future-work items (incremental matching, distributed, rule learning, formal verification) are fine but several "future" items (incremental hashing, reachability oracle, automorphism reconstruction) are now in-progress design, not open speculation — reposition. Fix placeholder repo/paclet URLs (800–801). |
| — | Acknowledgments (808–810) | **REVISE** | Thanks Gorard "for insights on uniqueness tree algorithms" (810). With UT removed, this attribution no longer matches the paper's content; drop or re-scope. |
| A | Appendix A — Complexity (825–850) | **REWRITE (A.1) / KEEP (A.2)** | A.1 UT construction + O(V^7) (828–839) goes with UT removal → replace with IR/WL complexity. A.2 pattern-matching complexity (841–850) is keepable. |
| B | Appendix B — Data Structures (852–868) | **KEEP (light revise)** | Lock-free structure complexity table is fine; add the arena/tier note and confirm the SparseBitset row reflects COW. |

---

## 2. Stale / incorrect claims to fix (quote + replacement)

### 2a. Uniqueness trees presented as production / polynomial (the biggest problem)

UT must be removed as the engine's canonicalizer everywhere. Replacement framing
(cite `reference/CANONICALIZATION.md`): **IR (McKay individualization–refinement) is
the exact reference canonicalizer; WL is the fast approximate hot-path hash; on a WL
collision the engine falls back to IR. UT is not used.**

- **Abstract, line 106:** "*polynomial-time graph canonicalization using uniqueness
  trees with $\BigO(V^7)$ complexity*" and "*configurable hash strategies (uniqueness
  trees, Weisfeiler-Lehman)*." → Replace with WL fast hash + IR exact fallback; drop
  the O(V^7) guarantee (IR is exponential worst-case like nauty, fast in practice).

- **Introduction, line 126:** "*Polynomial-time canonicalization: We implement
  uniqueness tree hashing~\cite{gorard2016uniqueness}, reducing state comparison from
  factorial to $\BigO(V^7)$ time while maintaining correctness for practical
  hypergraph instances.*" → Replace: WL color-refinement hash on the hot path (fast,
  may collide), exact IR canonicalization to confirm/deduplicate. No polynomial
  correctness guarantee is claimed.

- **§3.1 Uniqueness Tree Hashing, lines 195–232**, including **Theorem (Uniqueness
  Tree Complexity), lines 226–228:** "*Computing the uniqueness tree hash ... requires
  $\BigO(|V|^7)$ time in the worst case*" and its proof sketch (230–232). → **CUT**.
  Replace the whole subsection with an IR subsection (individualization–refinement,
  target-cell selection, automorphism pruning, canonical form) plus the WL hot-path
  hash.

- **§3.2 Incremental Uniqueness Trees, lines 234–260** (Bloom-filter affected-vertex
  scheme). → **CUT as written.** The engine's real incrementality story is (a) the
  delta path `child = parent − consumed + produced`, (b) incremental hashing is
  *designed but not yet wired* (`create_or_get_canonical_state` is handed
  `incr_consumed/incr_produced` and currently ignores them — `MEMORY_ARCHITECTURE_DESIGN.md`
  §4, `BACKLOG.md` §2). Do **not** claim a working incremental hash until it lands;
  present it as ongoing work or omit.

- **§3.4 Hash Strategy Selection, lines 289–302**, especially the enum
  (294–299: `UniquenessTree`, `IncrementalUniquenessTree`, `WL`, `Exact`) and line
  302 "*we default to incremental uniqueness trees, falling back to exact
  canonicalization*." → **REWRITE.** The real axis is the `StateCanonicalizationMode`
  {`None`, `Automatic`, `Full`} (see `CANONICALIZATION.md` Axis 1): `None` = no dedup;
  `Automatic` = fast content hash (WL, may false-merge); `Full` = exact (IR). There is
  no user-facing "pick UT vs WL" selector. The default exact path is WL-hash + IR.

- **§4.2 GPU Uniqueness Trees, lines 507–530** (per-block BFS tree hash). → **REWRITE**
  as GPU WL/IR canonicalization. (Also note `gpu/wl_hash.cu` has a known O(V·E) bug per
  `BACKLOG.md` §4 — do not over-claim GPU canonicalization performance.)

- **§7 Table (Hash strategy comparison), lines 665–681:** rows "Uniqueness Tree",
  "UT Incremental." → Replace rows with WL, WL+IR-fallback, IR-exact, None. Numbers are
  fabricated (see §3).

- **Conclusion, line 778:** "*Practical polynomial-time canonicalization via uniqueness
  trees with incremental computation.*" → "Exact canonicalization via IR with a fast WL
  hot-path hash and collision fallback."

- **Appendix A.1, lines 828–839**, incl. line 839 "*The $\BigO(|V|^7)$ bound
  from~\cite{gorard2016uniqueness}...*". → **REWRITE** for IR/WL.

- **Acknowledgments, line 810** (Gorard / UT). → Drop the UT-specific attribution.

- **`gorard2016uniqueness` citation** — remove from all UT contexts; keep only if the
  paper still needs it as related work, and if so it belongs only in §2.4 Related Work
  as a *rejected* approach, not as implemented infrastructure. (Cross-check with the
  project position in `reference/CANONICALIZATION.md`.)

### 2b. Event canonicalization naming (line 453–465)

The draft's `ByState` / `ByStateAndEdges` / `None` (457–463) predates the current
model. Replace with the `EventSignatureKeys` bitflag composition and the **four
conventions** in `reference/CANONICALIZATION.md` Axis 2: `None`, `States`
(= `EVENT_SIG_FULL`), `Automatic`+`Positional` (`EVENT_SIG_AUTOMATIC`, MultiwaySystem
parity), `Automatic`+`Canonical` (the principled IR edge-orbit merge). Explain the
Positional-vs-Canonical distinction (symmetric edge roles: e.g. the two
`{{1,1},{1,1}}` self-loop matches count as 2 under Positional, 1 under Canonical).

### 2c. GPU "megakernel" framing (§6, 471–505)

Line 476: "*runs a single persistent kernel throughout evolution*." The actual GPU
backend is a **host-driven, level-synchronised BFS step loop with four bounded kernel
phases per step** and a `PersistentEvolver` that keeps the device engine alive across
*calls* (`docs/ARCHITECTURE.md` line 66–68, `PAPER_RESULTS.md` §4). Reframe to match;
do not claim a device-resident persistent megakernel (that is the still-unbuilt
persistent-scheduler work, `BACKLOG.md` §4).

### 2d. Projected / placeholder performance numbers (all must go — see §3)

- **Abstract, line 106:** "*50--200$\times$ speedup over the reference Wolfram Language
  implementation on CPU, with additional 5--20$\times$ gains on GPU.*" — projection.
- **Introduction / Conclusion:** "50--200×" recurs at line 775.
- **§7.2 CPU figure (571–615):** data are explicitly placeholder (`main.tex:587`).
- **§7.3 GPU figure (621–659):** fabricated throughput curves.
- **§7.4 hash-strategy table (665–681):** fabricated (245/128/89/4200 ms, collisions).
- **§7.5 scaling figure (685–714):** fabricated speedup points.
- **§7.6 event-canon table (720–735):** fabricated event counts / overheads.
- **§5.1, line 422:** "*reducing memory usage by 3--5$\times$*" — projection.
- **§6/§7 GPU:** "*5--15$\times$ throughput*" (661) — projection.

### 2e. Experimental setup mismatch (§7.1, 563–567)

- **Hardware, line 564:** "*NVIDIA RTX 3090 (24GB VRAM)*" — real GPU runs are on an
  **RTX 4090** under WSL2 (`PAPER_RESULTS.md` header). CPU host must match the machine
  actually benchmarked.
- **Software, line 565:** "*CUDA 11.8, Wolfram Language 13.2*" — update to the versions
  actually used (project runs Wolfram 14.3). State WSL2 if that is the benchmark
  environment, and disclose the noise caveat (CV 10–40%, `PAPER_RESULTS.md`).

---

## 3. The benchmark problem — projections → measured tables

**Every quantitative result in §7 (and the abstract's headline numbers) is a projection
or an explicit placeholder and must be replaced with measured data.** The paper cannot
be submitted on projected numbers. `PAPER_RESULTS.md` is the current real-numbers
capture (noisy box; must be re-run on a quiet machine for final low-variance tables +
CIs). The tables the paper needs:

### Table T1 — Exactness + memory (from `tools/cost_matrix.cpp`)
Per workload, one row: `{canonical states, raw states, events, causal edges, branchial
edges, arena bytes, heap allocs, heap bytes}` and an **Exactness** column that must read
EXACT (checked against the brute-force oracle). cost_matrix already emits exactly these
(Full mode, single-threaded, deterministic memory = arena `bytes_allocated()`, not RSS).
This is the correctness + memory backbone of the evaluation.

### Table T2 — Wall-time speedup vs the Wolfram reference
Per-rule wall-time speedup of the C++ engine (1 thread and N threads) vs the Wolfram
`MultiwaySystem` paclet / brute-force reference, across increasing depth. Needs a
**timing harness** (not yet a committed tool — `profile_evolve` is single-threaded for
callgrind; a wall-time sweep harness must be added; GPU timing via `tools/bench_gpu_evolve`).
Disclose machine, thread count, and variance.

### Table T3 — De-heap result (headline systems number)
`heapAllocs 1567 → 69` on the corpus (`MEMORY_ARCHITECTURE_DESIGN.md` status,
`BACKLOG.md` §0). Before/after, per workload, from the cost_matrix global-`operator new`
counter. Pair with heap bytes (the ~2.8 MB fixed floor removed).

### Table T4 — Causal-closure reduction
`−28.5%` total arena on binary-growth from the base-layer closure rework (key-only
uint32 `ConcurrentIdSet`, `Anc` dropped, reconvergence skip, lazy empties), exact and
4/8/16-thread deterministic (`PAPER_RESULTS.md` §3, `MEMORY_ARCHITECTURE_DESIGN.md`).

### Table T5 — Copy-on-write states
COW SparseBitset: child shares immutable parent chunks, copies only the ~2 chunks a
consumed/produced edge touches vs memcpy-all-chunks (~20× fewer chunk copies per child).
Report chunk-copy counts and/or arena bytes before/after.

### Table T6 — Quotient exploration (headline algorithmic number)
Quotient vs full expansion, same canonical closure: **depth 6 ~4.1×, depth 7 ~16×**,
widening with depth; **45k vs 290k raw states at depth 7** on the Wolfram rule. No perf
regression vs the prior mode (paired −2.7%, CI spans 0). (`PAPER_RESULTS.md` §1.)

### Table T7 — GPU
Match kernel **~190 ms → ~4 ms at depth 7 (~47×)**; end-to-end depth 7 **~834 → ~252 ms**;
depth 8 **373 s → ~5.5 s**; lazy-index depth-8 **11.75 s → 5.25 s**. Branchial via
per-(state,edge) index. (`PAPER_RESULTS.md` §4.) On RTX 4090.

### Table T8 — Thread scaling
Real measured strong-scaling curve (replaces the fabricated §7.5 figure), with the
determinism note: expanded set + (input,output,rule) transition multiset identical
across thread counts and seeds.

### Table T9 — Ablation (proves each contribution)
Toggle each optimization off vs on: **COW off/on, incremental off/on, de-heap off/on,
reachability-oracle vs closure, quotient off/on, GPU lockstep vs persistent.** Per
`OBJECTIVE.md` §9 and `BACKLOG.md` §7, ablation numbers come from **CMake `#define`
ablation builds** (old paths kept, compiled out of the default binary, exercised by CI).
This scaffolding is **planned, not yet built** (`BACKLOG.md` §8) — flag it as a
prerequisite for the ablation table.

**Workloads for all tables:** the **rule-type corpus** — `reference/oracle_corpus.hpp`,
24 workloads: single/multi rule; productive/idempotent/reductive/mixed-arity;
single/mixed arity; 7 initial-state shapes (incl. proper-subset-of-orbit consumption and
repeated-LHS edges) — plus **larger depths** on the canonical Wolfram rule
`{{x,y,z}} → {{x,y,w},{y,z,w}}` for the scaling/GPU curves.

**Tools & oracle:**
- `tools/cost_matrix.cpp` → exactness + arena bytes + heap allocs/bytes (T1, T3).
- A wall-time timing harness (to add) → speedups (T2, T6); `tools/bench_gpu_evolve` → GPU (T7).
- Reference/oracle: `reference/MultiwayReference.wl` brute-force oracle +
  `reference/oracle_corpus.hpp`, cross-checked against the Wolfram `MultiwaySystem`
  paclet (`docs/ARCHITECTURE.md` "Validation").

**Honesty caveats to state in the paper:** current numbers are single-sample on a noisy
WSL2 box (CV 10–40%); final tables need a quiet machine with paired means + CIs
(`PAPER_RESULTS.md` "Open for the paper"). Do not present noisy singles as final.

---

## 4. Missing / incomplete sections to write

1. **Allocation architecture (new, genuine contribution).** The per-worker-cursor
   arena + hierarchical lifetime tiers (Tier D durable / Tier F frontier-transient /
   Tier T per-task) that take `malloc` off the concurrent path (heap 1567→69). The
   draft's "Memory Model" (436–451) covers only memory *ordering* — the *allocation*
   story is absent. Source: `MEMORY_ARCHITECTURE_DESIGN.md` §1, §3c. Explain why a
   single global arena is *not* the answer (removes contention, not the memory wall) and
   how per-worker bump cursors solved the 6× contention a naive single arena caused.

2. **Quotient exploration (headline algorithmic contribution).** Expand each canonical
   state once at its shortest reachable depth; cost ∝ canonical-state count, not
   provenance count; determinism via lock-free depth relaxation. Absent from the draft.
   Source: `PAPER_RESULTS.md` §1, `docs/ARCHITECTURE.md` line 92–94.

3. **Exact offline reconstruction of causal & branchial multisets.** The quotient
   skeleton discards multiplicity; it is recovered exactly offline, keyed by canonical
   **edge orbit under the automorphism group** (content classes suffice only when
   |Aut|=1). Validated on the 24-workload corpus incl. cases where Aut fuses edge
   contents. Source: `PAPER_RESULTS.md` §2, `MEMORY_ARCHITECTURE_DESIGN.md` §2b.

4. **Online transitive reduction (exact, deterministic).** The causal TR emits the
   unique minimal reduction with no redundant edge at any thread count; verified equal
   to the offline minimal TR on real workloads at 1/2/4/8/16 threads. Source:
   `PAPER_RESULTS.md` §3. (The draft never discusses TR.)

5. **Copy-on-write state representation.** Draft's Figure (395–420) shows states as
   bitset views over a shared edge pool but omits COW (share immutable parent chunks,
   copy only touched chunks). Add. Source: `MEMORY_ARCHITECTURE_DESIGN.md` §4 storage.

6. **IR canonicalization (replaces the UT section).** McKay individualization–refinement
   as the exact reference; automorphism generators + edge orbits already computed for
   orbit pruning (and reused for reconstruction, §4.3 above). Source:
   `reference/CANONICALIZATION.md`, `ir_canonicalization.*`. Note the honest limitation:
   IR blows up on high-automorphism states (cycles up to ~1100× slower) — which is *why*
   WL is the hot path (project verdict in memory / `tools/ir_vs_wl.cpp`).

7. **Measurement & validation methodology.** A dedicated subsection: the oracle corpus,
   cost_matrix (arena-bytes not RSS for reproducibility), the brute-force isomorphism
   oracle, the Wolfram `MultiwaySystem` cross-check, and the determinism gates. This is
   what lets the paper claim "exact, proven" rather than "fast, probably right."

8. **Uniform, unbiased subsampling (optional, if space).** Reservoir sampling
   (stratified, chi-square verified) and per-canonical-state exploration probability.
   Source: `PAPER_RESULTS.md` §5.

Design pieces that are **not yet built** — present as future/in-progress, do **not**
claim as done: incremental hashing (delta ignored today), the causal **reachability
oracle** (O(N²)→O(N·w), still the closure today), tier reclamation, automorphism
reconstruction as a default path, the persistent-kernel GPU scheduler, CMake ablation
scaffolding (`docs/MEMORY_ARCHITECTURE_DESIGN.md` "NOT STARTED", `BACKLOG.md`).

---

## 5. Prioritized checklist (recovered → submittable)

1. **Rip out UT.** Rewrite §3 around IR (exact reference) + WL (hot-path hash, IR
   fallback); delete the O(V^7) theorem (226–232), the incremental-UT/Bloom subsection
   (234–260), and the hash-strategy enum (294–299). Fix the abstract (106), intro (126),
   conclusion (778), appendix A.1 (828–839), and the Gorard acknowledgment (810). Cite
   `reference/CANONICALIZATION.md`.
2. **De-projection pass.** Strike every projected/placeholder number (§2d list) and mark
   each figure/table "TO MEASURE." Nothing quantitative ships until it is measured.
3. **Add the two missing headline contributions** as first-class sections: the
   allocation architecture (de-heap / per-worker arena / tiers) and quotient exploration
   + exact offline reconstruction.
4. **Fix the naming/framing errors:** event-canonicalization model (453–465 → bitflags +
   Positional/Canonical), GPU "megakernel" → level-synchronous step loop + PersistentEvolver
   (476), experimental setup (RTX 4090, correct CUDA/WL versions, WSL2 + noise caveat).
5. **Build the measurement harness** and produce T1/T3/T4/T5 (cost_matrix, ready now) and
   T2/T6/T7/T8 (add the wall-time timing harness; GPU via bench_gpu_evolve). Re-run on a
   **quiet machine** for paired means + CIs.
6. **Stand up the CMake `#define` ablation builds** (`BACKLOG.md` §8) and produce the
   ablation table T9 — the "prove each contribution" requirement of the charter.
7. **Add the methodology/validation section** (oracle corpus, cost_matrix, MultiwaySystem
   cross-check, determinism gates) so the exactness claims are backed.
8. **Verify the visualization** actually builds (`BACKLOG.md` §5) before its section
   ships; replace the screenshot placeholder (748–755) with a real figure.
9. **Housekeeping:** real repo/paclet URLs (800–801), reconcile future-work vs
   in-progress design, update `references.bib` (drop UT-as-implemented, add multiway /
   IR / reachability-oracle references), regenerate all figures from measured data.
10. **Final consistency pass:** keep this paper in sync with `docs/CODEMAP.md` +
    `docs/ARCHITECTURE.md` (the standing doc-sync rule); no claim in the paper that the
    docs/tests do not support.
