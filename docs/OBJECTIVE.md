# Objective & governing principles

This document is the charter for the overhaul. It is the standing statement of the aim and
the non-negotiable principles every change is measured against. The concrete architecture that
realises it is in [MEMORY_ARCHITECTURE_DESIGN.md](MEMORY_ARCHITECTURE_DESIGN.md) (to be
broadened into the full master plan); this document is the *why* and the *bar*.

---

**The aim is not memory alone. It is to drive this multiway hypergraph rewriting engine — CPU
*and* GPU — to the theoretical maximum of efficiency, as one cohesive, lock-free, fully-parallel,
incremental, cache-optimal system that is exhaustively proven correct, so that the paper is
strong (with ablation to prove each contribution), and the project lands as a clean, complete,
well-documented release. A deep drive to closure. No corners, no hacks, no stone unturned.**

## The performance principles (the non-negotiables)

1. **Lock-free / wait-free everywhere. No spinwait, anywhere.** Not just the current hot paths —
   the whole engine. No mutexes, no blocking, no busy-wait loops burning a core to poll. Every
   core (CPU thread or GPU warp/thread) is *always* doing useful work; a core that would spin
   instead steals or advances other work. Progress is guaranteed structurally (claim-winner,
   monotonic counters, reserve-then-take, error-flag wait-breaks), never by waiting.

2. **No phases. Nothing post-hoc.** Causal edge creation, branchial edge creation, and the
   transitive reduction are **fused inline into rewriting** — computed at the moment each event
   is created, via the online rendezvous, never as a second pass over a finished graph. This
   holds on **both** backends: no GPU "compute causal/branchial after evolve" phase, and — 
   critically — the causal **reachability oracle** that replaces the O(N²) closure must itself be
   maintained **online/incrementally**, not by batched relabeling. (Harder than a batched
   rebuild; the no-phases principle demands we solve it online, or explicitly justify any
   batching as a last resort. Online is the target; batching is flagged if unavoidable.)

3. **Parallelise everything** — pattern matching, rewriting, canonicalization, causal/branchial
   construction, deduplication, reconstruction. No inherently serial bottleneck left standing;
   where an algorithm looks serial (online TR), redesign it (reachability oracle) so it
   parallelises.

4. **Incrementalisation is foundational — the theoretical minimum of computation.** The delta
   path (child = parent − consumed + produced) is the *only* path; full recompute is the
   degenerate first-step case, not a second implementation. Nothing that can be reused is
   recomputed: not matches (forwarded/delta), not hashes (incremental — the delta is already
   handed in and currently ignored), not indices, not canonical forms. Every duplicated
   computation the cost analysis found is a defect against this.

5. **Absolute minimum computation *and* memory bandwidth; maximal cache locality.** Minimal
   representation (no dead fields, no fixed-cap padding, no leaked/retained-forever working
   data). Data laid out to keep the working set in L1/L2 and **not spill to L3/DRAM
   unnecessarily** — packed CSR/contiguous arrays instead of pointer-chasing linked lists,
   cache-line alignment, no false sharing, streaming access patterns, hot/cold field
   separation. Memory traffic (the forwarding deep-copies, the re-serialization, the closure)
   is minimized because bandwidth *is* the wall as much as capacity.

5a. **No `malloc`/heap on any concurrent path — de-heap the entire surface.** `malloc`/
   `::operator new` contend a global lock across workers and fragment. Every allocation is
   custom-allocator (arena) backed, placed in the **lifetime tier** that matches it (a single
   global arena is NOT the answer — it removes contention but not the memory wall). Reclamation
   happens at the generation boundary. Proven by the heap-allocation counter driving the
   hot-path heap to zero. See MEMORY_ARCHITECTURE_DESIGN §3c for the line-by-line inventory.

6. **One implementation, reused at every functionally-equivalent intersection.** CPU,
   GPU-lockstep, GPU-persistent are three *schedulers* over a single shared `hgcommon` kernel
   set. WL/IR, SCAN/EXPAND/SINK, delta/full, all bottom out in single sources of truth. No path
   may drift from another.

## Correctness and verification (the other non-negotiable)

7. **Exact, always. Proven, not hoped.** Every output — states, raw events, canonical events,
   causal edges, branchial edges, multiplicities, sampled subsets — bit-identical to the
   brute-force oracle and the authoritative Wolfram `MultiwaySystem`, across every mode, single-
   and multi-threaded, at every step.

8. **Tests that leave no bug possible.** Comprehensive coverage of every feature *and every
   feature combination*; **repeated, stress, high-thread-count, and randomized** runs to surface
   synchronization races (the "works single-threaded, fails at 128 threads" class); TSAN/ASAN
   gates; determinism verified across thread counts; differential tests for every new path vs
   the one it replaces (COW vs copy, oracle-reachability vs closure, reconstruction vs full
   materialisation, GPU vs CPU). No flakiness tolerated as "probably fine."

## Ablation and the paper

9. **Ablation built in, compiled out.** Old code paths are kept so the paper can show the
   contribution of each optimization (turn off COW, turn off incremental matching, turn off the
   reachability oracle, revert to full materialisation, lockstep vs persistent GPU) — but they
   are **behind CMake `#define`s and removed from the default build**, so the release binary is
   pure fast-path with zero ablation overhead, and each old path is still exercised by a CI
   configuration so it can't rot.

## Cohesion — the full feature surface as one whole

Every feature must be mapped, wired, and mutually reconciled — not just the hot loop:
evolution (multi-rule, multi-initial-state, genesis, budget, cooperative abort, overflow →
partial result); canonicalization None/Automatic/Full + WL fast-path + IR + event-canon
bitflags + run-stable CanonicalHash; quotient exploration; online transitive reduction;
uniform-random **reservoir sampling (unbiased)** + ExplorationProbability stochastic pruning +
multiplicity propagation; states/causal/branchial/evolution graphs + structure-only variants +
raw vs canonical events; CPU / GPU-lockstep / GPU-persistent / process-isolation binary; the
Wolfram paclet (HGEvolve, all IC generators, all analyses, plots, autocomplete, docs); and the
interactive 3D visualisation (compiling, wired into CMake with `BUILD_VISUALIZATION` in CI).

The known conflicts to resolve into coherence: determinism across thread counts under all
features; sampling ⊗ quotient ⊗ canonicalization (unbiased-of-*what*, preserved by
reconstruction); incremental matching ⊗ event-canon modes; automorphism reconstruction ⊗
causal/branchial/multiplicity/sampling (exact for *every* output); no-phases ⊗ GPU (online, not
batched); cache-optimal packed layout ⊗ lock-free concurrent append; overflow/partial-result ⊗
reconstruction; ablation flags ⊗ the combinatorial test matrix. The goal is that these stop
being separate, caveated code paths with hidden conflicts and become one cohesive design where
every feature composes with every other.

## The meta-goal

Strong paper (with ablation numbers, real not projected) · clean, well-documented, well-factored
public repo · visualisation building · comprehensive race-proof tests · CPU and GPU both at the
ceiling. **A deep drive to a strong release and closure.**
