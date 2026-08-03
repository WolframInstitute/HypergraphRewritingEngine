# P6.1 — Splitting rewrite semantics from hardware orchestration

**Status: proposed. No code moves until this is approved.**

This is the design document P6.1 asks for. It is written against measurements taken at
`00e21ee`, each of which is a command a reader can re-run, and it revises the item's own
premise where the measurement contradicts it.

---

## 1. What the item assumed, and what is actually true

The queue records the motivating measurement as:

> Measured input: `gpu → hypergraph` is only **27** references, so the GPU duplicates rather
> than depends.

That number no longer describes the tree. The real count is **zero**:

```
$ git grep -l '#include "hypergraph/' -- 'gpu/*'
gpu/tests/bench_cpu_vs_gpu.cpp
gpu/tests/test_gpu_vs_cpu_differential.cpp
```

Both are test files, and both include the host engine because their job is to compare against
it. **No GPU library file includes anything from `hypergraph/`.** The two engines are already
fully decoupled at the include level; they meet only in `common/include/hgcommon/`.

This matters because it changes what the item is for. There is no dependency to break. The
cost being paid is not coupling — it is that the same *rule* is written twice, and the copies
drift.

## 2. Where the line already falls

Every device file that implements rewrite SEMANTICS already routes through a shared core:

| device file | shared core it calls |
|---|---|
| `gpu/src/match.cu` | `hgcommon/join_core.hpp` |
| `gpu/src/rewrite.cu` | `hgcommon/rewrite_core.hpp` |
| `gpu/src/ir_canon.cu` | `hgcommon/ir_core.hpp` |
| `gpu/src/wl_hash.cu` | `hgcommon/wl_core.hpp` |
| `gpu/include/hg_gpu/event_identity.hpp` | `hgcommon/event_core.hpp` |

That is P1's work, and the done-line it reached is real: the join, the rewrite, both
canonicalizers and the event-identity lattice each have ONE body.

`hgcommon` is 11 headers, 1,990 lines. Nine are semantics; two (`park.hpp`,
`portable_intrinsics.hpp`) are platform shims and belong to orchestration, not to the rewrite
model. That is a naming problem, not a structural one.

## 3. The one rule still written twice

| | host | device |
|---|---|---|
| quotient reconstruction | `hypergraph.cpp`, 11 `qc_*` functions, **421 lines** | `quotient_expansion.hpp` 939 + `quotient_causal.hpp` 419, **1,358 lines** |

The device pair uses `hgcommon` only for the KEYS (`ir_core.hpp`, `slot_core.hpp`). The DP and
the rendezvous — which producer meets which consumer at which key, which `(instance, match)`
pair is claimed, which pair survives the reduction — are a second implementation.

They are the same rule. Read side by side, `qe_drive_instance` is `qc_add_instance`'s scan half
with the same `depth >= max_steps` guard and the same publish-then-scan fence;
`qe_drive_match` is `qc_capture_expansion`'s scan half with the same `d < max_steps` loop; and
`qe_reachable` is `qc_reachable` with the same id-order prune. What differs is not the rule but
the containers it runs over — `ConcurrentMap`/`LockFreeList`/arena against device hash tables
and pools — plus the fence spelling and the scratch discipline.

**Both copies have already drifted, and neither drift was found by a gate.** Both were found by
reading the two bodies against each other while writing this document:

1. `qe_reachable` had no visited set; `qc_reachable` (`hypergraph.cpp:976`) does. The device
   therefore expanded a node once per PATH into it rather than once, which is exponential in the
   worst case where the host is linear. Same answer, different bound. Measured after fixing it:
   no speedup outside this machine's run-to-run wall-clock drift, and the reason is worth
   keeping — the walk is over KEPT predecessors, so reduction leaves most events with one and
   the paths and the nodes stay close on ordinary workloads. It is a bound that was wrong, not a
   cost that was being paid.
2. The device cascades recurse once per reconstruction depth against a fixed per-thread stack,
   and a 7-step run on a two-consumed-edge rule returned an illegal memory access — and with a
   poisoned CUDA context, nothing at all for every later workload in the process. Fixed in
   `00e21ee` by sizing the stack from the depth and bounding the recursion, but the host has no
   such cap: it recurses the same way on an 8 MB thread stack, roughly two orders of magnitude
   further out. The two engines answer different questions at depth.

Neither is a bug in "the GPU". Both are what a second body costs.

## 4. Done-line

> The quotient reconstruction has one body. `hgcommon/quotient_core.hpp` holds the DP, the
> rendezvous and the reduction, templated on a container policy; `hypergraph.cpp` and
> `quotient_expansion.hpp` supply policies and nothing else. The line between semantics and
> orchestration is stated in one place and checked.

Not "one engine". Not "the GPU depends on the CPU". The two devices keep their own schedulers,
memory models and launch strategies — those are orchestration, they are genuinely different,
and merging them would be the mistake this document exists to avoid.

## 5. The shape, with its precedent

`join_core.hpp` is the precedent and it already works: one `HG_HD` backtracking-join body,
templated on candidate enumeration only, called by a 517-line host matcher and a 533-line
device kernel that share it. The differences that survive there are exactly the ones that must:
how you enumerate candidates, and what you do with a match.

The reconstruction wants the same treatment with a wider policy, because it stores state where
the join does not:

```
template <typename Policy>
struct QuotientCore {
    // Policy supplies, and NOTHING else:
    //   map_t / list_t / claim(key)      the containers and the exactly-once claim
    //   fence()                          seq_cst on host, __threadfence() on device
    //   scratch                          the reachability walk's stack and visited set
    //   record_overflow(kind)            what to do when a bounded resource runs out
    // The DP, the (instance, match) rendezvous, the id-order prune and the reduction live
    // HERE, once.
};
```

Two properties the policy must carry, both learned from the drifts above:

- **The reachability walk's scratch is a policy, not a constant.** The host has an unbounded
  vector and a visited set; the device has a fixed span. Making that a policy is what forces
  the device to declare its bound instead of silently having one.
- **Recursion is not part of the rule.** The cascade should be a work list in the shared body,
  so depth costs O(1) stack on both sides and the device's depth cap disappears rather than
  being merely reported. This is the change that makes §3's second drift impossible, not just
  detected.

## 6. Steps, each with a gate

| id | step | gate |
|---|---|---|
| P6.2a | Move `park.hpp` and `portable_intrinsics.hpp` out of `hgcommon` into a platform header set. `hgcommon` becomes semantics only. | both engines build; no behaviour change in any gate |
| P6.2b | `hgcommon/quotient_core.hpp`: the DP and the rendezvous as one body over a container policy. Host adapter lands with it; the host's `qc_*` bodies are deleted in the same commit. | `all_tests` green; `quotient_determinism_rate_probe` 0 at `--load 6`; the reconstruction's served relation unchanged on all 17 corpus workloads |
| P6.2c | Device adapter; `quotient_expansion.hpp`'s DP and rendezvous deleted in the same commit. | `gpu_differential_tests` green including `deep_cone_reduction_*`; `hg_gpu_tests` green |
| P6.2d | Convert the cascade to a work list inside the shared body. | the device depth cap is gone: `PastTheStackDepthItRecordsRatherThanFaults` becomes an equality test against the host at depth 80, with no `kScratchOverflow` |
| P6.2e | The reachability walk is one body, so the host's visited set and the device's cannot be two decisions. | the two walks are one function; `gpu_differential_tests` and `all_tests` green |

P6.2a is separable and cheap; it can land first and alone. P6.2b–c are the substance. P6.2d–e
are what turn "the copies agree today" into "there is nothing to disagree".

## 7. What is deliberately NOT shared

Stated so a later reader does not try:

- **Schedulers.** The host's work-stealing deques plus injector, and the device's persistent
  kernel with its ring buffer, solve the same problem on hardware with different costs. They
  are orchestration.
- **Memory.** Arenas with per-worker bump cursors against pre-allocated device pools with
  atomic claims. The allocation STRATEGY is hardware; what gets allocated is semantics and is
  already shared.
- **Overflow policy.** The device is capacity-bounded everywhere and reports partial work; the
  host grows. That difference is real and the policy argument is where it belongs.
- **The FFI and marshalling.** Already shared where it should be (`graph_marshal.hpp`,
  `hgmarshal::build_graph_data` via a Source adapter) and device-specific where it must be.

## 8. Cost, and the reason to do it anyway

P6.2b–c move ~1358 device lines and ~420 host lines behind one body. That is a large diff
against code that currently passes every gate, which is the strongest argument for NOT doing
it — and the reason to do it is in §3: both copies had already drifted, in ways no gate caught,
and one of them faulted the device on a seven-step run. The gates were not weak; they were
comparing two implementations on the workloads someone thought to write down. A rule with one
body has nothing to compare.
