# IR Verification Layer

## Architecture: 3-Tier System

### Tier 1: Hash (WL/UT) — hot path, always runs
Produces `uint64_t` for state deduplication during evolution. No false negatives (isomorphic → same hash). Rare false positives possible. `HashStrategy` enum: WL, UT, iUT, iWL.

### Tier 2: IR Verification — on hash collision or when edge correspondence needed
When two states hash-equal, IR confirms or rejects isomorphism. Provides exact edge correspondence as byproduct of vertex mappings. Replaces heuristic WL edge-signature correspondence in `find_edge_correspondence_dispatch()`. Used by event canonicalization (Automatic mode) for consumed/produced edge canonicalization. No stored canonical forms — mappings used transiently then discarded.

### Tier 3: Full Canonical Form Output — on explicit user request
`"ReturnCanonicalStates" -> True` triggers this. IR produces `CanonicalizationResult` with canonical form + vertex mapping. Returns lexicographically sorted, relabeled edges as the canonical representative.

## Options

- `"IRVerification" -> True/False` — enables Tier 2 (exact edge correspondence + collision verification)
- `"ReturnCanonicalStates" -> True/False` — enables Tier 3 (canonical form on demand)
- Both default to `False`
- In the Wolfram Language interface, `"CanonicalizeStates" -> Full` enables both automatically

## Performance Impact

The hot path is completely unchanged. Hashing uses WL/UT/iUT/iWL as before. IR code only runs when explicitly enabled.

### Hot path (every state, unchanged)
- **Hashing**: `compute_hash_with_cache_dispatch()` uses WL/UT — no IR involvement.
- **State dedup**: `create_or_get_canonical_state()` does hash → ConcurrentMap lookup. When `use_ir_verification_ == false` (default), the only overhead is a single bool branch that is always not-taken.

### IR verification enabled (`IRVerification -> True`)

Two places IR activates:

**1. Edge correspondence** (`find_edge_correspondence_dispatch()`)

Called only during event canonicalization when `EventKey_ConsumedEdges` or `EventKey_ProducedEdges` is set. Runs partition refinement + backtracking on both states.

- Low-symmetry graphs (most real cases): refinement produces discrete partition directly → O(V²·E), same order as WL with larger constant.
- High-symmetry graphs: backtracking explores automorphisms, worst case exponential but bounded by automorphism group size. Refined partition prunes almost all branches in practice.
- Replaces WL heuristic with exact answer — WL signatures can incorrectly match non-corresponding edges in graphs with symmetry.

**2. Hash collision verification** (`create_or_get_canonical_state()`)

Runs only when two states hash to the same value AND the insertion finds an existing entry.

- Truly isomorphic duplicates (common): IR confirms, small overhead per duplicate.
- False positives (rare WL collisions): IR correctly rejects, preventing incorrect state merging.
- Unique states (most insertions): never runs — the `was_inserted == true` fast path skips IR.

### Cost summary

| Scenario | IR off (default) | IR on |
|---|---|---|
| State hashing | WL/UT (unchanged) | WL/UT (unchanged) |
| State dedup (new state) | hash + map insert | same + 1 bool check |
| State dedup (duplicate) | hash + map lookup | same + IR isomorphism check |
| Edge correspondence | WL signature matching | IR canonical form matching |
| No event canonicalization | nothing | nothing |

## Brute-Force Canonicalizer Status

The old `Canonicalizer` class (`canonicalization.hpp/.cpp`) with O(V!) factorial-time brute-force search is **no longer used in any production code path**. It has been fully replaced by `IRCanonicalizer`:

- `compute_canonical_hash()`: fallback path (when `use_shared_tree_ == false`) now uses `IRCanonicalizer` instead of `Canonicalizer`. Note: `use_shared_tree_` is always `true` in practice — `disable_shared_tree()` is never called — so this fallback is currently dead code.
- `get_canonical_form_string()`: debug function now uses `IRCanonicalizer`.
- `find_edge_correspondence_dispatch()`: uses `IRCanonicalizer` when `use_ir_verification_` is set.
- `create_or_get_canonical_state()`: uses `IRCanonicalizer::are_isomorphic()` for collision verification.
- FFI `ReturnCanonicalStates` output: uses `IRCanonicalizer::canonicalize_edges()`.

The shared types (`CanonicalizationResult`, `CanonicalForm`, `VertexMapping`) have been extracted to `canonical_types.hpp`. All production code and tests now include `canonical_types.hpp` or `ir_canonicalization.hpp` (which includes it) instead of `canonicalization.hpp`.

The `Canonicalizer` class methods now early-return empty results. Their bodies are `#if 0` guarded. The class and `canonicalization.hpp/.cpp` remain only for `test_canonicalization.cpp` — these should be removed together.

Cross-validation tests that compared IR against brute-force have been removed. The benchmark file (`canonicalization_benchmarks.cpp`) and test helpers (`test_helpers.hpp`) have been switched to `IRCanonicalizer`.

## Key Files

- `hypergraph/include/hypergraph/ir_canonicalization.hpp` — IRCanonicalizer class
- `hypergraph/src/ir_canonicalization.cpp` — partition refinement + backtracking search
- `hypergraph/include/hypergraph/hypergraph.hpp` — `use_ir_verification_`, `return_canonical_states_` flags
- `hypergraph/src/hypergraph.cpp` — IR verification in `find_edge_correspondence_dispatch()` and `create_or_get_canonical_state()`
- `paclet_source/hypergraph_ffi.cpp` — `"IRVerification"` and `"ReturnCanonicalStates"` option parsing + canonical edges output
- `paclet/Kernel/HypergraphRewriting.wl` — `stateDisplayEdges` helper, state thumbnail rendering, tooltip display

## FFI Interface

Options passed via WXF `"Options"` association:

```
"IRVerification" -> True|False     (* enables Tier 2: exact edge correspondence + collision verification *)
"ReturnCanonicalStates" -> True|False  (* enables Tier 3: canonical form in output *)
```

When `ReturnCanonicalStates` is `True`, each state in the output includes an additional `"CanonicalEdges"` field containing the IR-canonicalized edge list (vertices relabeled to 0..n-1, edges lexicographically sorted). This is computed on demand during serialization, not during evolution.

## Wolfram Language Integration

When `"CanonicalizeStates" -> Full` is passed to `HGEvolve`, the WL layer automatically sets both `"IRVerification" -> True` and `"ReturnCanonicalStates" -> True`.

State plot thumbnails (in multiway graphs) use `stateDisplayEdges[data]` which returns `data["CanonicalEdges"]` when present, falling back to `Rest /@ data["Edges"]` otherwise. This means:
- `"CanonicalizeStates" -> Full`: thumbnails show IR-canonicalized hypergraphs
- `"CanonicalizeStates" -> None` or `Automatic`: thumbnails show original raw edges (unchanged behavior)

State tooltips also display the canonical edge list when available.

## Implementation Details

The `IRCanonicalizer` uses McKay-style individualization-refinement directly on the hypergraph (no incidence graph conversion). The `are_isomorphic()` method provides a cheaper yes/no answer by comparing canonical forms without building full correspondence mappings.

`HashStrategy::IR` was removed from the enum — IR is not a hash strategy but a verification/correspondence layer that sits alongside whatever hash strategy is active.
