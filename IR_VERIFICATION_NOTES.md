# IR Canonicalization

## User-Facing Options

Only two canonicalization options exist:

- `"CanonicalizeStates" -> None | Automatic | Full`
- `"CanonicalizeEvents" -> None | Full | Automatic | {keys...}`

These are independent of each other.

### CanonicalizeStates

| Mode | Evolution Dedup | Edge Correspondence | Output Edges | Output Dedup |
|---|---|---|---|---|
| None (default) | No | WL heuristic | Raw | No |
| Automatic | No (display-time grouping via ContentStateId) | WL heuristic | Raw | No |
| Full | Yes (IR canonical hash) | IR exact | IR-canonical (0..n-1) | Yes (one per canonical ID) |

**Full mode** uses McKay-style individualization-refinement (IRCanonicalizer) for:
1. **Evolution-time deduplication**: IR canonical hash as map key — isomorphic states collapse to one representative
2. **Exact edge correspondence**: `find_edge_correspondence_dispatch()` uses IR instead of WL heuristic
3. **Canonical output**: edges relabeled to contiguous 0..n-1, lexicographically sorted, deduplicated in output

**Automatic mode** does NOT perform evolution-time deduplication (matches MultiwaySystem reference behavior). It computes ContentStateId at output time for display-time grouping.

## Architecture

### State Deduplication (Full mode)

In `create_or_get_canonical_state()`, the map key is computed by `IRCanonicalizer::compute_canonical_hash()` which runs full partition refinement + backtracking to produce an exact isomorphism-invariant hash. Two states with the same IR hash ARE isomorphic — no additional verification is needed.

### Edge Correspondence (Full mode)

In `find_edge_correspondence_dispatch()`, when Full mode is active, both states are canonicalized via IR. Edge correspondence is derived from the canonical edge ordering — exact, replacing the WL signature heuristic which can fail on symmetric graphs.

### Canonical Output (Full mode)

At serialization time, `serialize_state_edges()` and the main states output loop run `IRCanonicalizer::canonicalize_edges()` to relabel vertices to 0..n-1 and sort edges. The output dedup filter emits one state per unique canonical ID.

## Performance

| Scenario | None/Automatic | Full |
|---|---|---|
| State hashing | WL/UT (hot path) | WL/UT (hot path) |
| State dedup map key | state_id / content_hash | IR canonical hash (partition refinement + backtracking) |
| Edge correspondence | WL signature matching | IR canonical form matching |
| Output edges | Raw | IR-canonicalized |

Full mode adds IR canonicalization cost per new state. For low-symmetry graphs (most real cases), partition refinement produces a discrete partition directly — O(V²·E). For high-symmetry graphs, backtracking explores automorphisms bounded by automorphism group size.

## Key Files

- `hypergraph/include/hypergraph/ir_canonicalization.hpp` — IRCanonicalizer class
- `hypergraph/src/ir_canonicalization.cpp` — partition refinement + backtracking search
- `hypergraph/include/hypergraph/hypergraph.hpp` — `StateCanonicalizationMode` enum, `is_full_canonicalization()` method
- `hypergraph/src/hypergraph.cpp` — IR hash in `create_or_get_canonical_state()`, IR edge correspondence in `find_edge_correspondence_dispatch()`
- `paclet_source/hypergraph_ffi.cpp` — `full_canonicalization` flag derived from `state_canon_mode == Full`, canonical edge output + state dedup
- `paclet/Kernel/HypergraphRewriting.wl` — `stateDisplayEdges` helper, state thumbnail rendering
