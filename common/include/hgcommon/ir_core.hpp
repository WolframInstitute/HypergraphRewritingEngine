#pragma once
#include "hgcommon/namespace.hpp"
// Shared CPU/GPU individualization-refinement canonical-hash core.
//
// ONE implementation, so the host engine and the CUDA port agree bit for bit on the exact
// canonical hash -- the same guarantee wl_core.hpp gives for the approximate one.
//
// Everything is flat and caller-provided: the partition is four uint32 arrays rather than a
// vector of per-cell vectors, the canonical form at a leaf is one contiguous buffer rather
// than a vector of per-edge vectors, and a search node snapshots two arrays rather than
// deep-copying a container of containers. That is what makes the routine callable from a
// device with no allocator, and it is also where the host time went: the search visits one
// node per individualization and one leaf per discrete partition, so a per-cell and per-edge
// allocation at each of those dominates on exactly the symmetric states that need the search.
//
// The caller flattens a state to LOCAL vertex indices [0, n_verts) in the wl_core convention:
//   ea[e]           = arity of edge e                          (e in [0, n_edges))
//   ev[eoff[e] + p] = local vertex index of edge e at position p (p in [0, ea[e]))
// ir_scratch_words() reports the buffer size needed for a given (n_verts, n_edges, total_occ,
// max_depth); the caller supplies that as one uint32 span and the core sub-allocates.

#include <cstdint>
#include "hgcommon/core.hpp"

namespace HG_NAMESPACE {
namespace common {

// Cells whose vertices all share an edge with the splitter are separated by a 64-bit key per
// incident edge: arity in bits 56-63, the vertex's own position in 48-55, and a bitmask of the
// positions the splitter's vertices occupy in 0-47. Exact for arity <= 48, and MAX_ARITY is 16.
constexpr uint32_t IR_SPOS_BITS = 48;

// Count trailing zeros, callable from host and device. Undefined for x == 0.
HG_HD inline uint32_t ir_ctz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__ffsll(static_cast<long long>(x)) - 1);
#elif defined(_MSC_VER)
    unsigned long i; _BitScanForward64(&i, x); return static_cast<uint32_t>(i);
#else
    return static_cast<uint32_t>(__builtin_ctzll(x));
#endif
}

HG_HD inline uint32_t ir_bitset_words(uint32_t n) { return (n + 63u) / 64u + 1u; }

// Individualization depth the search will explore before giving up. A path fixes at least one
// vertex per level, so n levels always suffice; the cap bounds scratch instead, and a state
// needing more reports IR_NEED_DEPTH so the caller can retry with a larger one.
constexpr uint32_t IR_MAX_DEPTH_DEFAULT = 64;

// Automorphisms retained for orbit pruning, as a per-caller BUDGET rather than a constant.
//
// Generators only SKIP branches automorphic to one already explored, and automorphic branches
// reach the same canonical form, so the budget costs search TIME on a symmetric state and
// never changes the result. It is not a tuning nicety: on a state of 30 isomorphic components
// a budget of 64 leaves the pruning too weak to collapse the search and it does not finish,
// where 512 completes in 5.4 s -- faster than the unbounded-generator host implementation's
// 6.9 s on the same state.
//
// So the two devices want different budgets, and that is exactly the split: the RESULT is
// shared, the search budget is orchestration. The host can afford generators * n_verts words
// of scratch; a device thread cannot.
constexpr uint32_t IR_HOST_GENERATORS   = 512;
constexpr uint32_t IR_DEVICE_GENERATORS = 32;

// Generators and depth blocks are the bulk of the scratch and are touched only when the
// initial refinement leaves a non-discrete partition. max_depth == 1 means "root only": it
// sizes for the common state, which is discrete after refinement, and reports IR_NEED_DEPTH
// for the rest so the caller can retry at a depth that searches.
HG_HD inline uint32_t ir_generator_cap(uint32_t max_depth, uint32_t max_generators) {
    return max_depth > 1 ? max_generators : 0u;
}

// IR_NEED_GENERATORS is returned ONLY when orbits were requested and the generator table
// filled. Generators serve two purposes and the distinction is the whole point: for SEARCH
// PRUNING a short table costs time and cannot change the canonical form, because automorphic
// branches reach the same form either way; for ORBITS it changes the answer, since orbits are
// fused over the generators found and a short table fuses less, yielding orbits that are too
// FINE. So a caller that asked only for a hash never sees this status, and a caller that asked
// for orbits is told to retry with a larger budget rather than handed a finer partition than
// the automorphism group licenses.
// Words a caller must provide for out_canonical_form: one arity word per edge plus one word
// per vertex occurrence, which is the form's own layout.
HG_HD inline uint32_t ir_canonical_form_words(uint32_t n_edges, uint32_t total_occ) {
    return n_edges + total_occ;
}

enum IrStatus : uint32_t {
    IR_OK = 0, IR_EMPTY = 1, IR_NEED_DEPTH = 2, IR_NEED_GENERATORS = 3
};

// Per-depth partition snapshot: lab, pos, cell_of, cstart, clen, plus the sorted target cell
// and its covered flags. A search node writes the next depth from the current one and refines
// it in place, so backtracking is a return -- no undo trail, and nothing is ever allocated.
// Sized to the word: the device takes this per state from its arena, and a search at the
// default depth of 64 multiplies every extra word here by 64 (measured: an eighth array per
// depth cost the device 12% on wolfram24 depth 7).
HG_HD inline uint64_t ir_depth_words(uint32_t n_verts) { return 7ull * n_verts + 8ull; }

// Total uint32 words of scratch for a given problem size.
HG_HD inline uint64_t ir_scratch_words(uint32_t n_verts, uint32_t n_edges,
                                       uint32_t total_occ, uint32_t max_depth,
                                       uint32_t max_generators = IR_HOST_GENERATORS) {
    const uint64_t n = n_verts, e = n_edges, occ = total_occ, d = max_depth;
    return
        (n + 1) + occ + occ + n             // occ_off, occ_edge, occ_pos, cursor
      + 2 * ir_bitset_words(n_verts)        // worklist, as uint64
      + e + e + e                           // inc_edges, edge_epoch, form_order
      + n + n + 2 * n                       // touched, on_touched, torder (2n: split staging)
      + n + n + (n + 1) + 2 * occ           // sig_off, sig_cnt, gstart, sig_buf as uint64
      + n + n + n + n + n + n               // path, first_path, labeling, first_labeling, inv, best_lab
      + 3 * (occ + e) + e                   // cur_form, best_form, first_form, best_order
      + d * ir_depth_words(n_verts)         // per-depth partition + cell + covered
      + uint64_t(ir_generator_cap(max_depth, max_generators)) * (n + 1)  // generators, row-major, and each one's fixed-prefix length
      + 64;                                 // alignment slack for the uint64 views
}

// Sub-allocator over the caller's span. No bounds checking on the fast path -- the caller
// sized the span with ir_scratch_words for the same (n, e, occ, depth).
struct IrScratch {
    uint32_t* base;
    uint64_t  used = 0;
    HG_HD uint32_t* u32(uint64_t count) { uint32_t* p = base + used; used += count; return p; }

    // Aligns on the ADDRESS, not on the index. Aligning the index only reaches an 8-byte
    // boundary when `base` is already 8-byte aligned, which a caller handing out fixed-stride
    // slices of one pool does not guarantee: a slot base of pool + i*stride with an odd stride
    // is 4-byte aligned for odd i, and a uint64 view of it faults on the device.
    HG_HD uint64_t* u64(uint64_t count) {
        while ((reinterpret_cast<uintptr_t>(base + used) & uintptr_t(7)) != 0) ++used;
        uint64_t* p = reinterpret_cast<uint64_t*>(base + used);
        used += 2 * count;
        return p;
    }
};

// Flat ordered partition. Cells are identified by an id, not by position: refinement keeps the
// largest piece at the split cell's id and appends the rest, and both the splitter worklist and
// the target-cell choice read ids in increasing order. Each id owns a contiguous run of `lab`.
struct IrPartition {
    uint32_t* lab;       // vertices, grouped by cell
    uint32_t* pos;       // pos[v] = index of v in lab
    uint32_t* cell_of;   // cell_of[v] = cell id
    uint32_t* cstart;    // cstart[c] = start of cell c in lab
    uint32_t* clen;      // clen[c] = length of cell c
    uint32_t  ncells;
    uint32_t  n;

    HG_HD bool is_discrete() const { return ncells == n; }
    HG_HD uint32_t first_non_singleton() const {
        for (uint32_t c = 0; c < ncells; ++c) if (clen[c] > 1) return c;
        return ncells;
    }
};

// Below this length, insertion sort. States here are small -- a few tens of vertices and
// edges -- and at that size heapsort's sift-down loses to a linear scan over data already
// close to sorted, while above it heapsort's bound is what keeps a large state off O(n^2).
constexpr uint32_t IR_SMALL_SORT = 24;

// Sort an index array. O(n log n) above the small-array threshold, no recursion, no
// allocation and no std::, so one routine serves host and device. Not stable; every
// comparator below either is a total order or has ties whose relative order does not reach
// the output.
template <class Cmp>
HG_HD inline void ir_heapsort_idx(uint32_t* a, uint32_t n, Cmp cmp) {
    if (n < 2) return;
    if (n <= IR_SMALL_SORT) {
        for (uint32_t i = 1; i < n; ++i) {
            const uint32_t key = a[i];
            uint32_t j = i;
            while (j > 0 && cmp(a[j - 1], key) > 0) { a[j] = a[j - 1]; --j; }
            a[j] = key;
        }
        return;
    }
    auto sift = [&](uint32_t root, uint32_t end) {
        for (;;) {
            uint32_t child = 2 * root + 1;
            if (child >= end) break;
            if (child + 1 < end && cmp(a[child], a[child + 1]) < 0) ++child;
            if (cmp(a[root], a[child]) >= 0) break;
            const uint32_t t = a[root]; a[root] = a[child]; a[child] = t;
            root = child;
        }
    };
    for (uint32_t i = n / 2; i-- > 0;) sift(i, n);
    for (uint32_t end = n; end-- > 1;) {
        const uint32_t t = a[0]; a[0] = a[end]; a[end] = t;
        sift(0, end);
    }
}

// Lexicographic compare of two sorted uint64 runs, shorter-is-smaller on a prefix. This is the
// order the per-vertex signature multisets are compared in.
HG_HD inline int ir_cmp_run(const uint64_t* a, uint32_t na, const uint64_t* b, uint32_t nb) {
    const uint32_t m = na < nb ? na : nb;
    for (uint32_t i = 0; i < m; ++i) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    if (na == nb) return 0;
    return na < nb ? -1 : 1;
}

// -----------------------------------------------------------------------------------------
// Occurrence CSR: per vertex, the (edge, position) pairs it appears at.
// -----------------------------------------------------------------------------------------
HG_HD inline void ir_build_occurrences(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev,
    uint32_t n_edges, uint32_t n_verts,
    uint32_t* occ_off, uint32_t* occ_edge, uint32_t* occ_pos, uint32_t* cursor)
{
    for (uint32_t v = 0; v <= n_verts; ++v) occ_off[v] = 0;
    for (uint32_t e = 0; e < n_edges; ++e)
        for (uint8_t p = 0; p < ea[e]; ++p) occ_off[ev[eoff[e] + p] + 1u]++;
    for (uint32_t v = 0; v < n_verts; ++v) occ_off[v + 1] += occ_off[v];
    for (uint32_t v = 0; v < n_verts; ++v) cursor[v] = occ_off[v];
    for (uint32_t e = 0; e < n_edges; ++e)
        for (uint8_t p = 0; p < ea[e]; ++p) {
            const uint32_t w = cursor[ev[eoff[e] + p]]++;
            occ_edge[w] = e;
            occ_pos[w]  = p;
        }
}

// -----------------------------------------------------------------------------------------
// SORTED-UNIQUE dense renumbering of ev[0..occ) in place. `verts` is caller scratch of at
// least occ words; on return its first (returned) words are the distinct raw ids ascending,
// and every ev[i] is that array's index of its raw id. One rule for both engines: the host's
// canonicalizer entry sorted a vector and the device's flattening deduplicated by linear scan
// and insertion-sorted -- the scan and the sort were both quadratic in a state's occurrence
// count, and two bodies of one convention drift. The convention itself (sorted, not
// encounter order) is load-bearing: the within-cell tie-break reads the numbering, so it
// selects which coset representative the canonical labelling is, and the RANKS the two
// engines exchange must come from the same numbering.
HG_HD inline uint32_t ir_renumber_sorted(uint32_t* ev, uint32_t occ, uint32_t* verts) {
    struct U32Cmp {
        HG_HD int operator()(uint32_t a, uint32_t b) const {
            return a < b ? -1 : (a > b ? 1 : 0);
        }
    };
    for (uint32_t i = 0; i < occ; ++i) verts[i] = ev[i];
    ir_heapsort_idx(verts, occ, U32Cmp{});
    uint32_t n = 0;
    for (uint32_t i = 0; i < occ; ++i)
        if (n == 0 || verts[i] != verts[n - 1]) verts[n++] = verts[i];
    for (uint32_t o = 0; o < occ; ++o) {
        uint32_t lo = 0, hi = n;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (verts[mid] < ev[o]) lo = mid + 1; else hi = mid;
        }
        ev[o] = lo;
    }
    return n;
}

// -----------------------------------------------------------------------------------------
// THE ORDER-SAFE LOOPS, as dispatchable bodies. On the host (and on every device caller that
// runs one thread per state) IrSerial runs them in place, byte-for-byte the loops they
// replace. Under the persistent kernel a warp policy (hg_gpu/ir_canon.hpp) fans iterations
// across the block's 32 lanes: these are the only loops handed over, because each either
// writes per-iteration-disjoint regions (a vertex's own signature run) or feeds a sort that
// normalises its write order (the splitter signature fill), so the lane schedule cannot reach
// any output. Everything order-carrying -- the incident-edge gather, the touched list, the
// cell splits, the search -- stays in the caller's serial code, because the within-cell
// tie-break reads those orders and the ranks read the tie-break.
struct IrParArgs {
    const uint8_t*  ea = nullptr;
    const uint32_t* eoff = nullptr;
    const uint32_t* ev = nullptr;
    const uint32_t* occ_off = nullptr;
    const uint32_t* occ_edge = nullptr;
    const uint32_t* occ_pos = nullptr;
    uint64_t* sig_buf = nullptr;
    const uint32_t* sig_off = nullptr;
    uint32_t* sig_cnt = nullptr;
    const uint32_t* touched = nullptr;
    const uint32_t* inc_edges = nullptr;
    const uint32_t* cell_of = nullptr;
    uint32_t* order = nullptr;
    uint32_t  S = 0;
};
enum IrParKind : uint32_t {
    IR_PAR_EXIT = 0,     // the warp policy's shutdown sentinel; never dispatched as work
    IR_PAR_INIT_SIGS,    // per vertex: fill + sort its degree-signature run, seed order
    IR_PAR_TOUCH_SORTS,  // per touched vertex: sort its splitter-signature run
    IR_PAR_SIG_FILL,     // per incident edge: append a key to every vertex on it
};
// How a body reserves a slot in a shared per-vertex counter: plain serially, atomic when the
// warp policy fans one kind's iterations across lanes.
template <bool kAtomic>
HG_HD inline uint32_t ir_par_reserve(uint32_t* p) {
#if defined(__CUDA_ARCH__)
    if constexpr (kAtomic) return atomicAdd(p, 1u);
#endif
    return (*p)++;
}
template <bool kAtomic>
HG_HD inline void ir_par_body(uint32_t kind, uint32_t i, const IrParArgs& a) {
    switch (kind) {
        case IR_PAR_INIT_SIGS: {
            const uint32_t s = a.occ_off[i], len = a.occ_off[i + 1] - s;
            for (uint32_t k = 0; k < len; ++k)
                a.sig_buf[s + k] =
                    (uint64_t(a.ea[a.occ_edge[s + k]]) << 32) | uint64_t(a.occ_pos[s + k]);
            isort_u64(a.sig_buf + s, len);
            a.order[i] = i;
            break;
        }
        case IR_PAR_TOUCH_SORTS: {
            const uint32_t u = a.touched[i];
            isort_u64(a.sig_buf + a.sig_off[u], a.sig_cnt[u]);
            break;
        }
        case IR_PAR_SIG_FILL: {
            const uint32_t e = a.inc_edges[i];
            const uint32_t arity = a.ea[e];
            uint64_t spos = 0;
            for (uint32_t p = 0; p < arity && p < IR_SPOS_BITS; ++p)
                if (a.cell_of[a.ev[a.eoff[e] + p]] == a.S) spos |= (uint64_t(1) << p);
            for (uint32_t p = 0; p < arity; ++p) {
                const uint32_t u = a.ev[a.eoff[e] + p];
                const uint32_t w = ir_par_reserve<kAtomic>(a.sig_cnt + u);
                a.sig_buf[a.sig_off[u] + w] =
                    (uint64_t(arity & 0xFF) << 56) | (uint64_t(p & 0xFF) << 48) | spos;
            }
            break;
        }
        default: break;
    }
}
struct IrSerial {
    HG_HD void run(uint32_t kind, uint32_t n, const IrParArgs& a) const {
        for (uint32_t i = 0; i < n; ++i) ir_par_body<false>(kind, i, a);
    }
};

// -----------------------------------------------------------------------------------------
// Initial partition: group vertices by degree signature -- the sorted multiset of
// (arity, position) over their occurrences -- ordering groups by signature and, within a
// signature, by vertex index. Both keys are structural, so the partition is equivariant.
// -----------------------------------------------------------------------------------------
template <class Par = IrSerial>
HG_HD inline void ir_initial_partition(
    const uint8_t* ea, const uint32_t* occ_off, const uint32_t* occ_edge, const uint32_t* occ_pos,
    uint32_t n_verts, IrPartition& pi,
    uint64_t* sig_buf, uint32_t* order, Par par = Par{})
{
    // sig_buf holds every vertex's signature contiguously at occ_off[v]; (arity, position)
    // packs into one uint64 so the multiset sorts with a scalar comparison. Each vertex's run
    // is its own, so the iterations fan out (IR_PAR_INIT_SIGS).
    {
        IrParArgs a;
        a.ea = ea; a.occ_off = occ_off; a.occ_edge = occ_edge; a.occ_pos = occ_pos;
        a.sig_buf = sig_buf; a.order = order;
        par.run(IR_PAR_INIT_SIGS, n_verts, a);
    }

    // Order by (signature, vertex). The signature is structural and label-independent; the
    // vertex tie-break makes it a total order, so the resulting cells are equivariant.
    struct SigCmp {
        const uint64_t* sig; const uint32_t* off;
        HG_HD int operator()(uint32_t a, uint32_t b) const {
            const int c = ir_cmp_run(sig + off[a], off[a + 1] - off[a],
                                     sig + off[b], off[b + 1] - off[b]);
            if (c != 0) return c;
            return a < b ? -1 : (a > b ? 1 : 0);
        }
    };
    ir_heapsort_idx(order, n_verts, SigCmp{sig_buf, occ_off});

    pi.n = n_verts;
    pi.ncells = 0;
    uint32_t w = 0;
    for (uint32_t i = 0; i < n_verts;) {
        const uint32_t c = pi.ncells++;
        const uint32_t a = order[i];
        pi.cstart[c] = w;
        uint32_t j = i;
        while (j < n_verts) {
            const uint32_t b = order[j];
            if (ir_cmp_run(sig_buf + occ_off[a], occ_off[a + 1] - occ_off[a],
                           sig_buf + occ_off[b], occ_off[b + 1] - occ_off[b]) != 0) break;
            pi.lab[w] = b; pi.pos[b] = w; pi.cell_of[b] = c;
            ++w; ++j;
        }
        pi.clen[c] = w - pi.cstart[c];
        i = j;
    }
}

// -----------------------------------------------------------------------------------------
// Refinement to the coarsest equitable partition finer than pi.
//
// Splitters are popped lowest-id-first from a bitset, which is a structurally determined order
// and therefore equivariant under vertex relabeling. Refining a cell by a splitter inspects
// only vertices sharing an edge with it, so a split costs O(boundary); keeping the largest
// piece at the split cell's id bounds the total to ~O(E log n).
// -----------------------------------------------------------------------------------------
template <class Par = IrSerial>
HG_HD inline void ir_refine(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev, uint32_t n_edges,
    const uint32_t* occ_off, const uint32_t* occ_edge,
    IrPartition& pi,
    uint64_t* worklist, uint32_t* inc_edges, uint32_t* edge_epoch,
    uint32_t* touched, uint32_t* on_touched, uint32_t* torder,   // torder holds 2n: order, then split staging
    uint32_t* sig_off, uint32_t* sig_cnt, uint32_t* gstart, uint64_t* sig_buf,
    uint32_t seed0 = 0xFFFFFFFFu, uint32_t seed1 = 0xFFFFFFFFu, Par par = Par{})
{
    const uint32_t n = pi.n;
    const uint32_t wl_words = ir_bitset_words(n);
    for (uint32_t w = 0; w < wl_words; ++w) worklist[w] = 0;
    if (seed0 == 0xFFFFFFFFu) {
        for (uint32_t c = 0; c < pi.ncells; ++c) worklist[c >> 6] |= (uint64_t(1) << (c & 63));
    } else {
        // The partition is an equitable one with one cell split in two (an individualisation):
        // every other cell is a splitter it was already stable against, so only the two new
        // cells can move anything, and the pieces they produce enqueue themselves below.
        // Seeding every cell instead re-signed the whole state per search node: measured on
        // disc-l3a2g2r2 depth 2 as 54% of all instructions in this function at 16.6k per node.
        worklist[seed0 >> 6] |= (uint64_t(1) << (seed0 & 63));
        if (seed1 != 0xFFFFFFFFu) worklist[seed1 >> 6] |= (uint64_t(1) << (seed1 & 63));
    }
    for (uint32_t e = 0; e < n_edges; ++e) edge_epoch[e] = 0;
    for (uint32_t v = 0; v < n; ++v) on_touched[v] = 0;
    uint32_t epoch = 0;

    for (;;) {
        // A discrete partition has nothing left to split, so every splitter still queued
        // would gather its incident edges and compute its signatures only to find no cell
        // divides. Refinement of a discrete partition is the identity.
        if (pi.ncells == n) break;

        // Pop the lowest set cell id.
        uint32_t S = 0xFFFFFFFFu;
        for (uint32_t w = 0; w < wl_words; ++w) {
            if (worklist[w]) {
                const uint32_t b = ir_ctz64(worklist[w]);
                worklist[w] &= worklist[w] - 1;
                S = w * 64u + b;
                break;
            }
        }
        if (S == 0xFFFFFFFFu) break;
        if (S >= pi.ncells) continue;

        // Edges incident to S, deduplicated by an epoch stamp.
        ++epoch;
        uint32_t n_inc = 0;
        for (uint32_t k = 0; k < pi.clen[S]; ++k) {
            const uint32_t s = pi.lab[pi.cstart[S] + k];
            for (uint32_t o = occ_off[s]; o < occ_off[s + 1]; ++o) {
                const uint32_t e = occ_edge[o];
                if (edge_epoch[e] != epoch) { edge_epoch[e] = epoch; inc_edges[n_inc++] = e; }
            }
        }
        if (n_inc == 0) continue;

        // Per touched vertex, the multiset of keys over the S-incident edges it lies on.
        // Counted first so each vertex's run is contiguous, then filled.
        uint32_t n_touched = 0;
        for (uint32_t i = 0; i < n_inc; ++i) {
            const uint32_t e = inc_edges[i];
            for (uint8_t p = 0; p < ea[e]; ++p) {
                const uint32_t u = ev[eoff[e] + p];
                if (!on_touched[u]) { on_touched[u] = 1; touched[n_touched++] = u; sig_cnt[u] = 0; }
                sig_cnt[u]++;
            }
        }
        uint32_t total = 0;
        for (uint32_t i = 0; i < n_touched; ++i) {
            const uint32_t u = touched[i];
            sig_off[u] = total; total += sig_cnt[u]; sig_cnt[u] = 0;
        }
        {
            IrParArgs a;
            a.ea = ea; a.eoff = eoff; a.ev = ev;
            a.sig_buf = sig_buf; a.sig_off = sig_off; a.sig_cnt = sig_cnt;
            a.inc_edges = inc_edges; a.cell_of = pi.cell_of; a.S = S;
            par.run(IR_PAR_SIG_FILL, n_inc, a);
            a.touched = touched;
            par.run(IR_PAR_TOUCH_SORTS, n_touched, a);
        }

        // Order touched vertices by (cell id, signature); both keys are structural. Ties are
        // vertices of one cell with equal signatures, which land in the same group -- their
        // relative order is not read.
        for (uint32_t i = 0; i < n_touched; ++i) torder[i] = touched[i];
        struct TouchedCmp {
            const uint32_t* cell_of; const uint64_t* sig;
            const uint32_t* off; const uint32_t* cnt;
            HG_HD int operator()(uint32_t a, uint32_t b) const {
                if (cell_of[a] != cell_of[b]) return cell_of[a] < cell_of[b] ? -1 : 1;
                return ir_cmp_run(sig + off[a], cnt[a], sig + off[b], cnt[b]);
            }
        };
        ir_heapsort_idx(torder, n_touched, TouchedCmp{pi.cell_of, sig_buf, sig_off, sig_cnt});

        // Each cell's touched vertices are now a contiguous run of torder; split that cell.
        uint32_t i = 0;
        while (i < n_touched) {
            const uint32_t C = pi.cell_of[torder[i]];
            uint32_t j = i;
            while (j < n_touched && pi.cell_of[torder[j]] == C) ++j;

            const uint32_t adjacent = j - i;
            const uint32_t leftover = pi.clen[C] - adjacent;

            // Group boundaries, found once. Equal signatures are contiguous after the sort,
            // so one scan records where each group starts and everything below reads gstart
            // instead of re-comparing the runs.
            uint32_t groups = 0;
            for (uint32_t k = i; k < j;) {
                gstart[groups++] = k;
                uint32_t m = k + 1;
                while (m < j && ir_cmp_run(sig_buf + sig_off[torder[m]], sig_cnt[torder[m]],
                                           sig_buf + sig_off[torder[k]], sig_cnt[torder[k]]) == 0) ++m;
                k = m;
            }
            gstart[groups] = j;

            if (groups + (leftover > 0 ? 1u : 0u) > 1) {
                // Rewrite C's run of lab as [leftover..., group0..., group1..., ...]. The
                // leftover vertices are the ones not touched, recovered by scanning C's run.
                const uint32_t cs = pi.cstart[C];
                uint32_t w = cs;
                for (uint32_t k = 0; k < pi.clen[C]; ++k) {
                    const uint32_t u = pi.lab[cs + k];
                    if (!on_touched[u]) { torder[n_touched + (w - cs)] = u; ++w; }
                }
                for (uint32_t k = i; k < j; ++k) torder[n_touched + (w - cs)] = torder[k], ++w;
                for (uint32_t k = 0; k < pi.clen[C]; ++k) {
                    pi.lab[cs + k] = torder[n_touched + k];
                    pi.pos[pi.lab[cs + k]] = cs + k;
                }

                // Piece boundaries within C's run: the leftover block, then one per signature
                // group. The largest keeps C's id so vertices referencing it need no revisit.
                uint32_t best_len = leftover, best_off = 0;
                for (uint32_t g = 0, off = leftover; g < groups; ++g) {
                    const uint32_t len = gstart[g + 1] - gstart[g];
                    if (len > best_len) { best_len = len; best_off = off; }
                    off += len;
                }

                // Assign ids: the winning piece keeps C, every other piece appends.
                auto assign = [&](uint32_t off, uint32_t len) {
                    if (len == 0) return;
                    uint32_t id;
                    if (off == best_off && len == best_len) { id = C; }
                    else {
                        id = pi.ncells++;
                        worklist[id >> 6] |= (uint64_t(1) << (id & 63));
                    }
                    pi.cstart[id] = cs + off;
                    pi.clen[id] = len;
                    for (uint32_t t = 0; t < len; ++t) pi.cell_of[pi.lab[cs + off + t]] = id;
                };
                assign(0, leftover);
                for (uint32_t g = 0, off = leftover; g < groups; ++g) {
                    const uint32_t len = gstart[g + 1] - gstart[g];
                    assign(off, len);
                    off += len;
                }
            }

            for (uint32_t k = i; k < j; ++k) on_touched[torder[k]] = 0;
            i = j;
        }
    }
}

// -----------------------------------------------------------------------------------------
// Canonical form at a discrete leaf: every edge relabeled and the edge list sorted.
// Stored flat as [arity, v0, v1, ...] runs so two forms compare with one linear scan.
// -----------------------------------------------------------------------------------------
HG_HD inline void ir_build_form(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev,
    uint32_t n_edges, const uint32_t* labeling, uint32_t* form, uint32_t* order)
{
    // order[] sorts edge indices by the relabeled vertex tuple, prefix-shorter first, ties
    // broken by INPUT INDEX. Tied edges have identical canonical content and so contribute the
    // same bytes to the form either way -- the tie-break does not move the hash. It is there so
    // that order[] is a well-defined permutation, which is what makes a per-edge canonical RANK
    // meaningful: rank is the position an edge takes here.
    for (uint32_t e = 0; e < n_edges; ++e) order[e] = e;
    struct EdgeCmp {
        const uint8_t* ea; const uint32_t* eoff; const uint32_t* ev; const uint32_t* labeling;
        HG_HD int operator()(uint32_t a, uint32_t b) const {
            const uint32_t la = ea[a], lb = ea[b], m = la < lb ? la : lb;
            for (uint32_t k = 0; k < m; ++k) {
                const uint32_t x = labeling[ev[eoff[a] + k]], y = labeling[ev[eoff[b] + k]];
                if (x != y) return x < y ? -1 : 1;
            }
            if (la != lb) return la < lb ? -1 : 1;
            return a < b ? -1 : (a > b ? 1 : 0);
        }
    };
    ir_heapsort_idx(order, n_edges, EdgeCmp{ea, eoff, ev, labeling});
    uint32_t w = 0;
    for (uint32_t i = 0; i < n_edges; ++i) {
        const uint32_t e = order[i];
        form[w++] = ea[e];
        for (uint32_t k = 0; k < ea[e]; ++k) form[w++] = labeling[ev[eoff[e] + k]];
    }
}

HG_HD inline int ir_cmp_form(const uint32_t* a, const uint32_t* b, uint32_t words) {
    for (uint32_t i = 0; i < words; ++i) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

// FNV-1a over (vertex count, then each canonical edge's vertices with a separator). This is
// the canonical hash: any two isomorphic states reach the same form and so the same value.
HG_HD inline uint64_t ir_hash_form(const uint32_t* form, uint32_t n_edges, uint32_t n_verts) {
    uint64_t h = FNV_OFFSET;
    h ^= uint64_t(n_verts); h *= FNV_PRIME;
    uint32_t w = 0;
    for (uint32_t e = 0; e < n_edges; ++e) {
        const uint32_t arity = form[w++];
        for (uint32_t k = 0; k < arity; ++k) { h ^= uint64_t(form[w++]); h *= FNV_PRIME; }
        h ^= 0xDEADBEEFu; h *= FNV_PRIME;
    }
    return h;
}


// -----------------------------------------------------------------------------------------
// Individualize: replace cell `target` with the singleton {v} followed by the rest, renumbering
// every cell so ids stay 0..ncells-1 in the source's id order. Written into a fresh depth's
// buffers, which is what lets a backtrack be a return rather than an undo.
// -----------------------------------------------------------------------------------------
HG_HD inline void ir_individualize(const IrPartition& src, IrPartition& dst,
                                   uint32_t target, uint32_t v) {
    uint32_t w = 0, nc = 0;
    for (uint32_t c = 0; c < src.ncells; ++c) {
        const uint32_t cs = src.cstart[c], cl = src.clen[c];
        if (c == target) {
            const uint32_t id = nc++;
            dst.cstart[id] = w; dst.clen[id] = 1;
            dst.lab[w] = v; dst.pos[v] = w; dst.cell_of[v] = id; ++w;
            if (cl > 1) {
                const uint32_t rid = nc++;
                dst.cstart[rid] = w;
                for (uint32_t k = 0; k < cl; ++k) {
                    const uint32_t u = src.lab[cs + k];
                    if (u == v) continue;
                    dst.lab[w] = u; dst.pos[u] = w; dst.cell_of[u] = rid; ++w;
                }
                dst.clen[rid] = w - dst.cstart[rid];
            }
        } else {
            const uint32_t id = nc++;
            dst.cstart[id] = w;
            for (uint32_t k = 0; k < cl; ++k) {
                const uint32_t u = src.lab[cs + k];
                dst.lab[w] = u; dst.pos[u] = w; dst.cell_of[u] = id; ++w;
            }
            dst.clen[id] = w - dst.cstart[id];
        }
    }
    dst.ncells = nc;
    dst.n = src.n;
}

struct IrResult {
    uint64_t hash;
    uint32_t status;    // IrStatus
    uint32_t n_verts;
};

// How much searching one state cost. Optional: passed as null, nothing is counted.
//
// The search's SIZE, not its duration. Wall clock on this hardware drifts more than 10% run to
// run, which is larger than most effects worth attributing, while these are exact and depend
// only on (state, max_depth, max_generators) -- so the same state reports the same numbers on
// host and device, and the two engines' only differing input is the generator cap.
//
// What they answer: whether the per-state cost is flat or has a tail. One thread canonicalizes
// one state start to finish on both engines, so a kernel's critical path is the MAX over its
// states, and splitting a state across threads is only worth building if a minority of states
// carries the work.
struct IrWork {
    uint32_t leaves;      // discrete partitions reached -- the search's size
    uint32_t nodes;       // individualizations, one refinement each
    uint32_t max_depth;   // deepest individualization reached
    uint32_t searched;    // 0 when the initial refinement was already discrete, so no search ran
};

// -----------------------------------------------------------------------------------------
// Canonical hash of a state.
//
// Refine; if the partition is discrete the labeling is read straight off it. Otherwise search
// by individualizing the lowest non-singleton cell, taking the lexicographically smallest
// canonical form over the leaves. Leaves whose form equals the first leaf's yield an
// automorphism, and target-cell vertices in an already-explored orbit under the automorphisms
// fixing the current path are skipped -- which collapses symmetric states from O(cell)
// branches to O(orbits) without changing the form that wins.
//
// The search is an explicit loop over depth-indexed buffers rather than recursion: a device
// has no stack to spare, and the host gets a bounded frame either way.
// -----------------------------------------------------------------------------------------
// out_edge_rank, when non-null, receives each input edge's CANONICAL RANK: the position it
// takes when the state's edges are ordered by (canonical content, input index) under the
// winning labeling. Ranks are a permutation of [0, n_edges), they are a property of the
// state's isomorphism class plus the input order, and event identity is defined over them --
// which is why the device needs them and why they come from the same pass as the hash rather
// than a second canonicalization.
//
// out_edge_orbit / out_edge_class, when non-null, receive each input edge's automorphism
// ORBIT id and canonical content CLASS id. The class of an edge is the index of its canonical
// content among the distinct contents in canonical order; two edges share an orbit iff some
// automorphism of the state maps one's content to the other's. The quotient causal DP keys on
// orbits, which is why they too come from this pass. Requires every edge to have arity >= 1
// (both engines' flatteners guarantee it): the orbit scratch overlays cur_form/first_form,
// whose size covers 2 * n_edges words only then.
template <class Par = IrSerial>
HG_HD inline IrResult ir_canonical_hash(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev,
    uint32_t n_edges, uint32_t n_verts, uint32_t total_occ,
    uint32_t* scratch, uint32_t max_depth, uint32_t* out_edge_rank = nullptr,
    uint32_t max_generators = IR_HOST_GENERATORS,
    uint32_t* out_edge_orbit = nullptr, uint32_t* out_edge_class = nullptr,
    uint32_t* out_canonical_form = nullptr, uint32_t* out_vertex_label = nullptr,
    IrWork* out_work = nullptr, Par par = Par{})
{
    IrResult out{0, IR_EMPTY, 0};
    if (out_work) { out_work->leaves = 0; out_work->nodes = 0;
                    out_work->max_depth = 0; out_work->searched = 0; }
    if (n_edges == 0 || n_verts == 0) return out;

    const uint32_t n = n_verts;
    const uint32_t form_words = total_occ + n_edges;

    IrScratch sc{scratch, 0};
    uint32_t* occ_off   = sc.u32(n + 1);
    uint32_t* occ_edge  = sc.u32(total_occ);
    uint32_t* occ_pos   = sc.u32(total_occ);
    uint32_t* cursor    = sc.u32(n);
    uint32_t* inc_edges = sc.u32(n_edges);
    uint32_t* edge_epoch= sc.u32(n_edges);
    uint32_t* form_order= sc.u32(n_edges);
    uint32_t* touched   = sc.u32(n);
    uint32_t* on_touched= sc.u32(n);
    uint32_t* torder    = sc.u32(2 * n);
    uint32_t* sig_off   = sc.u32(n + 1);
    uint32_t* sig_cnt   = sc.u32(n);
    uint32_t* gstart    = sc.u32(n + 1);
    uint32_t* path      = sc.u32(n);
    uint32_t* first_path = sc.u32(n);
    uint32_t* labeling  = sc.u32(n);
    uint32_t* first_lab = sc.u32(n);
    // The WINNING leaf's labelling. first_lab is the FIRST leaf's, which the generator
    // derivation needs; the winner is not necessarily the first or the last leaf visited, so
    // the vertex mapping has to be captured where the winner is chosen.
    uint32_t* best_lab  = sc.u32(n);
    uint32_t* inv       = sc.u32(n);
    uint32_t* cur_form  = sc.u32(form_words);
    uint32_t* best_form = sc.u32(form_words);
    uint32_t* first_form= sc.u32(form_words);
    uint32_t* best_order= sc.u32(n_edges);
    const uint32_t gen_cap = ir_generator_cap(max_depth, max_generators);
    uint32_t* gens      = sc.u32(uint64_t(gen_cap) * n);
    // Per generator: how many leading positions of the current path it fixes. Set when the
    // generator is recorded and kept current as the path changes, so a return at depth d
    // consults the generators with gen_fix >= d without re-walking the path for each.
    uint32_t* gen_fix   = sc.u32(gen_cap);
    uint32_t* depths    = sc.u32(uint64_t(max_depth) * ir_depth_words(n));
    uint64_t* worklist  = sc.u64(ir_bitset_words(n));
    uint64_t* sig_buf   = sc.u64(total_occ);

    // Per-depth block: five partition arrays, the sorted target cell, its covered flags, and
    // the frame scalars.
    auto block = [&](uint32_t d) -> uint32_t* { return depths + uint64_t(d) * ir_depth_words(n); };
    // A depth's partition over its block. `fresh` is the view of a block nothing has written
    // yet -- the one the initial partition or an individualisation is about to fill -- and it
    // reads no cell of it; `view` is the view of a block whose ncells has been stored.
    auto fresh = [&](uint32_t d) -> IrPartition {
        uint32_t* b = block(d);
        IrPartition p;
        p.lab = b; p.pos = b + n; p.cell_of = b + 2 * n;
        p.cstart = b + 3 * n; p.clen = b + 4 * n;
        p.ncells = 0; p.n = n;
        return p;
    };
    auto view = [&](uint32_t d) -> IrPartition {
        IrPartition p = fresh(d);
        p.ncells = block(d)[7 * n + 0];
        return p;
    };
    auto store_ncells = [&](uint32_t d, uint32_t v) { block(d)[7 * n + 0] = v; };
    auto cell_buf  = [&](uint32_t d) -> uint32_t* { return block(d) + 5 * n; };
    auto covered   = [&](uint32_t d) -> uint32_t* { return block(d) + 6 * n; };
    auto target_of = [&](uint32_t d) -> uint32_t& { return block(d)[7 * n + 1]; };
    auto next_of   = [&](uint32_t d) -> uint32_t& { return block(d)[7 * n + 2]; };
    auto cell_n_of = [&](uint32_t d) -> uint32_t& { return block(d)[7 * n + 3]; };
    auto chosen_of = [&](uint32_t d) -> uint32_t& { return block(d)[7 * n + 4]; };

    ir_build_occurrences(ea, eoff, ev, n_edges, n, occ_off, occ_edge, occ_pos, cursor);

    IrPartition pi = fresh(0);
    ir_initial_partition(ea, occ_off, occ_edge, occ_pos, n, pi, sig_buf, torder, par);
    ir_refine(ea, eoff, ev, n_edges, occ_off, occ_edge, pi,
              worklist, inc_edges, edge_epoch, touched, on_touched, torder,
              sig_off, sig_cnt, gstart, sig_buf, 0xFFFFFFFFu, 0xFFFFFFFFu, par);
    store_ncells(0, pi.ncells);

    uint32_t n_gens = 0;
    // Set when an automorphism was found and the table was already full. Only consulted when
    // orbits were asked for; see IR_NEED_GENERATORS.
    bool gens_truncated = false;
    bool has_best = false, has_first = false;
    uint32_t first_depth = 0;
    // AUTOMORPHISM BACKJUMP (the pruning rule every individualization-refinement
    // implementation carries): a leaf whose form equals the FIRST leaf's exhibits a group
    // element mapping this branch into the first branch. Every node still pending BELOW the
    // level where the current path diverged from the first path is an image of an explored
    // node under the group, and an image's leaf form is EQUAL -- it can affect neither the
    // minimum form nor the hash -- so the search unwinds straight to the divergence level
    // instead of sibling by sibling. Measured before this on a 32-leaf star (bigstar): 15,409
    // leaves and 165,181 nodes for a group needing 31 generators, the node count growing as
    // ~n^4 in the leaf count. Sound whether or not the generator table had room: the element
    // exists either way. 0xFFFFFFFFu = none pending.
    uint32_t backjump_to = 0xFFFFFFFFu;

    // A discrete partition names every vertex: the label of a vertex is the id of its
    // singleton cell, and refinement leaves ids contiguous from zero.
    auto leaf = [&](const IrPartition& p, uint32_t depth) {
        if (out_work) ++out_work->leaves;
        for (uint32_t v = 0; v < n; ++v) labeling[v] = p.cell_of[v];
        ir_build_form(ea, eoff, ev, n_edges, labeling, cur_form, form_order);
        if (!has_best || ir_cmp_form(cur_form, best_form, form_words) < 0) {
            for (uint32_t i = 0; i < form_words; ++i) best_form[i] = cur_form[i];
            // The winning leaf's edge order IS the canonical rank assignment, and the winner
            // is not necessarily the last leaf visited, so it is captured here.
            for (uint32_t i = 0; i < n_edges; ++i) best_order[i] = form_order[i];
            for (uint32_t v = 0; v < n; ++v) best_lab[v] = labeling[v];
            has_best = true;
        }
        if (!has_first) {
            for (uint32_t i = 0; i < form_words; ++i) first_form[i] = cur_form[i];
            for (uint32_t v = 0; v < n; ++v) first_lab[v] = labeling[v];
            for (uint32_t k = 0; k < depth; ++k) first_path[k] = path[k];
            first_depth = depth;
            has_first = true;
        } else if (ir_cmp_form(cur_form, first_form, form_words) == 0) {
            uint32_t fb = 0;
            const uint32_t common = depth < first_depth ? depth : first_depth;
            while (fb < common && path[fb] == first_path[fb]) ++fb;
            if (fb < backjump_to) backjump_to = fb;
            // sigma maps this leaf's naming back to the first's: an automorphism. Recording it
            // needs a free row; without one the automorphism is REAL but unrecorded, which is
            // what gens_truncated exists to report.
            if (n_gens >= gen_cap) {
                gens_truncated = true;
            } else {
                for (uint32_t vi = 0; vi < n; ++vi) inv[labeling[vi]] = vi;
                uint32_t* g = gens + uint64_t(n_gens) * n;
                bool identity = true;
                for (uint32_t u = 0; u < n; ++u) {
                    g[u] = inv[first_lab[u]];
                    if (g[u] != u) identity = false;
                }
                if (!identity) {
                    uint32_t k = 0;
                    while (k < depth && g[path[k]] == path[k]) ++k;
                    gen_fix[n_gens++] = k;
                }
            }
        }
    };

    auto emit_ranks = [&]() {
        if (!out_edge_rank) return;
        for (uint32_t i = 0; i < n_edges; ++i) out_edge_rank[best_order[i]] = i;
    };

    // Per-edge orbit and class of the winning form. Computed in INPUT space: the labeling is a
    // bijection applied position-wise, so two edges have equal canonical content iff their
    // input tuples are equal, and a generator (a vertex permutation over input indices) acts on
    // input tuples directly -- the winning labeling, which the search does not retain, is never
    // needed. Unions run generator-by-generator in discovery order and class-ascending within
    // one, and orbit ids follow ascending union-find root class id, so the assignment is a
    // deterministic function of the state alone. Runs entirely on scratch the search no longer
    // reads (cur_form, first_form, form_order); see the arity >= 1 requirement above.
    // The canonical form itself: for each edge in CANONICAL ORDER, its arity followed by its
    // canonically-labelled vertices -- total_occ + n_edges words, which is what
    // ir_canonical_form_words reports. This is the form the hash is taken over, so a caller
    // that needs the relabelled edges reads exactly what the identity was decided on rather
    // than reconstructing it from ranks and risking a second convention.
    auto emit_form = [&]() {
        if (out_canonical_form)
            for (uint32_t i = 0; i < form_words; ++i) out_canonical_form[i] = best_form[i];
        // local vertex index -> canonical label, under the winning labelling. n_verts words.
        if (out_vertex_label)
            for (uint32_t v = 0; v < n; ++v) out_vertex_label[v] = best_lab[v];
    };

    auto emit_orbits = [&]() {
        if (!out_edge_orbit && !out_edge_class) return;

        // Class id per canonical position: equal adjacent form runs share a class.
        uint32_t* pos_class = cur_form;                 // [n_edges]
        {
            uint32_t w = 0, prev = 0, cls = 0;
            for (uint32_t i = 0; i < n_edges; ++i) {
                const uint32_t len = 1u + best_form[w];
                if (i > 0) {
                    const uint32_t plen = 1u + best_form[prev];
                    bool eq = (plen == len);
                    for (uint32_t k = 0; eq && k < len; ++k)
                        eq = best_form[prev + k] == best_form[w + k];
                    if (!eq) ++cls;
                }
                pos_class[i] = cls;
                prev = w; w += len;
            }
        }

        // Per-edge class, and each class's canonical representative edge (first position).
        uint32_t* klass_of = first_form;                // [n_edges]
        uint32_t* rep_edge = cur_form + n_edges;        // [n_classes <= n_edges]
        uint32_t n_classes = 0;
        for (uint32_t i = 0; i < n_edges; ++i) {
            const uint32_t c = pos_class[i];
            klass_of[best_order[i]] = c;
            if (c + 1 > n_classes) { n_classes = c + 1; rep_edge[c] = best_order[i]; }
        }
        if (out_edge_class)
            for (uint32_t e2 = 0; e2 < n_edges; ++e2) out_edge_class[e2] = klass_of[e2];
        if (!out_edge_orbit) return;

        uint32_t* ufc = first_form + n_edges;           // [n_classes]
        for (uint32_t c = 0; c < n_classes; ++c) ufc[c] = c;
        auto cfind = [&](uint32_t x) {
            while (ufc[x] != x) { ufc[x] = ufc[ufc[x]]; x = ufc[x]; }
            return x;
        };

        if (n_gens > 0 && n_classes > 1) {
            // Edges sorted by input tuple, for the generator-image lookup. Duplicate tuples
            // are one class already, so any member of a tie serves.
            uint32_t* tuple_order = form_order;         // [n_edges]
            for (uint32_t e2 = 0; e2 < n_edges; ++e2) tuple_order[e2] = e2;
            struct TupCmp {
                const uint8_t* ea; const uint32_t* eoff; const uint32_t* ev;
                HG_HD int operator()(uint32_t a, uint32_t b) const {
                    const uint32_t la = ea[a], lb = ea[b], m = la < lb ? la : lb;
                    for (uint32_t k = 0; k < m; ++k) {
                        const uint32_t x = ev[eoff[a] + k], y = ev[eoff[b] + k];
                        if (x != y) return x < y ? -1 : 1;
                    }
                    if (la != lb) return la < lb ? -1 : 1;
                    return 0;
                }
            };
            ir_heapsort_idx(tuple_order, n_edges, TupCmp{ea, eoff, ev});

            // Compare edge `cand`'s tuple with g applied to `src`'s tuple, lazily.
            auto cmp_img = [&](uint32_t cand, const uint32_t* g, uint32_t src) -> int {
                const uint32_t lc = ea[cand], ls = ea[src], m = lc < ls ? lc : ls;
                for (uint32_t k = 0; k < m; ++k) {
                    const uint32_t x = ev[eoff[cand] + k], y = g[ev[eoff[src] + k]];
                    if (x != y) return x < y ? -1 : 1;
                }
                if (lc != ls) return lc < ls ? -1 : 1;
                return 0;
            };
            for (uint32_t gi = 0; gi < n_gens; ++gi) {
                const uint32_t* g = gens + uint64_t(gi) * n;
                for (uint32_t c = 0; c < n_classes; ++c) {
                    const uint32_t src = rep_edge[c];
                    uint32_t lo = 0, hi = n_edges;
                    while (lo < hi) {
                        const uint32_t mid = (lo + hi) >> 1;
                        if (cmp_img(tuple_order[mid], g, src) < 0) lo = mid + 1; else hi = mid;
                    }
                    // g is an automorphism of the state, so the image tuple IS one of its
                    // edges; the guard covers a truncated generator table only.
                    if (lo < n_edges && cmp_img(tuple_order[lo], g, src) == 0) {
                        const uint32_t d2 = klass_of[tuple_order[lo]];
                        const uint32_t a = cfind(c), b = cfind(d2);
                        if (a != b) ufc[a] = b;
                    }
                }
            }
        }

        // Orbits numbered by their SMALLEST member class, which is what makes the numbering
        // canonical: class ids come from the canonical form, so the smallest class in an orbit
        // is a property of the isomorphism class, while the union-find ROOT is a property of
        // the order the generators happened to union in. Walking c upward and numbering each
        // orbit the first time it is reached numbers by smallest member in one pass.
        // pos_class is dead by now, so its span holds the root->orbit map.
        uint32_t* orbit_of = pos_class;                 // [n_classes], indexed by root class
        for (uint32_t c = 0; c < n_classes; ++c) orbit_of[c] = 0xFFFFFFFFu;
        uint32_t next = 0;
        for (uint32_t c = 0; c < n_classes; ++c) {
            const uint32_t r = cfind(c);
            if (orbit_of[r] == 0xFFFFFFFFu) orbit_of[r] = next++;
        }
        for (uint32_t e2 = 0; e2 < n_edges; ++e2)
            out_edge_orbit[e2] = orbit_of[cfind(klass_of[e2])];
    };

    if (pi.is_discrete()) {
        leaf(pi, 0);
        emit_ranks();
        emit_orbits();
        emit_form();
        out.hash = ir_hash_form(best_form, n_edges, n);
        out.status = (out_edge_orbit && gens_truncated) ? IR_NEED_GENERATORS : IR_OK;
        out.n_verts = n;
        return out;
    }

    // Past the discrete fast path above, so this state pays for a search.
    if (out_work) out_work->searched = 1;

    uint32_t d = 0;
    bool returning = false;
    for (;;) {
        if (!returning) {
            IrPartition p = view(d);
            const uint32_t t = p.is_discrete() ? p.ncells : p.first_non_singleton();
            if (t >= p.ncells) {
                if (p.is_discrete()) leaf(p, d);
                if (d == 0) break;
                if (backjump_to < d) d = backjump_to; else --d;
                backjump_to = 0xFFFFFFFFu;
                returning = true; continue;
            }
            target_of(d) = t;
            uint32_t* cell = cell_buf(d);
            const uint32_t cl = p.clen[t];
            for (uint32_t k = 0; k < cl; ++k) cell[k] = p.lab[p.cstart[t] + k];
            struct AscCmp {
                HG_HD int operator()(uint32_t a, uint32_t b) const {
                    return a < b ? -1 : (a > b ? 1 : 0);
                }
            };
            ir_heapsort_idx(cell, cl, AscCmp{});
            cell_n_of(d) = cl;
            next_of(d) = 0;
            // Clear the flags this node can consult, which are exactly the target cell's
            // vertices: every read and every write below indexes cov by a member of `cell`.
            // Clearing all n instead costs the whole vertex set per search node, and the cell
            // is what refinement has already narrowed the choice down to.
            uint32_t* cov = covered(d);
            for (uint32_t k = 0; k < cl; ++k) cov[cell[k]] = 0;
        } else {
            returning = false;
            // The branch just explored is done; mark every target-cell vertex automorphic to
            // its representative, under the generators that fix the path above this node.
            const uint32_t v = chosen_of(d);
            // Mark v's orbit under the generators fixing the path above this node: a
            // breadth-first closure from v over those generators, costing the orbit's size
            // times the generator count rather than n times it (a union-find over all n
            // vertices per return was 388 M of this function's 1.06 G instructions on
            // disc-l3a2g2r2 depth 2). A generator fixing the path maps this node's partition,
            // and so its target cell, to itself, so every vertex reached lies in the cell
            // whose flags were cleared. The closure under the generators is the orbit under
            // the group they generate, every generator having finite order.
            // The queue borrows the refinement's touched-vertex scratch (n words): no
            // refinement runs during a return, and a per-depth queue would grow every depth
            // block the device takes per state.
            uint32_t* cov = covered(d);
            uint32_t* q   = touched;
            uint32_t  qn  = 0;
            if (!cov[v]) { cov[v] = 1; q[qn++] = v; }
            for (uint32_t qi = 0; qi < qn; ++qi) {
                const uint32_t u = q[qi];
                for (uint32_t gi = 0; gi < n_gens; ++gi) {
                    if (gen_fix[gi] < d) continue;
                    const uint32_t w = gens[uint64_t(gi) * n + u];
                    if (!cov[w]) { cov[w] = 1; q[qn++] = w; }
                }
            }
            ++next_of(d);
        }

        const uint32_t* cell = cell_buf(d);
        const uint32_t* cov = covered(d);
        while (next_of(d) < cell_n_of(d) && cov[cell[next_of(d)]]) ++next_of(d);
        if (next_of(d) >= cell_n_of(d)) {
            if (d == 0) break;
            --d; returning = true; continue;
        }

        const uint32_t v = cell[next_of(d)];
        chosen_of(d) = v;
        path[d] = v;
        // The path changed at depth d: every generator fixing at least the positions above
        // now fixes exactly d of them, or d + 1 if it fixes v too.
        for (uint32_t gi = 0; gi < n_gens; ++gi)
            if (gen_fix[gi] >= d) gen_fix[gi] = d + (gens[uint64_t(gi) * n + v] == v ? 1u : 0u);
        if (d + 1 >= max_depth) { out.status = IR_NEED_DEPTH; return out; }

        IrPartition child = fresh(d + 1);
        ir_individualize(view(d), child, target_of(d), v);
        // The singleton takes the target's id and the remainder the next one (cells are
        // renumbered in source order), so those two seed the refinement.
        ir_refine(ea, eoff, ev, n_edges, occ_off, occ_edge, child,
                  worklist, inc_edges, edge_epoch, touched, on_touched, torder,
                  sig_off, sig_cnt, gstart, sig_buf,
                  target_of(d), cell_n_of(d) > 1 ? target_of(d) + 1 : 0xFFFFFFFFu, par);
        store_ncells(d + 1, child.ncells);
        ++d;
        if (out_work) {
            ++out_work->nodes;
            if (d > out_work->max_depth) out_work->max_depth = d;
        }
    }

    if (!has_best) { out.status = IR_EMPTY; out.hash = fnv_hash(FNV_OFFSET, 0); return out; }
    emit_ranks();
    emit_orbits();
    emit_form();
    out.hash = ir_hash_form(best_form, n_edges, n);
    out.status = (out_edge_orbit && gens_truncated) ? IR_NEED_GENERATORS : IR_OK;
    out.n_verts = n;
    return out;
}

}  // namespace common
}  // namespace HG_NAMESPACE
