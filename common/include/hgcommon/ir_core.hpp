#pragma once
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

namespace hgcommon {

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

// Automorphisms retained for orbit pruning. Generators only SKIP branches that are automorphic
// to one already explored, and automorphic branches reach the same canonical form, so a cap
// costs search time on a highly symmetric state and never changes the result.
constexpr uint32_t IR_MAX_GENERATORS = 64;

// Generators and depth blocks are the bulk of the scratch and are touched only when the
// initial refinement leaves a non-discrete partition. max_depth == 1 means "root only": it
// sizes for the common state, which is discrete after refinement, and reports IR_NEED_DEPTH
// for the rest so the caller can retry at a depth that searches.
HG_HD inline uint32_t ir_generator_cap(uint32_t max_depth) {
    return max_depth > 1 ? IR_MAX_GENERATORS : 0u;
}

enum IrStatus : uint32_t { IR_OK = 0, IR_EMPTY = 1, IR_NEED_DEPTH = 2 };

// Per-depth partition snapshot: lab, pos, cell_of, cstart, clen, plus the sorted target cell
// and its covered flags. A search node writes the next depth from the current one and refines
// it in place, so backtracking is a return -- no undo trail, and nothing is ever allocated.
HG_HD inline uint64_t ir_depth_words(uint32_t n_verts) { return 7ull * n_verts + 8ull; }

// Total uint32 words of scratch for a given problem size.
HG_HD inline uint64_t ir_scratch_words(uint32_t n_verts, uint32_t n_edges,
                                       uint32_t total_occ, uint32_t max_depth) {
    const uint64_t n = n_verts, e = n_edges, occ = total_occ, d = max_depth;
    return
        (n + 1) + occ + occ + n             // occ_off, occ_edge, occ_pos, cursor
      + 2 * ir_bitset_words(n_verts)        // worklist, as uint64
      + e + e + e                           // inc_edges, edge_epoch, form_order
      + n + n + 2 * n                       // touched, on_touched, torder (2n: split staging)
      + n + n + (n + 1) + 2 * occ           // sig_off, sig_cnt, gstart, sig_buf as uint64
      + n + n + n + n + n                   // path, uf, labeling, first_labeling, inv
      + 3 * (occ + e)                       // cur_form, best_form, first_form
      + d * ir_depth_words(n_verts)         // per-depth partition + cell + covered
      + uint64_t(ir_generator_cap(max_depth)) * n   // generators, row-major
      + 64;                                 // alignment slack for the uint64 views
}

// Sub-allocator over the caller's span. No bounds checking on the fast path -- the caller
// sized the span with ir_scratch_words for the same (n, e, occ, depth).
struct IrScratch {
    uint32_t* base;
    uint64_t  used = 0;
    HG_HD uint32_t* u32(uint64_t count) { uint32_t* p = base + used; used += count; return p; }
    HG_HD uint64_t* u64(uint64_t count) {
        used = (used + 1) & ~uint64_t(1);                 // 8-byte align
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

// Heapsort over an index array. O(n log n) with no recursion, no allocation and no std::,
// so one routine serves host and device. Not stable; every comparator below either is a total
// order or has ties whose relative order does not reach the output.
template <class Cmp>
HG_HD inline void ir_heapsort_idx(uint32_t* a, uint32_t n, Cmp cmp) {
    if (n < 2) return;
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

HG_HD inline void ir_isort_u64(uint64_t* a, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t key = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > key) { a[j] = a[j - 1]; --j; }
        a[j] = key;
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
// Initial partition: group vertices by degree signature -- the sorted multiset of
// (arity, position) over their occurrences -- ordering groups by signature and, within a
// signature, by vertex index. Both keys are structural, so the partition is equivariant.
// -----------------------------------------------------------------------------------------
HG_HD inline void ir_initial_partition(
    const uint8_t* ea, const uint32_t* occ_off, const uint32_t* occ_edge, const uint32_t* occ_pos,
    uint32_t n_verts, IrPartition& pi,
    uint64_t* sig_buf, uint32_t* order)
{
    // sig_buf holds every vertex's signature contiguously at occ_off[v]; (arity, position)
    // packs into one uint64 so the multiset sorts with a scalar comparison.
    for (uint32_t v = 0; v < n_verts; ++v) {
        const uint32_t s = occ_off[v], len = occ_off[v + 1] - s;
        for (uint32_t k = 0; k < len; ++k) {
            const uint32_t e = occ_edge[s + k];
            sig_buf[s + k] = (uint64_t(ea[e]) << 32) | uint64_t(occ_pos[s + k]);
        }
        ir_isort_u64(sig_buf + s, len);
        order[v] = v;
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
HG_HD inline void ir_refine(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev, uint32_t n_edges,
    const uint32_t* occ_off, const uint32_t* occ_edge,
    IrPartition& pi,
    uint64_t* worklist, uint32_t* inc_edges, uint32_t* edge_epoch,
    uint32_t* touched, uint32_t* on_touched, uint32_t* torder,   // torder holds 2n: order, then split staging
    uint32_t* sig_off, uint32_t* sig_cnt, uint32_t* gstart, uint64_t* sig_buf)
{
    const uint32_t n = pi.n;
    const uint32_t wl_words = ir_bitset_words(n);
    for (uint32_t w = 0; w < wl_words; ++w) worklist[w] = 0;
    for (uint32_t c = 0; c < pi.ncells; ++c) worklist[c >> 6] |= (uint64_t(1) << (c & 63));
    for (uint32_t e = 0; e < n_edges; ++e) edge_epoch[e] = 0;
    for (uint32_t v = 0; v < n; ++v) on_touched[v] = 0;
    uint32_t epoch = 0;

    for (;;) {
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
        for (uint32_t i = 0; i < n_inc; ++i) {
            const uint32_t e = inc_edges[i];
            const uint32_t arity = ea[e];
            uint64_t spos = 0;
            for (uint32_t p = 0; p < arity && p < IR_SPOS_BITS; ++p)
                if (pi.cell_of[ev[eoff[e] + p]] == S) spos |= (uint64_t(1) << p);
            for (uint32_t p = 0; p < arity; ++p) {
                const uint32_t u = ev[eoff[e] + p];
                sig_buf[sig_off[u] + sig_cnt[u]++] =
                    (uint64_t(arity & 0xFF) << 56) | (uint64_t(p & 0xFF) << 48) | spos;
            }
        }
        for (uint32_t i = 0; i < n_touched; ++i)
            ir_isort_u64(sig_buf + sig_off[touched[i]], sig_cnt[touched[i]]);

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
    // order[] sorts edge indices by the relabeled vertex tuple, prefix-shorter first. Ties are
    // edges with identical canonical content, so their relative order does not reach the form.
    for (uint32_t e = 0; e < n_edges; ++e) order[e] = e;
    struct EdgeCmp {
        const uint8_t* ea; const uint32_t* eoff; const uint32_t* ev; const uint32_t* labeling;
        HG_HD int operator()(uint32_t a, uint32_t b) const {
            const uint32_t la = ea[a], lb = ea[b], m = la < lb ? la : lb;
            for (uint32_t k = 0; k < m; ++k) {
                const uint32_t x = labeling[ev[eoff[a] + k]], y = labeling[ev[eoff[b] + k]];
                if (x != y) return x < y ? -1 : 1;
            }
            if (la == lb) return 0;
            return la < lb ? -1 : 1;
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
HG_HD inline IrResult ir_canonical_hash(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev,
    uint32_t n_edges, uint32_t n_verts, uint32_t total_occ,
    uint32_t* scratch, uint32_t max_depth)
{
    IrResult out{0, IR_EMPTY, 0};
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
    uint32_t* uf        = sc.u32(n);
    uint32_t* labeling  = sc.u32(n);
    uint32_t* first_lab = sc.u32(n);
    uint32_t* inv       = sc.u32(n);
    uint32_t* cur_form  = sc.u32(form_words);
    uint32_t* best_form = sc.u32(form_words);
    uint32_t* first_form= sc.u32(form_words);
    const uint32_t gen_cap = ir_generator_cap(max_depth);
    uint32_t* gens      = sc.u32(uint64_t(gen_cap) * n);
    uint32_t* depths    = sc.u32(uint64_t(max_depth) * ir_depth_words(n));
    uint64_t* worklist  = sc.u64(ir_bitset_words(n));
    uint64_t* sig_buf   = sc.u64(total_occ);

    // Per-depth block: five partition arrays, the sorted target cell, its covered flags, and
    // the frame scalars.
    auto block = [&](uint32_t d) -> uint32_t* { return depths + uint64_t(d) * ir_depth_words(n); };
    auto view = [&](uint32_t d) -> IrPartition {
        uint32_t* b = block(d);
        IrPartition p;
        p.lab = b; p.pos = b + n; p.cell_of = b + 2 * n;
        p.cstart = b + 3 * n; p.clen = b + 4 * n;
        p.ncells = b[7 * n + 0]; p.n = n;
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

    IrPartition pi = view(0);
    ir_initial_partition(ea, occ_off, occ_edge, occ_pos, n, pi, sig_buf, torder);
    ir_refine(ea, eoff, ev, n_edges, occ_off, occ_edge, pi,
              worklist, inc_edges, edge_epoch, touched, on_touched, torder,
              sig_off, sig_cnt, gstart, sig_buf);
    store_ncells(0, pi.ncells);

    uint32_t n_gens = 0;
    bool has_best = false, has_first = false;

    // A discrete partition names every vertex: the label of a vertex is the id of its
    // singleton cell, and refinement leaves ids contiguous from zero.
    auto leaf = [&](const IrPartition& p) {
        for (uint32_t v = 0; v < n; ++v) labeling[v] = p.cell_of[v];
        ir_build_form(ea, eoff, ev, n_edges, labeling, cur_form, form_order);
        if (!has_best || ir_cmp_form(cur_form, best_form, form_words) < 0) {
            for (uint32_t i = 0; i < form_words; ++i) best_form[i] = cur_form[i];
            has_best = true;
        }
        if (!has_first) {
            for (uint32_t i = 0; i < form_words; ++i) first_form[i] = cur_form[i];
            for (uint32_t v = 0; v < n; ++v) first_lab[v] = labeling[v];
            has_first = true;
        } else if (ir_cmp_form(cur_form, first_form, form_words) == 0
                   && n_gens < gen_cap) {
            // sigma maps this leaf's naming back to the first's: an automorphism.
            for (uint32_t vi = 0; vi < n; ++vi) inv[labeling[vi]] = vi;
            uint32_t* g = gens + uint64_t(n_gens) * n;
            bool identity = true;
            for (uint32_t u = 0; u < n; ++u) {
                g[u] = inv[first_lab[u]];
                if (g[u] != u) identity = false;
            }
            if (!identity) ++n_gens;
        }
    };

    if (pi.is_discrete()) {
        leaf(pi);
        out.hash = ir_hash_form(best_form, n_edges, n);
        out.status = IR_OK;
        out.n_verts = n;
        return out;
    }

    uint32_t d = 0;
    bool returning = false;
    for (;;) {
        if (!returning) {
            IrPartition p = view(d);
            const uint32_t t = p.is_discrete() ? p.ncells : p.first_non_singleton();
            if (t >= p.ncells) {
                if (p.is_discrete()) leaf(p);
                if (d == 0) break;
                --d; returning = true; continue;
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
            uint32_t* cov = covered(d);
            for (uint32_t k = 0; k < n; ++k) cov[k] = 0;
        } else {
            returning = false;
            // The branch just explored is done; mark every target-cell vertex automorphic to
            // its representative, under the generators that fix the path above this node.
            const uint32_t v = chosen_of(d);
            for (uint32_t i = 0; i < n; ++i) uf[i] = i;
            auto find = [&](uint32_t x) {
                while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
                return x;
            };
            for (uint32_t gi = 0; gi < n_gens; ++gi) {
                const uint32_t* g = gens + uint64_t(gi) * n;
                bool fixes_path = true;
                for (uint32_t k = 0; k < d; ++k) if (g[path[k]] != path[k]) { fixes_path = false; break; }
                if (!fixes_path) continue;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t a = find(i), b = find(g[i]);
                    if (a != b) uf[a] = b;
                }
            }
            const uint32_t rv = find(v);
            uint32_t* cov = covered(d);
            const uint32_t* cell = cell_buf(d);
            for (uint32_t k = 0; k < cell_n_of(d); ++k)
                if (find(cell[k]) == rv) cov[cell[k]] = 1;
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
        if (d + 1 >= max_depth) { out.status = IR_NEED_DEPTH; return out; }

        IrPartition child = view(d + 1);
        ir_individualize(view(d), child, target_of(d), v);
        ir_refine(ea, eoff, ev, n_edges, occ_off, occ_edge, child,
                  worklist, inc_edges, edge_epoch, touched, on_touched, torder,
                  sig_off, sig_cnt, gstart, sig_buf);
        store_ncells(d + 1, child.ncells);
        ++d;
    }

    if (!has_best) { out.status = IR_EMPTY; out.hash = fnv_hash(FNV_OFFSET, 0); return out; }
    out.hash = ir_hash_form(best_form, n_edges, n);
    out.status = IR_OK;
    out.n_verts = n;
    return out;
}

}  // namespace hgcommon
