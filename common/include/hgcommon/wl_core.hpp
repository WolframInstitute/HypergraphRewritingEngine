#pragma once
// Shared CPU/GPU Weisfeiler-Leman canonical-hash core.
//
// ONE implementation, so the host engine and the CUDA port produce bit-identical
// WL hashes by construction (no two-copy drift). It runs single-threaded over a
// state that the caller has pre-flattened to LOCAL vertex indices [0, n_verts);
// GPU parallelism is across states (one thread per state), CPU is one call per
// state. The final fold is a commutative multiset sum, so the caller's choice of
// local-index ordering does not affect the result.
//
// All scratch is caller-provided (arena on the host, local/shared memory on the
// GPU). HG_HD makes every function callable from host and device.
#include <cstdint>
#include "hgcommon/core.hpp"  // HG_HD, FNV_OFFSET, fnv_hash, mix64

namespace hgcommon {

// Max WL colour-refinement rounds (shared so CPU and GPU stop identically).
constexpr uint32_t WL_MAX_REFINE_ITERS = 100;

// Insertion sort — small n, no std:: on device.
HG_HD inline void wl_isort(uint64_t* a, uint32_t n) {
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t key = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > key) { a[j] = a[j - 1]; --j; }
        a[j] = key;
    }
}

// Compute the WL canonical hash of a state given as local-index edges:
//   ea[e]           = arity of edge e            (e in [0, n_edges))
//   ev[eoff[e] + p] = local vertex index of edge e position p   (p in [0, ea[e]))
// Scratch (caller-sized): cur/nxt [n_verts]; occ_off [n_verts+1];
//   occ_edge/occ_pos [total_occ = Σ ea]; nbr [nbr_cap ≥ max vertex degree * (MAX_ARITY-1)];
//   dscr [n_verts]. max_iters bounds refinement. out_colours (optional, [n_verts])
//   receives the final per-vertex colours (for edge-correspondence caches).
HG_HD inline uint64_t wl_canonical_hash(
    const uint8_t* ea, const uint32_t* eoff, const uint32_t* ev,
    uint32_t n_edges, uint32_t n_verts, uint32_t max_iters,
    uint64_t* cur, uint64_t* nxt,
    uint32_t* occ_off, uint32_t* occ_edge, uint8_t* occ_pos,
    uint64_t* nbr, uint32_t nbr_cap,
    uint64_t* dscr,
    uint64_t* out_colours)
{
    if (n_verts == 0 || n_edges == 0) return 0;

    // --- Occurrence CSR: per vertex, its (edge, pos) occurrences (counting sort). ---
    for (uint32_t v = 0; v <= n_verts; ++v) occ_off[v] = 0;
    for (uint32_t e = 0; e < n_edges; ++e)
        for (uint8_t p = 0; p < ea[e]; ++p) occ_off[ev[eoff[e] + p] + 1u]++;
    for (uint32_t v = 0; v < n_verts; ++v) occ_off[v + 1] += occ_off[v];
    for (uint32_t v = 0; v < n_verts; ++v) dscr[v] = occ_off[v];  // write cursor
    for (uint32_t e = 0; e < n_edges; ++e)
        for (uint8_t p = 0; p < ea[e]; ++p) {
            uint32_t w = (uint32_t)dscr[ev[eoff[e] + p]]++;
            occ_edge[w] = e;
            occ_pos[w]  = p;
        }

    // --- Initial colours: FNV(offset, degree, SORTED (arity,pos) occurrences). ---
    for (uint32_t v = 0; v < n_verts; ++v) {
        uint64_t h = FNV_OFFSET;
        uint32_t deg = occ_off[v + 1] - occ_off[v];
        h = fnv_hash(h, deg);
        // Sort this vertex's (arity,pos) pairs (canonical form) via packed key.
        uint32_t cnt = 0;
        for (uint32_t j = occ_off[v]; j < occ_off[v + 1] && cnt < nbr_cap; ++j)
            nbr[cnt++] = ((uint64_t)ea[occ_edge[j]] << 8) | (uint64_t)occ_pos[j];
        wl_isort(nbr, cnt);
        for (uint32_t i = 0; i < cnt; ++i) {
            h = fnv_hash(h, nbr[i] >> 8);      // arity
            h = fnv_hash(h, nbr[i] & 0xFFu);   // pos
        }
        cur[v] = h;
    }

    // Distinct-colour count (partition-stabilization stop).
    // (uses dscr as sort scratch — CSR write-cursor use above is finished)
    #define HG_WL_NDISTINCT(dst) do { \
        for (uint32_t v = 0; v < n_verts; ++v) dscr[v] = cur[v]; \
        wl_isort(dscr, n_verts); \
        (dst) = 0; \
        for (uint32_t v = 0; v < n_verts; ++v) if (v == 0 || dscr[v] != dscr[v-1]) ++(dst); \
    } while (0)
    uint32_t prev_d; HG_WL_NDISTINCT(prev_d);

    // --- Refinement: neighbour multiset of fnv(colour, position), sorted then folded. ---
    for (uint32_t it = 0; it < max_iters; ++it) {
        for (uint32_t v = 0; v < n_verts; ++v) {
            uint64_t h = cur[v];
            uint32_t nn = 0;
            for (uint32_t j = occ_off[v]; j < occ_off[v + 1]; ++j) {
                uint32_t e = occ_edge[j];
                uint8_t mypos = occ_pos[j];
                for (uint8_t k = 0; k < ea[e]; ++k) {
                    if (k == mypos) continue;
                    if (nn < nbr_cap) nbr[nn++] = fnv_hash(cur[ev[eoff[e] + k]], k);
                }
            }
            wl_isort(nbr, nn);
            for (uint32_t i = 0; i < nn; ++i) h = fnv_hash(h, nbr[i]);
            nxt[v] = h;
        }
        for (uint32_t v = 0; v < n_verts; ++v) cur[v] = nxt[v];
        uint32_t d; HG_WL_NDISTINCT(d);
        if (d == prev_d) break;   // partition stabilised
        prev_d = d;
    }
    #undef HG_WL_NDISTINCT

    if (out_colours)
        for (uint32_t v = 0; v < n_verts; ++v) out_colours[v] = cur[v];

    // --- Final commutative fold: FNV(n_verts, n_edges, Σmix64(vcol), Σmix64(ehash)). ---
    uint64_t vsum = 0;
    for (uint32_t v = 0; v < n_verts; ++v) vsum += splitmix64(cur[v]);
    uint64_t esum = 0;
    for (uint32_t e = 0; e < n_edges; ++e) {
        uint64_t eh = fnv_hash(FNV_OFFSET, ea[e]);
        for (uint8_t p = 0; p < ea[e]; ++p) eh = fnv_hash(eh, cur[ev[eoff[e] + p]]);
        esum += splitmix64(eh);
    }
    uint64_t hash = FNV_OFFSET;
    hash = fnv_hash(hash, n_verts);
    hash = fnv_hash(hash, n_edges);
    hash = fnv_hash(hash, vsum);
    hash = fnv_hash(hash, esum);
    return hash;
}

}  // namespace hgcommon
