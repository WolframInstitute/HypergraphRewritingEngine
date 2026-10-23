#pragma once
#include "hgcommon/namespace.hpp"
// WHAT ONE CLASS ADDS TO ITS STEP'S "StepStatistics" POOL, one body for host and device.
//
// A step's pool is a multiset of (slot, key, weight) entries and a few integer totals. A slot is
// one pooled quantity (an invariant, the ball-growth dimension at a radius, the mean log
// geodesic count at a distance, the high or low double of a power sum of a per-vertex
// distribution, an arity, a degree, a class's multiplicity); the key is the value,
// and the weight is how many raw states (or classes, under "StepStatisticsWeighting" ->
// "Classes") carry it. Equal (slot, key) entries add their weights, saturating at 2^63 - 1, and
// the totals add or take extremes. Every operation commutes, so the pool is the same whichever
// thread or warp added which class in which order, and the step's record
// (stats::step_record) is a function of the pool alone: it reads each slot's entries in key
// order.
//
// sp_add_class enumerates a class's entries into a sink. The host's sink appends them to a job's
// list (paclet_source/paclet_support.cpp); the device's adds them to a hash table with atomics
// (gpu/src/step_pool.cu).
#include "hgcommon/core.hpp"
#include "hgcommon/state_invariants_core.hpp"
#include "hgcommon/quotient_multiplicity_core.hpp"

#include <cstdint>
#include <cstring>

namespace HG_NAMESPACE {
namespace common {

// The slots. Slots below SP_MULTIPLICITY and from SP_DIST_SUM on hold doubles as sp_key_of
// keys; SP_ARITY, SP_DEGREE and SP_MULTIPLICITY hold integers as themselves.
enum : uint32_t {
    SP_INVARIANTS = 0,           // the 49 "Invariants" in reply order, slots 0..48
    SP_NUM_INVARIANTS = 49,
    SP_ARITY = SP_NUM_INVARIANTS,        // a hyperedge's arity
    SP_DEGREE = SP_NUM_INVARIANTS + 1,   // a vertex's slot degree
    SP_MULTIPLICITY = SP_NUM_INVARIANTS + 2,  // a class's raw state count, weight 1 per class
    SP_DIST_SUM = SP_NUM_INVARIANTS + 3, // distribution k's sum of x^(j+1): slot SP_DIST_SUM +
                                         // 8 k + 2 j for its high double, + 1 for its low double
    SP_BALL = SP_DIST_SUM + 8 * SG_DISTS,  // the ball-growth dimension at radius r: slot
                                           // SP_BALL + 2 (r - 1); the mean log geodesic count
                                           // at distance d: slot SP_BALL + 2 (d - 1) + 1
};

HG_HD inline bool sp_integer_slot(uint32_t slot) {
    return slot == SP_ARITY || slot == SP_DEGREE || slot == SP_MULTIPLICITY;
}

// A double as a key whose unsigned order is the numeric order; -0.0 is 0.0.
HG_HD inline uint64_t sp_key_of(double v) {
    if (v == 0.0) v = 0.0;
    uint64_t b;
    memcpy(&b, &v, sizeof b);
    return (b >> 63) ? ~b : (b | (uint64_t{1} << 63));
}

HG_HD inline double sp_value_of(uint64_t key) {
    const uint64_t b = (key >> 63) ? (key & ~(uint64_t{1} << 63)) : ~key;
    double v;
    memcpy(&v, &b, sizeof v);
    return v;
}

// One entry of a step's pool.
struct SpEntry {
    uint32_t slot;
    uint64_t key;
    uint64_t weight;
};

// The per-vertex distributions' totals: counts and bins add, saturating; the extremes are taken
// over the classes with a vertex in the distribution.
struct SpDistTotals {
    uint64_t n = 0;
    uint64_t bins[SG_DIST_BINS] = {};
    uint64_t min_key = ~uint64_t{0};   // sp_key_of(min), ~0 when no class has a vertex
    uint64_t max_key = 0;
};

// Enumerates the entries of class record `r` carried `w` times (w > 0) and its multiplicity
// `raw` (its raw state count, 0 to add none) into `sink`:
//   sink.entry(slot, key, weight)   one weighted value
//   sink.dist(k, n, extremes_min_key, extremes_max_key)
//   sink.bin(k, b, n)               bin b of distribution k gains n
template <class Sink>
HG_HD inline void sp_add_class(const StateInvariantRecord& r, uint64_t w, uint64_t raw,
                               Sink& sink) {
    if (raw) sink.entry(SP_MULTIPLICITY, raw, 1);
    if (!w) return;
    const StateInvariantValues& s = r.v;
    const SgGeometry& g = r.g;
    auto put = [&](uint32_t slot, double v) { sink.entry(slot, sp_key_of(v), w); };
    // In reply order (stats::kInvariantNames).
    put(0, static_cast<double>(s.vertex_count));
    put(1, static_cast<double>(s.edge_count));
    put(2, static_cast<double>(s.max_degree));
    put(3, s.mean_degree);
    put(4, static_cast<double>(s.two_section_edge_count));
    put(5, static_cast<double>(s.components));
    put(6, static_cast<double>(s.cycle_rank));
    put(7, static_cast<double>(s.incidence_cycle_rank));
    put(8, static_cast<double>(s.incidence_diameter));
    put(9, s.incidence_mean_distance);
    put(10, s.largest_component_fraction);
    if (g.defined & SG_RADIUS) {
        put(11, static_cast<double>(g.radius));
        put(12, g.mean_eccentricity);
    }
    if (g.defined & SG_HAUSDORFF) put(13, g.hausdorff_dimension);
    if (g.defined & SG_RICCI) put(14, g.ricci_scalar);
    if (g.defined & SG_OLLIVIER) put(15, g.ollivier_ricci);
    if (g.defined & SG_DEGREE_ENTROPY) {
        put(16, g.degree_entropy);
        put(17, g.local_entropy);
    }
    if (g.defined & SG_MUTUAL_INFORMATION) put(18, g.mutual_information);
    if (g.defined & SG_FISHER) put(19, g.fisher_information);
    if (g.defined & SG_LARGEST_DIMENSION) {
        put(20, g.largest_dimension);
        put(21, g.local_dimension_max);
        put(22, g.local_dimension_sd);
    }
    if (g.defined & SG_OLLIVIER_MORAN) put(23, g.ollivier_moran_i);
    if (g.defined & SG_OLLIVIER_DEGREE) put(24, g.ollivier_degree_correlation);
    if (g.defined & SG_VOLUME_GROWTH)
        for (uint32_t k = 0; k < SG_GROWTH_VALUES; ++k) put(25 + k, g.growth[k]);
    if (g.defined & SG_GEODESIC) {
        put(31, g.geodetic_fraction);
        put(32, g.segment_entropy);
    }
    if (g.defined & SG_NONLAZY_OLLIVIER) put(33, g.nonlazy_ollivier);
    if (g.defined & SG_EULER) put(34, g.euler_characteristic);
    if (g.defined & SG_RIM) put(35, g.rim_fraction);
    if (g.defined & SG_CYCLE_CURVATURE) put(36, g.cycle_curvature_total);
    if (g.defined & SG_PARTICLE) {
        put(37, g.clustering);
        put(38, g.euler_excess);
        put(39, g.charge);
        put(40, g.charge_max);
    }
    if (g.defined & SG_DIMENSION_GRADIENT) put(41, g.dimension_gradient);
    if (g.defined & SG_TOPOLOGY) {
        for (uint32_t k = 0; k < 3; ++k) put(42 + k, g.clique_betti[k]);
        put(45, g.rips_betti1);
    }
    if (g.defined & SG_METRIC_TENSOR) put(46, g.metric_tensor_gap);
    if (g.defined & SG_HOLONOMY) put(47, g.holonomy_triangle);
    if (g.defined & SG_HOLONOMY_SQUARE) put(48, g.holonomy_square);

    const uint32_t* ar = r.arity_runs();
    for (uint32_t i = 0; i < r.num_arity_runs; ++i)
        sink.entry(SP_ARITY, ar[2 * i], qm_sat_mul(w, ar[2 * i + 1]));
    const uint32_t* dr = r.degree_runs();
    for (uint32_t i = 0; i < r.num_degree_runs; ++i)
        sink.entry(SP_DEGREE, dr[2 * i], qm_sat_mul(w, dr[2 * i + 1]));

    for (uint32_t k = 0; k < r.num_distributions && k < SG_DISTS; ++k) {
        const SgDistribution& d = r.distributions()[k];
        if (d.count == 0) continue;
        sink.dist(k, qm_sat_mul(d.count, w), sp_key_of(d.min), sp_key_of(d.max));
        for (uint32_t j = 0; j < 4; ++j) {
            put(SP_DIST_SUM + 8 * k + 2 * j, d.sum[j]);
            put(SP_DIST_SUM + 8 * k + 2 * j + 1, d.sum_lo[j]);
        }
        for (uint32_t b = 0; b < SG_DIST_BINS; ++b)
            if (d.bins[b]) sink.bin(k, b, qm_sat_mul(d.bins[b], w));
    }
    for (uint32_t b = 0; b < r.num_ball; ++b) put(SP_BALL + 2 * b, r.ball()[b]);
    for (uint32_t d = 0; d < r.num_geodesic; ++d) put(SP_BALL + 2 * d + 1, r.geodesic()[d]);
}

}  // namespace common
}  // namespace HG_NAMESPACE
