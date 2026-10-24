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
    SP_INVARIANTS = 0,           // the 62 "Invariants" in reply order, slots 0..61
    SP_NUM_INVARIANTS = 62,
    SP_ARITY = SP_NUM_INVARIANTS,        // a hyperedge's arity
    SP_DEGREE = SP_NUM_INVARIANTS + 1,   // a vertex's slot degree
    SP_MULTIPLICITY = SP_NUM_INVARIANTS + 2,  // a class's raw state count, weight 1 per class
    SP_DIST_SUM = SP_NUM_INVARIANTS + 3, // distribution k's sum of x^(j+1): slot SP_DIST_SUM +
                                         // 8 k + 2 j for its high double, + 1 for its low double
    SP_BALL = SP_DIST_SUM + 8 * SG_DISTS,  // the ball-growth dimension at radius r: slot
                                           // SP_BALL + 3 (r - 1); the mean log geodesic count
                                           // at distance d: slot SP_BALL + 3 (d - 1) + 1; the
                                           // orbital velocity at radius r: SP_BALL + 3 (r - 1) + 2
    SP_SERIES = 3,                         // the per-radius series interleaved from SP_BALL
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

// The SgGeometry::defined bit of "Invariants" slot `slot`; 0 for the slots below 11, which every
// record holds.
HG_HD inline uint32_t sp_invariant_bit(uint32_t slot) {
    if (slot < 11) return 0;
    if (slot <= 12) return SG_RADIUS;
    if (slot == 13) return SG_HAUSDORFF;
    if (slot == 14) return SG_RICCI;
    if (slot == 15) return SG_OLLIVIER;
    if (slot <= 17) return SG_DEGREE_ENTROPY;
    if (slot == 18) return SG_MUTUAL_INFORMATION;
    if (slot == 19) return SG_FISHER;
    if (slot <= 22) return SG_LARGEST_DIMENSION;
    if (slot == 23) return SG_OLLIVIER_MORAN;
    if (slot == 24) return SG_OLLIVIER_DEGREE;
    if (slot <= 30) return SG_VOLUME_GROWTH;
    if (slot <= 32) return SG_GEODESIC;
    if (slot == 33) return SG_NONLAZY_OLLIVIER;
    if (slot == 34) return SG_EULER;
    if (slot == 35) return SG_RIM;
    if (slot == 36) return SG_CYCLE_CURVATURE;
    if (slot <= 40) return SG_PARTICLE;
    if (slot == 41) return SG_DIMENSION_GRADIENT;
    if (slot <= 45) return SG_TOPOLOGY;
    if (slot == 46) return SG_METRIC_TENSOR;
    if (slot == 47) return SG_HOLONOMY;
    if (slot == 48) return SG_HOLONOMY_SQUARE;
    if (slot <= 51) return SG_BRANCH_ALIGNMENT;
    if (slot <= 55) return SG_GEODESIC_BUNDLE;
    if (slot <= 58) return SG_LENSING;
    return SG_ROTATION;
}

// The `want` bit of state_record that computes "Invariants" slot `slot`: its defined bit, except
// HolonomySquare (slot 48), which SG_HOLONOMY computes.
HG_HD inline uint32_t sp_invariant_want(uint32_t slot) {
    return slot == 48 ? SG_HOLONOMY : sp_invariant_bit(slot);
}

// "Invariants" slot `slot` (0..SP_NUM_INVARIANTS) of a state with values `s` and geometry `g`, in
// reply order (stats::kInvariantNames), into `out`. False when the state has no value there: its
// geometry bit (sp_invariant_bit) is not in g.defined.
HG_HD inline bool sp_invariant(const StateInvariantValues& s, const SgGeometry& g, uint32_t slot,
                               double& out) {
    const uint32_t bit = sp_invariant_bit(slot);
    if (bit != 0 && !(g.defined & bit)) return false;
    switch (slot) {
        case 0: out = static_cast<double>(s.vertex_count); break;
        case 1: out = static_cast<double>(s.edge_count); break;
        case 2: out = static_cast<double>(s.max_degree); break;
        case 3: out = s.mean_degree; break;
        case 4: out = static_cast<double>(s.two_section_edge_count); break;
        case 5: out = static_cast<double>(s.components); break;
        case 6: out = static_cast<double>(s.cycle_rank); break;
        case 7: out = static_cast<double>(s.incidence_cycle_rank); break;
        case 8: out = static_cast<double>(s.incidence_diameter); break;
        case 9: out = s.incidence_mean_distance; break;
        case 10: out = s.largest_component_fraction; break;
        case 11: out = static_cast<double>(g.radius); break;
        case 12: out = g.mean_eccentricity; break;
        case 13: out = g.hausdorff_dimension; break;
        case 14: out = g.ricci_scalar; break;
        case 15: out = g.ollivier_ricci; break;
        case 16: out = g.degree_entropy; break;
        case 17: out = g.local_entropy; break;
        case 18: out = g.mutual_information; break;
        case 19: out = g.fisher_information; break;
        case 20: out = g.largest_dimension; break;
        case 21: out = g.local_dimension_max; break;
        case 22: out = g.local_dimension_sd; break;
        case 23: out = g.ollivier_moran_i; break;
        case 24: out = g.ollivier_degree_correlation; break;
        case 25: case 26: case 27: case 28: case 29: case 30: out = g.growth[slot - 25]; break;
        case 31: out = g.geodetic_fraction; break;
        case 32: out = g.segment_entropy; break;
        case 33: out = g.nonlazy_ollivier; break;
        case 34: out = g.euler_characteristic; break;
        case 35: out = g.rim_fraction; break;
        case 36: out = g.cycle_curvature_total; break;
        case 37: out = g.clustering; break;
        case 38: out = g.euler_excess; break;
        case 39: out = g.charge; break;
        case 40: out = g.charge_max; break;
        case 41: out = g.dimension_gradient; break;
        case 42: case 43: case 44: out = g.clique_betti[slot - 42]; break;
        case 45: out = g.rips_betti1; break;
        case 46: out = g.metric_tensor_gap; break;
        case 47: out = g.holonomy_triangle; break;
        case 48: out = g.holonomy_square; break;
        case 49: case 50: case 51: out = g.alignment_moment[slot - 49]; break;
        case 52: out = g.proper_time; break;
        case 53: out = g.proper_time_variance; break;
        case 54: out = g.bundle_spread; break;
        case 55: out = g.geodesic_curvature; break;
        case 56: out = g.lensing_deflection; break;
        case 57: out = g.lensing_impact; break;
        case 58: out = g.lensing_ratio; break;
        case 59: out = g.orbital_velocity; break;
        case 60: out = g.rotation_exponent; break;
        case 61: out = g.rotation_residual; break;
        default: return false;
    }
    return true;
}

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
    auto put = [&](uint32_t slot, double v) { sink.entry(slot, sp_key_of(v), w); };
    for (uint32_t slot = 0; slot < SP_NUM_INVARIANTS; ++slot) {
        double v = 0.0;
        if (sp_invariant(r.v, r.g, slot, v)) put(slot, v);
    }

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
    for (uint32_t b = 0; b < r.num_ball; ++b) put(SP_BALL + SP_SERIES * b, r.ball()[b]);
    for (uint32_t d = 0; d < r.num_geodesic; ++d)
        put(SP_BALL + SP_SERIES * d + 1, r.geodesic()[d]);
    for (uint32_t k = 0; k < r.num_rotation; ++k)
        put(SP_BALL + SP_SERIES * k + 2, r.rotation()[k]);
}

}  // namespace common
}  // namespace HG_NAMESPACE
