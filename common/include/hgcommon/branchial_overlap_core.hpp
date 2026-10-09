#pragma once
#include "hgcommon/namespace.hpp"
// The values of the "StepStatisticsBranchial" overlap metrics, one body for both engines' replies
// (paclet_source/paclet_support.cpp, overlap_metrics). Each value is a function of counts: a pair
// of states A, B is (n11 = |A ∩ B|, |A|, |B|) inside a universe of |U| elements, and an element
// held by k states is k. The logarithms are det_math's, so the host and the device give the same
// double.
//
//   StateOverlap              n11 / |A ∪ B|, 0 when A ∪ B is empty
//   StateCosineSimilarity     n11 / sqrt(|A| |B|), 0 when A or B is empty
//   StateMutualInformation    max(0, sum over a, b in {0, 1} of p_ab log2(p_ab / (p_a p_b))),
//                             p_ab = n_ab / |U|, n10 = |A| - n11, n01 = |B| - n11,
//                             n00 = |U| - |A| - |B| + n11; 0 when U is empty
//   VertexSharpness, EdgeSharpness        1 / k
//   BranchEntropy, EdgeBranchEntropy      log2 k

#include <math.h>
#include <stdint.h>

#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"

namespace HG_NAMESPACE {
namespace common {

// n11 / |A ∪ B| from n11 and |A ∪ B|.
HG_HD inline double bo_jaccard_of_union(uint64_t both, uint64_t uni) {
    return uni == 0 ? 0.0 : static_cast<double>(both) / static_cast<double>(uni);
}

HG_HD inline double bo_cosine(uint64_t both, uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0.0;
    return static_cast<double>(both) /
           ::sqrt(dm_mul(static_cast<double>(a), static_cast<double>(b)));
}

// One term p_ab log2(p_ab / (p_a p_b)) of the mutual information, from counts over u elements;
// 0 when n_ab is 0.
HG_HD inline double bo_mi_term(uint64_t n_ab, uint64_t n_a, uint64_t n_b, uint64_t u) {
    if (n_ab == 0) return 0.0;
    const double ab = static_cast<double>(n_ab), uu = static_cast<double>(u);
    const double ratio = dm_mul(ab, uu) /
                         dm_mul(static_cast<double>(n_a), static_cast<double>(n_b));
    return dm_mul(ab / uu, dm_log2(ratio));
}

// The mutual information in bits of the indicators of A and B over U, with A, B subsets of U.
HG_HD inline double bo_mutual_information(uint64_t both, uint64_t a, uint64_t b, uint64_t u) {
    if (u == 0) return 0.0;
    const uint64_t n10 = a - both, n01 = b - both, n00 = u - a - b + both;
    const double sum = bo_mi_term(both, a, b, u) + bo_mi_term(n10, a, u - b, u) +
                       bo_mi_term(n01, u - a, b, u) + bo_mi_term(n00, u - a, u - b, u);
    return sum > 0.0 ? sum : 0.0;
}

HG_HD inline double bo_sharpness(uint64_t k) { return 1.0 / static_cast<double>(k); }

HG_HD inline double bo_branch_entropy(uint64_t k) {
    return dm_log2(static_cast<double>(k));
}

}  // namespace common
}  // namespace HG_NAMESPACE
