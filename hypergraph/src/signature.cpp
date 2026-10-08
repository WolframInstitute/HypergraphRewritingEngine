#include "hypergraph/signature.hpp"

// The bodies behind signature.hpp: EdgeSignature's accessors, its two builders and the
// compatibility predicate.

namespace HG_NAMESPACE {
namespace engine {

EdgeSignature EdgeSignature::from_pattern(const uint8_t* vars, uint8_t arity) {
    EdgeSignature sig;
    sig.arity = arity;
    std::memset(sig.pattern, 0, MAX_ARITY);

    if (arity == 0) return sig;

    // Map first occurrence of each variable to incrementing label
    uint8_t next_label = 0;
    uint8_t seen_vars[MAX_ARITY];
    uint8_t var_labels[MAX_ARITY];

    for (uint8_t i = 0; i < arity; ++i) {
        uint8_t var = vars[i];

        // Check if variable already seen
        uint8_t label = next_label;
        for (uint8_t j = 0; j < next_label; ++j) {
            if (seen_vars[j] == var) {
                label = var_labels[j];
                break;
            }
        }

        // If new variable, assign new label
        if (label == next_label) {
            seen_vars[next_label] = var;
            var_labels[next_label] = next_label;
            next_label++;
        }

        sig.pattern[i] = label;
    }

    return sig;
}

// =============================================================================
// EdgeSignature accessors
// =============================================================================

EdgeSignature EdgeSignature::from_edge(const VertexId* vertices, uint8_t arity) {
    EdgeSignature sig;
    sig.arity = arity;
    std::memset(sig.pattern, 0, MAX_ARITY);
    hgcommon::signature_pattern_from_vertices(vertices, arity, sig.pattern);
    return sig;
}

uint64_t EdgeSignature::hash() const { return hgcommon::signature_hash(arity, pattern); }

bool EdgeSignature::operator==(const EdgeSignature& other) const {
    if (arity != other.arity) return false;
    for (uint8_t i = 0; i < arity; ++i) {
        if (pattern[i] != other.pattern[i]) return false;
    }
    return true;
}

bool EdgeSignature::operator!=(const EdgeSignature& other) const {
    return !(*this == other);
}

uint8_t EdgeSignature::num_distinct() const {
    return hgcommon::signature_num_distinct(arity, pattern);
}

bool signature_compatible(const EdgeSignature& data_sig,
                          const EdgeSignature& pattern_sig) {
    return hgcommon::signature_compatible(data_sig.arity, data_sig.pattern,
                                          pattern_sig.arity, pattern_sig.pattern);
}

}  // namespace engine
}  // namespace HG_NAMESPACE
