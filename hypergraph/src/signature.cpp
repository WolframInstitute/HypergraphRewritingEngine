#include "hypergraph/signature.hpp"

// The bodies behind signature.hpp: EdgeSignature's accessors and its two builders, the
// compatibility predicate, and the set-partition enumeration that SignatureIndex::for_each_candidate
// drives per pattern edge per state. CompatibleSignatureCache's defaulted default constructor is
// NOT here: defining a defaulted special member out of class makes it user-provided, which would
// cost the type its trivial default construction -- RewriteRule holds an array of them.

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

CompatibleSignatureCache CompatibleSignatureCache::from_pattern(const EdgeSignature& pattern_sig) {
    CompatibleSignatureCache cache;
    cache.source_pattern_sig = pattern_sig;

    enumerate_compatible_signatures(
        pattern_sig,
        [](const EdgeSignature& sig, void* user_data) {
            auto* c = static_cast<CompatibleSignatureCache*>(user_data);
            if (c->count < MAX_CACHED_SIGS) {
                c->signatures[c->count++] = sig;
            } else {
                // More compatible signatures than the cache can hold (e.g. a
                // high-arity all-distinct pattern edge). Flag it so the consumer
                // enumerates the full set live instead of silently dropping the
                // tail — dropping it would MISS real matches.
                c->overflowed = true;
            }
        },
        &cache
    );

    return cache;
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

// =============================================================================
// Signature enumeration
// =============================================================================

namespace detail {

void enumerate_partitions_recursive(
    uint8_t num_classes,
    uint8_t current_class,
    uint8_t* merged_to,
    uint8_t max_partition_used,
    const EdgeSignature& pattern_sig,
    const uint8_t* var_to_class,
    SignatureVisitor visitor,
    void* user_data
) {
    if (current_class == num_classes) {
        // All classes assigned - generate signature
        EdgeSignature result;
        result.arity = pattern_sig.arity;
        std::memset(result.pattern, 0, MAX_ARITY);

        for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
            uint8_t pvar = pattern_sig.pattern[i];
            uint8_t pclass = var_to_class[pvar];
            result.pattern[i] = merged_to[pclass];
        }

        visitor(result, user_data);
        return;
    }

    // Try assigning current_class to each existing partition (0..max_partition_used)
    // or to a new partition (max_partition_used + 1)
    for (uint8_t partition = 0; partition <= max_partition_used + 1; ++partition) {
        merged_to[current_class] = partition;

        uint8_t new_max = max_partition_used;
        if (partition > max_partition_used) {
            new_max = partition;
        }

        enumerate_partitions_recursive(
            num_classes, current_class + 1, merged_to, new_max,
            pattern_sig, var_to_class, visitor, user_data
        );
    }
}

}  // namespace detail

void enumerate_compatible_signatures(
    const EdgeSignature& pattern_sig,
    SignatureVisitor visitor,
    void* user_data
) {
    if (pattern_sig.arity == 0) {
        visitor(pattern_sig, user_data);
        return;
    }

    // Find equivalence classes in pattern (positions that share a variable)
    // Pattern variable -> class id
    uint8_t var_to_class[MAX_ARITY];
    std::memset(var_to_class, 0xFF, MAX_ARITY);

    uint8_t num_classes = 0;

    for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
        uint8_t pvar = pattern_sig.pattern[i];
        if (var_to_class[pvar] == 0xFF) {
            var_to_class[pvar] = num_classes;
            num_classes++;
        }
    }

    // Enumerate all set partitions of {0, 1, ..., num_classes-1}
    // Each partition represents a way that distinct pattern variables can
    // collapse to the same data vertex
    uint8_t merged_to[MAX_ARITY];
    std::memset(merged_to, 0, MAX_ARITY);

    // First class always goes to partition 0
    merged_to[0] = 0;

    if (num_classes == 1) {
        // Only one class - just one signature possible
        EdgeSignature result;
        result.arity = pattern_sig.arity;
        std::memset(result.pattern, 0, MAX_ARITY);
        for (uint8_t i = 0; i < pattern_sig.arity; ++i) {
            result.pattern[i] = 0;  // All map to same partition
        }
        visitor(result, user_data);
        return;
    }

    detail::enumerate_partitions_recursive(
        num_classes, 1, merged_to, 0,  // start from class 1, class 0 is in partition 0
        pattern_sig, var_to_class, visitor, user_data
    );
}

}  // namespace engine
}  // namespace HG_NAMESPACE
