#include "hypergraph/signature.hpp"

// The two signature builders that run at rule registration. The ENUMERATION they feed
// (enumerate_compatible_signatures and its recursive helper) stays in the header: it is
// reached per pattern edge per state through SignatureIndex::for_each_candidate.

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

}  // namespace engine
}  // namespace HG_NAMESPACE
