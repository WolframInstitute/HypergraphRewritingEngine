// Host-side exact IR canonicalization for the GPU engine's fallback path.
// The GPU IR kernel degrades to a coarse hash on non-discrete (symmetric) or
// oversize states; those rare states are re-hashed here with the exact CPU
// IRCanonicalizer so GPU dedup never wrongly merges non-isomorphic states.
// (Compiled by the host C++ compiler and linked into hg_gpu.)
#include "hypergraph/ir_canonicalization.hpp"
#include <cstdint>
#include <vector>

namespace hg_gpu {

uint64_t ir_host_canonical_hash(const std::vector<std::vector<uint32_t>>& edges) {
    hypergraph::IRCanonicalizer ir;
    return ir.compute_canonical_hash(edges);
}

}  // namespace hg_gpu
