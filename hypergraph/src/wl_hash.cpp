#include "hypergraph/wl_hash.hpp"

// The non-template bodies behind wl_hash.hpp. The refinement itself is templated on the
// accessors the caller supplies (edge vertices, arities), so it stays in the header; what is
// here is the vertex-hash cache and the mixing primitives it and the refinement share.

namespace HG_NAMESPACE {
namespace engine {

VertexHashCache::VertexHashCache()
    : vertices(nullptr), hashes(nullptr), adjacency_ptr(nullptr), count(0), capacity(0) {}

uint64_t VertexHashCache::lookup(VertexId v) const {
    const VertexId* pos = std::lower_bound(vertices, vertices + count, v);
    if (pos != vertices + count && *pos == v) {
        return hashes[pos - vertices];
    }
    return 0;
}

void VertexHashCache::insert(VertexId v, uint64_t hash) {
    vertices[count] = v;
    hashes[count] = hash;
    ++count;
}

WLHash::WLHash(ConcurrentHeterogeneousArena* arena)
    : arena_(arena)
{}

uint64_t WLHash::fnv_combine(uint64_t h, uint64_t value) {
    return fnv_hash(h, value);
}

uint64_t WLHash::mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t WLHash::compute_edge_signature(
    const VertexId* vertices,
    uint8_t arity,
    const VertexHashCache& cache
) const {
    uint64_t sig = FNV_OFFSET;
    sig = fnv_combine(sig, arity);

    for (uint8_t i = 0; i < arity; ++i) {
        uint64_t vh = cache.lookup(vertices[i]);
        sig = fnv_combine(sig, vh);
    }

    return sig;
}

}  // namespace engine
}  // namespace HG_NAMESPACE
