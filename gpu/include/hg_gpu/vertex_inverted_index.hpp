#pragma once
#include "hgcommon/namespace.hpp"

#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// Vertex → list-of-edge-ids inverted index. For each VertexId in
// [0, max_vertices), maintains a LockFreeList of the EdgeIds that contain that
// vertex, each once: an edge in which the vertex occurs several times is
// inserted for its first occurrence only, so a walk of the list yields every
// incident edge exactly once.
//
// Used by the match kernel for "edges incident on this vertex" lookup
// during candidate generation: when a partial match has bound variable v
// and is searching for the next edge, the candidate set is the intersection
// of incident-edge lists for the variables shared between the matched and
// next pattern edges.
class VertexInvertedIndex {
public:
    struct DeviceView {
        typename LockFreeList<EdgeId>::DeviceView list;

        // Insert edge_id under vertex `v`, once per distinct vertex of the edge: for
        // {a, b, a} the caller inserts under a and b (first_occurrence below).
        __device__ uint32_t insert(VertexId v, EdgeId eid) const {
            return list.push(v, eid);
        }

        template <typename Fn>
        __device__ void for_each_incident(VertexId v, Fn fn) const {
            list.for_each(v, fn);
        }
    };

    VertexInvertedIndex(uint32_t max_vertices, uint32_t pool_capacity);

    DeviceView view() const;

    uint32_t max_vertices() const;
    uint32_t used() const;

    // `used_vertices` bounds the head reset to the ids a run actually minted; see
    // LockFreeList::clear. Vertex ids come from a monotone counter, so the prefix is exact.
    void clear(uint32_t used_vertices = 0xFFFFFFFFu);

private:
    LockFreeList<EdgeId> list_;
};

// True when position i of an edge is the first occurrence of its vertex, read through `at`.
template <typename At>
__host__ __device__ inline bool first_occurrence(At&& at, uint8_t i) {
    for (uint8_t j = 0; j < i; ++j)
        if (at(j) == at(i)) return false;
    return true;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE