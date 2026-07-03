#pragma once

#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>

namespace hg_gpu {

// Vertex → list-of-edge-ids inverted index. For each VertexId in
// [0, max_vertices), maintains a LockFreeList of EdgeIds that contain that
// vertex (counting multiplicity — a self-loop appears twice for the same
// vertex).
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

        // Insert edge_id under vertex `v`. Caller must invoke once per
        // (v, edge_id) occurrence — i.e. for an edge {a, b, c} the rewrite
        // kernel calls insert(a, eid), insert(b, eid), insert(c, eid).
        // For a self-loop {a, a} it calls insert twice (degree counts).
        __device__ uint32_t insert(VertexId v, EdgeId eid) {
            return list.push(v, eid);
        }

        template <typename Fn>
        __device__ void for_each_incident(VertexId v, Fn fn) const {
            list.for_each(v, fn);
        }
    };

    VertexInvertedIndex(uint32_t max_vertices, uint32_t pool_capacity)
        : list_(max_vertices, pool_capacity) {}

    DeviceView view() const { return DeviceView{list_.view()}; }

    uint32_t max_vertices() const { return list_.num_keys(); }
    uint32_t used()         const { return list_.pool_used_host(); }

    void clear() { list_.clear(); }

private:
    LockFreeList<EdgeId> list_;
};

}  // namespace hg_gpu
