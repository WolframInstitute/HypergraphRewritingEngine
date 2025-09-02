#include <hypergraph/thread_local_pool.hpp>
#include <hypergraph/pattern_matching_tasks.hpp>

namespace hypergraph {

// Thread-local storage definition for PartialMatchPool
thread_local std::vector<std::unique_ptr<PartialMatch>> PartialMatchPool::available_objects_;

std::unique_ptr<PartialMatch> PartialMatchPool::acquire() {
    if (!available_objects_.empty()) {
        auto obj = std::move(available_objects_.back());
        available_objects_.pop_back();
        
        // Reset object state
        obj->matched_edges.clear();
        obj->edge_map.clear();
        obj->assignment.clear();
        obj->next_pattern_edge_idx = 0;
        obj->anchor_vertex = 0;
        obj->available_edges.clear();
        
        return obj;
    }
    
    return std::make_unique<PartialMatch>();
}

void PartialMatchPool::release(std::unique_ptr<PartialMatch> obj) {
    if (obj) {
        available_objects_.push_back(std::move(obj));
    }
}

size_t PartialMatchPool::size() {
    return available_objects_.size();
}

void PartialMatchPool::clear() {
    available_objects_.clear();
}

} // namespace hypergraph