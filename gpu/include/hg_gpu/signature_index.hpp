#pragma once

#include "hg_gpu/lock_free_list.hpp"
#include "hg_gpu/types.hpp"

#include <cstdint>

namespace hg_gpu {

// Incremental signature index. Edges are bucketed by their signature hash
// (mod num_buckets); each bucket is a LockFreeList of EdgeIds. Lookup walks
// the bucket and the caller filters by comparing Edge::signature to the
// target signature hash — this keeps the index compact at the cost of a
// tiny per-candidate check for bucket collisions.
//
// Bucket count is a power of two so the modulus is a mask. Sized by the
// auto-tuner; a good default is ≥ 2× expected distinct signatures. Under
// reasonable distribution, false bucket collisions are rare and the filter
// is cheap.
class SignatureIndex {
public:
    struct DeviceView {
        typename LockFreeList<EdgeId>::DeviceView list;
        uint32_t mask;  // = num_buckets - 1

        __device__ uint32_t insert(EdgeId eid, uint64_t signature_hash) {
            return list.push(static_cast<uint32_t>(signature_hash) & mask, eid);
        }

        template <typename Fn>
        __device__ void for_each_in_bucket(uint64_t signature_hash, Fn fn) const {
            list.for_each(static_cast<uint32_t>(signature_hash) & mask, fn);
        }
    };

    SignatureIndex(uint32_t num_buckets_pow2, uint32_t max_edges)
        : list_(num_buckets_pow2, max_edges),
          mask_(num_buckets_pow2 - 1)
    {
        if ((num_buckets_pow2 & mask_) != 0 || num_buckets_pow2 == 0) {
            throw std::invalid_argument("SignatureIndex num_buckets must be a power of two ≥ 1");
        }
    }

    DeviceView view() const { return DeviceView{list_.view(), mask_}; }

    uint32_t num_buckets() const { return list_.num_keys(); }
    uint32_t used()        const { return list_.pool_used_host(); }

    void clear() { list_.clear(); }

private:
    LockFreeList<EdgeId> list_;
    uint32_t             mask_;
};

}  // namespace hg_gpu
