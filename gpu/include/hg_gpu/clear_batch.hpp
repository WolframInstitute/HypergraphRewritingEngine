#pragma once
#include "hgcommon/namespace.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

// Device regions filled with one byte value, cleared together by one kernel launch on the
// default stream, in stream order with the launches around it.
//
// A run's reset clears dozens of counters, prefixes and small arrays. One cudaMemset each was
// 154 synchronous API calls per evolve call (nsys, wpp at two steps, PersistentEvolver), each
// paying the API's fixed cost for a few bytes of work.
//
// A region whose pointer or length is not a multiple of 4 is cleared with cudaMemset at once.
struct ClearRegion {
    void*    ptr;
    uint64_t bytes;
    uint32_t word;   // the fill byte in all four bytes
};

class ClearBatch {
public:
    static constexpr uint32_t kMaxRegions = 64;

    ClearBatch() = default;
    ClearBatch(const ClearBatch&)            = delete;
    ClearBatch& operator=(const ClearBatch&) = delete;

    // A full batch is flushed before the region is added.
    void add(void* ptr, uint64_t bytes, uint8_t fill);
    // Launches the clear of every region added since the last flush.
    void flush();

    struct Set {
        ClearRegion r[kMaxRegions];
        uint32_t    n = 0;
    };

private:
    Set set_;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE
