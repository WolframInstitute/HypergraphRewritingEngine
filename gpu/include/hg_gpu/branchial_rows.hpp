#pragma once
#include "hgcommon/namespace.hpp"
// THE "StepStatisticsBranchial" OVERLAP ROWS ON THE DEVICE (docs/DEVICE_DESIGN.md 4.8): every row
// of one step's overlap pass, hgcommon::br_row on one warp per row, counted per row and key in
// shared memory and read back as (distance, |A|, |B|, shared, pairs) counts.

#include <cstdint>
#include <functional>

#include "hgcommon/branchial_rows_core.hpp"

namespace HG_NAMESPACE {
namespace gpu {

// add(distance, size_a, size_b, shared, n) for each count, in no particular order; the counts of
// one key may arrive in several calls. An empty `add` counts nothing. `shared_cells`, when given,
// receives each pair's shared count at distance 0 in its hgcommon::br_pair_index cell, the other
// cells 0. False, with nothing added or written, when no device is usable or the step's scratch
// does not fit in a quarter of the free device memory.
bool branchial_rows_device(
    const hgcommon::BrStep& step,
    const std::function<void(uint32_t, uint32_t, uint32_t, uint32_t, uint64_t)>& add,
    uint32_t* shared_cells = nullptr);

}  // namespace gpu
}  // namespace HG_NAMESPACE
