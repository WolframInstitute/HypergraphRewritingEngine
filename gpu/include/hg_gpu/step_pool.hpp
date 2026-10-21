#pragma once
#include "hgcommon/namespace.hpp"
// THE "StepStatistics" POOL ON THE DEVICE (docs/DEVICE_DESIGN.md 4.8): every class of every step
// adds its entries (hgcommon::sp_add_class) on one thread, the entries are sorted by (step,
// slot, key) and equal ones merged on the device, and the merged entries and each step's
// distribution totals are read back. The host finishes each step from them
// (stats::StepPoolFinish), as it does from its own pool, so the record is the host's bytes.

#include <array>
#include <cstdint>
#include <vector>

#include "hgcommon/step_pool_core.hpp"

namespace HG_NAMESPACE {
namespace gpu {

// One class of one step: its record at word `record_word` of the record pool (UINT64_MAX for the
// state with no edges), its pooling weight and its raw state count.
struct StepPoolPoint {
    uint32_t step;
    uint64_t record_word;
    uint64_t weight;
    uint64_t raw;
};

using StepPoolTotals = std::array<hgcommon::SpDistTotals, hgcommon::SG_DISTS>;

// The pool of every step below `steps`: entries[s] sorted by (slot, key) with equal keys merged,
// and totals[s]. False, with nothing written, when no device is usable or the pool does not fit
// in a quarter of the free device memory.
bool step_pool_device(const std::vector<uint64_t>& record_pool,
                      const std::vector<StepPoolPoint>& points, uint32_t steps,
                      std::vector<std::vector<hgcommon::SpEntry>>& entries,
                      std::vector<StepPoolTotals>& totals);

}  // namespace gpu
}  // namespace HG_NAMESPACE
