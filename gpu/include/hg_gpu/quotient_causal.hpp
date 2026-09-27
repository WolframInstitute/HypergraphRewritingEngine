#pragma once
#include "hgcommon/namespace.hpp"
// Whether the quotient route is on for a device run. On the route every canonicalized state
// also computes its edge ORBITS (state_key_device), which the replay's class frames key
// instances by (quotient_expansion.hpp). Orbits are what raise IR_NEED_GENERATORS, so this flag
// also decides which states hash and therefore which states dedup merges.

#include <cstdint>

namespace HG_NAMESPACE {
namespace gpu {

struct QcView {
    uint32_t enabled = 0;
};

class QcState {
public:
    explicit QcState(bool on) : on_(on) {}
    bool enabled() const { return on_; }
    QcView view() const { QcView q; q.enabled = on_ ? 1u : 0u; return q; }

private:
    bool on_ = false;
};

}  // namespace gpu
}  // namespace HG_NAMESPACE
