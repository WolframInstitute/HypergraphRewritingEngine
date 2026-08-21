#include "hgcommon/capacity.hpp"

// CapacityExhausted's constructor. The class is declared in hgcommon because the thrower (the
// engine's containers) and the catcher (the job system) are separate libraries and the job
// system must not depend on the engine to name what it caught. The BODY is here because
// hgcommon has no library of its own and job_system is linked by everything that throws it.

namespace HG_NAMESPACE {
namespace common {

CapacityExhausted::CapacityExhausted(const std::string& what) : std::length_error(what) {}

}  // namespace common
}  // namespace HG_NAMESPACE
