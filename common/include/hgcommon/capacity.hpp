#pragma once
#include "hgcommon/namespace.hpp"
//
// A CONFIGURED LIMIT WAS REACHED. Not a programmer mistake, and that distinction is the whole
// reason this type exists.
//
// The engine's containers are sized from configuration. Reaching one of those ceilings means the
// workload is larger than the arena it was given -- which is a thing users do on purpose, at the
// edge of what a machine holds -- and the answer they want is the part that fits plus a statement
// of why it stopped. The GPU already answers that way: an overflow flags EvolveResult.warnings
// and returns the PARTIAL result. The host threw, and a throw out of a worker propagates through
// evolve() and terminates a caller that did not wrap it, so the same event meant "here is your
// truncated graph" on one device and "your process is gone" on the other.
//
// A TYPE, NOT A MESSAGE. The job system already distinguishes an abort by comparing what() to a
// literal, which works and is exactly the coupling that breaks the day the wording changes.
// Classifying on the TYPE happens at the catch, where the type is still known, and a rename of
// the text cannot reach it.
//
// It lives in hgcommon because the thrower (the engine's containers) and the catcher (the job
// system) are separate libraries, and the job system must not depend on the engine to name what
// it caught.

#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace common {

class CapacityExhausted : public std::length_error {
public:
    explicit CapacityExhausted(const std::string& what) : std::length_error(what) {}
};

}  // namespace common
}  // namespace HG_NAMESPACE
