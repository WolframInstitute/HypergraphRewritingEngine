#pragma once
#include "hgcommon/namespace.hpp"
//
// The session's handle space and lifetime rule, independent of which engine holds the work.
//
// WHY THIS IS ITS OWN FILE, and why it exists before any engine is wired to it. The verbs are
// served for BOTH devices (D9), and the one thing that cannot be retrofitted is the SHAPE: a
// holder written around a `hypergraph::Hypergraph` would have to be replaced rather than
// extended when the device half arrives, and the two would be merged afterwards -- which is the
// defect this project spent a day removing from the canonicalizer, the causal DP and the replay.
// So the holder is an interface from the start and the slot never names a device.
//
// THE DECISIONS THIS ENCODES, from docs/FFI_INTERFACE_DESIGN:
//   D7  ONE SESSION AT A TIME. Not a map: a single slot. A second Open while one is live is an
//       error, not an eviction -- evicting would discard a caller's exploration silently.
//   D11 THE HANDLE IS AN OPAQUE uint64, minted by a per-worker counter, with 0 reserved for
//       "no session". Opaque so no client constructs one; zero-reserved so absence and a real
//       handle are never confused, the same discipline every other id space here follows.
//   D14 AN OVERFLOW INVALIDATES THE SESSION. `hg_gpu::PersistentEvolver` discards its engine on
//       any throw from `Engine::run` (gpu/src/evolve.cu), deliberately, because reusing a
//       poisoned engine produced the reported "works, then fails and never recovers until a
//       kernel reset". That is transparent for a stateless per-job evolver, whose next call
//       rebuilds. It is NOT transparent under a session: the discarded engine held the caller's
//       accumulated exploration. So the slot can be invalidated, and a handle that names an
//       invalidated session reports that rather than silently serving a fresh empty engine --
//       which would return a graph that had lost its history and satisfy every internal check.
//
// Handles are NOT reused after Close. A reissued handle would let a stale caller address a
// different session and be answered as if it were its own.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace ffi {

// A session's work, whichever engine performs it. The verbs are added here as they are served;
// what matters now is that the slot below owns one of THESE and not a device type.
class EngineHolder {
public:
    virtual ~EngineHolder() = default;

    // Extend the exploration by `steps` from the CURRENT frontier.
    //
    // This is the operation neither engine has today, and naming it here is the point. On the
    // device, `PersistentEvolver::run` takes a whole EvolveInput and evolves from
    // `in.initial_states` every call, so what persists across calls is the ALLOCATION, not the
    // explored graph (D13). A session needs the graph.
    virtual void extend(int steps) = 0;
};

// Why a live handle cannot be served.
enum class SessionState { None, Live, Invalidated };

class SessionError : public std::runtime_error {
public:
    explicit SessionError(const std::string& what) : std::runtime_error(what) {}
};

class SessionSlot {
public:
    static constexpr uint64_t kNoSession = 0;

    bool is_live() const { return state_ == SessionState::Live; }
    SessionState state() const { return state_; }
    uint64_t handle() const { return handle_; }

    // Take the holder and mint a handle. Refuses while one is live (D7): a caller that has not
    // closed its session has not finished with it.
    uint64_t open(std::unique_ptr<EngineHolder> holder) {
        if (!holder) throw SessionError("Open: no engine holder");
        if (state_ == SessionState::Live)
            throw SessionError("Open: a session is already live (" + std::to_string(handle_) +
                               "); this build serves one session at a time");
        holder_ = std::move(holder);
        handle_ = next_++;
        state_ = SessionState::Live;
        return handle_;
    }

    // The live session's engine. Throws rather than returning null, because every caller of this
    // is about to use it and a null check skipped once is a session served as a fresh engine.
    EngineHolder& engine(uint64_t handle) {
        require(handle);
        return *holder_;
    }

    // D14. The engine is gone; the handle stays addressable so the next verb on it can say so.
    void invalidate() {
        if (state_ != SessionState::Live) return;
        holder_.reset();
        state_ = SessionState::Invalidated;
    }

    void close(uint64_t handle) {
        require(handle);
        holder_.reset();
        handle_ = kNoSession;
        state_ = SessionState::None;
    }

private:
    void require(uint64_t handle) const {
        if (handle == kNoSession) throw SessionError("no session handle given");
        if (handle != handle_)
            throw SessionError("session " + std::to_string(handle) + " is not this worker's live "
                               "session");
        if (state_ == SessionState::Invalidated)
            throw SessionError("session " + std::to_string(handle) + " was invalidated: the run "
                               "overflowed and its engine was discarded, so the exploration it "
                               "held is gone. Open a new session");
        if (state_ != SessionState::Live)
            throw SessionError("session " + std::to_string(handle) + " is closed");
    }

    std::unique_ptr<EngineHolder> holder_;
    uint64_t handle_ = kNoSession;
    uint64_t next_ = 1;                       // 0 is reserved, and handles are never reused
    SessionState state_ = SessionState::None;
};

}  // namespace ffi
}  // namespace HG_NAMESPACE