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

#include "delivery_cursor.hpp"

#include "hgcommon/core.hpp"

#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace HG_NAMESPACE {
namespace ffi {

// A session's work, whichever engine performs it. The verbs are added here as they are served;
// what matters now is that the slot below owns one of THESE and not a device type.
class EngineHolder {
public:
    // Defined in paclet_support.cpp, so this polymorphic class has one translation unit
    // anchoring its vtable rather than a weak copy in every includer.
    virtual ~EngineHolder();

    // Extend the exploration by `steps` from the CURRENT frontier.
    //
    // This is the operation neither engine has today, and naming it here is the point. On the
    // device, `PersistentEvolver::run` takes a whole EvolveInput and evolves from
    // `in.initial_states` every call, so what persists across calls is the ALLOCATION, not the
    // explored graph (D13). A session needs the graph.
    //
    // `only_from`, when non-empty, names the frontier states to expand. Every other frontier
    // entry is RETAINED, so a later extend resumes it: a steered exploration narrows what runs
    // next, never what remains reachable.
    //
    // THE IDS HERE ARE THE HOLDER'S OWN RAW STATE IDS, not the effective ids the wire carries.
    // Translating between the two is a function of the canonicalization mode and the run's
    // content index, which the FFI already computes once for everything else it serialises;
    // giving the holder a second way to decide what a state's id is would be a second rule, and
    // the two would agree only until one of them was edited.
    virtual void extend(int steps, const std::vector<hgcommon::StateId>& only_from) = 0;

    // The states an extend would resume from, in the holder's raw id space. Meaningful only
    // between calls: during a run the frontier is the set of refusals so far, not a boundary.
    virtual std::vector<hgcommon::StateId> frontier() const = 0;

    // What this session has already been sent, so a Step can report what it ADDED. Not virtual
    // and not per-device: the record is of the SERIALISATION, which both devices marshal through
    // one build_graph_data, so a second copy per engine would be a second answer to a question
    // that has one.
    DeliveryCursor& delivery_cursor();

private:
    DeliveryCursor delivery_cursor_;
};

// Why a live handle cannot be served.
enum class SessionState { None, Live, Invalidated };

class SessionError : public std::runtime_error {
public:
    explicit SessionError(const std::string& what);
};

class SessionSlot {
public:
    static constexpr uint64_t kNoSession = 0;

    bool is_live() const;
    SessionState state() const;
    uint64_t handle() const;

    // Take the holder and mint a handle. Refuses while one is live (D7): a caller that has not
    // closed its session has not finished with it.
    // D7's refusal, in ONE place. Both devices refuse a second Open and both must say the same
    // thing: a caller matching on the message has no way to know which engine answered, and the
    // device's own wording drifted from this one until a test compiled against both found it.
    static std::string already_live_message(uint64_t live_handle);

    uint64_t open(std::unique_ptr<EngineHolder> holder);

    // The live session's engine. Throws rather than returning null, because every caller of this
    // is about to use it and a null check skipped once is a session served as a fresh engine.
    EngineHolder& engine(uint64_t handle);

    // D14. The engine is gone; the handle stays addressable so the next verb on it can say so.
    void invalidate();

    void close(uint64_t handle);

private:
    void require(uint64_t handle) const;

    std::unique_ptr<EngineHolder> holder_;
    uint64_t handle_ = kNoSession;
    uint64_t next_ = 1;                       // 0 is reserved, and handles are never reused
    SessionState state_ = SessionState::None;
};

}  // namespace ffi
}  // namespace HG_NAMESPACE