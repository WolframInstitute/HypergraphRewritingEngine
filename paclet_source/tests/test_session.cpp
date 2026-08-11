// The session handle space and lifetime rule, before any engine is wired to it.
//
// These assertions are the contract the verbs will be written against, so they are worth having
// while the slot is still small enough to read in one sitting. Each names the decision it comes
// from; see docs/FFI_INTERFACE_DESIGN.

#include <gtest/gtest.h>

#include "paclet_source/session.hpp"

#include <memory>
#include <vector>

namespace {

// Counts what a real holder would do, so a test can tell "the slot handed back the engine" from
// "the slot handed back something".
struct CountingHolder : hgffi::EngineHolder {
    int extended = 0;
    int* destroyed = nullptr;
    explicit CountingHolder(int* d = nullptr) : destroyed(d) {}
    ~CountingHolder() override { if (destroyed) ++*destroyed; }
    std::vector<hgcommon::StateId> pretend_frontier{7, 9};
    std::vector<hgcommon::StateId> last_selection;
    void extend(int steps, const std::vector<hgcommon::StateId>& only_from) override {
        extended += steps;
        last_selection = only_from;
    }
    std::vector<hgcommon::StateId> frontier() const override { return pretend_frontier; }
};

std::unique_ptr<hgffi::EngineHolder> holder(int* destroyed = nullptr) {
    return std::make_unique<CountingHolder>(destroyed);
}

}  // namespace

TEST(Session, ZeroIsNeverAHandle) {
    hgffi::SessionSlot s;
    EXPECT_FALSE(s.is_live());
    EXPECT_EQ(s.handle(), hgffi::SessionSlot::kNoSession);
    // D11: 0 means "no session", so it can never name one.
    EXPECT_THROW(s.engine(hgffi::SessionSlot::kNoSession), hgffi::SessionError);

    const uint64_t h = s.open(holder());
    EXPECT_NE(h, hgffi::SessionSlot::kNoSession);
    EXPECT_TRUE(s.is_live());
}

TEST(Session, OneAtATimeRefusesRatherThanEvicts) {
    hgffi::SessionSlot s;
    const uint64_t first = s.open(holder());
    // D7. Evicting would discard the first caller's exploration without telling it.
    EXPECT_THROW(s.open(holder()), hgffi::SessionError);
    // And the refusal left the live session untouched.
    EXPECT_TRUE(s.is_live());
    EXPECT_EQ(s.handle(), first);
    EXPECT_NO_THROW(s.engine(first));
}

TEST(Session, AHandleIsNeverReused) {
    hgffi::SessionSlot s;
    const uint64_t first = s.open(holder());
    s.close(first);
    const uint64_t second = s.open(holder());
    // A reissued handle would let a stale caller address a different session and be answered as
    // though it were its own.
    EXPECT_NE(second, first);
    EXPECT_THROW(s.engine(first), hgffi::SessionError);
    EXPECT_NO_THROW(s.engine(second));
}

TEST(Session, CloseReleasesTheEngine) {
    int destroyed = 0;
    hgffi::SessionSlot s;
    const uint64_t h = s.open(holder(&destroyed));
    EXPECT_EQ(destroyed, 0);
    s.close(h);
    EXPECT_EQ(destroyed, 1) << "Close must release the engine, or a worker leaks one per session";
    EXPECT_FALSE(s.is_live());
    EXPECT_THROW(s.close(h), hgffi::SessionError);
}

TEST(Session, AnInvalidatedSessionSaysSoRatherThanServingAFreshEngine) {
    int destroyed = 0;
    hgffi::SessionSlot s;
    const uint64_t h = s.open(holder(&destroyed));

    // D14: an overflow discarded the engine that held this caller's exploration.
    s.invalidate();
    EXPECT_EQ(destroyed, 1);
    EXPECT_FALSE(s.is_live());
    EXPECT_EQ(s.state(), hgffi::SessionState::Invalidated);

    // The handle stays addressable precisely so the next verb on it can report the loss. Serving
    // a fresh engine here would return a graph that had lost its history and pass every internal
    // consistency check -- which is the failure this state exists to prevent.
    try {
        s.engine(h);
        FAIL() << "an invalidated session must not hand back an engine";
    } catch (const hgffi::SessionError& e) {
        EXPECT_NE(std::string(e.what()).find("invalidated"), std::string::npos)
            << "the error must say the exploration is gone, not merely that the handle is bad: "
            << e.what();
    }

    // Invalidation is idempotent, and a new session is still possible afterwards.
    EXPECT_NO_THROW(s.invalidate());
    EXPECT_NE(s.open(holder()), h);
}

TEST(Session, TheSlotHandsBackTheEngineItWasGiven) {
    hgffi::SessionSlot s;
    auto owned = std::make_unique<CountingHolder>();
    CountingHolder* raw = owned.get();
    const uint64_t h = s.open(std::move(owned));

    s.engine(h).extend(3, {});
    s.engine(h).extend(4, {});
    EXPECT_EQ(raw->extended, 7) << "the slot must hand back the SAME holder, not any holder";
}

TEST(Session, AForeignHandleIsRefused) {
    hgffi::SessionSlot s;
    const uint64_t h = s.open(holder());
    EXPECT_THROW(s.engine(h + 1000), hgffi::SessionError);
    EXPECT_THROW(s.close(h + 1000), hgffi::SessionError);
    // The refusals did not disturb the live session.
    EXPECT_TRUE(s.is_live());
    EXPECT_NO_THROW(s.engine(h));
}

// A subset continuation reaches the holder UNCHANGED. The FFI resolves the caller's effective
// ids against the frontier and hands the holder its own raw ids; if the slot dropped or
// reordered them the holder would expand a different branch than the one named, and the result
// would still be a valid graph -- which is why this is checked by value rather than by outcome.
TEST(Session, TheSelectionReachesTheHolderExactly) {
    hgffi::SessionSlot s;
    auto owned = std::make_unique<CountingHolder>();
    CountingHolder* raw = owned.get();
    const uint64_t h = s.open(std::move(owned));

    s.engine(h).extend(1, {9});
    EXPECT_EQ(raw->last_selection, (std::vector<hgcommon::StateId>{9}));

    // An empty selection is "all of them", NOT "none of them". The two are one keystroke apart
    // and the second would silently converge an exploration that has work left.
    s.engine(h).extend(1, {});
    EXPECT_TRUE(raw->last_selection.empty());

    EXPECT_EQ(s.engine(h).frontier(), (std::vector<hgcommon::StateId>{7, 9}));
}
