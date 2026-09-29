// GenMC harness: raw counts from class multiplicities lose no mass and pass none twice.
//
// WHAT IS BEING PROVED. Under quotient exploration with raw counts (quotient_multiplicity_core.hpp)
// a class c at depth d carries a mass m(c, d), and every captured match j of c passes that mass
// on to its target class one depth deeper. Three sides meet through rv::QuotientMassMatch:
//
//   the capture of j:     record b_j (j is ready), fence, qm_pass(j) at every depth, drain
//   mass arriving at c:   add to m(c, d), fence, claim the point's queued flag, push if won
//   a queued point runs:  pop clears the flag, fence, read m(c, d), qm_pass every ready match
//
// qm_pass advances consumed_j(d) by compare-and-swap to the mass it read and passes the
// difference. Property, under RC11: when every thread has finished, j has passed on exactly the
// mass that arrived, so the target holds it and the event count equals it. A unit missed by
// every side is lost; a unit passed twice is a double count.
//
// This drives hgcommon::qm_pass, qm_credit and qm_drain. The Ctx is the harness's storage with
// the orders of Hypergraph::QmCtx (hypergraph.cpp): acq_rel on the mass CAS, acquire loads,
// acq_rel/acquire on the consumed CAS, acq_rel exchange to claim the flag and a release store
// to clear it, rendezvous_barrier<rv::QuotientMassMatch> as the fence.
//
// WHAT IS BOUNDED. One class c at depth 0, one match j into class t, max_steps 1 (mass at t is
// recorded and not passed on). Three threads: the capture of j and two arrivals at c, of 1 and
// 2 units, so an arrival can find the flag held by the other.
//
// CALIBRATION. -DCALIBRATE_NO_FENCE makes the Ctx fence a no-op: the capture can read no mass
// while each arrival's run reads j as not ready, and the mass is never passed on.
// -DCALIBRATE_CLEAR_LATE clears the queued flag after the point's passes (at the next pop)
// instead of before them: mass that arrives between the run's read and the clear finds the flag
// held and is left to a run that has already read.
//
// --disable-ipr: an arrival's failed exchange on the queued flag and the holder's clearing store
// are two writes with no order between them, which is the handoff itself. GenMC reports that as
// "unordered writes" and stops when in-place revisiting is on.
//
// GENMC-ARGS: --disable-estimation --disable-ipr
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh quotient_mass_match_rendezvous

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>

#include "genmc_support.hpp"
#include "hgcommon/quotient_multiplicity_core.hpp"
#include "hgcommon/rendezvous.hpp"

namespace {

constexpr uint64_t kC = 1;   // the class that receives mass
constexpr uint64_t kT = 2;   // j's target class

struct Match {
    uint32_t id = 0;
    uint64_t to_hash = kT;
    uint32_t rule = 0;
    uint32_t num_consumed = 0, num_produced = 0;
    const uint32_t* consumed_ptr() const { return nullptr; }
    const uint32_t* produced_ptr() const { return nullptr; }
};

Match g_j;
std::atomic<uint64_t> g_overlaps{0};      // qm_overlaps_[j]: b_j + 1 once ready
std::atomic<uint64_t> g_mass_c{0};        // m(c, 0)
std::atomic<uint64_t> g_mass_t{0};        // m(t, 1)
std::atomic<uint32_t> g_queued_c{0};      // QmPoint::queued of (c, 0)
std::atomic<uint64_t> g_consumed{0};      // consumed_j(0)
std::atomic<uint64_t> g_events{0};

void sat_add(std::atomic<uint64_t>& a, uint64_t d) {   // Hypergraph::qm_add
    uint64_t old = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(old, hgcommon::qm_sat_add(old, d), std::memory_order_acq_rel,
                                    std::memory_order_relaxed)) {}
}

struct Ctx {
    using Match = ::Match;
    uint64_t queue[2];
    uint32_t n = 0;

    uint32_t max_steps() const { return 1; }
    bool ready(const Match&, uint64_t& b) const {
        const uint64_t v = g_overlaps.load(std::memory_order_acquire);
        if (v == 0) return false;
        b = v - 1;
        return true;
    }
    uint64_t mass(uint64_t h, uint32_t) const {
        return (h == kC ? g_mass_c : g_mass_t).load(std::memory_order_acquire);
    }
    void add_mass(uint64_t h, uint32_t, uint64_t d) { sat_add(h == kC ? g_mass_c : g_mass_t, d); }
    uint64_t consumed(const Match&, uint32_t) { return g_consumed.load(std::memory_order_acquire); }
    bool advance(const Match&, uint32_t, uint64_t& expected, uint64_t desired) {
        return g_consumed.compare_exchange_strong(expected, desired, std::memory_order_acq_rel,
                                                  std::memory_order_acquire);
    }
    void count(uint64_t e, uint64_t) { sat_add(g_events, e); }
    hgcommon::EventSignatureKeys keys() const { return hgcommon::EVENT_SIG_NONE; }
    uint32_t frame_step(uint64_t, uint32_t f) const { return f; }
    void note_signature(const Match&, uint64_t, uint32_t) {}
    bool claim_queued(uint64_t, uint32_t) {
        return g_queued_c.exchange(1, std::memory_order_acq_rel) == 0;
    }
    void push(uint64_t h, uint32_t) { assert(n < 2); queue[n++] = h; }
    bool pop(uint64_t& h, uint32_t& d) {
#if defined(CALIBRATE_CLEAR_LATE)
        if (held) { g_queued_c.store(0, std::memory_order_release); held = false; }
        if (n == 0) return false;
        h = queue[--n];
        d = 0;
        held = true;
#else
        if (n == 0) return false;
        h = queue[--n];
        d = 0;
        g_queued_c.store(0, std::memory_order_release);
#endif
        return true;
    }
    bool held = false;
    template <class F> void for_each_match(uint64_t h, F&& f) { if (h == kC) f(g_j); }
    void fence() {
#if !defined(CALIBRATE_NO_FENCE)
        hgcommon::rendezvous_barrier<hgcommon::rv::QuotientMassMatch>();
#endif
    }
};

// qc_capture_expansion's multiplicity half: j ready, then a cascade that passes every depth.
void* capture(void*) {
    g_overlaps.store(1, std::memory_order_release);   // b_j = 0
    Ctx c;
    c.fence();
    for (uint32_t d = 0; d < c.max_steps(); ++d) hgcommon::qm_pass(c, g_j, kC, d);
    hgcommon::qm_drain(c);
    return nullptr;
}

// Mass arriving at (c, 0), as a parent's qm_pass credits it, then its cascade.
void* arrive(void* delta) {
    Ctx c;
    hgcommon::qm_credit(c, kC, 0, reinterpret_cast<uintptr_t>(delta));
    hgcommon::qm_drain(c);
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t0, t1, t2;
    pthread_create(&t0, nullptr, capture, nullptr);
    pthread_create(&t1, nullptr, arrive, reinterpret_cast<void*>(uintptr_t{1}));
    pthread_create(&t2, nullptr, arrive, reinterpret_cast<void*>(uintptr_t{2}));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);
    pthread_join(t2, nullptr);

    // Every unit that arrived at c crossed j once.
    assert(g_consumed.load(std::memory_order_relaxed) == 3);
    assert(g_mass_t.load(std::memory_order_relaxed) == 3);
    assert(g_events.load(std::memory_order_relaxed) == 3);
    return 0;
}
