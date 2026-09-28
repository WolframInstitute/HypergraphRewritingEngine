// GPUMC harness: the DEVICE's raw counts from class multiplicities lose no mass and pass none
// twice.
//
// Runs hgcommon/quotient_multiplicity_core.hpp ITSELF (qm_pass, qm_credit, qm_drain) with the
// storage orders of hg_gpu::DeviceQmCtx (gpu/include/hg_gpu/quotient_expansion.hpp): every RMW
// is a relaxed device-scope atomicCAS or atomicExch, every load a volatile read, and the fence
// is __threadfence(). The host's twin, verification/genmc/quotient_mass_match_rendezvous.cpp,
// checks the same protocol with the host's acquire/release orders under RC11; this checks it
// under the memory model the device runs, with each side in its own CTA.
//
//   the capture of j:     record b_j (j is ready), fence, qm_pass(j), drain
//   mass arriving at c:   add to m(c, 0), fence, claim the queued flag, push if won, drain
//   a queued point runs:  pop clears the flag, fence, read m(c, 0), qm_pass every ready match
//
// Property: when every thread has finished, j has passed on exactly the mass that arrived.
// One class c, one match j into t, max_steps 1, arrivals of 1 and 2 units.
//
// __threadfence() is cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_device), and
// that is what the fence below is: a seq_cst fence preceded by the device-scope annotation.
//
// Words are 64-bit: a 32-bit compare-exchange always reports failure in this GPUMC build (see
// run.sh). The device's queued flag is 32-bit and is only exchanged, never compared.
//
// CALIBRATED. -DCALIBRATE_NO_FENCE makes the fence a no-op, and the checker must report mass
// that j never passes on.
#include "hgcommon/quotient_multiplicity_core.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

extern "C" {
void __VERIFIER_memory_scope_device();
void __VERIFIER_thread_local_id(int);
void __VERIFIER_thread_group_id(int);
void __VERIFIER_thread_global_id(int);
void __VERIFIER_thread_kernel_id(int);
}

namespace {

constexpr uint64_t kC = 1;
constexpr uint64_t kT = 2;

struct Match {
    uint32_t id = 0;
    uint64_t to_hash = kT;
    uint32_t rule = 0;
    uint32_t num_consumed = 0, num_produced = 0;
    const uint32_t* consumed_ptr() const { return nullptr; }
    const uint32_t* produced_ptr() const { return nullptr; }
};

Match g_j;
uint64_t g_overlaps = 0;   // qm_overlaps[j]: b_j + 1 once ready
uint64_t g_mass_c = 0;
uint64_t g_mass_t = 0;
uint64_t g_queued_c = 0;
uint64_t g_consumed = 0;
uint64_t g_events = 0;

uint64_t load_dev(uint64_t* a) {           // a volatile read
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, __ATOMIC_RELAXED);
}
uint64_t cas_dev(uint64_t* a, uint64_t expected, uint64_t desired) {   // atomicCAS
    __VERIFIER_memory_scope_device();
    __atomic_compare_exchange_n(a, &expected, desired, /*weak=*/false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
uint64_t exch_dev(uint64_t* a, uint64_t v) {   // atomicExch
    __VERIFIER_memory_scope_device();
    return __atomic_exchange_n(a, v, __ATOMIC_RELAXED);
}
void threadfence() {
#if !defined(CALIBRATE_NO_FENCE)
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

void sat_add(uint64_t* a, uint64_t d) {    // qe_qm_add
    uint64_t old = load_dev(a);
    for (;;) {
        const uint64_t seen = cas_dev(a, old, hgcommon::qm_sat_add(old, d));
        if (seen == old) return;
        old = seen;
    }
}

struct Ctx {
    using Match = ::Match;
    uint64_t queue[2];
    uint32_t n = 0;

    uint32_t max_steps() const { return 1; }
    bool ready(const Match&, uint64_t& b) const {
        const uint64_t v = load_dev(&g_overlaps);
        if (v == 0) return false;
        b = v - 1;
        return true;
    }
    uint64_t mass(uint64_t h, uint32_t) const { return load_dev(h == kC ? &g_mass_c : &g_mass_t); }
    void add_mass(uint64_t h, uint32_t, uint64_t d) { sat_add(h == kC ? &g_mass_c : &g_mass_t, d); }
    uint64_t consumed(const Match&, uint32_t) { return load_dev(&g_consumed); }
    bool advance(const Match&, uint32_t, uint64_t& expected, uint64_t desired) {
        const uint64_t seen = cas_dev(&g_consumed, expected, desired);
        if (seen == expected) return true;
        expected = seen;
        return false;
    }
    void count(uint64_t e, uint64_t) { sat_add(&g_events, e); }
    hgcommon::EventSignatureKeys keys() const { return hgcommon::EVENT_SIG_NONE; }
    uint32_t frame_step(uint64_t, uint32_t f) const { return f; }
    void note_signature(uint64_t) {}
    bool claim_queued(uint64_t, uint32_t) { return exch_dev(&g_queued_c, 1) == 0; }
    void push(uint64_t h, uint32_t) { queue[n++] = h; }
    bool pop(uint64_t& h, uint32_t& d) {
        if (n == 0) return false;
        h = queue[--n];
        d = 0;
        exch_dev(&g_queued_c, 0);
        return true;
    }
    template <class F> void for_each_match(uint64_t h, F&& f) { if (h == kC) f(g_j); }
    void fence() { threadfence(); }
};

void ids(int g) {
    __VERIFIER_thread_global_id(g); __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(g);  __VERIFIER_thread_kernel_id(0);
}

// qe_capture_multiplicity: j ready through the overlaps map's insert, then the cascade.
void* capture(void*) {
    ids(0);
    cas_dev(&g_overlaps, 0, 1);   // b_j = 0
    Ctx c;
    c.fence();
    hgcommon::qm_pass(c, g_j, kC, 0);
    hgcommon::qm_drain(c);
    return nullptr;
}

void* arrive1(void*) {
    ids(1);
    Ctx c;
    hgcommon::qm_credit(c, kC, 0, 1);
    hgcommon::qm_drain(c);
    return nullptr;
}

void* arrive2(void*) {
    ids(2);
    Ctx c;
    hgcommon::qm_credit(c, kC, 0, 2);
    hgcommon::qm_drain(c);
    return nullptr;
}

}  // namespace

int main() {
    pthread_t a, b, d;
    pthread_create(&a, nullptr, capture, nullptr);
    pthread_create(&b, nullptr, arrive1, nullptr);
    pthread_create(&d, nullptr, arrive2, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    pthread_join(d, nullptr);
    assert(g_consumed == 3 && "mass that arrived was not passed on exactly once");
    assert(g_mass_t == 3);
    assert(g_events == 3);
    return 0;
}
