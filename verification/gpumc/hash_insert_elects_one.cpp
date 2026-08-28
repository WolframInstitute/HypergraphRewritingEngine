// GPUMC harness: two threads inserting the same key elect exactly one inserter, and it is the
// one whose value is stored.
//
// Runs hgcommon/hash_insert_core.hpp ITSELF -- the same hash_insert_claim body
// gpu/include/hg_gpu/hash_table.hpp drives -- rather than a model of it.
//
// WHY GPUMC RATHER THAN GenMC. The table is shared between CTAs and every access is at DEVICE
// scope. Scoped-RC11 admits behaviours RC11 does not, so GenMC would check a program the device
// does not run and call the map proved.
//
// WHAT IS BEING PROVED, and why both halves matter. `inserted` is not a courtesy flag: the device
// marks an event CANONICAL on it and points every later event at the STORED value, and
// qe.applied gates an APPLICATION on it. So two things have to hold together --
//
//   exactly one of the threads is told inserted, and
//   the value that is stored is that thread's
//
// -- because a run in which one thread reports inserted while another's value stands gives one
// signature two canonical events, and neither half alone forbids it.
//
// THE SHAPE THAT BREAKS IT. Electing on the KEY exchange instead of the value exchange. That
// elects one thread too, but a different one: the key winner can lose the value exchange, and
// then it reports inserted while carrying a stranger's value while the value's owner reports
// not-inserted. -DCALIBRATE_ELECT_ON_KEY makes claim_key's winner the inserter and the checker
// must report the stored value disagreeing with the elected thread's.
//
// THE VALUES ARE 64-BIT BECAUSE THE CHECKER CANNOT DO A 32-BIT ONE. Measured: with uint32_t
// values, every compare-exchange on the value word reports failure while reading exactly the
// expected value -- the trace shows the CAS read and no CAS write -- so no execution completes
// and the run reports "the key was inserted by no thread". The key CAS beside it, on a uint64_t,
// succeeds under the same orders. The device's values are 32-bit ids; the width is the checker's
// constraint and not the protocol's, and the election does not depend on it. A future harness
// that sees a CAS fail against its own expected value should suspect this before the code.
//
// THE BOUND. One slot, two threads, the same key, distinct values. That is the smallest shape
// containing the election: with one slot the probe cannot wander, so what is left is exactly the
// two exchanges and which thread wins each. Distinct values are what make "whose value stood"
// answerable at all -- most real callers offer a constant presence marker, under which every
// thread's value matches and the question cannot be posed.
#include "hgcommon/hash_insert_core.hpp"

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

constexpr uint32_t kCap = 1;
constexpr uint32_t kThreads = 2;
constexpr uint64_t kKey = 7;
constexpr uint32_t kEmpty = 0;
constexpr uint64_t kUnpublished = ~uint64_t{0};

uint64_t g_keys[kCap];
uint64_t g_values[kCap];

bool     g_inserted[kThreads];
uint64_t g_offered[kThreads];

uint64_t kload(uint64_t* a) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, __ATOMIC_ACQUIRE);
}
bool kcas(uint64_t* a, uint64_t* expected, uint64_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
uint64_t vload(uint64_t* a) {
    __VERIFIER_memory_scope_device();
    return __atomic_load_n(a, __ATOMIC_ACQUIRE);
}
bool vcas(uint64_t* a, uint64_t* expected, uint64_t desired) {
    __VERIFIER_memory_scope_device();
    return __atomic_compare_exchange_n(a, expected, desired, /*weak=*/false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

struct Ops {
    uint64_t value;
    uint64_t stood = 0;

    uint32_t capacity() const { return kCap; }
    uint32_t initial_slot() const { return 0; }
    uint32_t next_slot(uint32_t s) const { return (s + 1) % kCap; }

    hgcommon::KeyState key_state(uint32_t s) const {
        const uint64_t cur = kload(&g_keys[s]);
        if (cur == kKey)   return hgcommon::KeyState::Ours;
        if (cur == kEmpty) return hgcommon::KeyState::Empty;
        return hgcommon::KeyState::Other;
    }

    bool claim_key(uint32_t s) {
        uint64_t expected = kEmpty;
        const bool won = kcas(&g_keys[s], &expected, kKey);
#if defined(CALIBRATE_ELECT_ON_KEY)
        if (won) elected_on_key = true;
#endif
        return won;
    }

    hgcommon::InsertOutcome offer_value(uint32_t s) {
#if defined(CALIBRATE_ELECT_ON_KEY)
        // THE DEFECT: the key winner is called the inserter. It still offers its value, and it
        // still loses that exchange to whoever got there first -- so the thread reporting
        // inserted and the thread whose value stands are two different threads.
        uint64_t expect_v = kUnpublished;
        if (!vcas(&g_values[s], &expect_v, value)) stood = expect_v;
        return elected_on_key ? hgcommon::InsertOutcome::Inserted
                              : hgcommon::InsertOutcome::Present;
#else
        uint64_t expect_v = kUnpublished;
        if (vcas(&g_values[s], &expect_v, value))
            return hgcommon::InsertOutcome::Inserted;
        stood = expect_v;
        return hgcommon::InsertOutcome::Present;
#endif
    }

#if defined(CALIBRATE_ELECT_ON_KEY)
    bool elected_on_key = false;
#endif
};

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    __VERIFIER_thread_global_id(static_cast<int>(id));
    __VERIFIER_thread_local_id(0);
    __VERIFIER_thread_group_id(static_cast<int>(id));
    __VERIFIER_thread_kernel_id(0);

    // Distinct, and neither is the unpublished sentinel, so "whose value stood" has an answer.
    const uint64_t value = static_cast<uint64_t>(id) + 1;
    Ops ops{value};
    const hgcommon::InsertOutcome outcome = hgcommon::hash_insert_claim(ops);

    g_offered[id] = value;
    g_inserted[id] = (outcome == hgcommon::InsertOutcome::Inserted);
    return nullptr;
}

}  // namespace

int main() {
    for (uint32_t i = 0; i < kCap; ++i) { g_keys[i] = kEmpty; g_values[i] = kUnpublished; }

    pthread_t t[kThreads];
    for (long i = 0; i < static_cast<long>(kThreads); ++i)
        pthread_create(&t[i], nullptr, worker, reinterpret_cast<void*>(i));
    for (uint32_t i = 0; i < kThreads; ++i) pthread_join(t[i], nullptr);

    uint32_t winners = 0;
    long winner = -1;
    for (uint32_t i = 0; i < kThreads; ++i)
        if (g_inserted[i]) { ++winners; winner = i; }

    assert(winners == 1 && "the key was inserted by no thread, or by more than one");
    // The elected thread must be the one whose value is stored. Without this the count alone is
    // satisfied by electing on the key exchange, which is a different thread.
    assert(g_values[0] == g_offered[winner] &&
           "the thread told it inserted is not the one whose value stands");
    return 0;
}
