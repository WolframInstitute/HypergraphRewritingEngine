// GENMC-ARGS: --disable-estimation
// GENMC-CALIBRATE: -DHG_CALIBRATE_RUNSIG_UNGATED
// GENMC-CALIBRATE: -DCALIBRATE_KEY_BEFORE_STEP
//
// GenMC harness: a match's cached run-signature key (hgcommon::qr_cached_key, qr_cache_key) is
// read only for the output step it was claimed for.
//
// THE PROTOCOL. Under event signature keys the replay claims each application's event class on
// its signature values (Hypergraph::claim_replay_event). The values are a function of the match,
// its from class and the output step, so the claimed key is kept on the match for one output
// step: the step cell is set once from QR_NO_STEP by compare-and-swap, and only the thread whose
// swap succeeded stores the key (release). A reader loads the key (acquire), then the step, and
// uses the key only when the step is its own. The core is the shared one; the cells are the
// host's (claim_replay_event's Cells in hypergraph/src/hypergraph.cpp): step relaxed, key
// acquire/release.
//
// THE SHAPE. One match. Thread A applies it at output step 1, thread B at output step 2; each
// claims the class key for its step (101 and 202 here, standing for keyed_claim's result) when
// the cache does not answer, and offers it to the cache. Main then reads both steps.
//
// THE PROPERTY. Every key returned for step s is the key of step s: in either thread, and in main
// after both joined.
//
// CALIBRATED. -DHG_CALIBRATE_RUNSIG_UNGATED (in the core) stores the key whether or not the swap
// succeeded, so the step-2 key can land under step 1. -DCALIBRATE_KEY_BEFORE_STEP makes the
// cells' swap store the key first and then try the step, so a reader can read a key whose step
// was never set by its writer.
//
// THE MEMORY ORDERS ARE NOT WHAT THE PROPERTY RESTS ON. The step cell takes one value besides
// QR_NO_STEP over its life, so a reader that reads the key with a stale step misses the cache
// and claims; the property holds with every access relaxed. It rests on the gate: one writer,
// the one that set the step.
#include "genmc_support.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <pthread.h>

#include "hypergraph/atomic_compat.hpp"
#include "hgcommon/quotient_replay_core.hpp"

namespace {

uint64_t g_key = 0;
uint32_t g_step = hgcommon::QR_NO_STEP;

struct Cells {
    uint64_t pending = 0;   // CALIBRATE_KEY_BEFORE_STEP: the key a swap stores before trying
    uint32_t step_load() const {
        return hgcommon::atomic_ref<uint32_t>(g_step).load(std::memory_order_relaxed);
    }
    bool step_cas(uint32_t expected, uint32_t desired) {
#if defined(CALIBRATE_KEY_BEFORE_STEP)
        hgcommon::atomic_ref<uint64_t>(g_key).store(pending, std::memory_order_release);
#endif
        return hgcommon::atomic_ref<uint32_t>(g_step).compare_exchange_strong(
            expected, desired, std::memory_order_relaxed);
    }
    uint64_t key_load() const {
        return hgcommon::atomic_ref<uint64_t>(g_key).load(std::memory_order_acquire);
    }
    void key_store(uint64_t key) {
        hgcommon::atomic_ref<uint64_t>(g_key).store(key, std::memory_order_release);
    }
};

uint64_t class_key(uint32_t step) { return 100u * step + step; }

// claim_replay_event's use of the cache: a cached key, or the claimed one, offered to the cache.
uint64_t key_for(uint32_t step) {
    Cells c;
    uint64_t k = 0;
    if (hgcommon::qr_cached_key(c, step, k)) return k;
    k = class_key(step);
#if defined(CALIBRATE_KEY_BEFORE_STEP)
    c.pending = k;
#endif
    hgcommon::qr_cache_key(c, step, k);
    return k;
}

void* apply_step1(void*) {
    assert(key_for(1) == class_key(1) && "step 1 read another step's key");
    return nullptr;
}

void* apply_step2(void*) {
    assert(key_for(2) == class_key(2) && "step 2 read another step's key");
    return nullptr;
}

}  // namespace

int main() {
    pthread_t a, b;
    pthread_create(&a, nullptr, apply_step1, nullptr);
    pthread_create(&b, nullptr, apply_step2, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    assert(key_for(1) == class_key(1) && "step 1 read another step's key after the run");
    assert(key_for(2) == class_key(2) && "step 2 read another step's key after the run");
    return 0;
}
