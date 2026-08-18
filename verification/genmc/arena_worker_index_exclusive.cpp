// GenMC harness: the arena's worker-index registry never gives one index to two live holders.
//
// WHY THIS PROPERTY. ConcurrentHeterogeneousArena's fast path is a PER-WORKER bump cursor, and
// allocate_local bumps a PLAIN, non-atomic offset:
//     size_t new_offset = aligned + size;
//     if (new_offset <= c.capacity) { c.offset = new_offset; ... }
// That is safe only because cursors_[wi] is private to one thread. If two live threads ever held
// the same wi they would bump one cursor without synchronisation and could be handed the SAME
// address -- and two callers holding one Node* is sufficient to orphan a LockFreeList chain,
// which is the shape lock_free_list_completeness takes as a precondition rather than tests.
// This harness is where that precondition is discharged.
//
// WHY IT NEEDED A DIFFERENT APPROACH. verification/genmc/README.md says of the arena.hpp
// substitutions: "no harness in this directory makes a claim about [the worker-index allocator].
// If a harness is ever written *for* that allocator, it cannot use these substitutions and needs
// a different approach." Under HG_VERIFICATION arena_worker_index() is replaced by a monotonic
// counter that never releases and never reuses, precisely so the registry -- an aggregate global
// of MAX_ARENA_WORKERS atomics -- never enters the module. So this harness does not call
// arena_worker_index(). It constructs ArenaWorkerRegistry as a LOCAL, which is the real shipped
// code and not a model of it, and drives acquire()/release() directly.
//
// WHY THE BOUND IS OVERRIDDEN. acquire() scans every slot, and each slot is an atomic location,
// so enumerating 256 of them enumerates the scan rather than the property. HG_MAX_ARENA_WORKERS
// is set to 2 here: with two threads and two slots, exclusivity is the tightest it can be -- any
// sharing is immediately visible, and the exhaustion path (-1) is also on the interleaving.
//
// WHAT IS BOUNDED. Two threads, one acquire each, no release, two slots. Neither may be -1 (two
// slots exist for two threads) and the two indices must differ. This is a statement about every
// execution of THIS program under RC11.
//
// CALIBRATION. Replacing acquire()'s compare_exchange_strong with a plain load-then-store
// (read in_use_[i], if false store true and return i) must make this harness report the safety
// violation -- that is exactly the check-then-act the CAS exists to close.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
//
// Build/run: verification/genmc/run.sh arena_worker_index_exclusive

#define HG_MAX_ARENA_WORKERS 2

#include <pthread.h>
#include <cassert>

#include "genmc_support.hpp"
#include "hypergraph/arena.hpp"

namespace {

hypergraph::ArenaWorkerRegistry* g_registry;
int g_index[2];

void* worker(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    // Acquire and HOLD. Releasing here would let the two lifetimes not overlap, and a registry
    // that hands the same index to two threads in sequence is correct -- it is simultaneous
    // holding that breaks the private-cursor invariant.
    g_index[id] = g_registry->acquire();
    return nullptr;
}

}  // namespace

int main() {
    // A LOCAL, not the arena_worker_registry() singleton: the singleton is a function-local
    // static whose aggregate-global materialisation is the thing the substitution exists to
    // avoid. The type and its logic are the shipped ones either way.
    hypergraph::ArenaWorkerRegistry registry;
    g_registry = &registry;
    g_index[0] = g_index[1] = -2;

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, worker, reinterpret_cast<void*>(0L));
    pthread_create(&t1, nullptr, worker, reinterpret_cast<void*>(1L));
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // Two slots exist for two threads, so neither may be turned away.
    assert(g_index[0] >= 0);
    assert(g_index[1] >= 0);
    assert(g_index[0] < HG_MAX_ARENA_WORKERS);
    assert(g_index[1] < HG_MAX_ARENA_WORKERS);

    // THE INVARIANT allocate_local RESTS ON: distinct live holders, distinct cursors.
    assert(g_index[0] != g_index[1]);
    return 0;
}
