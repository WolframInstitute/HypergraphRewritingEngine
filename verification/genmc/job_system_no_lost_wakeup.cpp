// GENMC-LINK: job_system
// GENMC-ARGS: --check-liveness --disable-ipr --disable-sr
// GENMC-CALIBRATE: -DHG_PARK_GATE_WEAK_ORDERS
// GENMC-CALIBRATE: -DHG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER
// GENMC-CALIBRATE: -DHG_PARK_GATE_SEARCH_LEAVE_LATE
//
// --disable-ipr --disable-sr: the submitter's store of the job flag and a worker's claiming exchange are two
// writes with no order between them, which is the push and the claim themselves. GenMC reports
// that as "unordered writes" and stops when in-place revisiting or symmetry reduction is on.
//
// GenMC harness: a worker that parks is always woken, so no submitted job is left unclaimed --
// under the searching-worker throttle, where a submit skips its wake while a worker searches.
//
// IT INCLUDES THE PROTOCOL, and drives it: hgcommon/park_gate.hpp, the header JobSystem's worker
// loop and submit path call. The protocol is a unit and not checked inside a running JobSystem
// because starting one prunes to 8,952 lines and segfaulted GenMC v0.17.0.
//
// WHAT IS BEING PROVED. wake_one SKIPS the wake when the idle count reads zero, or when a worker
// is searching and the submitter's backlog is at most twice the searchers. Each skip is safe only
// because of the order on both sides:
//
//     the submitter  publishes the job, passes the barrier, reads the idle and searching counts
//     the worker     leaves the searching count, announces itself idle, passes the barrier,
//                    takes one last look for work
//
// Each side writes one location and reads the other, so at least one must observe the other -- a
// guarantee under sequential consistency and NOT under acquire/release. A failure is a worker
// asleep with a job queued and nobody left to wake it: the engine hangs.
//
// THE SHAPE. Two workers run JobSystem's idle path in a loop -- search_begin, one steal attempt,
// then search_found or park_unless -- until the job is claimed; the claimer releases the other
// worker with wake_all. The submitter is a thread that is not a worker, so its backlog is zero.
// The interleaving the throttle adds is one worker parked (idle) and the other counted as a
// searcher when the submit reads the counts: the submit skips its wake, and the searcher must
// find the job, by its steal or by its last look. Two workers searching at once also exercises
// the bound (the second is Declined and goes to the last look). search_found is called with
// `more` false: its wake adds a worker and no property rests on it.
//
// HOW PARKING IS REPRESENTED. Under HG_PARK_VERIFICATION park_if_equal spins on the word instead
// of entering a futex, which is a conforming park -- the contract permits spurious return and
// every caller re-tests -- and it is something the checker can see. A wake that never comes is
// then a loop that never exits, which --check-liveness reports.
//
// CALIBRATED, three ways, each reported as a non-terminating spinloop:
//   HG_PARK_GATE_WEAK_ORDERS                 both sides at release/acquire, no barrier
//   HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER  the searcher leaves the count after the barrier, so
//                                            the submit can read it as searching while its last
//                                            look misses the job (RC11 only)
//   HG_PARK_GATE_SEARCH_LEAVE_LATE           the searcher leaves the count after its last look
//
// WHAT IS BOUNDED. Two workers, one submitter, one job, one domain. The per-domain fallback is the
// subject of the _domains harness beside this one.
#include "hgcommon/park_gate.hpp"

#include <atomic>
#include <cassert>

#include <pthread.h>


namespace {

hgcommon::ParkGate::Domain g_domains[1];
hgcommon::ParkGate        g_gate;

// Stands for "a job is reachable in some deque"; exchange is the claim. The deque itself is
// verified separately and its internals are not what this protocol turns on.
std::atomic<bool> g_job{false};
std::atomic<int>  g_claims{0};

bool take() {
    if (!g_job.exchange(false, std::memory_order_acq_rel)) return false;
    g_claims.fetch_add(1, std::memory_order_release);
    return true;
}

void* worker(void*) {
    using Search = hgcommon::ParkGate::Search;
    bool woken = false;
    while (g_claims.load(std::memory_order_acquire) == 0) {
        const Search s = g_gate.search_begin(woken);
        bool got = s != Search::Declined && take();
        if (got) {
            if (s == Search::Counted) g_gate.search_found(0, /*more=*/false);
        } else {
            // The last look is take(); keep_waiting is false once the job has been claimed.
            got = g_gate.park_unless(0, s == Search::Counted, [] { return take(); },
                                     [] { return g_claims.load(std::memory_order_acquire) == 0; });
        }
        if (got) {
            g_gate.wake_all();   // release the other worker
            break;
        }
        woken = true;
    }
    return nullptr;
}

void* submitter(void*) {
    // The push comes first: wake_one's barrier sits between it and the reads of the counts.
    g_job.store(true, std::memory_order_release);
    g_gate.wake_one(0);  // not a worker: backlog 0
    return nullptr;
}

}  // namespace

int main() {
    g_gate.seat(g_domains, 1, /*workers=*/2, /*throttled=*/true);

    pthread_t w0, w1, s;
    pthread_create(&w0, nullptr, worker, nullptr);
    pthread_create(&w1, nullptr, worker, nullptr);
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(s, nullptr);
    // A lost wakeup leaves the job unclaimed with both workers spinning in their parks, which the
    // liveness check reports.
    pthread_join(w0, nullptr);
    pthread_join(w1, nullptr);

    assert(g_claims.load(std::memory_order_relaxed) == 1);
    return 0;
}
