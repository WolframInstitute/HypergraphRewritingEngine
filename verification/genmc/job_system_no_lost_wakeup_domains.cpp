// GENMC-LINK: job_system
// GENMC-ARGS: --check-liveness --disable-ipr --disable-sr
// GENMC-CALIBRATE: -DHG_PARK_GATE_NO_REMOTE_SCAN
// GENMC-CALIBRATE: -DHG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER
// --disable-ipr --disable-sr: as in job_system_no_lost_wakeup.cpp, the job flag's store and claim are
// unordered writes by design.
//
// A WORKER PARKED IN ANOTHER CACHE DOMAIN IS STILL WOKEN, under the searching-worker throttle.
//
// job_system_no_lost_wakeup.cpp proves the pairing for ONE park word. Workers park on a word PER
// CACHE DOMAIN, so a submitter wakes within its own domain first and scans the others only when no
// worker there is idle. The pairing argument has to survive that scan: the submitter reads a
// DIFFERENT location than the one the worker announced itself on, and a stale read of the remote
// domain's idle count is a way to skip a wake that the single-word harness cannot express. The
// throttle adds the searching count, which is global, so a searcher in one domain makes a submit
// in another skip its wake; the searcher's last look must then find the job across domains.
//
// THE SHAPE. Two workers, both in domain 1, run JobSystem's idle path in a loop (search_begin,
// one steal attempt, search_found or park_unless) until the job is claimed; the claimer releases
// the other worker with wake_all. The submitter runs in domain 0, where nobody parks, is not a
// worker, and has no backlog, so every wake it issues is delivered by the scan.
//
// IT INCLUDES THE PROTOCOL, and drives it: hgcommon/park_gate.hpp, the header JobSystem drives.
//
// CALIBRATION, each reported by --check-liveness as a non-terminating spin:
//   HG_PARK_GATE_NO_REMOTE_SCAN              a submitter whose own domain has nobody idle gives
//                                            up instead of scanning the others
//   HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER  the searcher leaves the count after the barrier
//
// TWO DOMAINS is the smallest shape that has a remote domain at all, and the property does not
// depend on the width: the submitter's scan visits every domain, so a third adds interleavings
// without adding a case. Two workers in the remote domain are the smallest shape in which one is
// parked while the other searches.

#include "hgcommon/park_gate.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

hgcommon::ParkGate::Domain g_domains[2];
hgcommon::ParkGate        g_gate;

std::atomic<bool> g_job{false};
std::atomic<int>  g_claims{0};

bool take() {
    if (!g_job.exchange(false, std::memory_order_acq_rel)) return false;
    g_claims.fetch_add(1, std::memory_order_release);
    return true;
}

void* worker(void* arg) {
    using Search = hgcommon::ParkGate::Search;
    const unsigned home = static_cast<unsigned>(reinterpret_cast<uintptr_t>(arg));
    bool woken = false;
    while (g_claims.load(std::memory_order_acquire) == 0) {
        const Search s = g_gate.search_begin(woken);
        bool got = s != Search::Declined && take();
        if (got) {
            if (s == Search::Counted) g_gate.search_found(home, /*more=*/false);
        } else {
            got = g_gate.park_unless(home, s == Search::Counted, [] { return take(); },
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
    // Pushed first; wake_one's barrier sits between the push and the reads of the counts.
    g_job.store(true, std::memory_order_release);
    g_gate.wake_one(0);   // not a worker, backlog 0; home domain 0, so only the scan reaches 1
    return nullptr;
}

}  // namespace

int main() {
    g_gate.seat(g_domains, 2, /*workers=*/2, /*throttled=*/true);

    pthread_t w0, w1, s;
    pthread_create(&w0, nullptr, worker, reinterpret_cast<void*>(uintptr_t{1}));
    pthread_create(&w1, nullptr, worker, reinterpret_cast<void*>(uintptr_t{1}));
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(s, nullptr);
    pthread_join(w0, nullptr);
    pthread_join(w1, nullptr);

    assert(g_claims.load(std::memory_order_relaxed) == 1);
    return 0;
}
