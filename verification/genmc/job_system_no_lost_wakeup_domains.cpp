// GENMC-LINK: job_system
// GENMC-ARGS: --check-liveness
// A WORKER PARKED IN ANOTHER CACHE DOMAIN IS STILL WOKEN.
//
// job_system_no_lost_wakeup.cpp proves the pairing for ONE park word: a submitter that reads
// idle_workers_ as zero skips the wake, and the seq_cst fence on both sides forbids the submitter
// and a parking worker from both reading stale. That harness models the protocol as it was when
// every worker parked on the same word.
//
// WHAT CHANGED AND WHY IT NEEDS ITS OWN HARNESS. Workers now park on a word PER CACHE DOMAIN, so
// a submitter wakes within its own domain first and scans the others only when no worker there is
// idle. The pairing argument has to survive that scan: the submitter now reads a DIFFERENT
// location than the one the worker announced itself on, and a stale read of the remote domain's
// idle count is a new way to skip a wake that the single-word harness cannot express.
//
// The scan is what makes it sound, and this is the property: a worker parked in domain 1 while
// the only submitter runs in domain 0 is still woken, or else finds the job on its last look.
// Breaking it is a worker asleep with a job queued and nothing left to wake it -- the same
// liveness failure the original harness reports, reached by a route only domains create.
//
// IT INCLUDES THE PROTOCOL, and drives it. This was a transcription of the two sides with their
// memory orders copied out of JobSystem, which verifies the transcription and stops describing
// the engine the moment the engine moves. It now drives hgcommon::ParkGate itself -- the same
// header JobSystem's worker loop and wake_one_worker drive.
//
// CALIBRATION. HG_HARNESS_DEFINES=-DHG_PARK_GATE_NO_REMOTE_SCAN removes the fallback scan from
// ParkGate::wake_one, so a submitter whose own domain has nobody idle gives up instead of looking
// at the others. --check-liveness then reports a non-terminating spin: a worker asleep in
// domain 1 with a job queued in domain 0. The defect is reinstated in the real path, not in a
// copy of it.
//
// TWO DOMAINS AND ONE WORKER EACH is the smallest shape that has a remote domain at all, and the
// property does not depend on the width: the submitter's scan visits every domain, so a third
// adds interleavings without adding a case.

#include "hgcommon/park_gate.hpp"

#include <atomic>
#include <cassert>
#include <pthread.h>

namespace {

// Two domains: the submitter runs in 0, the worker parks in 1. That is the smallest shape in
// which the fallback scan is the only thing that can deliver the wake.
hgcommon::ParkGate::Domain g_domains[2];
hgcommon::ParkGate        g_gate;

std::atomic<bool> g_work_available{false};
bool g_worker_got_work = false;

void* worker(void*) {
    auto look = [] { return g_work_available.load(std::memory_order_acquire); };
    const bool took = g_gate.park_unless(1, look, [] { return true; });
    g_worker_got_work = took || g_work_available.load(std::memory_order_acquire);
    return nullptr;
}

void* submitter(void*) {
    // Pushed first; wake_one's barrier sits between the push and the read of the idle counts.
    g_work_available.store(true, std::memory_order_release);
    g_gate.wake_one(0);   // home domain 0 has nobody idle -- only the scan can reach domain 1
    return nullptr;
}

}  // namespace

int main() {
    g_gate.seat(g_domains, 2);

    pthread_t w, s;
    pthread_create(&w, nullptr, worker, nullptr);
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(w, nullptr);
    pthread_join(s, nullptr);

    assert(g_worker_got_work);
    return 0;
}
