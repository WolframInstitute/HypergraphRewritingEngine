// GENMC-LINK: job_system
// GENMC-ARGS: --check-liveness
//
// GenMC harness: a worker that parks is always woken, so no submitted job is left unclaimed.
//
// IT INCLUDES THE PROTOCOL, and drives it. Until hgcommon/park_gate.hpp existed this was a
// TRANSCRIPTION -- the two sides copied out of JobSystem's worker loop with the same memory
// orders, and a comment asking the next reader to re-read both sides together whenever either
// changed. A harness maintained that way verifies the transcription, and stops describing the
// engine the moment the engine moves. Driving the real ParkGate cannot drift.
//
// It could not include the old shape for a real reason: the protocol lived inside a worker loop
// that spawns its own threads and blocks in a futex. That is still true of JobSystem -- measured,
// constructing one prunes to 2,659 lines and verifies, starting one prunes to 8,952 and segfaults
// GenMC v0.17.0 -- which is why the protocol is a unit and not checked in place.
//
// WHAT IS BEING PROVED. wake_one SKIPS the wake when the idle count reads zero, and that is safe
// only because of what the two sides do in which order:
//
//     the submitter  publishes the job, then reads the idle count
//     the worker     announces itself idle, then takes one last look for work
//
// Each writes one location and reads the other, so at least one must observe the other -- a
// guarantee under sequential consistency and NOT under acquire/release, where both may read stale
// and both conclude there is nothing to do. A failure is a worker asleep with a job queued and
// nobody left to wake it: the engine hangs.
//
// HOW PARKING IS REPRESENTED. Under HG_PARK_VERIFICATION park_if_equal spins on the word instead
// of entering a futex, which is a conforming park -- the contract permits spurious return and
// every caller re-tests -- and it is something the checker can see. A wake that never comes is
// then a loop that never exits, which --check-liveness reports.
//
// CALIBRATED. The property rests entirely on the seq_cst pairing, so that is what the calibration
// removes: HG_HARNESS_DEFINES=-DHG_PARK_GATE_WEAK_ORDERS drops both sides to release/acquire and
// the checker reports a non-terminating spinloop. The counterexample is store buffering exactly as
// described above -- the submitter reads the idle count as its initial zero while the worker reads
// the work flag as its initial zero, both stale -- so the submitter skips the wake and the worker
// parks with the job queued.
//
// WHAT IS BOUNDED. One worker, one submitter, one job, one domain. Two workers contending for one
// job is a different question (which one claims it) and is not what the wake protocol is about;
// the per-domain fallback is the subject of the _domains harness beside this one.
#include "hgcommon/park_gate.hpp"

#include <atomic>
#include <cassert>

#include <pthread.h>


namespace {

hgcommon::ParkGate::Domain g_domains[1];
hgcommon::ParkGate        g_gate;

// Stands for "a job is reachable in some deque". The deque itself is verified separately and its
// internals are not what this protocol turns on.
std::atomic<bool> g_work_available{false};

bool g_worker_got_work = false;

void* worker(void*) {
    // look() is the worker's last exhaustive search; keep_waiting() is false once it must leave.
    // Neither is part of the protocol, which is why the gate takes them from the caller.
    auto look = [] { return g_work_available.load(std::memory_order_acquire); };
    const bool took = g_gate.park_unless(0, look, [] { return true; });

    // Either it found the job on its last look, or it parked and was woken to find it there.
    g_worker_got_work = took || g_work_available.load(std::memory_order_acquire);
    return nullptr;
}

void* submitter(void*) {
    // The push comes first: wake_one's barrier sits between it and the read of the idle counts.
    g_work_available.store(true, std::memory_order_release);
    g_gate.wake_one(0);
    return nullptr;
}

}  // namespace

int main() {
    g_gate.seat(g_domains, 1);

    pthread_t w, s;
    pthread_create(&w, nullptr, worker, nullptr);
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(w, nullptr);
    pthread_join(s, nullptr);

    // Reaching here at all is most of the property: a lost wakeup leaves the worker spinning and
    // the join never completes, which the liveness check reports.
    //
    // The rest is that the worker did not wake to an empty world.
    assert(g_worker_got_work);
    return 0;
}
