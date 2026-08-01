// GENMC-ARGS: --check-liveness
//
// GenMC harness: a worker that parks is always woken, so no submitted job is left unclaimed.
//
// THIS ONE IS A TRANSCRIPTION, NOT AN INCLUDE. Every other harness in this directory includes the
// engine header and calls its own functions. This one cannot: the protocol lives inside
// JobSystem's worker loop, which spawns its own threads and blocks in park_if_equal, and neither
// std::thread's libstdc++ machinery nor syscall(SYS_futex, ...) is something the checker can run.
// So the two sides are transcribed here, with the SAME memory orders as
// job_system/include/job_system/job_system.hpp. Those orders are the entire content of the
// property -- if they drift there and not here, this harness verifies something the engine no
// longer does. Re-read both sides together when either changes.
//
// WHAT IS BEING PROVED. wake_one_worker() SKIPS the wake when idle_workers_ reads zero, and the
// header states why that is safe:
//
//     "The two sides are a store-then-load on different locations: the submitter pushes then
//      reads idle_workers_; a parking worker increments idle_workers_ then looks for work. At
//      least one of them must observe the other, so a worker that this call skips is a worker
//      whose own final look for work happens after the push and therefore finds it."
//
// That is store buffering, and "at least one must observe the other" is a guarantee ONLY under
// sequential consistency -- under acquire/release both sides can read stale and both conclude
// there is nothing to do. Both sides use seq_cst, so the claim should hold; this harness is what
// makes that a checked fact rather than a stated one. A failure is a worker asleep with a job
// queued and nobody left to wake it: the engine hangs.
//
// HOW PARKING IS REPRESENTED. park_if_equal blocks while the word still holds the value the
// caller sampled. That is transcribed as a spin on the same condition, which is what the futex
// does, and --check-liveness reports a spin that can never exit. A lost wakeup therefore shows up
// as a liveness violation rather than as a failed assertion.
//
// WHAT IS BOUNDED. One worker, one submitter, one job. Two workers contending for one job is a
// different question (which one claims it) and is not what the wake protocol is about.

#include <pthread.h>
#include <atomic>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"

namespace {

// Transcribed from JobSystem. work_available_ stands for "a job is reachable in some deque"; the
// deque itself is verified separately and its internals are not what this protocol turns on.
std::atomic<bool> g_work_available{false};
std::atomic<int> g_idle_workers{0};
std::atomic<uint32_t> g_work_seq{0};

bool g_worker_got_work = false;

// job_system.hpp:232-245, worker loop.
void* worker(void*) {
    // "Announce the park BEFORE sampling the counter and taking the last look for work, so a
    // submitter either sees this worker as parked and wakes it, or is ordered before that last
    // look and is found by it."
    g_idle_workers.fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t seq = g_work_seq.load(std::memory_order_acquire);

    if (g_work_available.load(std::memory_order_acquire)) {
        g_idle_workers.fetch_sub(1, std::memory_order_relaxed);
        g_worker_got_work = true;
        return nullptr;
    }

    // park_if_equal(work_seq_, seq): block while the word still reads what was sampled. If the
    // submitter has already bumped it, this falls through immediately -- that is what makes a
    // wake issued before the park not a lost one.
    while (g_work_seq.load(std::memory_order_acquire) == seq) {
        // Spin. --check-liveness reports an execution in which this can never exit, which is
        // exactly a worker asleep with a job queued.
    }

    g_idle_workers.fetch_sub(1, std::memory_order_relaxed);

    // Woken: the job is there to be taken.
    g_worker_got_work = g_work_available.load(std::memory_order_acquire);
    return nullptr;
}

// job_system.hpp:124-128, wake_one_worker(), preceded by the submit that makes work reachable.
void* submitter(void*) {
    g_work_available.store(true, std::memory_order_release);

    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g_idle_workers.load(std::memory_order_seq_cst) <= 0) return nullptr;   // skip the wake
    g_work_seq.fetch_add(1, std::memory_order_release);
    return nullptr;
}

}  // namespace

int main() {
    pthread_t w, s;
    pthread_create(&w, nullptr, worker, nullptr);
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(w, nullptr);
    pthread_join(s, nullptr);

    // Reaching here at all is most of the property: a lost wakeup leaves the worker spinning and
    // the join never completes, which the liveness check reports.
    //
    // The rest is that the worker did not wake to an empty world -- it either took the job on its
    // last look or was woken to find it still there.
    assert(g_worker_got_work);
    return 0;
}
