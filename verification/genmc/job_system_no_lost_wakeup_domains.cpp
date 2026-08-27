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
// CALIBRATION. Deleting the fallback scan in submitter() -- returning instead of trying domain 1
// -- makes --check-liveness report a non-terminating spin, which is the check confirming it can
// still see the failure it is looking for.
//
// TWO DOMAINS AND ONE WORKER EACH is the smallest shape that has a remote domain at all, and the
// property does not depend on the width: the submitter's scan visits every domain, so a third
// adds interleavings without adding a case.

#include <atomic>
#include <pthread.h>
#include <cassert>
#include <cstdint>

namespace {

// The job, reachable once published. Standing in for a deque, whose internals are verified
// separately and are not what this protocol turns on.
std::atomic<bool> g_work_available{false};

// The global count, read first by the submitter as the cheap early-out.
std::atomic<int> g_idle_workers{0};

// Per domain: the park word and the count of workers parked on it. Two domains, and the
// submitter belongs to domain 0 while the worker belongs to domain 1 -- the case the single-word
// harness cannot construct.
constexpr int kDomains = 2;
std::atomic<uint32_t> g_domain_seq[kDomains];
std::atomic<int> g_domain_idle[kDomains];

constexpr int kSubmitterDomain = 0;
constexpr int kWorkerDomain = 1;

bool g_worker_got_work = false;

// job_system.hpp, worker loop: announce on BOTH counts before sampling the sequence and taking
// the last look, so a submitter either sees this worker parked or is ordered before that look.
void* worker(void*) {
    g_idle_workers.fetch_add(1, std::memory_order_seq_cst);
    g_domain_idle[kWorkerDomain].fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t seq = g_domain_seq[kWorkerDomain].load(std::memory_order_acquire);

    // find_work_exhaustive: reaches every worker's deque and the injector, so a job published in
    // ANY domain is found here. That is why a skipped wake is not automatically a lost one.
    if (g_work_available.load(std::memory_order_acquire)) {
        g_domain_idle[kWorkerDomain].fetch_sub(1, std::memory_order_relaxed);
        g_idle_workers.fetch_sub(1, std::memory_order_relaxed);
        g_worker_got_work = true;
        return nullptr;
    }

    while (g_domain_seq[kWorkerDomain].load(std::memory_order_acquire) == seq) {
        // Spin. --check-liveness reports an execution where this cannot exit: a worker asleep
        // in one domain while a job sits queued in another.
    }

    g_domain_idle[kWorkerDomain].fetch_sub(1, std::memory_order_relaxed);
    g_idle_workers.fetch_sub(1, std::memory_order_relaxed);
    g_worker_got_work = g_work_available.load(std::memory_order_acquire);
    return nullptr;
}

// job_system.hpp, wake_one_worker(), preceded by the submit that makes the work reachable.
void* submitter(void*) {
    g_work_available.store(true, std::memory_order_release);

    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g_idle_workers.load(std::memory_order_seq_cst) <= 0) return nullptr;   // nobody parked

    // Home domain first: a job pushed here is warm here.
    if (g_domain_idle[kSubmitterDomain].load(std::memory_order_seq_cst) > 0) {
        g_domain_seq[kSubmitterDomain].fetch_add(1, std::memory_order_release);
        return nullptr;
    }

    // THE SCAN, and the reason the property survives domains. Nobody idle at home does not mean
    // nobody is idle: without this the worker in domain 1 is never woken and the job is stranded.
    for (int d = 0; d < kDomains; ++d) {
        const int k = (kSubmitterDomain + 1 + d) % kDomains;
        if (g_domain_idle[k].load(std::memory_order_seq_cst) <= 0) continue;
        g_domain_seq[k].fetch_add(1, std::memory_order_release);
        return nullptr;
    }
    return nullptr;
}

}  // namespace

int main() {
    for (int d = 0; d < kDomains; ++d) {
        g_domain_seq[d].store(0, std::memory_order_relaxed);
        g_domain_idle[d].store(0, std::memory_order_relaxed);
    }

    pthread_t w, s;
    pthread_create(&w, nullptr, worker, nullptr);
    pthread_create(&s, nullptr, submitter, nullptr);
    pthread_join(w, nullptr);
    pthread_join(s, nullptr);

    // The job was published, so the worker must have it: either its last look found it, or a
    // wake reached it across the domain boundary.
    assert(g_worker_got_work);
    return 0;
}
