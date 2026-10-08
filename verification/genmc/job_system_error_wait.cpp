// GENMC-LINK: job_system
// GENMC-ARGS: --unroll=64 --disable-estimation
// GENMC-DEFINES: -DHG_JOB_QUEUE_CAPACITY=4 -DHG_JOB_INJECTOR_CAPACITY=16 -DHG_JOB_MAX_POOLS=4 -DHG_JOB_CHUNK_SLOTS=4 -DHG_JOB_ERROR_MESSAGE_CAP=32
// GENMC-CALIBRATE: -DHG_ERROR_WAIT_EARLY
//
// GenMC harness: after a job fails, wait_for_completion returns only once no worker is inside a
// job.
//
// WHAT IS BEING PROVED. A failing job latches the error and stops every worker; each worker
// discards what its queues hold and leaves. The caller reads the results as soon as the wait
// returns, so a wait that returned while a worker was still inside a job would hand it results
// that job is still writing. The wait therefore ends on the count of workers that have left. The
// failing job writes a flag after it latches the error; the property is that the flag is set
// when the wait returns.
//
// HOW A JOB FAILS. The checker interprets no C++ exception, so the failing job calls
// fail_current_job_for_verification(), the latch run_job's catch clauses reach on a
// CapacityExhausted throw (JobSystem::latch_error), compiled only under HG_VERIFICATION.
//
// CALIBRATED. Returning from the wait as soon as the error is seen, without waiting for the
// workers to leave (HG_HARNESS_DEFINES=-DHG_ERROR_WAIT_EARLY, read in wait_for_completion), makes
// the checker report the assertion below.
//
// WHAT IS BOUNDED. One worker and one job. With two workers and two jobs the checker started
// 827,000 executions in 30 minutes and completed none. The queues are 4 and 16 entries, with 4
// pools of 4-slot chunks and a 32-byte error message; --unroll=64 bounds the loops, and run.sh
// fails the run if no execution completes or every one was cut at that bound.
#include "job_system/job_system.hpp"

#include <atomic>
#include <cassert>

namespace {

enum class JobKind { Work };

std::atomic<int> g_after_latch{0};

}  // namespace

int main() {
    job_system::JobSystem<JobKind> js(1);
    js.start();
    js.submit(job_system::make_job<JobKind>([&js] {
        js.fail_current_job_for_verification();
        g_after_latch.store(1, std::memory_order_relaxed);
    }, JobKind::Work));
    js.wait_for_completion();
    assert(js.get_error_type() == job_system::ErrorType::CapacityExhausted);
    assert(g_after_latch.load(std::memory_order_relaxed) == 1 &&
           "the wait returned while the failing job was still running");
    js.shutdown();
    return 0;
}
