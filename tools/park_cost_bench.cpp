// What a park/unpark handoff costs, and what it costs to wake nobody.
//
// WHY THIS EXISTS. The engine parks idle workers rather than spinning them, and the primitive it
// parks on is chosen per platform: raw SYS_futex on Linux, WaitOnAddress on Windows,
// os_sync_wait_on_address on macOS, with std::atomic::wait and a shared condition variable as
// fallbacks (see hgcommon/park.hpp). Those are the same ALGORITHM -- an address-based wait -- so
// a scaling difference between two platforms cannot be attributed to one of them taking a lock
// without measuring what the primitive actually costs. Measured on this box, the same commit
// built for Windows and for Linux is indistinguishable on one thread (57.965 ms against 58.169
// ms on wpp depth 6) and diverges as threads are added (17.0 against 15.9 at eight), which is a
// per-handoff cost and nothing else. This reports that cost directly instead of inferring it.
//
// THE CONTROL IS PART OF THE MEASUREMENT. A number in nanoseconds means nothing without a floor
// to read it against, so the same ping-pong runs twice: once handing off through a spin on the
// atomic, and once handing off through park/unpark. The spin arm never enters the kernel, so it
// measures the cache-line transfer between two cores and nothing more. The difference between
// the arms is the primitive; the spin arm alone says whether the timer resolves the scale at
// all. An instrument that cannot separate those two is not evidence.
//
// The third arm is the case a busy job system hits most: unpark_one on an address where nothing
// is parked. A wake that finds no waiter is pure overhead, and whether it costs a syscall or a
// predictable branch differs per backend.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "hgcommon/park.hpp"

namespace {

const char* backend_name() {
    switch (hgcommon::park_backend()) {
        case hgcommon::ParkBackend::Futex:         return "futex (SYS_futex, Linux)";
        case hgcommon::ParkBackend::WaitOnAddress: return "WaitOnAddress (Windows)";
        case hgcommon::ParkBackend::OsSync:        return "os_sync_wait_on_address (macOS)";
        case hgcommon::ParkBackend::StdAtomicWait: return "std::atomic::wait";
        case hgcommon::ParkBackend::CondVar:       return "condition_variable (takes a lock)";
    }
    return "unknown";
}

double ns_per(std::chrono::steady_clock::duration d, uint64_t n) {
    return std::chrono::duration<double, std::nano>(d).count() / static_cast<double>(n);
}

// Hand a token back and forth `rounds` times. Each round is TWO handoffs, one per direction.
// `park` selects whether the waiting side sleeps on the address or spins on it; everything else
// about the two arms is identical, so their difference is the primitive and not the loop.
double pingpong(uint64_t rounds, bool park) {
    std::atomic<uint32_t> turn{0};
    std::atomic<bool> ready{false};

    auto side = [&](uint32_t mine, uint32_t theirs) {
        for (uint64_t i = 0; i < rounds; ++i) {
            // Spurious wakes are permitted by every backend, so the predicate is re-tested in a
            // loop rather than assumed to hold once the wait returns.
            while (turn.load(std::memory_order_acquire) != mine) {
                if (park) hgcommon::park_if_equal(turn, theirs);
                else      std::this_thread::yield();
            }
            turn.store(theirs, std::memory_order_release);
            if (park) hgcommon::unpark_one(turn);
        }
    };

    std::thread other([&] {
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        side(1, 0);
    });

    ready.store(true, std::memory_order_release);
    const auto t0 = std::chrono::steady_clock::now();
    side(0, 1);
    const auto t1 = std::chrono::steady_clock::now();
    other.join();
    return ns_per(t1 - t0, rounds * 2);   // two handoffs per round
}

// A wake that finds nothing parked. No thread ever waits on this address.
double lone_wake(uint64_t n) {
    std::atomic<uint32_t> addr{0};
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < n; ++i) hgcommon::unpark_one(addr);
    const auto t1 = std::chrono::steady_clock::now();
    return ns_per(t1 - t0, n);
}

}  // namespace

int main(int argc, char** argv) {
    const uint64_t rounds = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
    const uint64_t wakes  = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 200000;

    std::printf("park backend: %s%s\n", backend_name(),
                hgcommon::park_is_lock_free() ? "" : "  [NOT lock free]");

    // Warm both arms before timing: the first handoff pays thread start-up and the first fault on
    // the shared line, which is not what either arm is reporting.
    (void)pingpong(rounds / 10 + 1, false);
    (void)pingpong(rounds / 10 + 1, true);
    (void)lone_wake(wakes / 10 + 1);

    const double spin = pingpong(rounds, false);
    const double park = pingpong(rounds, true);
    const double wake = lone_wake(wakes);

    std::printf("spin handoff      : %8.1f ns   (control: cache line between two cores, no kernel)\n", spin);
    std::printf("park/unpark handoff: %8.1f ns   (%.1fx the control)\n", park, park / spin);
    std::printf("wake with no waiter: %8.1f ns\n", wake);
    std::printf("primitive cost per handoff: %.1f ns  (park arm minus control)\n", park - spin);
    return 0;
}
