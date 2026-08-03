#pragma once

// Park/unpark on a 32-bit address, calling the OS primitive directly.
//
// std::atomic<T>::wait would express this in one line, and the standard requires only that it
// block until notified -- it says nothing about how. libstdc++ shows what that licenses: with
// _GLIBCXX_HAVE_PLATFORM_WAIT it issues SYS_futex, and WITHOUT it the waiter pool holds a
// std::mutex and a __condvar and the wait takes the lock. So on a platform lacking that macro,
// "we replaced the condition variable with std::atomic::wait" replaces a condition variable we
// can see with one we cannot. Same for libc++ and the MSVC STL: each has a lock-free path and
// a lock-based fallback, and which one is compiled is not visible from the call site.
//
// This engine's charter is that nothing waits on a lock, so the blocking primitive is named
// here rather than inherited. Each backend is the platform's own address-wait syscall:
//
//   Linux    futex(FUTEX_WAIT_PRIVATE / FUTEX_WAKE_PRIVATE)
//   Windows  WaitOnAddress / WakeByAddressSingle / WakeByAddressAll
//   macOS    os_sync_wait_on_address / os_sync_wake_by_address_{any,all}
//            (public since macOS 14.4; earlier SDKs fall through below)
//   fallback std::atomic::wait -- WHICH MAY TAKE A LOCK. park_backend() reports it so a build
//            can assert against shipping on it, rather than discovering it at runtime.
//
// The waited-on type is fixed at uint32_t because that is what every backend accepts: futex
// compares a 32-bit word, and WaitOnAddress takes a size of 1, 2, 4 or 8 with 4 the portable
// choice. It also avoids a trap in the library facility -- libstdc++ routes any type that is
// not the platform wait type through a PROXY waiter, adding indirection, so a counter declared
// as size_t would silently take the slower path.

#include <atomic>
#include <cstdint>

#if defined(__linux__)
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#  define HG_PARK_FUTEX 1
#elif defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX   // windows.h's min/max macros poison std::numeric_limits<T>::max() etc.
#  endif
#  include <windows.h>
#  if defined(_MSC_VER)
     // WaitOnAddress/WakeByAddress* live in Synchronization.lib, which MSVC does
     // not link by default (MinGW's default library set resolves them).
#    pragma comment(lib, "synchronization")
#  endif
#  define HG_PARK_WAIT_ON_ADDRESS 1
#elif defined(__APPLE__) && __has_include(<os/os_sync_wait_on_address.h>)
#  include <os/os_sync_wait_on_address.h>
#  define HG_PARK_OS_SYNC 1
#endif

// A macOS target too old for BOTH of its options.
//
// os_sync_wait_on_address arrived in the 14.4 SDK, so a build against an older SDK does not
// select HG_PARK_OS_SYNC. The fallthrough is std::atomic::wait/notify -- which libc++ gates
// behind _LIBCPP_AVAILABILITY_SYNC and marks UNAVAILABLE below macOS 11. The cross toolchain
// deploys to 10.15 (cmake/toolchains/macos-cross.cmake), so both are out and the build fails to
// compile rather than to link.
//
// Raising the deployment target would fix it and would also drop macOS 10.15 users, which is a
// support decision and not this header's to take. atomic_compat.hpp met the same SDK the same way
// -- work around, do not bump -- so this follows it: a condition-variable park, which is what
// std::atomic::wait is on a platform without a native wait-on-address anyway.
#if defined(__APPLE__) && !defined(HG_PARK_OS_SYNC) && \
    defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED < 110000
#  define HG_PARK_CONDVAR 1
#  include <condition_variable>
#  include <mutex>
#endif

namespace hgcommon {

enum class ParkBackend { Futex, WaitOnAddress, OsSync, StdAtomicWait, CondVar };

// Which primitive this build uses. StdAtomicWait is the only one that may block on a lock.
constexpr ParkBackend park_backend() {
#if defined(HG_PARK_FUTEX)
    return ParkBackend::Futex;
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    return ParkBackend::WaitOnAddress;
#elif defined(HG_PARK_OS_SYNC)
    return ParkBackend::OsSync;
#elif defined(HG_PARK_CONDVAR)
    return ParkBackend::CondVar;
#else
    return ParkBackend::StdAtomicWait;
#endif
}

// True when blocking is known not to take a lock.
constexpr bool park_is_lock_free() {
    return park_backend() != ParkBackend::StdAtomicWait && park_backend() != ParkBackend::CondVar;
}

#if defined(HG_PARK_CONDVAR)
// One bucket per address, striped. A wake notifies the whole bucket rather than one waiter, so
// two addresses sharing a bucket cost a spurious wake and never a lost one -- and park_if_equal's
// contract already allows spurious returns.
namespace detail {
struct ParkBucket {
    std::mutex m;
    std::condition_variable cv;
};
inline ParkBucket& park_bucket(const void* addr) {
    static ParkBucket buckets[64];
    return buckets[(reinterpret_cast<uintptr_t>(addr) >> 4) & 63u];
}
}  // namespace detail
#endif

// Block while *addr == expected. May return spuriously; callers must re-check, which they
// have to do anyway since the value can change and change back.
inline void park_if_equal(const std::atomic<uint32_t>& addr, uint32_t expected) {
#if defined(HG_PARK_FUTEX)
    ::syscall(SYS_futex, reinterpret_cast<const uint32_t*>(&addr),
              FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    uint32_t compare = expected;
    ::WaitOnAddress(const_cast<volatile void*>(static_cast<const volatile void*>(&addr)),
                    &compare, sizeof(uint32_t), INFINITE);
#elif defined(HG_PARK_OS_SYNC)
    ::os_sync_wait_on_address(const_cast<void*>(static_cast<const void*>(&addr)),
                              static_cast<uint64_t>(expected), sizeof(uint32_t),
                              OS_SYNC_WAIT_ON_ADDRESS_NONE);
#elif defined(HG_PARK_CONDVAR)
    auto& b = detail::park_bucket(&addr);
    std::unique_lock<std::mutex> lk(b.m);
    // Re-check under the lock: without it a wake between the caller's load and this wait is lost.
    if (addr.load(std::memory_order_acquire) == expected) b.cv.wait(lk);
#else
    addr.wait(expected, std::memory_order_acquire);
#endif
}

inline void unpark_one(const std::atomic<uint32_t>& addr) {
#if defined(HG_PARK_FUTEX)
    ::syscall(SYS_futex, reinterpret_cast<const uint32_t*>(&addr),
              FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    ::WakeByAddressSingle(const_cast<void*>(static_cast<const void*>(&addr)));
#elif defined(HG_PARK_OS_SYNC)
    ::os_sync_wake_by_address_any(const_cast<void*>(static_cast<const void*>(&addr)),
                                  sizeof(uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
#elif defined(HG_PARK_CONDVAR)
    auto& b = detail::park_bucket(&addr);
    { std::lock_guard<std::mutex> lk(b.m); }   // order the wake after the waiter's re-check
    b.cv.notify_all();                          // bucket-wide, so a share costs a spurious wake
#else
    // wait() is const-qualified but notify_one()/notify_all() are not; the wake mutates no
    // atomic value, and this API takes const& to mirror the futex path, so cast it away.
    const_cast<std::atomic<uint32_t>&>(addr).notify_one();
#endif
}

inline void unpark_all(const std::atomic<uint32_t>& addr) {
#if defined(HG_PARK_FUTEX)
    ::syscall(SYS_futex, reinterpret_cast<const uint32_t*>(&addr),
              FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0);
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    ::WakeByAddressAll(const_cast<void*>(static_cast<const void*>(&addr)));
#elif defined(HG_PARK_OS_SYNC)
    ::os_sync_wake_by_address_all(const_cast<void*>(static_cast<const void*>(&addr)),
                                  sizeof(uint32_t), OS_SYNC_WAKE_BY_ADDRESS_NONE);
#elif defined(HG_PARK_CONDVAR)
    auto& b = detail::park_bucket(&addr);
    { std::lock_guard<std::mutex> lk(b.m); }
    b.cv.notify_all();
#else
    const_cast<std::atomic<uint32_t>&>(addr).notify_all();
#endif
}

}  // namespace hgcommon
