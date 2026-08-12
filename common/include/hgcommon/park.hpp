#pragma once
#include "hgcommon/namespace.hpp"

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
//            (public since the 14.4 SDK). An older SDK falls through to std::atomic::wait,
//            which libc++ marks unavailable below macOS 11 -- so the cross toolchain's
//            deployment floor is 11.0, and a 14.4+ SDK is what selects the native path.
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

// THE FALLBACK IS CHOSEN BY THE FEATURE MACRO, NOT BY A VERSION NUMBER. std::atomic<T>::wait is
// C++20, and a toolchain can have the header and refuse the call: libc++ annotates it
// `availability(macosx, strict, introduced=11.0)` and, when its own checks decide the target
// cannot reach it, leaves __cpp_lib_atomic_wait UNDEFINED while the declaration stays visible.
// Testing the deployment target instead is how the macOS legs failed while configured at exactly
// the version libc++ names -- os_sync_wait_on_address needs SDK 14.4+, atomic wait was refused,
// and NEITHER of the two primitives was reachable.
//
// So: the library says whether the feature is usable, and this asks it.
#if !defined(HG_PARK_FUTEX) && !defined(HG_PARK_WAIT_ON_ADDRESS) && !defined(HG_PARK_OS_SYNC)
#  if defined(__cpp_lib_atomic_wait)
#    define HG_PARK_STD_ATOMIC_WAIT 1
#  else
#    define HG_PARK_CONDVAR 1
#    include <condition_variable>
#    include <mutex>
#  endif
#endif

namespace HG_NAMESPACE {
namespace common {

#if defined(HG_PARK_CONDVAR)
namespace detail {
// One mutex and condition variable shared by every parked address. Function-local so it has no
// static-initialisation order to depend on, and process-lifetime so a wake never touches a
// destroyed object.
struct ParkCondVar { std::mutex m; std::condition_variable cv; };
inline ParkCondVar& park_condvar() { static ParkCondVar g; return g; }
}  // namespace detail
#endif

enum class ParkBackend { Futex, WaitOnAddress, OsSync, StdAtomicWait, CondVar };

// Which primitive this build uses. StdAtomicWait is the only one that may block on a lock.
constexpr ParkBackend park_backend() {
#if defined(HG_PARK_FUTEX)
    return ParkBackend::Futex;
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    return ParkBackend::WaitOnAddress;
#elif defined(HG_PARK_OS_SYNC)
    return ParkBackend::OsSync;
#elif defined(HG_PARK_STD_ATOMIC_WAIT)
    return ParkBackend::StdAtomicWait;
#else
    return ParkBackend::CondVar;
#endif
}

// True when blocking is known not to take a lock. Both fallbacks may: std::atomic::wait is
// permitted to be implemented with one, and CondVar is one by construction.
constexpr bool park_is_lock_free() {
    return park_backend() != ParkBackend::StdAtomicWait &&
           park_backend() != ParkBackend::CondVar;
}


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
#elif defined(HG_PARK_STD_ATOMIC_WAIT)
    addr.wait(expected, std::memory_order_acquire);
#else
    // THE PORTABLE FLOOR, and it takes a lock -- park_is_lock_free() reports that rather than
    // hiding it. One shared mutex and condition variable for every address: this backend is
    // reached only where the platform offers no address-wait at all, the waiter is an IDLE
    // worker with nothing to do, and a wake is O(waiters) rather than O(1). Correct everywhere,
    // and slower where it is the only thing that compiles.
    //
    // The predicate is re-read under the lock, so the store-then-wake a waker performs cannot
    // land between this thread's check and its wait.
    auto& g = detail::park_condvar();
    std::unique_lock<std::mutex> lk(g.m);
    if (addr.load(std::memory_order_acquire) == expected) g.cv.wait(lk);
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
#elif defined(HG_PARK_STD_ATOMIC_WAIT)
    // wait() is const-qualified but notify_one()/notify_all() are not; the wake mutates no
    // atomic value, and this API takes const& to mirror the futex path, so cast it away.
    const_cast<std::atomic<uint32_t>&>(addr).notify_one();
#else
    (void)addr;   // one shared variable: a waiter on any address may be the one woken
    auto& g = detail::park_condvar();
    std::lock_guard<std::mutex> lk(g.m);
    g.cv.notify_all();   // notify_ONE could wake a waiter on a different address and lose this
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
#elif defined(HG_PARK_STD_ATOMIC_WAIT)
    const_cast<std::atomic<uint32_t>&>(addr).notify_all();
#else
    (void)addr;
    auto& g = detail::park_condvar();
    std::lock_guard<std::mutex> lk(g.m);
    g.cv.notify_all();
#endif
}

}  // namespace common
}  // namespace HG_NAMESPACE
