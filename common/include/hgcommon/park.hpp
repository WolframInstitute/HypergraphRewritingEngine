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

// ONLY THE SELECTION IS HERE. The platform headers each backend calls into -- futex.h,
// windows.h, os_sync_wait_on_address.h, condition_variable -- are included by park.cpp, so a
// translation unit that parks does not also parse windows.h. park_backend() is constexpr and
// callers assert on its value, so the choice itself must be visible.
#if defined(__linux__)
#  define HG_PARK_FUTEX 1
#elif defined(_WIN32)
#  define HG_PARK_WAIT_ON_ADDRESS 1
#elif defined(__APPLE__) && __has_include(<os/os_sync_wait_on_address.h>)
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
#  endif
#endif

namespace HG_NAMESPACE {
namespace common {

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
void park_if_equal(const std::atomic<uint32_t>& addr, uint32_t expected);

// Wake one waiter on this address.
void unpark_one(const std::atomic<uint32_t>& addr);

// Wake every waiter on this address.
void unpark_all(const std::atomic<uint32_t>& addr);

}  // namespace common
}  // namespace HG_NAMESPACE
