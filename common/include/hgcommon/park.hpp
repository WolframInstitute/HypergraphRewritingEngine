#pragma once
#include "hgcommon/namespace.hpp"

// Park/unpark on a 32-bit address, calling the OS primitive directly.
//
// std::atomic<T>::wait would express this in one line, and the standard requires only that it
// block until notified -- it says nothing about how. libstdc++ shows what that licenses: with
// _GLIBCXX_HAVE_PLATFORM_WAIT it issues SYS_futex, and WITHOUT it the waiter pool holds a
// std::mutex and a __condvar and the wait takes the lock, which is why that rung is gone. So on a platform lacking that macro,
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
// NO MUTEX RUNG. There used to be one below this: a shared std::mutex and condition_variable
// for every parked address, selected when a platform offered none of the three address-waits.
// It was the only lock in the tree, and the engine's design is that there is none -- a claim
// the paper makes and the GPU table was fixed to honour. A platform that cannot wait on an
// address does not get a lock here; it gets a compile error, which is the honest answer and
// keeps "no mutex anywhere" checkable by grep rather than by argument.
#if !defined(HG_PARK_FUTEX) && !defined(HG_PARK_WAIT_ON_ADDRESS) && !defined(HG_PARK_OS_SYNC)
#  if defined(__cpp_lib_atomic_wait)
#    define HG_PARK_STD_ATOMIC_WAIT 1
#  else
#    error "no address-wait primitive on this platform; the engine takes no lock as a fallback"
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

// A LOCK DOES NOT ENTER THE BUILD UNNOTICED. Until this, park_is_lock_free() was a fact a caller
// could ask for and nothing refused a build on, so a platform that fell through to
// std::atomic::wait would ship a waiter pool holding a std::mutex and a __condvar behind a call
// named park -- the exact substitution this file exists to prevent, arriving silently through the
// standard library rather than through anything greppable.
//
// It fires at COMPILE time and in every translation unit that parks, which is what makes "no
// mutex anywhere" a property of the build rather than of a reviewer's attention. A bring-up on a
// platform with no address-wait syscall can still proceed by saying so, and park_backend() then
// still reports which primitive it got.
// Spelled as a constant in the idiom of std::atomic<T>::is_always_lock_free, and for the same
// reason: the answer is fixed by the build rather than discovered at runtime, so it belongs where
// a caller can static_assert on it too, not only where this file happens to check it.
inline constexpr bool park_is_always_lock_free = park_is_lock_free();

#if !defined(HG_PARK_ALLOW_LOCKING_FALLBACK)
static_assert(park_is_always_lock_free,
              "park() would take a lock in this build: std::atomic::wait is permitted to hold a "
              "mutex and a condition variable in its waiter pool, and the engine's charter is "
              "that nothing waits on a lock. Define HG_PARK_ALLOW_LOCKING_FALLBACK to accept it "
              "deliberately.");
#endif

// THE SAME SUBSTITUTION ARRIVES THROUGH std::atomic ITSELF. An atomic whose type is too wide for
// the target's native compare-exchange is implemented with a lock table, and it says so only if
// asked: the call sites look identical to the lock-free ones. These are the widths the engine's
// hot paths depend on -- the 32-bit park word, the 64-bit counters and keys, and the pointers the
// deques and segment tables publish -- so a target that would silently lock any of them fails
// here rather than in a profile.
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "std::atomic<uint32_t> takes a lock on this target; the park word is 32-bit");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "std::atomic<uint64_t> takes a lock on this target; counters and keys are 64-bit");
static_assert(std::atomic<void*>::is_always_lock_free,
              "std::atomic<pointer> takes a lock on this target; the deques and the segment "
              "tables publish pointers");


// Block while *addr == expected. May return spuriously; callers must re-check, which they
// have to do anyway since the value can change and change back.
void park_if_equal(const std::atomic<uint32_t>& addr, uint32_t expected);

// Wake one waiter on this address.
void unpark_one(const std::atomic<uint32_t>& addr);

// Wake every waiter on this address.
void unpark_all(const std::atomic<uint32_t>& addr);

}  // namespace common
}  // namespace HG_NAMESPACE
