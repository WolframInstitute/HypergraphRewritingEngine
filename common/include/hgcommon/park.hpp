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

namespace hgcommon {

enum class ParkBackend { Futex, WaitOnAddress, OsSync, StdAtomicWait };

// Which primitive this build uses. StdAtomicWait is the only one that may block on a lock.
constexpr ParkBackend park_backend() {
#if defined(HG_PARK_FUTEX)
    return ParkBackend::Futex;
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
    return ParkBackend::WaitOnAddress;
#elif defined(HG_PARK_OS_SYNC)
    return ParkBackend::OsSync;
#else
    return ParkBackend::StdAtomicWait;
#endif
}

// True when blocking is known not to take a lock.
constexpr bool park_is_lock_free() { return park_backend() != ParkBackend::StdAtomicWait; }

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
#else
    const_cast<std::atomic<uint32_t>&>(addr).notify_all();
#endif
}

}  // namespace hgcommon
