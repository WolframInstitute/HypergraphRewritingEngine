#include "hgcommon/park.hpp"

// The three park/unpark bodies, and the platform headers they call into. park.hpp names only
// the backend SELECTION, so a translation unit that parks parses <atomic> and <cstdint> rather
// than windows.h or linux/futex.h. The selection macros are already defined by the time this
// include returns, so the #if arms below agree with park_backend() by construction.
//
// This lives under job_system because job_system is the only consumer of park.hpp and the only
// target that could hold a body for it; hgcommon is headers with no library of its own.

#if defined(HG_PARK_FUTEX)
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#elif defined(HG_PARK_WAIT_ON_ADDRESS)
#  ifndef NOMINMAX
#    define NOMINMAX   // windows.h's min/max macros poison std::numeric_limits<T>::max() etc.
#  endif
#  include <windows.h>
#  if defined(_MSC_VER)
     // WaitOnAddress/WakeByAddress* live in Synchronization.lib, which MSVC does
     // not link by default (MinGW's default library set resolves them).
#    pragma comment(lib, "synchronization")
#  endif
#elif defined(HG_PARK_OS_SYNC)
#  include <os/os_sync_wait_on_address.h>
#endif

namespace HG_NAMESPACE {
namespace common {


void park_if_equal(const std::atomic<uint32_t>& addr, uint32_t expected) {
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
#endif
}

void unpark_one(const std::atomic<uint32_t>& addr) {
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
#endif
}

void unpark_all(const std::atomic<uint32_t>& addr) {
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
#endif
}

}  // namespace common
}  // namespace HG_NAMESPACE
