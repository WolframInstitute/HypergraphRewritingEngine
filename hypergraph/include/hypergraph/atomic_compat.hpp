#pragma once

// Portable atomic-view over a plain (non-atomic) object.
//
// The engine keeps some State fields as plain scalars (so State stays trivially copyable and
// single-threaded paths touch them directly) but accesses them atomically from concurrent paths.
// std::atomic_ref (C++20) is the standard tool, but the OSXCross SDK's bundled libc++ predates it,
// so the macOS cross build cannot use std::atomic_ref directly. This header selects std::atomic_ref
// when the library provides it (Linux/Windows) and otherwise falls back to the compiler's __atomic
// builtins, which every GCC/Clang target supports. The two paths are semantically identical for the
// operations this codebase uses (load / store / compare_exchange_weak / compare_exchange_strong).

#include <atomic>
#include <cstdint>

namespace hg {

#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L

template <class T>
using atomic_ref = std::atomic_ref<T>;

#else

template <class T>
class atomic_ref {
    T* ptr_;
    // std::memory_order enumerators carry the same underlying values as the __ATOMIC_* macros on
    // GCC and Clang (relaxed=0 … seq_cst=5), which is the only place this fallback compiles.
    static constexpr int mo(std::memory_order o) noexcept { return static_cast<int>(o); }

public:
    explicit atomic_ref(T& obj) noexcept : ptr_(&obj) {}

    T load(std::memory_order o = std::memory_order_seq_cst) const noexcept {
        T v;
        __atomic_load(ptr_, &v, mo(o));
        return v;
    }

    void store(T v, std::memory_order o = std::memory_order_seq_cst) const noexcept {
        __atomic_store(ptr_, &v, mo(o));
    }

    bool compare_exchange_strong(T& expected, T desired,
                                 std::memory_order succ = std::memory_order_seq_cst,
                                 std::memory_order fail = std::memory_order_seq_cst) const noexcept {
        return __atomic_compare_exchange(ptr_, &expected, &desired,
                                         /*weak=*/false, mo(succ), mo(fail));
    }

    bool compare_exchange_weak(T& expected, T desired,
                               std::memory_order succ = std::memory_order_seq_cst,
                               std::memory_order fail = std::memory_order_seq_cst) const noexcept {
        return __atomic_compare_exchange(ptr_, &expected, &desired,
                                         /*weak=*/true, mo(succ), mo(fail));
    }

    T fetch_add(T v, std::memory_order o = std::memory_order_seq_cst) const noexcept {
        return __atomic_fetch_add(ptr_, v, mo(o));
    }

    T exchange(T v, std::memory_order o = std::memory_order_seq_cst) const noexcept {
        T old;
        __atomic_exchange(ptr_, &v, &old, mo(o));
        return old;
    }
};

#endif

}  // namespace hg
