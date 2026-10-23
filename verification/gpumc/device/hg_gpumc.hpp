#pragma once
// The CUDA device built-ins a GPUMC harness needs to compile the engine's device sources as they
// are. Each harness thread is one CUDA thread; a harness declares its grid with launch() and each
// thread enters with enter(). This directory is on the include path ahead of the CUDA toolkit's,
// so <cuda_runtime.h>, <cuda/atomic> and <cooperative_groups.h> resolve here.
//
// ATOMICS AND FENCES. atomicAdd and the other atomic* functions are relaxed read-modify-writes at
// device scope; the *_block forms are at block scope. __threadfence() is a seq_cst fence at device
// scope and __threadfence_block() one at block scope. cuda::atomic_ref (cuda/atomic) carries the
// memory order it is given and the scope of its template argument; thread_scope_system is checked
// at device scope (one device). Every access is preceded by the scope annotation GPUMC reads, and
// each is always inlined, so the annotation and the access are adjacent in the caller.
//
// BARRIERS. __syncthreads() and __syncwarp() are a pthread barrier over the block's threads, at
// block scope. The engine's persistent kernels run one warp per block, so the two coincide. A
// thread that leaves while the others of its block wait at a barrier leaves them blocked.
//
// WARP COLLECTIVES. A shuffle, ballot or match writes the calling lane's value to a per-block
// slot, waits at the block barrier, reads, and waits again. A block of N harness threads is a warp
// of N lanes: __activemask() is the full mask when N > 1 and one lane when N = 1. The kernels
// stride their loops by 32 lanes, so a harness gives every loop at most N items. A shuffle from a
// lane at or past N reads lane N - 1 (the lane a full-warp scan or broadcast reads last); a
// shuffle-down from such a lane gives 0 (an idle lane of a reduction).
//
// __shared__ is a static: one copy per program. A harness that runs a function declaring
// __shared__ variables runs it on one block at a time.
//
// Plain loads and stores are non-atomic, so GPUMC reports a race on any that conflict without
// synchronisation; a volatile access is non-atomic here as well.
//
// CALIBRATIONS of the engine's own code. HG_GPUMC_CALIBRATE_NO_THREADFENCE makes every
// __threadfence() a no-op. HG_GPUMC_CALIBRATE_SPLIT_RMW makes atomicAdd, atomicMin, atomicMax,
// atomicCAS and cuda::atomic_ref's compare-exchanges and fetch_add a load and a separate store,
// so two threads can both take the value either one read.
#include <pthread.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

extern "C" {
void __VERIFIER_memory_scope_device(void);
void __VERIFIER_memory_scope_work_group(void);
void __VERIFIER_thread_local_id(int);
void __VERIFIER_thread_group_id(int);
void __VERIFIER_thread_global_id(int);
void __VERIFIER_thread_kernel_id(int);
}

#define HG_GPUMC_INLINE inline __attribute__((always_inline))

namespace hg_gpumc {

constexpr unsigned kMaxBlocks = 4;
constexpr unsigned kMaxLanes = 8;

inline thread_local unsigned t_lane = 0;
inline thread_local unsigned t_block = 0;
inline unsigned g_block_lanes[kMaxBlocks];
inline unsigned g_blocks = 1;
inline pthread_barrier_t g_bar[kMaxBlocks];
inline uint64_t g_slot[kMaxBlocks][kMaxLanes];

// Before any thread starts: block b has lanes[b] threads.
inline void launch(unsigned blocks, const unsigned* lanes) {
    g_blocks = blocks;
    for (unsigned b = 0; b < blocks; ++b) {
        g_block_lanes[b] = lanes[b];
        pthread_barrier_init(&g_bar[b], nullptr, lanes[b]);
    }
}

// The calling thread's block size.
inline unsigned lanes() { return g_block_lanes[t_block]; }

// First call of each harness thread.
inline void enter(unsigned block, unsigned lane) {
    t_block = block;
    t_lane = lane;
    __VERIFIER_thread_global_id(static_cast<int>(block * kMaxLanes + lane));
    __VERIFIER_thread_local_id(static_cast<int>(lane));
    __VERIFIER_thread_group_id(static_cast<int>(block));
    __VERIFIER_thread_kernel_id(0);
}

inline void block_barrier() {
    __VERIFIER_memory_scope_work_group();
    pthread_barrier_wait(&g_bar[t_block]);
}

// A collective's value as one 64-bit word, by conversion: a byte copy into a wider word would be
// a mixed-size access, which GPUMC refuses.
template <class T>
inline uint64_t to_bits(T v) {
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value || std::is_pointer<T>::value,
                  "a warp collective moves an integer, an enum or a pointer");
    if constexpr (std::is_pointer<T>::value) return reinterpret_cast<uint64_t>(v);
    else return static_cast<uint64_t>(v);
}
template <class T>
inline T from_bits(uint64_t b) {
    if constexpr (std::is_pointer<T>::value) return reinterpret_cast<T>(b);
    else return static_cast<T>(b);
}

inline unsigned lane_of(unsigned src) { return src < lanes() ? src : lanes() - 1; }

template <class T>
inline T shfl(T v, unsigned src, unsigned width) {
    g_slot[t_block][t_lane] = to_bits(v);
    block_barrier();
    const unsigned base = width >= 32 ? 0 : (t_lane / width) * width;
    const uint64_t r = g_slot[t_block][lane_of(base + src % width)];
    block_barrier();
    return from_bits<T>(r);
}

// A source past lane 31 returns the caller's value, as on the device. A source between the block's
// last lane and 31 is a lane the model leaves out, which in a warp reduction is idle: it gives 0.
template <class T>
inline T shfl_down(T v, unsigned delta) {
    const unsigned src = t_lane + delta;
    g_slot[t_block][t_lane] = to_bits(v);
    block_barrier();
    const uint64_t r = src >= 32 ? g_slot[t_block][t_lane] : src < lanes() ? g_slot[t_block][src] : 0;
    block_barrier();
    return from_bits<T>(r);
}

// Bit i set when lane i's `key` equals the calling lane's; *all when every lane's does.
inline uint32_t match_mask(uint64_t key, bool* all) {
    g_slot[t_block][t_lane] = key;
    block_barrier();
    uint32_t m = 0;
    bool same = true;
    for (unsigned i = 0; i < lanes(); ++i) {
        if (g_slot[t_block][i] == key) m |= 1u << i;
        else same = false;
    }
    block_barrier();
    if (all) *all = same;
    return m;
}

inline uint32_t ballot(bool p) {
    g_slot[t_block][t_lane] = p ? 1u : 0u;
    block_barrier();
    uint32_t m = 0;
    for (unsigned i = 0; i < lanes(); ++i)
        if (g_slot[t_block][i]) m |= 1u << i;
    block_barrier();
    return m;
}

template <class T>
inline T exclusive_sum(T v) {
    g_slot[t_block][t_lane] = to_bits(v);
    block_barrier();
    T s = 0;
    for (unsigned i = 0; i < t_lane; ++i) s += from_bits<T>(g_slot[t_block][i]);
    block_barrier();
    return s;
}

// threadIdx, blockIdx, blockDim and gridDim: x reads the calling thread's lane, block, block size
// and grid size. Properties (-fdeclspec), so the names stay usable as struct members.
struct LaneIdx {
    unsigned get_x() const { return t_lane; }
    __declspec(property(get = get_x)) unsigned x;
    unsigned y = 0, z = 0;
};
struct BlockIdx {
    unsigned get_x() const { return t_block; }
    __declspec(property(get = get_x)) unsigned x;
    unsigned y = 0, z = 0;
};
struct BlockDim {
    unsigned get_x() const { return lanes(); }
    __declspec(property(get = get_x)) unsigned x;
    unsigned y = 1, z = 1;
};
struct GridDim {
    unsigned get_x() const { return g_blocks; }
    __declspec(property(get = get_x)) unsigned x;
    unsigned y = 1, z = 1;
};

}  // namespace hg_gpumc

inline const hg_gpumc::LaneIdx threadIdx{};
inline const hg_gpumc::BlockIdx blockIdx{};
inline const hg_gpumc::BlockDim blockDim{};
inline const hg_gpumc::GridDim gridDim{};

#define __device__
#define __host__
#define __global__
#define __forceinline__ HG_GPUMC_INLINE
#define __noinline__
#define __grid_constant__
#define __shared__ static
#define __launch_bounds__(...)

// ---- atomics ---------------------------------------------------------------------------------

#define HG_GPUMC_RMW(name, builtin)                                 \
    template <class T, class U>                                     \
    HG_GPUMC_INLINE T name(T* a, U v) {                             \
        __VERIFIER_memory_scope_device();                           \
        return builtin(a, static_cast<T>(v), __ATOMIC_RELAXED);     \
    }                                                               \
    template <class T, class U>                                     \
    HG_GPUMC_INLINE T name##_block(T* a, U v) {                     \
        __VERIFIER_memory_scope_work_group();                       \
        return builtin(a, static_cast<T>(v), __ATOMIC_RELAXED);     \
    }
#if defined(HG_GPUMC_CALIBRATE_SPLIT_RMW)
template <class T, class U>
HG_GPUMC_INLINE T atomicAdd(T* a, U v) {
    __VERIFIER_memory_scope_device();
    const T old = __atomic_load_n(a, __ATOMIC_RELAXED);
    __VERIFIER_memory_scope_device();
    __atomic_store_n(a, static_cast<T>(old + static_cast<T>(v)), __ATOMIC_RELAXED);
    return old;
}
template <class T, class U>
HG_GPUMC_INLINE T atomicAdd_block(T* a, U v) {
    return atomicAdd(a, v);
}
#else
HG_GPUMC_RMW(atomicAdd, __atomic_fetch_add)
#endif
HG_GPUMC_RMW(atomicSub, __atomic_fetch_sub)
HG_GPUMC_RMW(atomicOr, __atomic_fetch_or)
HG_GPUMC_RMW(atomicAnd, __atomic_fetch_and)
HG_GPUMC_RMW(atomicXor, __atomic_fetch_xor)
#if defined(HG_GPUMC_CALIBRATE_SPLIT_RMW)
#define HG_GPUMC_SPLIT_MINMAX(name, keep_old)                                          \
    template <class T, class U>                                                       \
    HG_GPUMC_INLINE T name(T* a, U v) {                                               \
        __VERIFIER_memory_scope_device();                                             \
        const T old = __atomic_load_n(a, __ATOMIC_RELAXED);                           \
        const T val = static_cast<T>(v);                                              \
        __VERIFIER_memory_scope_device();                                             \
        __atomic_store_n(a, (keep_old) ? old : val, __ATOMIC_RELAXED);                \
        return old;                                                                   \
    }                                                                                 \
    template <class T, class U>                                                       \
    HG_GPUMC_INLINE T name##_block(T* a, U v) {                                       \
        return name(a, v);                                                            \
    }
HG_GPUMC_SPLIT_MINMAX(atomicMin, old <= val)
HG_GPUMC_SPLIT_MINMAX(atomicMax, old >= val)
#undef HG_GPUMC_SPLIT_MINMAX
#else
HG_GPUMC_RMW(atomicMin, __atomic_fetch_min)
HG_GPUMC_RMW(atomicMax, __atomic_fetch_max)
#endif
#undef HG_GPUMC_RMW

template <class T, class U>
HG_GPUMC_INLINE T atomicExch(T* a, U v) {
    __VERIFIER_memory_scope_device();
    return __atomic_exchange_n(a, static_cast<T>(v), __ATOMIC_RELAXED);
}

template <class T, class U, class V>
HG_GPUMC_INLINE T atomicCAS(T* a, U expected, V desired) {
    T e = static_cast<T>(expected);
#if defined(HG_GPUMC_CALIBRATE_SPLIT_RMW)
    __VERIFIER_memory_scope_device();
    const T seen = __atomic_load_n(a, __ATOMIC_RELAXED);
    if (seen == e) {
        __VERIFIER_memory_scope_device();
        __atomic_store_n(a, static_cast<T>(desired), __ATOMIC_RELAXED);
    }
    return seen;
#else
    __VERIFIER_memory_scope_device();
    __atomic_compare_exchange_n(a, &e, static_cast<T>(desired), false, __ATOMIC_RELAXED,
                                __ATOMIC_RELAXED);
    return e;
#endif
}

HG_GPUMC_INLINE void __threadfence() {
#if !defined(HG_GPUMC_CALIBRATE_NO_THREADFENCE)
    __VERIFIER_memory_scope_device();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}
HG_GPUMC_INLINE void __threadfence_block() {
    __VERIFIER_memory_scope_work_group();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// ---- barriers and warp collectives -----------------------------------------------------------

inline void __syncthreads() { hg_gpumc::block_barrier(); }
inline void __syncwarp(unsigned = 0xFFFFFFFFu) { hg_gpumc::block_barrier(); }
inline unsigned __activemask() { return hg_gpumc::lanes() > 1 ? 0xFFFFFFFFu : 1u; }

template <class T>
inline T __shfl_sync(unsigned, T v, int src, int width = 32) {
    return hg_gpumc::shfl(v, static_cast<unsigned>(src), static_cast<unsigned>(width));
}
template <class T>
inline T __shfl_down_sync(unsigned, T v, unsigned delta, int = 32) {
    return hg_gpumc::shfl_down(v, delta);
}
inline unsigned __ballot_sync(unsigned, int p) { return hg_gpumc::ballot(p != 0); }
inline int __any_sync(unsigned, int p) { return hg_gpumc::ballot(p != 0) != 0; }
inline int __all_sync(unsigned, int p) {
    return hg_gpumc::ballot(p != 0) == ((1u << hg_gpumc::lanes()) - 1u);
}
template <class T>
inline unsigned __match_any_sync(unsigned, T v) {
    return hg_gpumc::match_mask(hg_gpumc::to_bits(v), nullptr);
}
template <class T>
inline unsigned __match_all_sync(unsigned, T v, int* pred) {
    bool all = false;
    const unsigned m = hg_gpumc::match_mask(hg_gpumc::to_bits(v), &all);
    *pred = all ? 1 : 0;
    return all ? m : 0u;
}

// ---- integer intrinsics and the device's global min/max --------------------------------------

inline int __popc(unsigned v) { return __builtin_popcount(v); }
inline int __popcll(unsigned long long v) { return __builtin_popcountll(v); }
inline int __ffs(unsigned v) { return __builtin_ffs(static_cast<int>(v)); }
inline int __ffsll(unsigned long long v) { return __builtin_ffsll(static_cast<long long>(v)); }
inline int __clz(unsigned v) { return v ? __builtin_clz(v) : 32; }
inline int __clzll(unsigned long long v) { return v ? __builtin_clzll(v) : 64; }
inline unsigned __brev(unsigned v) {
    unsigned r = 0;
    for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
    return r;
}
inline long long clock64() { return 0; }
inline void __nanosleep(unsigned) {}

template <class A, class B>
inline typename std::common_type<A, B>::type min(A a, B b) {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(a) : static_cast<C>(b);
}
template <class A, class B>
inline typename std::common_type<A, B>::type max(A a, B b) {
    using C = typename std::common_type<A, B>::type;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(b) : static_cast<C>(a);
}
