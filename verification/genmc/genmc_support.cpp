// Definitions the C++ runtime supplies on a real target and GenMC's interpreter does not,
// compiled and linked into every composed harness by run.sh.
//
// GenMC executes LLVM IR directly. It models the allocator, threads and atomics; everything a
// program normally gets from crtbegin.o, libsupc++ and the dynamic loader is simply absent, and
// the interpreter stops with "unknown external function" or "could not resolve external global"
// the first time one is reached. Constructing a Hypergraph reaches three of them.
//
// Each definition below is EXACTLY what the checker needs to keep executing and nothing more. All
// three concern storage layout or process teardown; none is reachable from any inter-thread
// ordering, so none changes the set of executions the checker explores.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <unordered_map>
#include <utility>

// ALIGNED ALLOCATION. Types the engine over-aligns to a cache line -- the map entry arrays, the
// per-worker blocks -- compile to operator new[](size_t, align_val_t), which GenMC does not
// implement. The unaligned operator new satisfies it: alignment beyond the natural alignment of the
// object exists to keep two workers' writes off one cache line, and the checker has no cache lines.
// The natural alignment atomics DO require comes with the allocation either way. Routing through
// operator new rather than malloc is what makes it visible: GenMC models the C++ allocation
// functions, and run.sh takes the SYSTEM stdlib.h rather than the checker's, so a direct malloc
// call reaches the interpreter as an unknown external function.
//
// ZERO IS CLAMPED TO ONE. The standard requires operator new(0) to return a distinct non-null
// pointer, and GenMC's address allocator refuses a zero-size request outright (an internal check,
// not a diagnostic). One byte satisfies both: the pointer is distinct, and nothing reads it,
// because a zero-size object has nothing to read.
void* operator new(std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void* operator new[](std::size_t n, std::align_val_t) { return ::operator new(n ? n : 1); }
void operator delete(void* p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { ::operator delete(p); }

extern "C" {

// THE TRANSLATION UNIT HANDLE that __cxa_atexit takes so a shared object can run its destructors
// when it is unloaded. crtbegin.o defines it; under the checker there is no loader and its value
// is never dereferenced, only passed along.
void* __dso_handle = nullptr;

// STATIC DESTRUCTOR REGISTRATION. A destructor registered here would run after main returns,
// which is after every thread has been joined -- there is no concurrency left for it to
// participate in. Recording nothing and reporting success leaves the checker with the same set of
// executions and one less external symbol to resolve.
int __cxa_atexit(void (*)(void*), void*, void*) { return 0; }
// The C allocation functions over the checker's own primitives. Every engine allocation but
// one goes through operator new, which the interpreter models; the job slot pool calls
// std::malloc through <cstdlib>, which the checker's C runtime override does not reach, and the
// call arrived at the interpreter as an unknown external (the default evolve arm, main:57).
void* __VERIFIER_malloc(std::size_t);
void __VERIFIER_free(void*);
void* malloc(std::size_t n) noexcept { return __VERIFIER_malloc(n ? n : 1); }
void free(void* p) noexcept { if (p) __VERIFIER_free(p); }
void* calloc(std::size_t count, std::size_t size) noexcept {
    const std::size_t n = count * size;
    auto* p = static_cast<unsigned char*>(__VERIFIER_malloc(n ? n : 1));
    for (std::size_t i = 0; i < n; ++i) p[i] = 0;
    return p;
}
// The guard of a function-local static (`static T x = ...;` reached on the evolve path).
// acquire returns 1 to the thread that will initialise (it takes the guard 0 -> 1); a thread
// that finds it taken spins until release stores 2 -- the loop is a plain re-read the
// checker's SpinAssume pass turns into an assume, so no execution is lost to it.
int __cxa_guard_acquire(std::uint64_t* guard) noexcept {
    auto* g = reinterpret_cast<std::atomic<char>*>(guard);
    char expected = 0;
    if (g->compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return 1;
    while (g->load(std::memory_order_acquire) != 2) {}
    return 0;
}
void __cxa_guard_release(std::uint64_t* guard) noexcept {
    reinterpret_cast<std::atomic<char>*>(guard)->store(2, std::memory_order_release);
}
void __cxa_guard_abort(std::uint64_t* guard) noexcept {
    reinterpret_cast<std::atomic<char>*>(guard)->store(0, std::memory_order_release);
}
// std::this_thread::yield() is this call. A yield is a scheduling hint with no memory
// semantics; under the checker every interleaving is explored regardless, so it is nothing.
int sched_yield() noexcept { return 0; }
// The libc string and memory functions the engine's normal path can reach, each a plain loop
// the interpreter executes: std::equal on integer ranges (memcmp), std::string construction
// from a literal (strlen), and the exception-message comparison in run_job (strcmp).
int memcmp(const void* a, const void* b, std::size_t n) noexcept {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    for (std::size_t i = 0; i < n; ++i)
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
std::size_t strlen(const char* s) noexcept {
    std::size_t n = 0;
    while (s[n] != '\0') ++n;
    return n;
}
int strcmp(const char* a, const char* b) noexcept {
    std::size_t i = 0;
    while (a[i] != '\0' && a[i] == b[i]) ++i;
    return static_cast<unsigned char>(a[i]) - static_cast<unsigned char>(b[i]);
}

}  // extern "C"

// THE ABI's TYPE-INFO VTABLES, which libstdc++ declares as `[0 x ptr]` -- an external global of
// SIZE ZERO. GenMC materialises every external global through an address allocator that refuses a
// zero-size request outright, as an internal check rather than a diagnostic, so a module
// referencing one aborts the run before it starts. Any translation unit with a polymorphic class
// references them; std::thread's state class is enough, which is why a harness that constructs a
// JobSystem is fine and one that STARTS it is not.
//
// Only their ADDRESS is used: a typeinfo object's first word is this vtable's address offset past
// its header, and nothing here dereferences it, because nothing in a harness does a dynamic_cast
// or catches by base. Four words with the right symbol names give the linker something of
// non-zero size to point at.
//
// Spelled through asm labels because the names are Itanium-mangled and cannot be written as C++
// identifiers.
// External linkage on purpose: these must DEFINE the symbols libstdc++ declares, and an internal
// one would leave the zero-size declaration standing beside it.
void* g_class_type_info_vtable[4] asm("_ZTVN10__cxxabiv117__class_type_infoE") = {};
void* g_si_class_type_info_vtable[4] asm("_ZTVN10__cxxabiv120__si_class_type_infoE") = {};
void* g_vmi_class_type_info_vtable[4] asm("_ZTVN10__cxxabiv121__vmi_class_type_infoE") = {};

// libstdc++ keeps two things out of its headers that the engine's normal path reaches: the
// byte hash behind std::hash<std::string>/<std::thread::id>, and the prime rehash policy of
// std::unordered_map/set. Neither has a body in the module, so each is defined here with the
// semantics the container relies on: a hash that is a function of the bytes, and a bucket
// policy that grows the table before the load factor is exceeded.
namespace std {
size_t _Hash_bytes(const void* p, size_t n, size_t seed) {
    const auto* b = static_cast<const unsigned char*>(p);
    size_t h = seed ^ 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
    return h;
}
namespace __detail {
size_t _Prime_rehash_policy::_M_next_bkt(size_t n) const {
    size_t b = 2;
    while (b < n) b *= 2;
    b += 1;  // odd, so a power-of-two-poor hash still spreads
    _M_next_resize = static_cast<size_t>(static_cast<float>(b) * _M_max_load_factor);
    return b;
}
pair<bool, size_t> _Prime_rehash_policy::_M_need_rehash(size_t n_bkt, size_t n_elt, size_t n_ins) const {
    if (n_elt + n_ins > _M_next_resize) {
        const float min_bkts = static_cast<float>(n_elt + n_ins) / _M_max_load_factor;
        if (min_bkts >= static_cast<float>(n_bkt)) {
            const size_t want = static_cast<size_t>(min_bkts) + 1;
            return {true, _M_next_bkt(want > n_bkt * 2 ? want : n_bkt * 2)};
        }
        _M_next_resize = static_cast<size_t>(static_cast<float>(n_bkt) * _M_max_load_factor);
    }
    return {false, 0};
}
}  // namespace __detail
}  // namespace std
