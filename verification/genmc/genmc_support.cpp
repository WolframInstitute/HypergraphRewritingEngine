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

#include <cstddef>
#include <new>

// ALIGNED ALLOCATION. Types the engine over-aligns to a cache line -- the map entry arrays, the
// per-worker blocks -- compile to operator new[](size_t, align_val_t), which GenMC does not
// implement. The unaligned operator new satisfies it: alignment beyond the natural alignment of the
// object exists to keep two workers' writes off one cache line, and the checker has no cache lines.
// The natural alignment atomics DO require comes with the allocation either way. Routing through
// operator new rather than malloc is what makes it visible: GenMC models the C++ allocation
// functions, and run.sh takes the SYSTEM stdlib.h rather than the checker's, so a direct malloc
// call reaches the interpreter as an unknown external function.
void* operator new(std::size_t n, std::align_val_t) { return ::operator new(n); }
void* operator new[](std::size_t n, std::align_val_t) { return ::operator new(n); }
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

}  // extern "C"
