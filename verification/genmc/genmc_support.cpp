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
