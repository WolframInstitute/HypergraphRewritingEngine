// mingw-w64 corrupts the heap when a worker thread exits. This is the smallest program that
// shows it, and it contains no engine code at all.
//
// WHAT IT IS FOR. `bench_cpu_evolve.exe 1 1 1 wpp` built with mingw-w64 exits 116 -- WSL's
// truncation of STATUS_HEAP_CORRUPTION, 0xC0000374 -- deterministically, with no output, after
// the work has completed. Linux is clean on the same source under valgrind and under the arena's
// own ASan container annotations, and MSVC is clean on Windows. Until this file existed the only
// account of that was a comment in build_all_platforms.sh asserting a mechanism nothing checked,
// which is what decides that the shipped Windows x86-64 artifacts are built natively with MSVC
// and that the mingw cross-build is a warned fallback. A shipping decision resting on an
// unverifiable comment is what this replaces.
//
// THE SHAPE, and every knob is one step away from the engine's own:
//   - `scratch()` holds a function-local `static thread_local` whose destructor frees a chain of
//     1 MB blocks. That is hypergraph/src/arena.cpp's `worker_scratch()`, whose
//     ConcurrentHeterogeneousArena destructor walks its block chain calling ::operator delete.
//   - `Guard` is a second function-local `static thread_local` whose destructor hands something
//     back to a process-lifetime static. That is job_system/src/job_pool.cpp's `TlsGuard`,
//     which calls release_pool() into a registry that outlives every worker.
//   - The worker thread exits, which is when both destructors run.
//
// WHAT THE KNOBS ESTABLISH, measured on this machine with mingw-w64 13-posix, 16 blocks, one
// worker, eight rounds, three runs each:
//
//   (no knobs)                                              exit 0, 0, 0
//   TWO_TLS                                                 exit 0, 0, 0
//   TWO_TLS GUARD_ALLOCATES                                 exit 0, 0, 0
//   TWO_TLS GUARD_TOUCHES_STATIC                            exit 0, 0, 0
//   TWO_TLS GUARD_TOUCHES_STATIC GUARD_ALLOCATES            exit 116, 116, 116
//   TWO_TLS SPLIT_FN GUARD_TOUCHES_STATIC GUARD_ALLOCATES   exit 0, 0, 0
//   THREE_TLS                                               exit 0, 0, 0
//
// So it takes TWO thread_local destructors declared in the SAME FUNCTION, the second of which
// both ALLOCATES and PUBLISHES the allocation into a static that outlives the thread. Split the
// two across functions and it is clean. Drop either half of the second destructor's work and it
// is clean.
//
// READ THAT AS BOUNDED, because the manifestation is heap-layout sensitive and every knob moves
// the layout. The SAME source and flags built as `t.exe` corrupts 3 of 3 and built as
// `two_tls_static_alloc.exe` is clean 3 of 3 -- the output filename alone decides it. The table
// above therefore holds the binary name constant (run.sh builds every cell as t.exe in its own
// directory), and with that held the map reproduces exactly. It still cannot separate "this knob
// is necessary" from "this knob shifted the layout enough to hide it", so a CLEAN cell is clean
// at that layout and nothing more. The same caution applies to any engine configuration that
// looks clean under mingw: the plan already recorded that one extra argv hides the engine's own
// failure, which is this same sensitivity seen from the other side.
//
// This is a sufficient condition, not the whole broken space, and it is not a claim that the
// engine's own thread_local set is this exact instance. What it establishes is the toolchain
// split.
//
// THE CONTROL. The identical source with the CORRUPTING knob set, built by MSVC 14.42 at /O2
// /MT on the same machine, is clean 5 of 5 where mingw is 116 ten times out of ten. The
// variable is the toolchain.

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

struct Registry {
    std::vector<void*> released;
    ~Registry() { for (void* p : released) ::operator delete(p); }
};

// A process-lifetime static, touched before any thread_local is constructed so that it outlives
// them -- the ordering job_pool.cpp's tls_pool() establishes for the same reason.
Registry& registry() { static Registry r; return r; }

struct BlockChain {
    std::vector<void*> blocks;

    void grow(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            void* p = ::operator new(1024 * 1024);
            // Touch a byte per page so the pages are real and nothing is optimised away.
            for (size_t j = 0; j < 1024 * 1024; j += 4096) static_cast<char*>(p)[j] = 1;
            blocks.push_back(p);
        }
    }
    ~BlockChain() { for (void* p : blocks) ::operator delete(p); }
};

struct Guard {
    ~Guard() {
#ifdef GUARD_TOUCHES_STATIC
        registry().released.push_back(
#ifdef GUARD_ALLOCATES
            ::operator new(64)
#else
            nullptr
#endif
        );
#elif defined(GUARD_ALLOCATES)
        void* p = ::operator new(64);
        ::operator delete(p);
#else
        // A non-trivial destructor that does nothing else, so "non-trivial" alone is testable.
        volatile int sink = 0; (void)sink;
#endif
    }
};

struct Third { std::vector<int> v{1, 2, 3}; ~Third() { v.clear(); } };

#if defined(TWO_TLS) && defined(SPLIT_FN)
void touch_guard() { static thread_local Guard g; (void)g; }
#endif

BlockChain& scratch(size_t nblocks) {
    static thread_local BlockChain chain;
#if defined(TWO_TLS) && !defined(SPLIT_FN)
    static thread_local Guard guard; (void)guard;
#endif
#ifdef THREE_TLS
    static thread_local Third third; (void)third;
#endif
    if (chain.blocks.empty()) chain.grow(nblocks);
    return chain;
}

}  // namespace

int main(int argc, char** argv) {
    const size_t nblocks  = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 16;
    const size_t nthreads = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1;
    const size_t rounds   = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 8;

    (void)registry();

    for (size_t r = 0; r < rounds; ++r) {
        std::vector<std::thread> workers;
        for (size_t t = 0; t < nthreads; ++t)
            workers.emplace_back([nblocks] {
#if defined(TWO_TLS) && defined(SPLIT_FN)
                touch_guard();
#endif
                scratch(nblocks);
            });
        for (auto& w : workers) w.join();
    }

    std::printf("ok: %zu round(s) of %zu worker(s), %zu MB each\n", rounds, nthreads, nblocks);
    std::fflush(stdout);
    return 0;
}
