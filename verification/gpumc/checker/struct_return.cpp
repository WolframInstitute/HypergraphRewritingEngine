// Checker test for the SROA pass in gpumc-fixes.patch: a 16-byte struct returned by value.
//
// The x86-64 ABI returns {uint64_t, bool, bool} in two registers: the callee stores the fields to
// a stack temporary and loads it as one 128-bit integer, the caller stores the integer and loads
// the fields. Without SROA the 128-bit load reads the 64-bit store of the first field.
//
// gpumc_docker:latest reports "Mixed-size accesses detected: tried to read event with a 128-bit
// access!"; gpumc_docker:hg-fixes reports one complete execution and no error.
// GPUMC-CELL: patched PASS
#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

struct Result {
    uint64_t value;
    bool inserted;
    bool overflowed;
};

__attribute__((noinline)) Result make(uint64_t v) { return Result{v, true, false}; }

uint64_t g_value;
bool g_inserted;

void* run(void*) {
    const Result r = make(41);
    g_value = r.value + 1;
    g_inserted = r.inserted && !r.overflowed;
    return nullptr;
}

}  // namespace

int main() {
    pthread_t t;
    pthread_create(&t, nullptr, run, nullptr);
    pthread_join(t, nullptr);
    assert(g_value == 42 && g_inserted);
    return 0;
}
