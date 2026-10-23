// Checker test for the race recording in gpumc-fixes.patch: a data race whose access is in an
// included header is reported, and the assertion after the joins is still checked.
//
// Two threads store to one word with plain stores, both in race_in_header.hpp, with nothing
// ordering them; then main asserts. The run must report the race (both header lines) and the
// assertion.
//
// gpumc_docker:latest reports neither (2 complete executions, no error): it drops a race whose
// detecting access is not in the input file, and then returns without restoring the interpreter
// state it saved to describe the race.
// GPUMC-CELL: patched FAIL
#include "race_in_header.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

uint32_t g_word;

void* in_header(void*) {
    race_in_header_store_two(&g_word);
    return nullptr;
}

void* also_in_header(void*) {
    race_in_header_store_one(&g_word);
    return nullptr;
}

}  // namespace

int main() {
    pthread_t a, b;
    pthread_create(&a, nullptr, in_header, nullptr);
    pthread_create(&b, nullptr, also_in_header, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    assert(g_word == 7u && "the assertion after the joins is checked");
    return 0;
}
