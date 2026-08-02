#pragma once
//
// ONE CUDA error check.
//
// The rule is "if a CUDA call did not return cudaSuccess, throw a runtime_error naming what was
// attempted and what the driver said". It was written seventeen times: eight file-local copies,
// one per .cu, and nine more as private statics inside Pool, QcState, DeviceArena, EngineState,
// RingBuffer, HashTable, QeState, TerminationDetector and DeviceLockFreeList. Every copy was
// byte-identical apart from a hand-written module name in the message.
//
// THE MODULE NAME IS NOW THE CALL SITE. A literal like "hg_gpu::run_match_kernel" has to be
// maintained by hand and silently lies as soon as the code around it moves or is renamed;
// __FILE__ and __LINE__ cannot. That also makes the message strictly more useful than what it
// replaces -- it names the exact line, not the enclosing module.
//
// A macro rather than a function because the CUDA sources are compiled as C++17, where there is
// no std::source_location to capture the call site through a default argument.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace hg_gpu {

// Out of line from the check itself so the success path is a single comparison with no string
// machinery for the optimiser to carry through it.
[[noreturn]] inline void cuda_fail(cudaError_t err, const char* what,
                                   const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + what + ": " +
                             cudaGetErrorString(err));
}

inline void cuda_check_at(cudaError_t err, const char* what, const char* file, int line) {
    if (err != cudaSuccess) cuda_fail(err, what, file, line);
}

}  // namespace hg_gpu

#define HG_CUDA_CHECK(err, what) ::hg_gpu::cuda_check_at((err), (what), __FILE__, __LINE__)
