#pragma once
#include "hgcommon/namespace.hpp"

// THE HOST MEMORY LIMIT: Tuning "MemoryLimit", HG_TUNE_MEMORY_LIMIT, in bytes.
//
// The limit is the job's "Tuning" value, else the environment's, else half of physical memory,
// and never more than physical memory. Half leaves the other half for the process that reads
// the reply: the Wolfram kernel or Python decodes the reply into its own expressions, which take
// memory of the same order as the engine's result.
//
// Two mechanisms enforce it, and both read resolved_memory_limit():
// - The engine's arenas (hypergraph/arena.hpp, JobMemoryLimit) refuse to map a block that would
//   take the bytes they hold, with the end-of-run reserve, above the limit. This applies in
//   every process: hg_evolve, the paclet's library and the Python module.
// - hg_evolve also installs an operating-system limit at start (install_process_memory_limit),
//   which covers every allocation of the process. Windows: a Job Object with
//   JOB_OBJECT_LIMIT_PROCESS_MEMORY and JOB_OBJECT_LIMIT_JOB_MEMORY (committed memory). Linux:
//   RLIMIT_DATA (private writable mappings, heap and anonymous mmap). RLIMIT_AS also counts
//   address space reserved and never committed: the CUDA runtime reserves tens of GB of it and
//   does not initialise under a lowered RLIMIT_AS. macOS enforces neither RLIMIT_DATA nor
//   RLIMIT_AS on mmap and has no job objects, so there the arenas' check is the only one.
// An allocation over either limit fails with std::bad_alloc, and the evolution stops with a
// partial result and the "memory exhausted" warning (ParallelEvolutionEngine::raise_worker_error).

#include <cstdint>

#include "hgcommon/tuning.hpp"

namespace HG_NAMESPACE {
namespace common {

inline constexpr uint64_t kNoMemoryLimit = ~uint64_t{0};

// Half of `physical`; kNoMemoryLimit when physical memory is unknown (0).
constexpr uint64_t derived_memory_limit(uint64_t physical) {
    return physical ? physical / 2 : kNoMemoryLimit;
}

// `requested`, at most `physical` when physical memory is known.
constexpr uint64_t clamp_memory_limit(uint64_t requested, uint64_t physical) {
    return physical && requested > physical ? physical : requested;
}

// The limit for a job with overrides `job`: its "MemoryLimit", else HG_TUNE_MEMORY_LIMIT, else
// derived_memory_limit(physical memory); then clamp_memory_limit.
uint64_t resolved_memory_limit(const TuningOverrides& job, TuneSource* source = nullptr);

enum class MemoryLimitBackend {
    JobObject,   // Windows
    RlimitData,  // Linux
    None         // macOS and anything else: the arenas' check only
};

constexpr MemoryLimitBackend memory_limit_backend() {
#if defined(HG_VERIFICATION)
    return MemoryLimitBackend::None;
#elif defined(_WIN32)
    return MemoryLimitBackend::JobObject;
#elif defined(__linux__)
    return MemoryLimitBackend::RlimitData;
#else
    return MemoryLimitBackend::None;
#endif
}

const char* memory_limit_backend_name(MemoryLimitBackend b);

// Puts this process under the operating-system limit of `bytes`. On Linux the hard RLIMIT_DATA is
// lowered to physical memory, so a later job can raise the limit up to physical memory and no
// further. False where the platform has no backend or the call fails; the arenas' check applies
// either way. Called once, by hg_evolve's main.
bool install_process_memory_limit(uint64_t bytes);

// Changes the installed limit to `bytes`. False when none is installed or the call fails.
bool set_process_memory_limit(uint64_t bytes);

// The installed limit in bytes; 0 when none is installed.
uint64_t process_memory_limit();

}  // namespace common
}  // namespace HG_NAMESPACE
