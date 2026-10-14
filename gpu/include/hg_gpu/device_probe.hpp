#pragma once
#include "hgcommon/namespace.hpp"
//
// Whether this process can run the port's kernels on CUDA device 0.
//
// Host-only: no CUDA header, so the paclet's C++ front end includes it without nvcc.

#include <cstdint>
#include <string>

namespace HG_NAMESPACE {
namespace gpu {

// Every hardware property of device 0 that a tuning value is derived from. No other code calls
// cudaGetDeviceProperties or cudaDeviceGetAttribute for these.
struct DeviceProbe {
    bool usable = false;
    // Empty when usable. Otherwise one of: "no CUDA-capable device is detected"; "no NVIDIA
    // driver is installed"; "the NVIDIA driver supports CUDA 11.4 and this build needs CUDA 12.0
    // or later"; "compute capability 6.1 of <name> is not in this build's list (7.5, 8.9)"; or
    // the CUDA runtime's own error string.
    std::string reason;
    std::string name;        // device 0's name; empty when no device was found
    int major = 0;           // device 0's compute capability
    int minor = 0;
    // Filled when cudaGetDeviceProperties succeeded, 0 otherwise.
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int max_blocks_per_sm = 0;
    int regs_per_sm = 0;                  // 32-bit registers
    uint64_t shared_per_sm = 0;           // bytes
    uint64_t shared_per_block_optin = 0;  // bytes a block may opt in to
    uint64_t l2_bytes = 0;
    uint64_t persisting_l2_max = 0;       // bytes an access policy window may set aside
    uint64_t total_bytes = 0;             // device memory, as cudaMemGetInfo reports it
    int driver_version = 0;               // e.g. 12040 for CUDA 12.4
    int runtime_version = 0;
};

// Checks the device count, the driver version and whether a kernel image in this binary loads on
// device 0. The checks run on the first call; later calls return the same result.
const DeviceProbe& probe_device();

// probe_device(), or std::runtime_error("no usable GPU: <reason>") when the device is not usable.
// Read by every value derived from the device's properties.
const DeviceProbe& usable_device();

// Blocks a persistent kernel launches when the caller does not choose: eight per SM, or
// HG_GPU_PERSISTENT_BLOCKS when set (gpu/src/persistent.cu).
//
// A persistent kernel's blocks do not retire and get replaced -- they live for the whole
// evolution -- so the grid IS the worker count. Exposed because the IR arena has to be sized
// from the same number: each resident worker holds its own slot, so arena demand scales with the
// grid, and a caller sizing the arena off the state budget alone starves it as soon as the grid
// grows. See gpu/src/persistent.cu for the sweep behind eight-per-SM.
uint32_t default_persistent_grid();

// Threads the device can hold resident: SMs x maximum threads per SM. The driver reserves the
// per-thread stack (cudaLimitStackSize) for this many threads, whatever grid is launched.
uint64_t device_resident_threads();

// The device-memory ceiling of a run: `requested` when non-zero, else 90% of
// DeviceProbe::total_bytes (gpu/src/evolve.cu).
uint64_t resolved_memory_cap(uint64_t requested);

}  // namespace gpu
}  // namespace HG_NAMESPACE
