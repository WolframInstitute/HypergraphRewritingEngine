#pragma once
#include "hgcommon/namespace.hpp"
//
// Whether this process can run the port's kernels on CUDA device 0.
//
// Host-only: no CUDA header, so the paclet's C++ front end includes it without nvcc.

#include <string>

namespace HG_NAMESPACE {
namespace gpu {

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
};

// Checks the device count, the driver version and whether a kernel image in this binary loads on
// device 0. The checks run on the first call; later calls return the same result.
const DeviceProbe& probe_device();

}  // namespace gpu
}  // namespace HG_NAMESPACE
