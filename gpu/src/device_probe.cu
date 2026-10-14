#include "hg_gpu/device_probe.hpp"

#include <cuda_runtime.h>

#include <string>

// CMAKE_CUDA_ARCHITECTURES of this build with commas, e.g. "75,80,86,89,90,100,120"
// (gpu/CMakeLists.txt).
#ifndef HG_GPU_ARCH_LIST
#define HG_GPU_ARCH_LIST ""
#endif

namespace HG_NAMESPACE {
namespace gpu {

namespace {

// Compiled for the same architectures as every other kernel in hg_gpu, so it loads on device 0
// exactly when they do.
__global__ void k_device_probe() {}

// 12040 -> "12.4".
std::string cuda_version_string(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

// "75,89-real,native" -> "7.5, 8.9, native".
std::string arch_list_string(const std::string& list) {
    std::string out, item;
    auto flush = [&] {
        if (item.empty()) return;
        const std::string digits = item.substr(0, item.find('-'));
        std::string shown = item;
        if (digits.size() >= 2 && digits.find_first_not_of("0123456789") == std::string::npos)
            shown = digits.substr(0, digits.size() - 1) + "." + digits.substr(digits.size() - 1);
        out += (out.empty() ? "" : ", ") + shown;
        item.clear();
    };
    for (char c : list) {
        if (c == ';' || c == ',' || c == ' ') flush();
        else item += c;
    }
    flush();
    return out;
}

DeviceProbe run_probe() {
    DeviceProbe p;
    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorInsufficientDriver) {
        int driver = 0;
        cudaDriverGetVersion(&driver);
        p.reason = driver == 0
            ? std::string("no NVIDIA driver is installed")
            : "the NVIDIA driver supports CUDA " + cuda_version_string(driver) +
              " and this build needs CUDA " + std::to_string(CUDART_VERSION / 1000) +
              ".0 or later";
    } else if (count_err == cudaErrorNoDevice || (count_err == cudaSuccess && count == 0)) {
        p.reason = "no CUDA-capable device is detected";
    } else if (count_err != cudaSuccess) {
        p.reason = cudaGetErrorString(count_err);
    } else {
        cudaDeviceProp prop{};
        const cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
        if (prop_err != cudaSuccess) {
            p.reason = cudaGetErrorString(prop_err);
        } else {
            p.name = prop.name;
            p.major = prop.major;
            p.minor = prop.minor;
            cudaFuncAttributes attrs{};
            const cudaError_t image_err =
                cudaFuncGetAttributes(&attrs, reinterpret_cast<const void*>(&k_device_probe));
            if (image_err == cudaErrorNoKernelImageForDevice ||
                image_err == cudaErrorInvalidDeviceFunction) {
                p.reason = "compute capability " + std::to_string(p.major) + "." +
                           std::to_string(p.minor) + " of " + p.name +
                           " is not in this build's list (" + arch_list_string(HG_GPU_ARCH_LIST) +
                           ")";
            } else if (image_err != cudaSuccess) {
                p.reason = cudaGetErrorString(image_err);
            }
        }
    }
    cudaGetLastError();  // the failed calls above leave a non-sticky error behind
    p.usable = p.reason.empty();
    return p;
}

}  // namespace

const DeviceProbe& probe_device() {
    static const DeviceProbe probe = run_probe();
    return probe;
}

}  // namespace gpu
}  // namespace HG_NAMESPACE
