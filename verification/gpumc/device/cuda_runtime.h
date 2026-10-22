#pragma once
// The host runtime API the device headers name. A GPUMC harness never calls it: every function
// returns cudaSuccess and does nothing.
#include "hg_gpumc.hpp"

#include <cstddef>

enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
struct dim3 {
    unsigned x = 1, y = 1, z = 1;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};

inline cudaError_t cudaMalloc(void**, size_t) { return cudaSuccess; }
template <class T>
inline cudaError_t cudaMalloc(T**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind,
                                   cudaStream_t = nullptr) {
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = nullptr) {
    return cudaSuccess;
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return ""; }
inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    *free_bytes = *total_bytes = 0;
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* d) {
    *d = 0;
    return cudaSuccess;
}
inline cudaError_t cudaGetDeviceCount(int* n) {
    *n = 1;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceReset() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMallocHost(void**, size_t) { return cudaSuccess; }
template <class T>
inline cudaError_t cudaMallocHost(T**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFreeHost(void*) { return cudaSuccess; }
inline cudaError_t cudaDriverGetVersion(int* v) {
    *v = 0;
    return cudaSuccess;
}
inline cudaError_t cudaRuntimeGetVersion(int* v) {
    *v = 0;
    return cudaSuccess;
}
struct cudaDeviceProp {
    char name[256] = {};
    int major = 0, minor = 0, multiProcessorCount = 0, maxThreadsPerMultiProcessor = 0;
    int maxBlocksPerMultiProcessor = 0, regsPerMultiprocessor = 0, l2CacheSize = 0;
    int persistingL2CacheMaxSize = 0;
    size_t sharedMemPerBlockOptin = 0, sharedMemPerMultiprocessor = 0;
};
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) { return cudaSuccess; }

enum cudaDeviceAttr { cudaDevAttrMaxAccessPolicyWindowSize, cudaDevAttrMultiProcessorCount };
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) {
    *v = 0;
    return cudaSuccess;
}
enum cudaLimit { cudaLimitPersistingL2CacheSize, cudaLimitStackSize };
inline cudaError_t cudaDeviceSetLimit(cudaLimit, size_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceGetLimit(size_t* v, cudaLimit) {
    *v = 0;
    return cudaSuccess;
}
inline cudaError_t cudaCtxResetPersistingL2Cache() { return cudaSuccess; }

struct cudaFuncAttributes {
    size_t localSizeBytes = 0, sharedSizeBytes = 0, constSizeBytes = 0;
    int maxThreadsPerBlock = 0, numRegs = 0;
};
template <class F>
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, F) {
    return cudaSuccess;
}
template <class F>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int, size_t) {
    *n = 1;
    return cudaSuccess;
}

enum cudaAccessProperty { cudaAccessPropertyNormal, cudaAccessPropertyStreaming,
                          cudaAccessPropertyPersisting };
struct cudaAccessPolicyWindow {
    void* base_ptr = nullptr;
    size_t num_bytes = 0;
    float hitRatio = 0;
    cudaAccessProperty hitProp = cudaAccessPropertyNormal, missProp = cudaAccessPropertyNormal;
};
enum cudaLaunchAttributeID { cudaLaunchAttributeAccessPolicyWindow };
struct cudaLaunchAttributeValue {
    cudaAccessPolicyWindow accessPolicyWindow;
};
struct cudaLaunchAttribute {
    cudaLaunchAttributeID id;
    cudaLaunchAttributeValue val;
};
struct cudaLaunchConfig_t {
    dim3 gridDim, blockDim;
    size_t dynamicSmemBytes = 0;
    cudaStream_t stream = nullptr;
    cudaLaunchAttribute* attrs = nullptr;
    unsigned numAttrs = 0;
};
template <class F, class... A>
inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t*, F, A&&...) {
    return cudaSuccess;
}

// A load and a store through the L2: plain accesses here.
template <class T>
inline T __ldcg(const T* p) {
    return *p;
}
template <class T>
inline void __stcg(T* p, T v) {
    *p = v;
}
