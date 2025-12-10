// CUDA Warmup - runs before all tests to pay the WSL2 GPU initialization cost
// This is a gtest "Environment" that initializes CUDA once at startup

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>

// Minimal warmup kernel
__global__ void warmup_kernel() {
    // Empty - just triggers CUDA initialization
}

class CudaWarmupEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // Force CUDA initialization by doing a minimal operation
        // This pays the ~2s WSL2 overhead once, before any tests run

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            fprintf(stderr, "CUDA warmup: No CUDA devices found\n");
            return;
        }

        // Set device and trigger context creation
        cudaSetDevice(0);

        // Launch a minimal kernel to fully initialize
        warmup_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();

        // Print device info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("CUDA initialized: %s (SM %d.%d, %zu MB)\n",
               prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024));
    }

    void TearDown() override {
        // Optional: reset device at end
        cudaDeviceReset();
    }
};

// Register the environment - gtest will call SetUp() before any tests
// This uses a constructor that runs before main()
static int register_cuda_warmup = []() {
    ::testing::AddGlobalTestEnvironment(new CudaWarmupEnvironment());
    return 0;
}();
