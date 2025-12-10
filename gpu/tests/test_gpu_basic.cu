#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>

// Basic CUDA sanity tests to verify GPU is working

__global__ void simple_kernel(int* output) {
    output[threadIdx.x] = threadIdx.x * 2;
}

TEST(GPU_Basic, CudaWorks) {
    int* d_output;
    int h_output[32];

    cudaError_t err = cudaMalloc(&d_output, 32 * sizeof(int));
    ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);

    simple_kernel<<<1, 32>>>(d_output);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err);

    err = cudaMemcpy(h_output, d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(h_output[i], i * 2);
    }

    cudaFree(d_output);
}

// Test struct with device pointers
struct TestStruct {
    int* data;
    int size;
};

__global__ void struct_kernel(TestStruct s) {
    // s is copied to device - s.data points to device memory
    if (threadIdx.x < s.size) {
        s.data[threadIdx.x] = threadIdx.x * 3;
    }
}

TEST(GPU_Basic, StructPassedByValue) {
    TestStruct s;
    s.size = 32;

    cudaError_t err = cudaMalloc(&s.data, 32 * sizeof(int));
    ASSERT_EQ(err, cudaSuccess);

    // Pass struct BY VALUE - this copies the struct to device
    struct_kernel<<<1, 32>>>(s);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess);

    int h_output[32];
    err = cudaMemcpy(h_output, s.data, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(h_output[i], i * 3);
    }

    cudaFree(s.data);
}

__global__ void struct_kernel_ptr(TestStruct* s) {
    // s is a pointer to device memory containing the struct
    if (threadIdx.x < s->size) {
        s->data[threadIdx.x] = threadIdx.x * 5;
    }
}

TEST(GPU_Basic, StructPassedByPointer) {
    TestStruct h_s;
    h_s.size = 32;

    cudaError_t err = cudaMalloc(&h_s.data, 32 * sizeof(int));
    ASSERT_EQ(err, cudaSuccess);

    // Allocate device copy of struct
    TestStruct* d_s;
    err = cudaMalloc(&d_s, sizeof(TestStruct));
    ASSERT_EQ(err, cudaSuccess);

    // Copy struct to device
    err = cudaMemcpy(d_s, &h_s, sizeof(TestStruct), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // Pass pointer to device struct
    struct_kernel_ptr<<<1, 32>>>(d_s);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess);

    int h_output[32];
    err = cudaMemcpy(h_output, h_s.data, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    for (int i = 0; i < 32; i++) {
        EXPECT_EQ(h_output[i], i * 5);
    }

    cudaFree(h_s.data);
    cudaFree(d_s);
}
