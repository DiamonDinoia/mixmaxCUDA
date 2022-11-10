//
// Created by mbarbone on 11/3/22.
//
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

#include <chrono>
#include <iostream>

#include "clean.h"
#include "mixmax/mixmax.h"
#include "original.h"

#define BLOCKS (82 * 16)
//#define BLOCKS 128
#define THREADS 128
#define SIZE (BLOCKS * THREADS)
#define TESTS (1 << 20)
#define iterations (1 << 20)

using namespace clean;
using MixMaxRng17 = MIXMAX::MixMaxRng17;

#define CUDA_CALL(x)                                        \
    do {                                                    \
        if ((x) != cudaSuccess) {                           \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define CURAND_CALL(x)                                      \
    do {                                                    \
        if ((x) != CURAND_STATUS_SUCCESS) {                 \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

class GPURandom {
   public:
    __device__ explicit GPURandom(unsigned long seed) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &state);
    }

    __device__ inline double getUniform() { return curand_uniform_double(&state); }

   private:
    curandState state{};
};
__global__ void initialize_rngs(uint64_t seed, GPURandom* curand_rngs, rng_state_t* mixmax_rngs,
                                MixMaxRng17* mixmax_opts) {
    curand_rngs[blockIdx.x * blockDim.x + threadIdx.x] = GPURandom{seed};
    mixmax_rngs[blockIdx.x * blockDim.x + threadIdx.x] = rng_state_t{};
    mixmax_opts[blockIdx.x * blockDim.x + threadIdx.x] = MixMaxRng17{seed};
}

__global__ void run_curdand(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto rng     = curand_rngs[blockIdx.x * blockDim.x + threadIdx.x];
    double value = 0;
    for (long i = 0; i < iterations; ++i) {
        value += rng.getUniform();
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void run_mtgp32(curandStateMtgp32* state, double* results) {
    auto rng     = state[blockIdx.x];
    double value = 0;
    for (long i = 0; i < iterations; ++i) {
        value += curand_uniform_double(&rng);
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void run_mixmax(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto rng     = mixmax_rngs[blockIdx.x * blockDim.x + threadIdx.x];
    double value = 0;
    for (long i = 0; i < iterations; ++i) {
        value += rng.getfloat();
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void run_mixmax2(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto rng     = mixmax_rngs[blockIdx.x * blockDim.x + threadIdx.x];
    double value = 0;
    for (long i = 0; i < iterations; ++i) {
        value += rng.getfloat2();
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

__global__ void run_mixmax_opt(MixMaxRng17* rngs, double* results) {
    auto rng     = rngs[blockIdx.x * blockDim.x + threadIdx.x];
    double value = 0;
    for (long i = 0; i < iterations; ++i) {
        value += rng.getFloat();
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

void test_curand(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto start = std::chrono::steady_clock::now();
    run_curdand<<<BLOCKS, THREADS>>>(curand_rngs, mixmax_rngs, results);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "CURAND required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_clean(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto start = std::chrono::steady_clock::now();
    run_mixmax<<<BLOCKS, THREADS>>>(curand_rngs, mixmax_rngs, results);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "CLEAN  required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_clean_2(GPURandom* curand_rngs, rng_state_t* mixmax_rngs, double* results) {
    auto start = std::chrono::steady_clock::now();
    run_mixmax2<<<BLOCKS, THREADS>>>(curand_rngs, mixmax_rngs, results);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "CLEAN 2 required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_opt(MixMaxRng17* mixmax_opts, double* results) {
    auto start = std::chrono::steady_clock::now();
    run_mixmax_opt<<<BLOCKS, THREADS>>>(mixmax_opts, results);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "OPTIMIZED 2 required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_mtgp32(curandStateMtgp32* state, double* results) {
    auto start = std::chrono::steady_clock::now();
    run_mtgp32<<<BLOCKS, THREADS>>>(state, results);
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "MTGP32 required " << timeRequired.count() << " milliseconds" << std::endl;
}

__global__ void runRNG(uint64_t* results) {
    MixMaxRng17 rng{};
    for (uint64_t i = 0; i < TESTS; ++i) {
        results[i] = rng();
    }
}

void check_result() {
    uint64_t* gpu_results;
    CUDA_CALL(cudaMalloc(&gpu_results, sizeof(uint64_t) * TESTS));
    runRNG<<<1, 1>>>(gpu_results);
    CUDA_CALL(cudaDeviceSynchronize());
    std::vector<uint64_t> results(TESTS);
    CUDA_CALL(cudaMemcpy(results.data(), gpu_results, sizeof(uint64_t) * TESTS, cudaMemcpyDefault));
    original::rng_state_t original{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2};
    original.sumtot = original::iterate_raw_vec(original.V, original.sumtot);
    for (uint64_t i = 0; i < TESTS; ++i) {
        const auto reference = original::flat(&original);
        if (results[i] == reference) {
            std::cout << "ERROR: " << results[i] << "!=" << reference << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "TESTS [OK]" << std::endl;
}
void printDevProp(cudaDeviceProp devProp) {
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i) printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %lu\n", devProp.totalConstMem);
    printf("Texture alignment:             %lu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Max Threads Per Multi Processor:     %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Max Blocks Per Multi Processor:     %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}
int main(const int argc, const char** argv) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printDevProp(deviceProp);
    GPURandom* curand_rngs;
    rng_state_t* mixmax_rngs;
    MixMaxRng17* mixmax_opts;
    curandStateMtgp32* devMTGPStates;
    mtgp32_kernel_params* devKernelParams;
    double* results;
    CUDA_CALL(cudaMalloc(&curand_rngs, sizeof(GPURandom) * SIZE));
    CUDA_CALL(cudaMalloc(&mixmax_rngs, sizeof(rng_state_t) * SIZE));
    CUDA_CALL(cudaMalloc(&mixmax_opts, sizeof(MixMaxRng17) * SIZE));
    CUDA_CALL(cudaMalloc(&devMTGPStates, sizeof(curandStateMtgp32) * BLOCKS));
    CUDA_CALL(cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)));
    CUDA_CALL(cudaMalloc(&results, sizeof(results) * SIZE));

//    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
//    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, BLOCKS, 42));

    initialize_rngs<<<BLOCKS, THREADS>>>(42, curand_rngs, mixmax_rngs, mixmax_opts);
    CUDA_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 5; ++i) {
        test_curand(curand_rngs, mixmax_rngs, results);
        test_clean(curand_rngs, mixmax_rngs, results);
        test_clean_2(curand_rngs, mixmax_rngs, results);
        test_opt(mixmax_opts, results);
//        test_mtgp32(devMTGPStates, results);
    }
    cudaFree(curand_rngs);
    cudaFree(mixmax_rngs);
    cudaFree(mixmax_opts);
    cudaFree(results);
    cudaFree(devMTGPStates);
    cudaFree(devKernelParams);
    check_result();
    return 0;
}