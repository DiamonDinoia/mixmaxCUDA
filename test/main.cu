//
// Created by mbarbone on 11/3/22.
//
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <mixmax/mixmax.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <mixmax.hpp>
#undef MULWU
#undef MOD_MERSENNE
#include <numeric>
#include <random>

constexpr auto ITERATIONS = 1ULL << 24;
constexpr auto TESTS      = 1ULL << 20;
constexpr auto RUNS       = 5;

using namespace MIXMAX;

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
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

template <typename T = curandState>
class GPURandom {
   public:
    __device__ explicit GPURandom(unsigned long seed) {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &state);
    }

    __device__ inline double operator()() { return curand_uniform_double(&state); }

   private:
    T state{};
};

template <uint N>
class MixMaxGPU : public internal::MixMaxRng<N> {
   public:
    __host__ __device__ inline double operator()() { return internal::MixMaxRng<N>::Uniform(); }
};

template <typename T>
__global__ void initialize_rngs(uint64_t seed, T* curand_rngs) {
    curand_rngs[blockIdx.x * blockDim.x + threadIdx.x] = T{seed};
}

template <typename T>
__global__ void rngKernel(T* rngs, double* results) {
    auto idx     = std::is_same<curandStateMtgp32, T>() ? blockIdx.x : blockIdx.x * blockDim.x + threadIdx.x;
    auto rng     = rngs[idx];
    double value = 0;
    for (long i = 0; i < ITERATIONS; ++i) {
        if constexpr (std::is_same<curandStateMtgp32, T>()) {
            value += curand_uniform_double(&rng);
        } else {
            value += rng();
        }
    }
    results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

template <typename T>
T stdev(const std::vector<T>& vec) {
    const size_t sz = vec.size();
    if (sz == 1) {
        return 0.0;
    }
    // Calculate the mean
    const T mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;
    // Now calculate the variance
    auto variance_func = [&mean, &sz](T accumulator, const T& val) {
        return accumulator + ((val - mean) * (val - mean) / (sz - 1));
    };
    return std::sqrt(std::accumulate(vec.begin(), vec.end(), 0.0, variance_func));
}

template <typename T>
void Benchmack(const std::string& message, const uint64_t threads, const uint64_t blocks, const uint64_t seed) {

    const auto size = threads * blocks;
    T* gpuRGNs;
    double* results;
    mtgp32_kernel_params* devKernelParams;
    CUDA_CALL(cudaMalloc(&results, sizeof(double) * size));

    // RNG initialization
    if constexpr (std::is_same<curandStateMtgp32, T>()) {
        CUDA_CALL(cudaMalloc(&gpuRGNs, sizeof(T) * blocks));
        CUDA_CALL(cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)));
        CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));
        CURAND_CALL(curandMakeMTGP32KernelState(gpuRGNs, mtgp32dc_params_fast_11213, devKernelParams, blocks, seed));
    } else {
        CUDA_CALL(cudaMalloc(&gpuRGNs, sizeof(T) * size));
        initialize_rngs<<<blocks, threads>>>(seed, gpuRGNs);
    }

    std::vector<double> times;
    double total_time = 0.;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    times.reserve(RUNS);
    for (auto i = 0ULL; i < RUNS; ++i) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventRecord(start));
        rngKernel<<<blocks, threads>>>(gpuRGNs, results);
        CUDA_CALL(cudaEventRecord(stop));
        CUDA_CALL(cudaEventSynchronize(stop));
        float elapsedTime = 0;
        CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
        times.emplace_back(elapsedTime);
        total_time += elapsedTime;
    }

    CUDA_CALL(cudaFree(gpuRGNs));
    CUDA_CALL(cudaFree(results));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    if constexpr (std::is_same<curandStateMtgp32, T>()) {
        CUDA_CALL(cudaFree(devKernelParams));
    }
    std::cout << message << " required " << total_time / RUNS << "+-" << stdev(times) << " ms" << std::endl;
}

__global__ void runRNG(uint64_t* results, uint32_t seed1, uint32_t seed2, uint32_t seed3, uint32_t seed4) {
    MIXMAX::MixMaxRng240 rng{seed1, seed2, seed3, seed4};
    for (uint64_t i = 0; i < TESTS; ++i) {
        results[i] = rng();
    }
}

void check_result() {
    const auto seed1 = std::random_device()();
    const auto seed2 = std::random_device()();
    const auto seed3 = std::random_device()();
    const auto seed4 = std::random_device()();
    uint64_t* gpu_results;
    CUDA_CALL(cudaMalloc(&gpu_results, sizeof(uint64_t) * TESTS));
    runRNG<<<1, 1>>>(gpu_results, seed1, seed2, seed3, seed4);
    CUDA_CALL(cudaDeviceSynchronize());
    std::vector<uint64_t> results(TESTS);
    CUDA_CALL(cudaMemcpy(results.data(), gpu_results, sizeof(uint64_t) * TESTS, cudaMemcpyDeviceToHost));
    mixmax_engine gen{seed1, seed2, seed3,
                      seed4};  // Create a Mixmax object and initialize the RNG with four 32-bit seeds 0,0,0,1
    for (uint64_t i = 0; i < TESTS; ++i) {
        const auto reference = gen();
        if (results[i] != reference) {
            std::cout << "ERROR: " << results[i] << "!=" << reference << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    CUDA_CALL(cudaFree(gpu_results));
    std::cout << "TESTS [OK]" << std::endl;
}

int main(const int argc, const char** argv) {
    check_result();
    const auto seed = std::random_device()();
    auto threads    = 128;
    auto blocks     = 82 * 12;
    std::cout << "Using seed " << seed << std::endl;
    std::cout << "Using blocks " << blocks << " threads " << threads << std::endl;
    Benchmack<GPURandom<curandStatePhilox4_32_10>>("Philox4_32_10", threads, blocks, seed);
    Benchmack<GPURandom<curandStateMRG32k3a>>("MRG32k3a", threads, blocks, seed);
    Benchmack<GPURandom<curandStateXORWOW>>("XORWOW", threads, blocks, seed);
    Benchmack<MixMaxGPU<240>>("MixMaxGPU<240>", threads, blocks, seed);
    Benchmack<MixMaxGPU<17>>("MixMaxGPU<17>", threads, blocks, seed);
    Benchmack<MixMaxGPU<8>>("MixMaxGPU<8>", threads, blocks, seed);
    threads = 256;
    blocks  = 128;
    std::cout << "Using blocks " << blocks << " threads " << threads << std::endl;
    Benchmack<curandStateMtgp32>("MTGP32", threads, blocks, seed);
    Benchmack<GPURandom<curandStatePhilox4_32_10>>("Philox4_32_10", threads, blocks, seed);
    Benchmack<GPURandom<curandStateMRG32k3a>>("MRG32k3a", threads, blocks, seed);
    Benchmack<GPURandom<curandStateXORWOW>>("XORWOW", threads, blocks, seed);
    Benchmack<MixMaxGPU<240>>("MixMaxGPU<240>", threads, blocks, seed);
    Benchmack<MixMaxGPU<17>>("MixMaxGPU<17>", threads, blocks, seed);
    Benchmack<MixMaxGPU<8>>("MixMaxGPU<8>", threads, blocks, seed);
    return 0;
}