//
// Created by mbarbone on 11/3/22.
//
#include <mixmax/mixmax.h>

#include <chrono>
#include <iostream>
#include <mixmax.hpp>
#undef MULWU
#undef MOD_MERSENNE
#include <functional>
#include <random>
#include <thread>

constexpr auto ITERATIONS = 1ULL << 31;
constexpr auto RUNS       = 5;

const auto seed1          = std::random_device()();
const auto seed2          = std::random_device()();
const auto seed3          = std::random_device()();
const auto seed4          = std::random_device()();

double test_original() {
    mixmax_engine gen{seed1, seed2, seed3, seed4};
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (auto i = 0ULL; i < ITERATIONS; ++i) {
        result = gen();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    return timeRequired.count();
}

double test_opt() {
    MIXMAX::MixMaxRng240 rng{seed1, seed2, seed3, seed4};
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (auto i = 0ULL; i < ITERATIONS; ++i) {
        result = rng();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    return timeRequired.count();
}

void test_seeding() {
    MIXMAX::MixMaxRng240 rng{seed1, seed2, seed3, seed4};
    mixmax_engine gen{seed1, seed2, seed3,
                      seed4};  // Create a Mixmax object and initialize the RNG with four 32-bit seeds 0,0,0,1
    for (auto i = 0ULL; i < ITERATIONS; ++i) {
        if (rng() != gen()) {
            throw std::runtime_error("RNG WRONG RESULT");
        }
    }
    std::cout << "ORIGINAL " << rng() << std::endl;
    std::cout << "OPT " << gen() << std::endl;
}

template <typename T>
T variance(const std::vector<T>& vec) {
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
    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

void benchmack(std::function<double()> func, const std::string& message) {
    double time = 0.;
    std::vector<double> times;
    times.reserve(RUNS);
    for (auto i = 0ULL; i < RUNS; ++i) {
        const auto current_time = func();
        times.emplace_back(current_time);
        time += current_time;
    }
    std::cout << message << " required " << time / RUNS << " milliseconds" << std::endl;
    std::cout << message << " stdev " << variance(times) << " milliseconds" << std::endl;
}

int main(int argc, char** argv) {
    test_seeding();
    benchmack(test_original, "ORIGINAL");
    benchmack(test_opt, "OPTIMIZED");
    return 0;
}