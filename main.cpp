//
// Created by mbarbone on 11/3/22.
//
#include <chrono>
#include <iostream>

#include "clean.h"
#include "mixmax/mixmax.h"
#include "original.h"
#include "mixmax.hpp"
#include <random>

constexpr auto iterations = 1ULL << 30;

void test_original() {
    original::rng_state_t original{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 2};
    original.sumtot = original::iterate_raw_vec(original.V, original.sumtot);
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        result = original::flat(&original);
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    std::cout << "Original required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_clean() {
    clean::rng_state_t clean{};
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations + 1; ++i) {
        result = clean.get();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    std::cout << "Clean required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_clean_2() {
    clean::rng_state_t clean{};
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations + 1; ++i) {
        result = clean.get2();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    std::cout << "Clean2 required " << timeRequired.count() << " milliseconds" << std::endl;
}

void test_opt() {
    MIXMAX::MixMaxRng17 rng{};
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        result = rng();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    std::cout << "OPTIMIZED required " << timeRequired.count() << " milliseconds" << std::endl;
}
void test_seeding() {
    const auto seed1 = std::random_device()();
    const auto seed2 = std::random_device()();
    const auto seed3 = std::random_device()();
    const auto seed4 = std::random_device()();
    MIXMAX::MixMaxRng240 rng{seed1, seed2, seed3, seed4};
    mixmax_engine gen{seed1, seed2, seed3, seed4};  // Create a Mixmax object and initialize the RNG with four 32-bit seeds 0,0,0,1
    for (int i = 0; i < iterations; ++i) {
        if (rng() != gen()) {
            throw std::runtime_error("RNG WRONG RESULT");
        }
    }
    std::cout << "ORIGINAL " << rng() << std::endl;
    std::cout << "OPT " << gen() << std::endl;
}

void test_branching() {
    mixmax_engine gen{};  // Create a Mixmax object and initialize the RNG with four 32-bit seeds 0,0,0,1
    gen.seed_spbox(42);
    MIXMAX::MixMaxRng240 rng{42};
    for (int i = 0; i < iterations; ++i) {
        if (rng() != gen()) {
            throw std::runtime_error("RNG WRONG RESULT");
        }
    }
}

int main(int argc, char **argv) {
    test_branching();
    test_original();
    test_clean();
    test_clean_2();
    test_opt();
    test_seeding();
    return 0;
}