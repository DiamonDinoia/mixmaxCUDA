//
// Created by mbarbone on 11/3/22.
//
#include <chrono>
#include <iostream>

#include "MixMaxRng.h"
#include "clean.h"
#include "original.h"

constexpr auto iterations = 1 << 30;

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
    RNG::MixMaxRng17 rng{};
    std::cout << rng << std::endl;
    uint64_t result;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations + 1; ++i) {
        result = rng();
    }
    auto end = std::chrono::steady_clock::now();
    //
    std::chrono::duration<double, std::milli> timeRequired = (end - start);
    std::cout << "result " << result << std::endl;
    std::cout << "OPTIMIZED required " << timeRequired.count() << " milliseconds" << std::endl;
}

int main(int argc, char **argv) {
    test_original();
    test_clean();
    test_clean_2();
    test_opt();
    return 0;
}