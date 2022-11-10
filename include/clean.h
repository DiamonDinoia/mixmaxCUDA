//
// Created by mbarbone on 11/3/22.
//

#ifndef MIXMAX__CLEAN_H_
#define MIXMAX__CLEAN_H_

#include <cstdint>
#include <iomanip>
#include <ostream>

namespace clean {

#define likely(expr) __builtin_expect(!!(expr), 1)

constexpr uint64_t M61(0x1FFFFFFFFFFFFFFF);

inline constexpr uint64_t Rotate_61bit(uint64_t aVal, std::size_t aSize) {
    return ((aVal << aSize) & M61) | (aVal >> (61 - aSize));
}

inline constexpr uint64_t MOD_MERSENNE(uint64_t aVal) { return (aVal & M61) + (aVal >> 61); }

inline constexpr double reinterpret(u_int64_t a) {
    union {
        uint64_t nVal;
        double flVal;
    } tOffset = {a};
    return tOffset.flVal;
}

constexpr uint64_t onemask = 0x3FF0000000000000;

struct rng_state_t {
    uint64_t V[16];
    int counter;
    inline constexpr rng_state_t()
        : V{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, counter(15), SumOverNew(1), PartialSumOverOld(0) {}
    uint64_t SumOverNew, PartialSumOverOld;

    // Update per call method
    inline constexpr uint64_t get() {
        if (likely(counter != 15)) {
            counter += 1;
            uint64_t RotatedPreviousPartialSumOverOld(Rotate_61bit(PartialSumOverOld, 36));
            PartialSumOverOld = MOD_MERSENNE(PartialSumOverOld + V[counter]);
            V[counter]        = MOD_MERSENNE(V[counter - 1] + PartialSumOverOld + RotatedPreviousPartialSumOverOld);
        } else {
            counter           = 0;
            PartialSumOverOld = V[0];
            V[0]              = MOD_MERSENNE(SumOverNew + PartialSumOverOld);
        }

        SumOverNew = MOD_MERSENNE(SumOverNew + V[counter]);

        return V[counter];
    }

    inline constexpr double getfloat() {
        const auto u       = get();
        const uint64_t tmp = (u >> 9) | onemask;  // bits between 52 and 62 dont affect the result!
        const double d     = reinterpret(tmp);
        return d - 1.0;
    }

    // Batch-update, more like the original
    inline constexpr uint64_t get2() {
        if (likely(counter != 15)) {
            counter += 1;
        } else {
            PartialSumOverOld = V[0];
            auto lV = V[0] = MOD_MERSENNE(SumOverNew + PartialSumOverOld);
            SumOverNew     = MOD_MERSENNE(SumOverNew + lV);
#pragma GCC unroll 15
            for (int i(1); i != 16; ++i) {
                const auto lRotatedPreviousPartialSumOverOld = Rotate_61bit(PartialSumOverOld, 36);
                PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + V[i]);
                lV = V[i]  = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
                SumOverNew = MOD_MERSENNE(SumOverNew + lV);
            }
            counter = 0;
        }
        return V[counter];
    }

    inline constexpr double getfloat2() {
        const auto u   = get2();
        uint64_t tmp   = (u >> 9) | onemask;  // bits between 52 and 62 dont affect the result!
        const double d = reinterpret(tmp);
        return d - 1.0;
    }
};

}  // namespace clean
#endif  // MIXMAX__CLEAN_H_
