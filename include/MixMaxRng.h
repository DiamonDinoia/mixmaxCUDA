//
// Created by mbarbone on 11/3/22.
//

#ifndef MIXMAX_INCLUDE_MIXMAXRNG_H_
#define MIXMAX_INCLUDE_MIXMAXRNG_H_

#include <cstdint>
#include <ostream>

namespace RNG {

namespace {

#ifdef __CUDACC__

#define MIXMAX_HOST_AND_DEVICE __host__ __device__
#define MIXMAX_HOST __host__
#define MIXMAX_DEVICE __device__
#define MIXMAX_KERNEL __global__
#define MIXMAX_CLASS_ALIGN __align__(16)

#else

#define MIXMAX_HOST_AND_DEVICE
#define MIXMAX_HOST
#define MIXMAX_DEVICE
#define MIXMAX_KERNEL
#define MIXMAX_CLASS_ALIGN

#endif

#define likely(expr) __builtin_expect(!!(expr), 1)
};  // namespace

template <uint8_t M, uint8_t N = M - 1>
class MIXMAX_CLASS_ALIGN MixMaxRng {
   public:
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng() {
        for (auto& element : V) {
            element = 1;
        }
        counter    = N - 1;
        SumOverNew = 1;
    }
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(uint64_t seed) {
#ifndef __CUDA_ARCH__
        // a 64-bit LCG from Knuth line 26, in combination with a bit swap is used to seed
        const uint64_t MULT64 = 6364136223846793005ULL;
        uint64_t sum_total = 0, overflow = 0;
        uint64_t l = seed;
        for (unsigned long& i : V) {
            l *= MULT64;
            l = (l << 32) ^ (l >> 32);
            i = l & M61;
            sum_total += i;
            if (sum_total < i) {
                overflow++;
            }
        }
        counter    = N;  // set the counter to N if iteration should happen right away
        SumOverNew = MOD_MERSENNE(MOD_MERSENNE(sum_total) + (overflow << 3));
#else
        union unpack {
            struct {
                uint32_t low;   // lower 32 bits
                uint32_t high;  // upper 32 bits
            } split;
            uint64_t element;
        };
        unpack seeds{};
        seeds.element         = seed;

        const uint64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
        unpack streams{};
        streams.element = stream;
        appplyBigSkip(seeds.split.high, seeds.split.low, streams.split.high, streams.split.low);
#endif
    }

    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t streamID) {
        appplyBigSkip(clusterID, machineID, runID, streamID);
    }

    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(uint64_t seed, uint64_t stream) {
        union unpack {
            struct {
                uint32_t low;   // lower 32 bits
                uint32_t high;  // upper 32 bits
            } split;
            uint64_t element;
        };
        unpack seeds{};
        seeds.element = seed;
        unpack streams{};
        streams.element = stream;
        appplyBigSkip(seeds.split.high, seeds.split.low, streams.split.high, streams.split.low);
    }

    /**
     * Using CPP 17 features to return int or float depending on the need
     * @tparam T
     * @return
     */
    MIXMAX_HOST_AND_DEVICE
    inline constexpr uint64_t operator()() noexcept {
        if (likely(counter != (N - 1))) {
            counter += 1;
        } else {
            updateState();
        }
        return V[counter];
    }

    MIXMAX_HOST_AND_DEVICE
    inline constexpr double getFloat() noexcept {
        const auto u = operator()();
        auto f       = static_cast<const double>(u);
        return f * INV_MERSBASE;
    }

    const uint64_t* GetV() const { return V; }
    void setV(const uint64_t* v) {
        for (auto i = 0; i < N; i++) {
            V[i] = v[i];
        }
    }
    uint64_t GetSumOverNew() const { return SumOverNew; }
    void SetSumOverNew(uint64_t sum_over_new) { SumOverNew = sum_over_new; }
    uint8_t GetCounter() const { return counter; }
    void SetCounter(uint8_t counter) { MixMaxRng::counter = counter; }

   private:
    alignas(16) uint64_t V[N];
    uint64_t SumOverNew;
#ifndef __CUDA_ARCH__
    uint32_t counter;
#else
    uint8_t counter;
#endif
    //
    static constexpr double INV_MERSBASE = 0.43368086899420177360298E-18;
    static constexpr uint8_t BITS        = 61U;
    static constexpr uint64_t M61        = 0x1FFFFFFFFFFFFFFF;

    MIXMAX_HOST_AND_DEVICE
    static inline constexpr uint64_t Rotate_61bit(const uint64_t aVal, const std::size_t aSize) {
        return ((aVal << aSize) & M61) | (aVal >> (61 - aSize));
    }
    MIXMAX_HOST_AND_DEVICE
    static inline constexpr uint64_t MOD_MERSENNE(uint64_t aVal) { return (aVal & M61) + (aVal >> 61); }

    MIXMAX_HOST_AND_DEVICE
    inline constexpr void updateState() {
        static_assert(N == 16 || N == 7);
        uint64_t PartialSumOverOld = V[0];
        auto lV = V[0] = MOD_MERSENNE(SumOverNew + PartialSumOverOld);
        SumOverNew     = MOD_MERSENNE(SumOverNew + lV);
        if constexpr (N == 16) {
#pragma unroll 15
            for (int i = 1; i < N; ++i) {
                const auto lRotatedPreviousPartialSumOverOld = Rotate_61bit(PartialSumOverOld, 36);
                PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + V[i]);
                lV = V[i]  = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
                SumOverNew = MOD_MERSENNE(SumOverNew + lV);
            }
        } else if constexpr (N == 7) {
#pragma unroll 6
            for (int i = 1; i < N; ++i) {
                const auto lRotatedPreviousPartialSumOverOld = Rotate_61bit(PartialSumOverOld, 36);
                PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + V[i]);
                lV = V[i]  = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
                SumOverNew = MOD_MERSENNE(SumOverNew + lV);
            }
        }
        counter = 0;
    }

#if defined(__x86_64__)
    MIXMAX_HOST_AND_DEVICE
    inline static constexpr uint64_t mod128(__uint128_t s) {
        uint64_t s1 = ((((uint64_t)s) & M61) + (((uint64_t)(s >> 64)) * 8) + (((uint64_t)s) >> BITS));
        return MOD_MERSENNE(s1);
    }
    MIXMAX_HOST_AND_DEVICE
    inline static constexpr uint64_t fmodmulM61(uint64_t cum, uint64_t a, uint64_t b) {
        __uint128_t temp = (__uint128_t)a * (__uint128_t)b + cum;
        return mod128(temp);
    }
#else
    MIXMAX_HOST_AND_DEVICE
    uint64_t MixMaxRng::fmodmulM61(uint64_t cum, uint64_t s, uint64_t a) {
        const uint64_t MASK32 = 0xFFFFFFFFULL;
        uint64_t o, ph, pl, ah, al;
        o = (s)*a;
        ph = ((s) >> 32);
        pl = (s)&MASK32;
        ah = a >> 32;
        al = a & MASK32;
        o = (o & M61) + ((ph * ah) << 3) + ((ah * pl + al * ph + ((al * pl) >> 32)) >> 29);
        o += cum;
        o = (o & M61) + ((o >> 61));
        return o;
    }
#endif

    MIXMAX_HOST_AND_DEVICE
    void appplyBigSkip(uint32_t clusterID, uint32_t machineID, uint32_t runID, uint32_t streamID) {
        /*
         * makes a derived state vector, Vout, from the mother state vector Vin
         * by skipping a large number of steps, determined by the given seeding ID's
         * it is mathematically guaranteed that the substreams derived in this way from the SAME (!!!) Vin will not
         * collide provided
         * 1) at least one bit of ID is different
         * 2) less than 10^100 numbers are drawn from the stream
         * (this is good enough : a single CPU will not exceed this in the lifetime of the universe, 10^19 sec,
         * even if it had a clock cycle of Planch time, 10^44 Hz )
         * Caution: never apply this to a derived vector, just choose some mother vector Vin, for example the unit
         * ector by seed_vielbein(X,0), and use it in all your runs, just change runID to get completely nonoverlapping
         * treams of random numbers on a different day. clusterID and machineID are provided for the benefit of large
         * rganizations who wish to ensure that a simulation which is running in parallel on a large number of clusters
         * nd machines will have non-colliding source of random numbers. did I repeat it enough times? the
         * on-collision guarantee is absolute, not probabilistic
         */
        constexpr uint64_t skipMat17[128][17] =
#include "mixmax_skip_N17.h"
            ;
        constexpr uint64_t skipMat8[128][8] =
#include "mixmax_skip_N8.h"
            ;
        const uint64_t* skipMat[128];

        for (int i = 0; i < 128; i++) {
            if constexpr (N == 7) {
                skipMat[i] = skipMat8[i];
            }
            if constexpr (N == 16) {
                skipMat[i] = skipMat17[i];
            }
        }

        V[0] = 1;
        for (int i = 1; i < N; i++) {
            V[i]       = 0;
            SumOverNew = MOD_MERSENNE(SumOverNew + i);
        }

        u_int32_t IDvec[4] = {streamID, runID, machineID, clusterID};
        for (auto IDindex = 0; IDindex < 4; IDindex++) {  // go from lower order to higher order ID
            auto id = IDvec[IDindex];
            auto r  = 0;
            while (id) {
                if (id & 1) {
                    auto& rowPtr    = skipMat[r + IDindex * 8 * sizeof(uint32_t)];
                    uint64_t cum[N] = {0};

                    for (auto j = 0; j < N; j++) {  // j is lag, enumerates terms of the poly
                        // for zero lag Y is already given
                        auto coeff = rowPtr[j];  // same coeff for all i
                        for (auto i = 0; i < N; i++) {
                            cum[i] = fmodmulM61(cum[i], coeff, V[i]);
                        }
                        SumOverNew = fmodmulM61(cum[N - 1], coeff, V[N - 1]);
                        updateState();
                    }
                    for (auto i = 0; i < N; i++) {
                        V[i] = cum[i];
                    }
                }
                id = (id >> 1);
                r++;  // bring up the r-th bit in the ID
            }
        }
        for (unsigned long i : V) {
            SumOverNew += MOD_MERSENNE(SumOverNew + i);
        }
        counter = 0;
    }

   public:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const MixMaxRng& rng) {
        os << "V: ";
        for (const auto& elem : rng.V) {
            os << elem << " ";
        }
        os << " counter: " << static_cast<u_int32_t>(rng.counter) << " SumOverNew: " << rng.SumOverNew;
        return os;
    }
#endif
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return 0x1FFFFFFFFFFFFFFF; }
};

using MixMaxRng8  = MixMaxRng<8>;
using MixMaxRng17 = MixMaxRng<17>;

}  // namespace RNG

#endif  // MIXMAX_INCLUDE_MIXMAXRNG_H_
