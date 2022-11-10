//
// Created by mbarbone on 11/3/22.
//

#ifndef MIXMAX_INCLUDE_MIXMAXRNG_H_
#define MIXMAX_INCLUDE_MIXMAXRNG_H_

#include <cstdint>
#include <ostream>

namespace MIXMAX {

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

};  // namespace

template <uint8_t M>
class MIXMAX_CLASS_ALIGN MixMaxRng {
   public:
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng() {
        for (auto& element : m_State) {
            element = 1;
        }
        m_SumOverNew = 1;
        updateState();
        m_counter = 1;
    }
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(uint64_t seed) {
#ifndef __CUDA_ARCH__
        // a 64-bit LCG from Knuth line 26, in combination with a bit swap is used to seed
        const uint64_t MULT64 = 6364136223846793005ULL;
        uint64_t sum_total = 0, overflow = 0;
        uint64_t l = seed;
        for (unsigned long& i : m_State) {
            l *= MULT64;
            l = (l << 32) ^ (l >> 32);
            i = l & M61;
            sum_total += i;
            if (sum_total < i) {
                overflow++;
            }
        }
        m_counter    = N;  // set the counter to N if iteration should happen right away
        m_SumOverNew = MOD_MERSENNE(MOD_MERSENNE(sum_total) + (overflow << 3));
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
        if (m_counter == N) {
            updateState();
        }
        return m_State[m_counter++];
    }

    MIXMAX_HOST_AND_DEVICE
    inline constexpr double getFloat() noexcept {
        const auto u = operator()();
        return static_cast<double>(u) * INV_MERSBASE;
    }

   private:
    // Constants
    static constexpr double INV_MERSBASE = 0.43368086899420177360298E-18;
    // The state is M-1 because the last element is stored in the variable SumOverNew outside the vector
    static constexpr uint8_t N    = M - 1;
    static constexpr uint8_t BITS = 61U;
    static constexpr uint64_t M61 = 0x1FFFFFFFFFFFFFFF;
    // RNG state
    alignas(16) uint64_t m_State[N];
    uint64_t m_SumOverNew;
    uint32_t m_counter;
    //

    MIXMAX_HOST_AND_DEVICE
    static inline constexpr uint64_t ROTATE_61(const uint64_t aVal, const std::size_t aSize) {
        return ((aVal << aSize) & M61) | (aVal >> (61 - aSize));
    }
    MIXMAX_HOST_AND_DEVICE
    static inline constexpr uint64_t MOD_MERSENNE(uint64_t aVal) { return (aVal & M61) + (aVal >> 61); }

    MIXMAX_HOST_AND_DEVICE
    inline constexpr void updateState() {
        static_assert(N == 16 || N == 7);
        uint64_t PartialSumOverOld = m_State[0];
        auto lV = m_State[0] = MOD_MERSENNE(m_SumOverNew + PartialSumOverOld);
        m_SumOverNew         = MOD_MERSENNE(m_SumOverNew + lV);
#ifdef __CUDA_ARCH__
#pragma unroll N - 1
#endif
        for (int i = 1; i < N; ++i) {
            const auto lRotatedPreviousPartialSumOverOld = ROTATE_61(PartialSumOverOld, 36);
            PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + m_State[i]);
            lV = m_State[i] = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
            m_SumOverNew    = MOD_MERSENNE(m_SumOverNew + lV);
        }
        m_counter = 0;
    }

#if defined(__x86_64__)
    MIXMAX_HOST_AND_DEVICE
    inline static constexpr uint64_t MOD_128(const __uint128_t s) {
        uint64_t s1 = (static_cast<uint64_t>(s) & M61) + (static_cast<uint64_t>(s >> 64) * 8) +
                      (static_cast<uint64_t>(s) >> BITS);
        return MOD_MERSENNE(s1);
    }
    MIXMAX_HOST_AND_DEVICE
    inline static constexpr uint64_t F_MOD_MUL_M61(const uint64_t cum, const uint64_t a, const uint64_t b) {
        const auto temp = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b) + cum;
        return MOD_128(temp);
    }
#else
    MIXMAX_HOST_AND_DEVICE
    inline static constexpr uint64_t F_MOD_MUL_M61(const uint64_t cum, const uint64_t s, const uint64_t a) {
        static const uint64_t MASK32 = 0xFFFFFFFFULL;
        //
        auto o = (s)*a;
        const auto ph = ((s) >> 32);
        const auto pl = (s)&MASK32;
        const auto ah = a >> 32;
        const auto al = a & MASK32;
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
#include "mixmax_skip_N17.c"
            ;
        constexpr uint64_t skipMat8[128][8] =
#include "mixmax_skip_N8.c"
            ;
        const uint64_t* skipMat[128];

        for (int i = 0; i < 128; i++) {
            if (N == 7) {
                skipMat[i] = skipMat8[i];
            }
            if (N == 16) {
                skipMat[i] = skipMat17[i];
            }
        }

        m_State[0] = m_SumOverNew = 0;
        for (int i = 1; i < N; i++) {
            m_State[i]   = 0;
            m_SumOverNew = MOD_MERSENNE(m_SumOverNew + m_State[i]);
        }

        m_SumOverNew       = 1;
        u_int32_t IDvec[4] = {streamID, runID, machineID, clusterID};
        for (auto IDindex = 0; IDindex < 4; IDindex++) {  // go from lower order to higher order ID
            auto id = IDvec[IDindex];
            auto r  = 0;
            while (id) {
                if (id & 1) {
                    const auto& rowPtr = skipMat[r + IDindex * 8 * sizeof(uint32_t)];
                    uint64_t cum[M];
                    for (int i = 0; i < M; i++) {
                        cum[i] = 0;
                    }
                    for (auto j = 0; j < M; j++) {  // j is lag, enumerates terms of the poly
                        // for zero lag Y is already given
                        auto coeff = rowPtr[j];  // same coeff for all i
                        for (auto i = 0; i < M; i++) {
                            cum[i] = F_MOD_MUL_M61(cum[i], coeff, i == 0 ? m_SumOverNew : m_State[i - 1]);
                        }
                        updateState();
                    }
                    m_SumOverNew = cum[0];
                    for (auto i = 0; i < N; i++) {
                        m_State[i] = cum[i + 1];
                    }
                }
                id = (id >> 1);
                r++;  // bring up the r-th bit in the ID
            }
        }
        m_counter = 0;
    }

   public:
#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const MixMaxRng& rng) {
        os << "V: ";
        for (const auto& elem : rng.m_State) {
            os << elem << " ";
        }
        os << " counter: " << static_cast<u_int32_t>(rng.m_counter) << " SumOverNew: " << rng.m_SumOverNew;
        return os;
    }
#endif
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return 0x1FFFFFFFFFFFFFFF; }
};

using MixMaxRng8  = MixMaxRng<8>;
using MixMaxRng17 = MixMaxRng<17>;

}  // namespace MIXMAX

#endif  // MIXMAX_INCLUDE_MIXMAXRNG_H_
