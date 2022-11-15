//
// Created by mbarbone on 11/3/22.
//

#ifndef MIXMAX_INCLUDE_MIXMAXRNG_H_
#define MIXMAX_INCLUDE_MIXMAXRNG_H_

#include <cstdint>
#include <ostream>

namespace MIXMAX {

#if __cplusplus >= 201402L
#define MIXMAX_CONSTEXPR constexpr
#else
#define MIXMAX_CONSTEXPR
#endif

#define MIXMAX_STRINGIFY_(...) #__VA_ARGS__
#define MIXMAX_STRINGIFY(...) MIXMAX_STRINGIFY_(__VA_ARGS__)
#if defined(__CUDA_ARCH__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define MIXMAX_GCC
#elif defined(__GNUC__) || defined(__GNUG__)
#define MIXMAX_GCC GCC
#endif
#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#pragma nv_diag_suppress 1675
#elif defined(__CUDA_ARCH__)
#pragma diag_suppress 1675
#endif
#define MIXMAX_UNROLL(...) _Pragma(MIXMAX_STRINGIFY(MIXMAX_GCC unroll __VA_ARGS__))

#ifdef __CUDACC__

#define MIXMAX_HOST_AND_DEVICE __host__ __device__
#define MIXMAX_HOST __host__
#define MIXMAX_DEVICE __device__
#define MIXMAX_KERNEL __global__

#else

#define MIXMAX_HOST_AND_DEVICE
#define MIXMAX_HOST
#define MIXMAX_DEVICE
#define MIXMAX_KERNEL

#endif

namespace internal {

enum { SMALL = 8, MEDIUM = 17, LARGE = 240 };

/**
 * Figure of merit is entropy: best generator overall is N=240
 * Vector size  | period q
 *     N        | log10(q)  |   entropy
 *              |
 * ------------------------------------|
 *       8      |    129    |    220.4 |
 *      17      |    294    |    374.3 |
 *     240      |   4389    |   8679.2 |
 */
template <std::uint8_t M>
class MixMaxRng {
   public:
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng() {
        validateTemplate();
        seedZero();
    }

    /**
     * Basic seeding function.
     * On GPU seeding is expensive so it is recommended to do so in another kernel
     * @param seed
     */
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(std::uint64_t seed) {
        validateTemplate();
        if (seed == 0) {
            seedZero();
            return;
        }
#ifndef __CUDA_ARCH__
        // On CPU seed using a 64-bit LCG from Knuth line 26, in combination with a bit swap
        seedLCG(seed);
#else
        // ON GPU Using GPU Thread id to seed different streams
        const std::uint64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
        unpackAndBigSkip(seed, stream);
#endif
    }

    /**
     * Seeding with 256 bits.
     * This guarantees that there is no collision between different streams if at least one of the 256 bits is different
     */
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(std::uint32_t clusterID, std::uint32_t machineID, std::uint32_t runID, std::uint32_t streamID) {
        validateTemplate();
        appplyBigSkip(clusterID, machineID, runID, streamID);
    }

    /**
     * Seeding with 256 bits.
     * This guarantees that there is no collision between different streams if at least one of the 256 bits is different
     * Useful in a parallel application where stream can be the CPU/GPU thread id
     */
    MIXMAX_HOST_AND_DEVICE
    MixMaxRng(std::uint64_t seed, std::uint64_t stream) {
        validateTemplate();
        unpackAndBigSkip(seed, stream);
    }

    /**
     * Generates an uniform 64bits integer
     */
    MIXMAX_HOST_AND_DEVICE
    inline MIXMAX_CONSTEXPR std::uint64_t operator()() noexcept {
        if (m_Counter == N) {
            updateState();
        }
        return m_State[m_Counter++];
    }

    /**
     * Generates an uniform double between 0,1
     * It is faster than using a std::uniform_distribution()
     */
    MIXMAX_HOST_AND_DEVICE
    inline MIXMAX_CONSTEXPR double Uniform() noexcept {
        const auto u = operator()();
        return static_cast<double>(u) * INV_MERSBASE;
    }

    // For compatibility with std::random
    static constexpr std::uint64_t min() noexcept { return 0; }
    static constexpr std::uint64_t max() noexcept { return 0x1FFFFFFFFFFFFFFF; }
    using result_type = std::uint64_t;

    MIXMAX_HOST_AND_DEVICE
    MIXMAX_CONSTEXPR MixMaxRng(const uint64_t* state, uint64_t sum_over_new, uint32_t counter)
        : m_SumOverNew(sum_over_new), m_Counter(counter) {
        for (auto i = 0; i < N; ++i) {
            m_State[i] = state[i];
        }
    }

#ifdef __CUDACC__
    MIXMAX_HOST_AND_DEVICE
    constexpr MixMaxRng(MixMaxRng const& other) : m_SumOverNew(other.m_SumOverNew), m_Counter(other.m_Counter) {
        for (auto i = 0; i < N; ++i) {
            m_State[i] = other.m_State[i];
        }
    }
    MIXMAX_HOST_AND_DEVICE
    constexpr MixMaxRng(MixMaxRng&& other) noexcept : m_SumOverNew(other.m_SumOverNew), m_Counter(other.m_Counter) {
        for (auto i = 0; i < N; ++i) {
            m_State[i] = other.m_State[i];
        }
    }

    MIXMAX_HOST_AND_DEVICE
    MIXMAX_CONSTEXPR MixMaxRng& operator=(MixMaxRng&& other) noexcept {
        m_SumOverNew = other.m_SumOverNew;
        m_Counter    = other.m_Counter;
        for (auto i = 0; i < N; ++i) {
            m_State[i] = other.m_State[i];
        }
    }

    MIXMAX_HOST_AND_DEVICE
    MIXMAX_CONSTEXPR MixMaxRng& operator=(MixMaxRng const& other) {
        m_SumOverNew = other.m_SumOverNew;
        m_Counter    = other.m_Counter;
        for (auto i = 0; i < N; ++i) {
            m_State[i] = other.m_State[i];
        }
    }
#else
    constexpr MixMaxRng(MixMaxRng const& other) = default;
    constexpr MixMaxRng(MixMaxRng&& other) noexcept = default;
    MIXMAX_CONSTEXPR MixMaxRng& operator=(MixMaxRng&& other) noexcept = default;
    MIXMAX_CONSTEXPR MixMaxRng& operator=(MixMaxRng const& other) = default;
#endif

   private:
    // Constants
    static constexpr double INV_MERSBASE{0.43368086899420177360298E-18};
    // The state is M-1 because the last element is stored in the variable m_SumOverNew outside the vector
    //    enum { N = M - 1, BITS = 61U, M61 = 0x1FFFFFFFFFFFFFFF };
    static constexpr std::uint8_t N{M - 1};
    static constexpr std::uint8_t BITS{61U};
    static constexpr std::uint64_t M61{0x1FFFFFFFFFFFFFFF};
    // RNG state
    std::uint64_t m_State[N];
    std::uint64_t m_SumOverNew;
    std::uint8_t m_Counter;

    /**
     * Table of parameters for MIXMAX
     * Vector size  |
     *     N        |      SPECIAL       |   SHIFT_ROTATION   |           MOD_MULSPEC
     *              |
     * -------------------------------------------------------------------------------------
     *       8      |         0          |         53         |                none
     *      17      |         0          |         36         |                none
     *     240      | 487013230256099140 |         51         |   fmodmulM61( 0, SPECIAL , (k) )
     */
    enum { SMALL_SHIFT = 53U, MEDIUM_SHIFT = 36U, LARGE_SHIFT = 51U };
    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == LARGE>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE static inline constexpr std::uint8_t SHIFT_ROTATION() noexcept {
        return LARGE_SHIFT;
    }

    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == MEDIUM>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE static inline constexpr std::uint8_t SHIFT_ROTATION() noexcept {
        return MEDIUM_SHIFT;
    }

    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == SMALL>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE static inline constexpr std::uint8_t SHIFT_ROTATION() noexcept {
        return SMALL_SHIFT;
    }

    MIXMAX_HOST_AND_DEVICE
    static inline constexpr std::uint64_t ROTATE_61(const std::uint64_t aVal) noexcept {
        return ((aVal << SHIFT_ROTATION()) & M61) | (aVal >> (61U - SHIFT_ROTATION()));
    }

    MIXMAX_HOST_AND_DEVICE
    static inline constexpr std::uint64_t MOD_MERSENNE(std::uint64_t aVal) noexcept {
        return (aVal & M61) + (aVal >> 61U);
    }

    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == LARGE>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE MIXMAX_CONSTEXPR inline void updateState() noexcept {
        auto PartialSumOverOld    = m_State[0];
        auto oldPartialSumOverOld = PartialSumOverOld;
        auto lV = m_State[0] = MOD_MERSENNE(m_SumOverNew + PartialSumOverOld);
        m_SumOverNew         = MOD_MERSENNE(m_SumOverNew + lV);
        MIXMAX_UNROLL(238)
        for (std::uint8_t i = 1U; i < N; ++i) {
            const auto lRotatedPreviousPartialSumOverOld = ROTATE_61(PartialSumOverOld);
            PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + m_State[i]);
            lV = m_State[i] = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
            m_SumOverNew    = MOD_MERSENNE(m_SumOverNew + lV);
        }
        oldPartialSumOverOld = F_MOD_MUL_M61(0, 487013230256099140, oldPartialSumOverOld);
        m_State[1]           = MOD_MERSENNE(oldPartialSumOverOld + m_State[1]);
        m_SumOverNew         = MOD_MERSENNE(m_SumOverNew + oldPartialSumOverOld);
        m_Counter            = 0;
    }

    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == MEDIUM>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE MIXMAX_CONSTEXPR inline void updateState() noexcept {
        auto PartialSumOverOld = m_State[0];
        auto lV = m_State[0] = MOD_MERSENNE(m_SumOverNew + PartialSumOverOld);
        m_SumOverNew         = MOD_MERSENNE(m_SumOverNew + lV);
        MIXMAX_UNROLL(15)
        for (std::uint8_t i = 1U; i < N; ++i) {
            const auto lRotatedPreviousPartialSumOverOld = ROTATE_61(PartialSumOverOld);
            PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + m_State[i]);
            lV = m_State[i] = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
            m_SumOverNew    = MOD_MERSENNE(m_SumOverNew + lV);
        }
        m_Counter = 0;
    }

    template <std::uint8_t STATE_SIZE = M, typename std::enable_if<STATE_SIZE == SMALL>::type* = nullptr>
    MIXMAX_HOST_AND_DEVICE MIXMAX_CONSTEXPR inline void updateState() noexcept {
        auto PartialSumOverOld = m_State[0];
        auto lV = m_State[0] = MOD_MERSENNE(m_SumOverNew + PartialSumOverOld);
        m_SumOverNew         = MOD_MERSENNE(m_SumOverNew + lV);
        MIXMAX_UNROLL(6)
        for (std::uint8_t i = 1U; i < N; ++i) {
            const auto lRotatedPreviousPartialSumOverOld = ROTATE_61(PartialSumOverOld);
            PartialSumOverOld                            = MOD_MERSENNE(PartialSumOverOld + m_State[i]);
            lV = m_State[i] = MOD_MERSENNE(lV + PartialSumOverOld + lRotatedPreviousPartialSumOverOld);
            m_SumOverNew    = MOD_MERSENNE(m_SumOverNew + lV);
        }
        m_Counter = 0;
    }

#if defined(__x86_64__) && \
    (!defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 5))
    MIXMAX_HOST_AND_DEVICE
    static inline constexpr std::uint64_t MOD_128(const __uint128_t s) noexcept {
        return MOD_MERSENNE((static_cast<std::uint64_t>(s) & M61) + (static_cast<std::uint64_t>(s >> 64) * 8) +
                            (static_cast<std::uint64_t>(s) >> BITS));
    }
    MIXMAX_HOST_AND_DEVICE
    static inline constexpr std::uint64_t F_MOD_MUL_M61(const std::uint64_t cum, const std::uint64_t a,
                                                        const std::uint64_t b) noexcept {
        return MOD_128(static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b) + cum);
    }
#else
    MIXMAX_HOST_AND_DEVICE
    static inline MIXMAX_CONSTEXPR std::uint64_t F_MOD_MUL_M61(const std::uint64_t cum, const std::uint64_t s,
                                                               const std::uint64_t a) noexcept {
        constexpr std::uint64_t MASK32 = 0xFFFFFFFFULL;
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
    void appplyBigSkip(std::uint32_t clusterID, std::uint32_t machineID, std::uint32_t runID,
                       std::uint32_t streamID) noexcept {
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
         * ector by seed_vielbein(X,0), and use it in all your runs, just change runID to get completely
         * nonoverlapping treams of random numbers on a different day. clusterID and machineID are provided for the
         * benefit of large rganizations who wish to ensure that a simulation which is running in parallel on a
         * large number of clusters nd machines will have non-colliding source of random numbers. did I repeat it
         * enough times? the on-collision guarantee is absolute, not probabilistic
         */
        static constexpr std::uint64_t skipMat240[128][240] =
#include "mixmax/private/mixmax_skip_N240.c"
            ;
        static constexpr std::uint64_t skipMatrix17[128][17] =
#include "mixmax/private/mixmax_skip_N17.c"
            ;
        static constexpr std::uint64_t skipMatrix8[128][8] =
#include "mixmax/private/mixmax_skip_N8.c"
            ;
        const std::uint64_t* skipMat[128];

        for (int i = 0; i < 128; i++) {
            if MIXMAX_CONSTEXPR (M == SMALL) {
                skipMat[i] = skipMatrix8[i];
            }
            if MIXMAX_CONSTEXPR (M == MEDIUM) {
                skipMat[i] = skipMatrix17[i];
            }
            if MIXMAX_CONSTEXPR (M == LARGE) {
                skipMat[i] = skipMat240[i];
            }
        }

        m_SumOverNew = 1;
        for (int i = 0; i < N; i++) {
            m_State[i] = 0;
        }
        u_int32_t idVector[4] = {streamID, runID, machineID, clusterID};
        std::uint64_t cumulativeVector[M];
        for (auto idIndex = 0; idIndex < 4; idIndex++) {  // go from lower order to higher order ID
            auto currentID = idVector[idIndex], skipIndex = 0U;
            for (; currentID > 0; currentID >>= 1, skipIndex++) {  // bring up the r-th bit in the ID
                if (!(currentID & 1)) {
                    continue;
                }
                const auto& skipVector = skipMat[skipIndex + idIndex * 8 * sizeof(std::uint32_t)];
                for (int i = 0; i < M; i++) {
                    cumulativeVector[i] = 0;
                }
                for (auto j = 0; j < M; j++) {  // j is lag, enumerates terms of the poly
                    // for zero lag Y is already given
                    const auto skipElement = skipVector[j];  // same skipElement for all i
                    for (auto i = 0; i < M; i++) {
                        cumulativeVector[i] =
                            F_MOD_MUL_M61(cumulativeVector[i], skipElement, i == 0 ? m_SumOverNew : m_State[i - 1]);
                    }
                    updateState();
                }
                m_SumOverNew = cumulativeVector[0];
                for (auto i = 0; i < N; i++) {
                    m_State[i] = cumulativeVector[i + 1];
                }
            }
        }
        m_Counter = 0;
    }

    /**
     * Sets the generator to the unary vector
     */
    MIXMAX_HOST_AND_DEVICE
    void seedZero() noexcept {
        for (auto& element : m_State) {
            element = 1;
        }
        m_SumOverNew = 1;
        updateState();
        // Skip the first element
        m_Counter = 1;
    }

    /**
     * a 64-bit LCG from Knuth line 26, in combination with a bit swap is used to seed
     */
    void seedLCG(std::uint64_t seed) noexcept {
        static constexpr std::uint64_t MULT64 = 6364136223846793005ULL;
        std::uint64_t overflow                = 0;
        seed *= MULT64;
        seed         = (seed << 32) ^ (seed >> 32);
        m_SumOverNew = seed & M61;
        for (auto& currentState : m_State) {
            seed *= MULT64;
            seed         = (seed << 32) ^ (seed >> 32);
            currentState = seed & M61;
            m_SumOverNew += currentState;
            overflow += m_SumOverNew < currentState;
        }
        m_SumOverNew = MOD_MERSENNE(MOD_MERSENNE(m_SumOverNew) + (overflow << 3));
        m_Counter    = N;
    }

    MIXMAX_HOST_AND_DEVICE
    void unpackAndBigSkip(std::uint64_t seed, std::uint64_t stream) noexcept {
        const std::uint32_t seed_low    = seed & 0xFFFFFFFF;
        const std::uint32_t seed_high   = (seed & (0xFFFFFFFFUL << 32)) >> 32;
        const std::uint32_t stream_low  = stream & 0xFFFFFFFF;
        const std::uint32_t stream_high = (stream & (0xFFFFFFFFUL << 32)) >> 32;
        appplyBigSkip(stream_high, stream_low, seed_high, seed_low);
    }
    /**
     * Do not compile if state size is not valid
     */
    MIXMAX_HOST_AND_DEVICE
    void validateTemplate() {
        static_assert(M == LARGE || M == MEDIUM || M == SMALL, "State must be either 8, 17, 240");
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
};
}  // namespace internal

using MixMaxRng8   = internal::MixMaxRng<internal::SMALL>;
using MixMaxRng17  = internal::MixMaxRng<internal::MEDIUM>;
using MixMaxRng240 = internal::MixMaxRng<internal::LARGE>;

}  // namespace MIXMAX

// Always clean-up defines
#undef MIXMAX_HOST_AND_DEVICE
#undef MIXMAX_HOST
#undef MIXMAX_DEVICE
#undef MIXMAX_KERNEL
#undef MIXMAX_UNROLL
#undef MIXMAX_STRINGIFY_
#undef MIXMAX_STRINGIFY
#ifdef MIXMAX_GCC
#undef MIXMAX_GCC
#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#pragma nv_diag_default 1675
#elif defined(__CUDA_ARCH__)
#pragma diag_default 1675
#endif
#endif

#endif  // MIXMAX_INCLUDE_MIXMAXRNG_H_
