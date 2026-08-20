/*
    Copyright 2018 Ludwig Jens Papenfort

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-08-08  Added selectable native fixed-size real half-complex codelets
 *               through N=32 beside the retained FFTW backend.
 *   2026-08-08  Added fused native N=6..20 CHEB/COSSIN line transforms that
 *               consume strided samples and emit final coefficients or
 *               collocation values.
 *   2026-08-09  Extended the fused forward and inverse paths through the six
 *               COS/SIN parity families.
 *   2026-08-08  Replaced the packed complex wrapper with direct real-input
 *               R2HC/HC2R DAGs for every selected even transform size.
 *   2026-08-08  Extended fixed-size native coverage through N=32 with
 *               N=11, N=13, and N=15 real primitive schedules.
 *   2026-08-08  Restricted both public backends to the production contract:
 *               even transform lengths N=2..32.
 *   2026-08-09  Removed FFTW from production and replaced its allocation
 *               ownership with first-party aligned native storage; an
 *               explicit build can retain FFTW as a test-only oracle.
 *   2026-08-10  Added plan-owned GCC/x86 workspaces and a narrow cached COSSIN
 *               forward dispatcher for production N=10--16 line batches.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace Kadath
{
    enum class r2hc_backend
    {
        /** Test-only oracle when enabled; otherwise produces an explicit error. */
        fftw,
        native
    };

    enum class r2hc_direction
    {
        forward,
        inverse
    };

    enum class native_spectral_family
    {
        cheb,
        cheb_even,
        cheb_odd,
        cossin,
        cos,
        sin,
        cos_even,
        cos_odd,
        sin_even,
        sin_odd
    };

    /**
     * Reusable state for Celephais' real discrete Fourier transforms.
     *
     * The coefficient kernels use the half-complex layout traditionally used
     * by real transforms: real mode k is stored at k and imaginary mode k at
     * n-k.  Both transforms are unnormalised, so HC2R(R2HC(x)) is n*x.
     * Only even transform lengths 2..32 are accepted; other lengths throw
     * std::invalid_argument before a buffer is created.  The native backend
     * uses compile-time-sized code paths.  The
     * N=22, N=26, and N=30 paths use fixed real N=11, N=13, and N=15 primitive
     * schedules, respectively.  Native is the only production backend.
     * CELEPHAIS_FFT_BACKEND may be unset or `native`.  The `fftw` value is
     * available only in an explicit test-oracle build and otherwise fails with
     * a migration error rather than silently changing results.
     *
     * One instance owns one mutable input/output buffer.  As with the former
     * transform cache, a single instance is not safe for concurrent execution;
     * Celephais' production model is one compute thread per MPI rank.
     *
     * @ingroup fft
     */
    class r2hc_precomp_t
    {
      private:
        struct aligned_buffer_deleter
        {
            void operator()(double* pointer) const noexcept;
        };

        using aligned_buffer = std::unique_ptr<double[], aligned_buffer_deleter>;

        aligned_buffer buffer_storage_;
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
        mutable std::vector<double> scratch_;
#else
        std::vector<double> scratch_;
#endif
        std::vector<double> cos_table_;
        std::vector<double> sin_table_;
        [[maybe_unused]] void* oracle_plan_;
        const r2hc_backend backend_;
        const r2hc_direction direction_;
        const bool fused_spectral_enabled_;

        static int checked_transform_size(int n);
        static int checked_backend_transform_size(int n, r2hc_backend backend);
        static aligned_buffer allocate_aligned_buffer(int n);
        static r2hc_backend configured_backend();
        static bool configured_fused_spectral();
        static constexpr bool is_cos_sin_parity_family(
            native_spectral_family family) noexcept
        {
            return family == native_spectral_family::cos
                   || family == native_spectral_family::sin
                   || family == native_spectral_family::cos_even
                   || family == native_spectral_family::cos_odd
                   || family == native_spectral_family::sin_even
                   || family == native_spectral_family::sin_odd;
        }
        static constexpr bool fused_path_passes_microbenchmark_gate(
            int transform_size, native_spectral_family family,
            r2hc_direction direction) noexcept
        {
            // Production-shaped line timings retain ordinary COSSIN at N=6/8;
            // the GCC/x86 persistent-workspace route below additionally admits
            // the production N=10--16 range.  For the COS/SIN parity families,
            // COS and COS_EVEN forward lose at N=20, while only the odd-family
            // inverse kernels keep winning at N=18/20.  Keeping this gate inline
            // makes every declined probe, including the N>20 range rejection
            // above, a zero-call fallback.
            if (family == native_spectral_family::cossin)
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
                return transform_size == 6 || transform_size == 8
                       || (transform_size >= 10 && transform_size <= 16);
#else
                return transform_size == 6 || transform_size == 8;
#endif
            if (is_cos_sin_parity_family(family)) {
                if (direction == r2hc_direction::inverse)
                    return transform_size <= 16
                           || family == native_spectral_family::cos_odd
                           || family == native_spectral_family::sin_odd;
                if (transform_size == 20)
                    return family != native_spectral_family::cos
                           && family != native_spectral_family::cos_even;
                return true;
            }
            if (direction == r2hc_direction::inverse
                && (transform_size == 18 || transform_size == 20))
                return family != native_spectral_family::cheb
                       && family != native_spectral_family::cheb_even;
            return true;
        }
        bool execute_fused_forward(native_spectral_family family,
                                   const double* src, double* dst,
                                   int nbr_in, int nbr_out, int stride) const;
        bool execute_fused_inverse(native_spectral_family family,
                                   const double* src, double* dst,
                                   int nbr_in, int nbr_out, int stride) const;
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
        bool execute_cached_cossin_forward(const double* src, double* dst,
                                           int nbr_in, int nbr_out,
                                           int stride) const;
#endif
        void execute_r2hc_generic();
        void execute_hc2r_generic();

      public:
        const int transform_size;

        /** Tables retained by the coefficient fold/unfold kernels. */
        std::vector<double> sin_pi_over_n;
        std::vector<double> sin_pi_i_over_2n;
        std::vector<double> sin_half_pi_i_over_n;
        const double sin_pi_quarter;

        /** Mutable in-place half-complex transform buffer. */
        double* const buffer;

        r2hc_precomp_t(int n, r2hc_direction direction);
        r2hc_precomp_t(int n, r2hc_direction direction, r2hc_backend backend);
        ~r2hc_precomp_t();

        r2hc_precomp_t(const r2hc_precomp_t&) = delete;
        r2hc_precomp_t& operator=(const r2hc_precomp_t&) = delete;
        r2hc_precomp_t(r2hc_precomp_t&&) = delete;
        r2hc_precomp_t& operator=(r2hc_precomp_t&&) = delete;

        r2hc_backend backend() const noexcept;

        /**
         * Try the native line-level fusion for a supported family and shape.
         * Returns false without touching either line when the backend, runtime
         * flag, direction, transform size, or line extents require fallback.
         */
        bool try_execute_fused_forward(native_spectral_family family,
                                       const double* src, double* dst,
                                       int nbr_in, int nbr_out, int stride) const
        {
            if (transform_size < 6 || transform_size > 20)
                return false;
            if (!fused_spectral_enabled_ || direction_ != r2hc_direction::forward
                || !fused_path_passes_microbenchmark_gate(
                    transform_size, family, r2hc_direction::forward))
                return false;
            return execute_fused_forward(family, src, dst, nbr_in, nbr_out, stride);
        }
        bool try_execute_fused_inverse(native_spectral_family family,
                                       const double* src, double* dst,
                                       int nbr_in, int nbr_out, int stride) const
        {
            if (transform_size < 6 || transform_size > 20)
                return false;
            if (!fused_spectral_enabled_ || direction_ != r2hc_direction::inverse
                || !fused_path_passes_microbenchmark_gate(
                    transform_size, family, r2hc_direction::inverse))
                return false;
            return execute_fused_inverse(family, src, dst, nbr_in, nbr_out, stride);
        }
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) \
    && !defined(__clang__)
        /**
         * Try the narrow cached-plan route for growing ordinary COSSIN lines.
         * Only the production N=10,12,14,16 forward plans are accepted.
         */
        bool try_execute_cached_cossin_forward(const double* src, double* dst,
                                               int nbr_in, int nbr_out,
                                               int stride) const
        {
            if (!fused_spectral_enabled_
                || direction_ != r2hc_direction::forward)
                return false;
            if (transform_size != 10 && transform_size != 12
                && transform_size != 14 && transform_size != 16)
                return false;
            return execute_cached_cossin_forward(
                src, dst, nbr_in, nbr_out, stride);
        }
#endif
        void execute_r2hc();
        void execute_hc2r();
    };
} // namespace Kadath
