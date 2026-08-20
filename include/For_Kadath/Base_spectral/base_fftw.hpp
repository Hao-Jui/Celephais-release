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

#pragma once

#include <fftw3.h>
#include <math.h>
#include <stdexcept>
#include <vector>

namespace Kadath
{
    /**
     * @struct fftw_precomp_t
     * @brief Precomputed FFTW plan and buffer for efficient repeated transforms
     *
     * @details
     * This structure encapsulates the precomputation strategy for FFTW (Fastest Fourier
     * Transform in the West) operations. FFTW achieves optimal performance through a
     * two-phase approach:
     *
     * 1. **Planning phase**: Analyzes the transform size and measures different algorithms
     *    to determine the fastest approach for the specific hardware
     * 2. **Execution phase**: Applies the precomputed optimal strategy repeatedly
     *
     * ## Performance Benefits
     *
     * Planning a transform can be expensive (milliseconds to seconds depending on size
     * and FFTW_MEASURE flags), but executing a planned transform is extremely fast
     * (microseconds). For repeated transforms of the same size:
     *
     * - **Without precomputation**: Plan + Execute on every call (slow)
     * - **With precomputation**: Plan once, Execute many times (fast)
     *
     * Typical speedup: 10-1000x for small transforms called repeatedly.
     *
     * ## Memory Management
     *
     * The buffer is allocated using fftw_alloc_real() rather than standard malloc/new
     * to ensure:
     * - Proper memory alignment for SIMD instructions (SSE, AVX)
     * - Optimal cache line alignment
     * - Compatibility with FFTW's internal memory requirements
     *
     * Both plan and buffer are automatically cleaned up in the destructor, making this
     * a RAII-compliant resource handle.
     *
     * ## Thread Safety
     *
     * FFTW planning is NOT thread-safe. If using in multi-threaded code:
     * - Create all plans before spawning threads, OR
     * - Protect plan creation with a mutex
     * - Plan execution IS thread-safe once plans are created
     *
     * ## Usage Pattern
     *
     * Typically stored in static containers or class members for reuse:
     * @code
     * // Store precomputed transforms by size
     * std::unordered_map<int, fftw_precomp_t> cosine_transforms;
     *
     * // First use: creates and stores the plan
     * auto& transform = cosine_transforms.emplace(
     *     std::piecewise_construct,
     *     std::forward_as_tuple(n),
     *     std::forward_as_tuple(n, FFTW_REDFT00)
     * ).first->second;
     *
     * // Subsequent uses: directly access precomputed plan
     * std::copy(input, input + n, transform.buffer);
     * fftw_execute(transform.plan);
     * std::copy(transform.buffer, transform.buffer + n, output);
     * @endcode
     *
     * ## FFTW Transform Types
     *
     * Common fftw_r2r_kind values for spectral methods:
     * - FFTW_REDFT00: DCT-I (Chebyshev extrema points)
     * - FFTW_REDFT10: DCT-II (Chebyshev interior points)
     * - FFTW_REDFT01: DCT-III (inverse of DCT-II)
     * - FFTW_RODFT00: DST-I (sine transform)
     *
     * @note This struct is non-copyable due to FFTW plan semantics
     * @warning Do not manually free buffer or destroy plan - destructor handles cleanup
     *
     * @see FFTW documentation: http://www.fftw.org/
     * @ingroup fft
     */
    struct fftw_precomp_t {
      private:
        static int checked_transform_size(int n)
        {
            if (n <= 0)
                throw std::invalid_argument("FFTW transform size must be positive");
            return n;
        }

      public:
        /** Number of points used by the FFTW plan and trigonometric tables. */
        const int transform_size;

        /** sin(M_PI * i / transform_size), evaluated once in the original order. */
        std::vector<double> sin_pi_over_n;

        /** sin(M_PI * i / 2. / transform_size), evaluated once in the original order. */
        std::vector<double> sin_pi_i_over_2n;

        /** sin(M_PI / 2. * i / transform_size), kept distinct for bit compatibility. */
        std::vector<double> sin_half_pi_i_over_n;

        /** sin(M_PI * transform_size / 4 / transform_size), preserving evaluation order. */
        const double sin_pi_quarter;

        /**
         * @brief Aligned buffer for transform input/output
         *
         * @details
         * FFTW performs in-place transforms, using this buffer for both input and output.
         * The buffer is allocated via fftw_alloc_real() to ensure proper alignment for
         * vectorized operations (SSE/AVX).
         *
         * **Memory Alignment**: Typically 16-byte (SSE) or 32-byte (AVX) aligned
         *
         * **Usage Pattern**:
         * 1. Copy input data to buffer
         * 2. Execute plan (modifies buffer in-place)
         * 3. Copy output data from buffer
         *
         * @note Never access this pointer after object destruction
         * @warning Do not free manually - managed by destructor
         */
        double* buffer;

        /**
         * @brief Precomputed FFTW execution plan
         *
         * @details
         * Contains the optimized algorithm selection and internal data structures
         * determined during the planning phase. The plan is specific to:
         * - Transform size (n)
         * - Transform type (FFTW_REDFT00, etc.)
         * - Buffer memory address
         *
         * **Plan Persistence**: Valid until explicitly destroyed or program termination
         *
         * **Execution**: Thread-safe after creation
         * @code
         * fftw_execute(plan);  // Transforms buffer in-place
         * @endcode
         *
         * @note Plans are opaque handles - internal structure is FFTW implementation detail
         * @warning Do not destroy manually - managed by destructor
         */
        fftw_plan plan;

        // =========================================================================
        // Constructor
        // =========================================================================
        /**
         * @brief Constructs precomputed transform for specified size and type
         *
         * @param[in] n                Number of real-valued data points
         * @param[in] FFTW_TRANSFORM   Type of real-to-real transform to perform
         *
         * @details
         * **Construction Steps**:
         * 1. Allocates aligned buffer for n doubles using fftw_alloc_real()
         * 2. Creates FFTW plan using FFTW_MEASURE for optimal performance
         * 3. Stores both for repeated use
         *
         * **Transform Types** (common in spectral methods):
         *
         * | Type | Description | Boundary Conditions |
         * |------|-------------|---------------------|
         * | FFTW_REDFT00 | DCT-I | Even extension, both boundaries |
         * | FFTW_REDFT10 | DCT-II | Even extension, one boundary |
         * | FFTW_REDFT01 | DCT-III | Inverse of DCT-II |
         * | FFTW_RODFT00 | DST-I | Odd extension, both boundaries |
         *
         * @post buffer allocated and properly aligned
         * @post plan created and ready for execution
         *
         * @note Planning may temporarily allocate additional memory for measurements
         * @warning Planning is NOT thread-safe - create all plans before threading
         *
         * Example:
         * @code
         * // Create DCT-II transform for 64 points (Chebyshev spectral method)
         * fftw_precomp_t dct2(64, FFTW_REDFT10);
         *
         * // Use repeatedly
         * for(int iter = 0; iter < 1000; ++iter) {
         *     std::copy(input_data, input_data + 64, dct2.buffer);
         *     fftw_execute(dct2.plan);  // Transform in-place
         *     process_spectral_coefficients(dct2.buffer);
         * }
         * @endcode
         */
        fftw_precomp_t(int const n, fftw_r2r_kind FFTW_TRANSFORM)
            : transform_size(checked_transform_size(n)), sin_pi_over_n(transform_size + 1),
              sin_pi_i_over_2n(transform_size + 1), sin_half_pi_i_over_n(transform_size + 1),
              sin_pi_quarter(sin(M_PI * n / 4 / n)), buffer(fftw_alloc_real(transform_size)),
              plan(fftw_plan_r2r_1d(transform_size, buffer, buffer, FFTW_TRANSFORM, FFTW_MEASURE))
        {
            for (int i = 0; i <= transform_size; ++i) {
                sin_pi_over_n[i] = sin(M_PI * i / n);
                sin_pi_i_over_2n[i] = sin(M_PI * i / 2. / n);
                sin_half_pi_i_over_n[i] = sin(M_PI / 2. * i / n);
            }
        }

        fftw_precomp_t(const fftw_precomp_t&) = delete;
        fftw_precomp_t& operator=(const fftw_precomp_t&) = delete;

        // =========================================================================
        // Destructor
        // =========================================================================
        /**
         * @brief Destructor - properly releases FFTW resources
         *
         * @details
         * Ensures proper cleanup of FFTW resources in correct order:
         * 1. Destroy plan first (releases internal FFTW structures)
         * 2. Free buffer second (releases aligned memory)
         *
         * **RAII Compliance**: Automatic resource management guarantees no leaks
         *
         * **Thread Safety**: Safe to destroy from any thread (after final use)
         *
         * @post plan invalidated - do not use after destruction
         * @post buffer freed - pointer becomes dangling
         *
         * @note Order matters: plan must be destroyed before freeing buffer
         * @warning Undefined behavior if plan or buffer accessed after destruction
         */
        ~fftw_precomp_t()
        {
            fftw_destroy_plan(plan); // Release FFTW internal structures
            fftw_free(buffer);       // Release aligned memory
        }
    };
} // namespace Kadath
