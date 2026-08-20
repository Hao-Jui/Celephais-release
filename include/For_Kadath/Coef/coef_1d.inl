/*
    Copyright 2017 Philippe Grandclement
              2018 Ludwig Jens Papenfort

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
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

/**
 * @file coef_1d.cpp
 * @brief Implementation of spectral coefficient transformations using FFTW3
 * @author Philippe Grandclement, Ludwig Jens Papenfort
 * @date 2017-2018
 *
 * This file provides spectral coefficient transformation routines for converting
 * between physical space (values at collocation points) and spectral space
 * (expansion coefficients) using different orthogonal basis functions.
 *
 * The implementations leverage FFTW3 (Fastest Fourier Transform in the West)
 * library for efficient FFT operations. Plans and buffers are cached for reuse
 * to avoid repeated expensive FFTW planning operations.
 *
 * Supported basis types:
 * - Chebyshev polynomials (standard, even parity, odd parity)
 * - Legendre polynomials (standard, even parity, odd parity)
 * - Fourier series (cosine, sine, combined cosine-sine)
 * - Parity-adapted variants for exploiting function symmetries
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/base_fftw.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#include <fftw3.h>
#include "math.h"
#include <unordered_map>
#include <tuple>
#include "For_Kadath/Array/memory.hpp"

namespace Kadath
{

    // ============================================================================
    // FFTW Plan Management
    // ============================================================================

    /**
     * @brief Cache for precomputed FFTW plans and buffers
     *
     * FFTW3 computes an optimal FFT algorithm for each transform size through a
     * planning phase that can be computationally expensive. This map caches these
     * precomputed plans indexed by size, allowing reuse across multiple calls.
     *
     * Key benefits:
     * - Avoids repeated planning overhead
     * - Significantly improves performance for repeated transforms
     * - Thread-safe access after initialization (FFTW plans are thread-safe)
     */
    std::unordered_map<int, fftw_precomp_t> fftw_precomp_map;

    /**
     * @brief Get or create FFTW plan and buffer for specified transform size
     * @param n Size of the transform
     * @return Reference to precomputed FFTW plan and buffer structure
     *
     * If a plan for size n doesn't exist in the cache, creates a new R2HC
     * (real-to-half-complex) plan and stores it. Otherwise returns the cached plan.
     *
     * @note R2HC transform is used for real input data, producing half-complex output
     * @note The returned reference remains valid for the program lifetime
     */
    fftw_precomp_t& coef_1d_fftw(int n)
    {
        auto precomp_it = fftw_precomp_map.find(n);
        if (precomp_it == fftw_precomp_map.end())
            precomp_it =
                fftw_precomp_map
                    .emplace(std::piecewise_construct, std::forward_as_tuple(n), std::forward_as_tuple(n, FFTW_R2HC))
                    .first;
        return (*precomp_it).second;
    }

    // ============================================================================
    // Memory Management for Temporary Buffers
    // ============================================================================

    /**
     * @brief Static memory buffers for temporary coefficient computations
     * @note Prevents repeated heap allocation/deallocation during transformations
     * @note Two buffers serve the two simultaneous scratch roles used by some transforms
     */
    // coef_mem storage is intentionally defined only in
    // src/Coef/coef_1d.cpp.  Keeping definitions out of this archival inline
    // implementation prevents accidental per-TU scratch pools if a consumer
    // includes it directly.

    // ============================================================================
    // Function Declarations
    // ============================================================================

    void coef_i_1d(int, Array<double>&);

    /**
     * @brief Placeholder for unimplemented basis transformations
     * @param tab Array to transform (unused)
     * @note Aborts execution if called
     */
    void coef_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("Coef_1d not implemented.");
    }

    // ============================================================================
    // Chebyshev Transformations
    // ============================================================================

    /**
     * @brief Transform from Chebyshev collocation values to spectral coefficients
     * @param tab Input: function values at Chebyshev-Gauss-Lobatto points;
     *            Output: Chebyshev polynomial expansion coefficients
     * @pre tab must be 1-dimensional with size nr >= 2
     *
     * Computes Chebyshev coefficients T_k using a symmetry-exploiting FFT algorithm.
     * The transformation leverages the relationship between Chebyshev polynomials
     * and cosine functions, applying even/odd decomposition before FFT.
     *
     * Algorithm steps:
     * 1. Apply symmetry operations exploiting T_k(-x) = (-1)^k * T_k(x)
     * 2. Perform FFT on the symmetrized data
     * 3. Extract even-indexed coefficients (k=0,2,4,...) directly from FFT
     * 4. Compute odd-indexed coefficients (k=1,3,5,...) via recurrence relation
     * 5. Adjust first odd coefficient to satisfy boundary constraints
     *
     * @note Collocation points are x_i = cos(πi/(n-1)) for i=0,...,n-1
     */
    void coef_1d_cheb(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        double fmoins0 = 0.5 * (tab(0) - tab(nr - 1));

        auto& fftw_data = coef_1d_fftw(nr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        // Apply symmetry transformations
        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (tab(i) + tab(nr - 1 - i));
            double fms = 0.5 * (tab(i) - tab(nr - 1 - i)) * sin(M_PI * i / (nr - 1));
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (tab(0) + tab(nr - 1));
        tab_auxi[(nr - 1) / 2] = tab((nr - 1) / 2);

        fftw_execute(p);

        // Extract even coefficients (pairs)
        tab.set(0) = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            tab.set(i) = 2 * tab_auxi[(i / 2)] / (nr - 1);
        tab.set(nr - 1) = tab_auxi[(nr - 1) / 2] / (nr - 1);

        // Compute odd coefficients via recurrence
        tab.set(1) = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            tab.set(i) = tab(i - 2) - 4 * tab_auxi[nr - 1 - (i - 1) / 2] / (nr - 1);
            som += tab(i);
        }

        // Compute c_1 coefficient
        double c1 = -(fmoins0 + som) / ((nr - 1) / 2);

        // Adjust all odd coefficients
        tab.set(1) = c1;
        for (int i = 3; i < nr; i += 2)
            tab.set(i) += c1;
    }

    /**
     * @brief Transform to even-parity Chebyshev spectral coefficients
     * @param tab Input: function values at collocation points;
     *            Output: even Chebyshev polynomial coefficients
     * @pre tab must be 1-dimensional
     * @pre Function must have even parity: f(-x) = f(x)
     *
     * Exploits even parity to compute only coefficients of even Chebyshev
     * polynomials T_0, T_2, T_4, .... Odd coefficients are identically zero.
     * This reduces computation and improves accuracy for symmetric functions.
     */
    void coef_1d_cheb_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        double fmoins0 = -0.5 * (tab(0) - tab(nr - 1));

        auto& fftw_data = coef_1d_fftw(nr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        // Apply even-parity symmetry
        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (tab(i) + tab(nr - 1 - i));
            double fms = 0.5 * (-tab(i) + tab(nr - 1 - i)) * sin(M_PI * i / (nr - 1));
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (tab(0) + tab(nr - 1));
        tab_auxi[(nr - 1) / 2] = tab((nr - 1) / 2);

        fftw_execute(p);

        // Extract coefficients
        tab.set(0) = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            tab.set(i) = 2 * tab_auxi[(i / 2)] / (nr - 1);
        tab.set(nr - 1) = tab_auxi[(nr - 1) / 2] / (nr - 1);

        tab.set(1) = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            tab.set(i) = tab(i - 2) + 4 * tab_auxi[nr - 1 - i / 2] / (nr - 1);
            som += tab(i);
        }

        double c1 = (fmoins0 - som) / ((nr - 1) / 2);

        tab.set(1) = c1;
        for (int i = 3; i < nr; i += 2)
            tab.set(i) += c1;
    }

    /**
     * @brief Transform to odd-parity Chebyshev spectral coefficients
     * @param tab Input: function values at collocation points;
     *            Output: odd Chebyshev polynomial coefficients
     * @pre tab must be 1-dimensional
     * @pre Function must have odd parity: f(-x) = -f(x)
     *
     * Exploits odd parity to compute only coefficients of odd Chebyshev
     * polynomials T_1, T_3, T_5, .... Even coefficients are identically zero.
     * Applies sin(πi/2(n-1)) weighting before transformation, followed by
     * recurrence relations to obtain the final coefficients.
     */
    void coef_1d_cheb_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        double* cf = coef_mem::get_mem(0, nr);

        // Apply odd-parity weighting
        for (int i = 0; i < nr; i++)
            cf[i] = tab(i) * sin(M_PI / 2. * i / (nr - 1));

        double fmoins0 = -0.5 * (cf[0] - cf[nr - 1]);

        auto& fftw_data = coef_1d_fftw(nr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (cf[i] + cf[nr - 1 - i]);
            double fms = 0.5 * (-cf[i] + cf[nr - 1 - i]) * sin(M_PI * i / (nr - 1));
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (cf[0] + cf[nr - 1]);
        tab_auxi[(nr - 1) / 2] = cf[(nr - 1) / 2];

        fftw_execute(p);

        cf[0] = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            cf[i] = 2 * tab_auxi[(i / 2)] / (nr - 1);
        cf[nr - 1] = tab_auxi[(nr - 1) / 2] / (nr - 1);

        cf[1] = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            cf[i] = cf[i - 2] + 4 * tab_auxi[nr - 1 - i / 2] / (nr - 1);
            som += cf[i];
        }

        double c1 = (fmoins0 - som) / ((nr - 1) / 2);

        cf[1] = c1;
        for (int i = 3; i < nr; i += 2)
            cf[i] += c1;

        // Apply recurrence relation for odd basis
        cf[0] = 2 * cf[0];
        for (int i = 1; i < nr - 1; i++)
            cf[i] = 2 * cf[i] - cf[i - 1];
        cf[nr - 1] = 0;

        for (int i = 0; i < nr; i++)
            tab.set(i) = cf[i];
    }

    // ============================================================================
    // Legendre Transformations
    // ============================================================================

    // Forward declarations for Legendre helper functions
    double coloc_leg(int, int);
    double weight_leg(int, int);
    double gamma_leg(int, int);
    double leg(int, double);
    double coloc_leg_parity(int, int);
    double weight_leg_parity(int, int);
    double gamma_leg_even(int, int);
    double gamma_leg_odd(int, int);

    /**
     * @brief Transform from Gauss-Legendre collocation to Legendre coefficients
     * @param tab Input: function values at Gauss-Legendre points;
     *            Output: Legendre polynomial expansion coefficients
     * @pre tab must be 1-dimensional
     *
     * Computes Legendre polynomial coefficients P_k using direct Gauss-Legendre
     * quadrature. This method evaluates the orthogonal projection:
     *
     *   c_k = (2k+1)/2 * ∫_{-1}^{1} f(x) P_k(x) dx
     *
     * using Gauss-Legendre collocation points and weights for exact integration
     * of polynomials up to degree 2n-1.
     *
     * @note This is an O(n²) algorithm, slower than FFT-based Chebyshev transforms
     * @note More accurate for smooth functions due to exponential convergence
     */
    void coef_1d_leg(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) += tab(j) * leg(i, coloc_leg(j, nbr)) * weight_leg(j, nbr) / gamma_leg(i, nbr);
        tab = res;
    }

    /**
     * @brief Transform to even-parity Legendre spectral coefficients
     * @param tab Input: function values at collocation points;
     *            Output: even Legendre polynomial coefficients
     * @pre tab must be 1-dimensional
     * @pre Function has even parity: f(-x) = f(x)
     *
     * Only even Legendre polynomials P_0, P_2, P_4, ... contribute to expansion.
     */
    void coef_1d_leg_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) +=
                    tab(j) * leg(i * 2, coloc_leg_parity(j, nbr)) * weight_leg_parity(j, nbr) / gamma_leg_even(i, nbr);
        tab = res;
    }

    /**
     * @brief Transform to odd-parity Legendre spectral coefficients
     * @param tab Input: function values at collocation points;
     *            Output: odd Legendre polynomial coefficients
     * @pre tab must be 1-dimensional
     * @pre Function has odd parity: f(-x) = -f(x)
     *
     * Only odd Legendre polynomials P_1, P_3, P_5, ... contribute to expansion.
     */
    void coef_1d_leg_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr - 1; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) += tab(j) * leg(i * 2 + 1, coloc_leg_parity(j, nbr)) * weight_leg_parity(j, nbr) /
                              gamma_leg_odd(i, nbr);
        tab = res;
    }

    // ============================================================================
    // Fourier (Cos/Sin) Transformations
    // ============================================================================

    /**
     * @brief Transform to combined cosine-sine Fourier coefficients
     * @param tab Input: function values at equally-spaced points;
     *            Output: interleaved cosine and sine coefficients
     * @pre tab must be 1-dimensional with size >= 4
     *
     * Computes both cosine and sine Fourier coefficients in a single FFT operation.
     * Expansion: f(x) = Σ [a_k cos(kx) + b_k sin(kx)]
     *
     * Output coefficient ordering:
     * [a_0, b_0=0, a_1, b_1, a_2, b_2, ..., a_{n/2}, b_{n/2}=0]
     *
     * @note Sine coefficients at k=0 and k=n/2 are always zero by construction
     * @note For nbr <= 3, sets coefficients to zero (insufficient points)
     */
    void coef_1d_cossin(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            auto& fftw_data = coef_1d_fftw(nbr - 2);
            double* cf = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 0; i < nbr - 2; i++)
                cf[i] = tab(i);
            fftw_execute(p);

            int index = 0;
            double* pcos = cf;
            double* psin = cf + nbr - 3;

            tab.set(index) = *pcos / double(nbr - 2);
            index++;
            pcos++;
            tab.set(index) = 0;
            index++;

            for (int i = 1; i < nbr / 2 - 1; i++) {
                tab.set(index) = 2. * (*pcos) / double(nbr - 2); // cosines
                index++;
                pcos++;
                tab.set(index) = -2. * (*psin) / double(nbr - 2); // sines
                index++;
                psin--;
            }

            tab.set(index) = *pcos / double(nbr - 2);
            index++;
            tab.set(index) = 0;
        } else {
            tab.set(1) = 0;
            tab.set(2) = 0;
        }
    }

    /**
     * @brief Transform to pure cosine Fourier coefficients
     * @param tab Input: function values at collocation points;
     *            Output: cosine series coefficients
     * @pre tab must be 1-dimensional
     *
     * Computes coefficients for pure cosine expansion: f(x) = Σ a_k cos(kx)
     * Exploits symmetry properties to efficiently extract only cosine terms.
     */
    void coef_1d_cos(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Symmetry taken into account
        double fmoins0 = 0.5 * (tab(0) - tab(nbr - 1));

        auto& fftw_data = coef_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab(i) + tab(nbr - 1 - i));
            double fms = 0.5 * (tab(i) - tab(nbr - 1 - i)) * sin(M_PI * i / (nbr - 1));
            tab_auxi[i] = fp + fms;
            tab_auxi[nbr - 1 - i] = fp - fms;
        }

        tab_auxi[0] = 0.5 * (tab(0) + tab(nbr - 1));
        tab_auxi[(nbr - 1) / 2] = tab((nbr - 1) / 2);

        fftw_execute(p);

        tab.set(0) = tab_auxi[0] / (nbr - 1);
        for (int i = 2; i < nbr - 1; i += 2)
            tab.set(i) = 2 * tab_auxi[i / 2] / (nbr - 1);
        tab.set(nbr - 1) = tab_auxi[(nbr - 1) / 2] / (nbr - 1);

        tab.set(1) = 0;
        double som = 0;
        for (int i = 3; i < nbr; i += 2) {
            tab.set(i) = tab(i - 2) + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
            som += tab(i);
        }

        double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

        tab.set(1) = c1;
        for (int i = 3; i < nbr; i += 2)
            tab.set(i) += c1;
    }

    /**
     * @brief Transform to sine Fourier coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     *
     * Pure sine expansion exploiting symmetry properties.
     */
    void coef_1d_sin(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Symmetry taken into account
        auto& fftw_data = coef_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab(i) + tab(nbr - 1 - i)) * sin(M_PI * i / (nbr - 1));
            double fms = 0.5 * (tab(i) - tab(nbr - 1 - i));
            tab_auxi[i] = fp + fms;
            tab_auxi[nbr - 1 - i] = fp - fms;
        }

        tab_auxi[0] = 0.5 * (tab(0) + tab(nbr - 1));
        tab_auxi[(nbr - 1) / 2] = tab((nbr - 1) / 2);

        fftw_execute(p);

        tab.set(0) = 0;
        for (int i = 2; i < nbr - 1; i += 2)
            tab.set(i) = -2 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
        tab.set(nbr - 1) = 0;

        tab.set(1) = 2 * tab_auxi[0] / (nbr - 1);
        for (int i = 3; i < nbr; i += 2)
            tab.set(i) = tab(i - 2) + 4 * tab_auxi[i / 2] / (nbr - 1);
    }

    /**
     * @brief Transform to even-parity cosine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     */
    void coef_1d_cos_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double fmoins0 = 0.5 * (tab(0) - tab(nbr - 1));

            auto& fftw_data = coef_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab(i) + tab(nbr - 1 - i));
                double fms = 0.5 * (tab(i) - tab(nbr - 1 - i)) * sin(M_PI * i / (nbr - 1));
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (tab(0) + tab(nbr - 1));
            tab_auxi[(nbr - 1) / 2] = tab((nbr - 1) / 2);

            fftw_execute(p);

            tab.set(0) = tab_auxi[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                tab.set(i) = 2 * tab_auxi[i / 2] / (nbr - 1);
            tab.set(nbr - 1) = tab_auxi[(nbr - 1) / 2] / (nbr - 1);

            tab.set(1) = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                tab.set(i) = tab(i - 2) + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
                som += tab(i);
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            tab.set(1) = c1;
            for (int i = 3; i < nbr; i += 2)
                tab.set(i) += c1;
        }
    }

    /**
     * @brief Transform to odd-parity cosine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     */
    void coef_1d_cos_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double* cf = coef_mem::get_mem(0, nbr);

            for (int i = 0; i < nbr - 1; i++)
                cf[i] = tab(i) * sin(M_PI * (nbr - 1 - i) / 2. / (nbr - 1));
            cf[nbr - 1] = 0;
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            auto& fftw_data = coef_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin(M_PI * i / (nbr - 1));
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            tab_auxi[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            fftw_execute(p);

            cf[0] = tab_auxi[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                cf[i] = 2 * tab_auxi[i / 2] / (nbr - 1);
            cf[nbr - 1] = tab_auxi[(nbr - 1) / 2];

            cf[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = cf[i - 2] + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
                som += cf[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] - cf[i - 1];
            cf[nbr - 1] = 0;

            for (int i = 0; i < nbr; i++)
                tab.set(i) = cf[i];
        }
    }

    /**
     * @brief Transform to even-parity sine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     */
    void coef_1d_sin_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            auto& fftw_data = coef_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab(i) + tab(nbr - 1 - i)) * sin(M_PI * i / (nbr - 1));
                double fms = 0.5 * (tab(i) - tab(nbr - 1 - i));
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (tab(0) - tab(nbr - 1));
            tab_auxi[(nbr - 1) / 2] = tab((nbr - 1) / 2);

            fftw_execute(p);

            tab.set(0) = 0.;
            for (int i = 2; i < nbr - 1; i += 2)
                tab.set(i) = -2 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
            tab.set(nbr - 1) = 0;

            tab.set(1) = 2 * tab_auxi[0] / (nbr - 1);
            for (int i = 3; i < nbr; i += 2)
                tab.set(i) = tab(i - 2) + 4 * tab_auxi[i / 2] / (nbr - 1);
        }
    }

    /**
     * @brief Transform to odd-parity sine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     */
    void coef_1d_sin_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double* cf = coef_mem::get_mem(0, nbr);
            cf[0] = 0;
            for (int i = 1; i < nbr; i++)
                cf[i] = tab(i) * sin(M_PI * i / 2. / (nbr - 1));
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            auto& fftw_data = coef_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin(M_PI * i / (nbr - 1));
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            tab_auxi[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            fftw_execute(p);

            cf[0] = tab_auxi[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                cf[i] = 2 * tab_auxi[i / 2] / (nbr - 1);
            cf[nbr - 1] = tab_auxi[(nbr - 1) / 2] / (nbr - 1);

            cf[1] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = cf[i - 2] + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
                som += cf[i];
            }

            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] + cf[i - 1];
            cf[nbr - 1] = 0;

            for (int i = 0; i < nbr; i++)
                tab.set(i) = cf[i];
        }
    }

    /**
     * @brief Transform to even-parity combined cosine-sine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     *
     * Exploits even parity by doubling the array size and applying symmetry.
     */
    void coef_1d_cossin_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Copy values into double-size array with symmetry
        Array<double> tab2(2 * nbr - 2);
        for (int i = 0; i < nbr - 2; i++) {
            tab2.set(i) = tab(i);
            tab2.set(i + nbr - 2) = tab(i);
        }

        coef_1d_cossin(tab2);

        // Extract every other pair of coefficients
        int conte = 0;
        for (int k = 0; k < nbr - 1; k += 2) {
            tab.set(k) = tab2(conte);
            tab.set(k + 1) = tab2(conte + 1);
            conte += 4;
        }
    }

    /**
     * @brief Transform to odd-parity combined cosine-sine coefficients
     * @param tab Input: values at collocation points; Output: spectral coefficients
     * @pre tab must be 1-dimensional
     *
     * Exploits odd parity using antisymmetric extension in doubled array.
     */
    void coef_1d_cossin_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Copy values into double-size array with antisymmetry
        Array<double> tab2(2 * nbr - 2);
        tab2 = 0;
        for (int i = 0; i < nbr - 2; i++) {
            tab2.set(i) = tab(i);
            tab2.set(i + nbr - 2) = -tab(i);
        }

        coef_1d_cossin(tab2);

        // Extract coefficient pairs starting at offset 2
        int conte = 2;
        for (int k = 0; k < nbr - 3; k += 2) {
            tab.set(k) = tab2(conte);
            tab.set(k + 1) = tab2(conte + 1);
            conte += 4;
        }
        tab.set(nbr - 2) = 0.;
        tab.set(nbr - 1) = 0.;
    }

    // ============================================================================
    // Dispatcher Function
    // ============================================================================

    /**
     * @brief Main dispatch function for 1D coefficient transformations
     * @param base Basis type identifier (from base_spectral.hpp constants)
     * @param tab Array to transform (in-place modification)
     *
     * This function uses a static function pointer table initialized on first call
     * to efficiently dispatch to the appropriate transformation routine based on
     * the basis type.
     *
     * Supported basis types:
     * - CHEB, CHEB_EVEN, CHEB_ODD: Chebyshev polynomials
     * - LEG, LEG_EVEN, LEG_ODD: Legendre polynomials
     * - COS, COS_EVEN, COS_ODD: Cosine bases
     * - SIN, SIN_EVEN, SIN_ODD: Sine bases
     * - COSSIN, COSSIN_EVEN, COSSIN_ODD: Combined cosine-sine bases
     *
     * @note Function pointer table is initialized once on first call for efficiency
     * @note Unimplemented basis types will abort with error message
     */
    void coef_1d(int base, Array<double>& tab)
    {
        static void (*coef_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // First call: initialize function pointer table
        if (premier_appel) {
            premier_appel = false;

            // Default all entries to unimplemented handler
            for (int i = 0; i < NBR_MAX_BASE; i++)
                coef_1d[i] = coef_1d_pasprevu;

            // Register implemented transformations
            coef_1d[CHEB] = coef_1d_cheb;
            coef_1d[CHEB_EVEN] = coef_1d_cheb_even;
            coef_1d[CHEB_ODD] = coef_1d_cheb_odd;
            coef_1d[COSSIN] = coef_1d_cossin;
            coef_1d[COS_EVEN] = coef_1d_cos_even;
            coef_1d[COS_ODD] = coef_1d_cos_odd;
            coef_1d[SIN_EVEN] = coef_1d_sin_even;
            coef_1d[SIN_ODD] = coef_1d_sin_odd;
            coef_1d[COS] = coef_1d_cos;
            coef_1d[SIN] = coef_1d_sin;
            coef_1d[LEG] = coef_1d_leg;
            coef_1d[LEG_EVEN] = coef_1d_leg_even;
            coef_1d[LEG_ODD] = coef_1d_leg_odd;
            coef_1d[COSSIN_EVEN] = coef_1d_cossin_even;
            coef_1d[COSSIN_ODD] = coef_1d_cossin_odd;
        }

        // Dispatch to appropriate transformation
        coef_1d[base](tab);
    }

} // namespace Kadath
