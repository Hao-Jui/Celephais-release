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

/**
 * @file coef_i_1d.cpp
 * @brief Implementation of inverse spectral coefficient transformations using FFTW3
 * @author Philippe Grandclement, Ludwig Jens Papenfort
 * @date 2017-2018
 *
 * This file provides inverse spectral transformation routines for converting
 * from spectral space (expansion coefficients) back to physical space (function
 * values at collocation points). These are the inverse operations of the forward
 * transforms implemented in coef_fftw.cpp.
 *
 * The transformations leverage FFTW3's HC2R (half-complex-to-real) transforms
 * which efficiently invert the R2HC (real-to-half-complex) transforms used in
 * the forward direction. This ensures exact reconstruction of function values
 * from their spectral representations.
 *
 * Supported inverse transforms:
 * - Chebyshev polynomials (standard, even parity, odd parity)
 * - Legendre polynomials (standard, even parity, odd parity)
 * - Fourier series (cosine, sine, combined cosine-sine)
 * - Parity-adapted variants for symmetric/antisymmetric functions
 *
 * Mathematical operation: Given spectral coefficients {c_k}, reconstruct function
 * values at collocation points via f(x_i) = Σ_k c_k φ_k(x_i) where φ_k are the
 * orthogonal basis functions (Chebyshev, Legendre, or Fourier).
 *
 * Key property: For any basis, applying forward transform followed by inverse
 * transform (or vice versa) returns the original values (up to numerical precision).
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/base_fftw.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#include <fftw3.h>
#include <unordered_map>
#include <tuple>

namespace Kadath
{

    // ============================================================================
    // FFTW Plan Management for Inverse Transforms
    // ============================================================================

    /**
     * @brief Cache for precomputed inverse FFTW plans and buffers
     *
     * Stores HC2R (half-complex-to-real) FFTW plans for efficient inverse
     * transforms. Kept separate from the forward transform cache (fftw_precomp_map)
     * to avoid conflicts between forward (R2HC) and inverse (HC2R) plan types.
     *
     * Key benefits:
     * - Eliminates repeated expensive FFTW planning for inverse transforms
     * - Maintains separate namespace for forward vs inverse operations
     * - Thread-safe after initialization (FFTW plans are reentrant)
     *
     * @note HC2R is the mathematical inverse of the R2HC used in forward transforms
     * @note Both caches persist for the lifetime of the program
     */
    std::unordered_map<int, fftw_precomp_t> fftw_precomp_map_i;

    /**
     * @brief Get or create inverse FFTW plan and buffer for specified size
     * @param n Size of the inverse transform
     * @return Reference to precomputed HC2R FFTW plan and buffer structure
     *
     * If a plan for size n doesn't exist in the cache, creates a new HC2R
     * (half-complex-to-real) plan and stores it. Otherwise returns cached plan.
     *
     * @note HC2R transform converts half-complex frequency data to real spatial data
     * @note The returned reference remains valid for the program lifetime
     */
    fftw_precomp_t& coef_i_1d_fftw(int n)
    {
        auto precomp_it = fftw_precomp_map_i.find(n);
        if (precomp_it == fftw_precomp_map_i.end())
            precomp_it =
                fftw_precomp_map_i
                    .emplace(std::piecewise_construct, std::forward_as_tuple(n), std::forward_as_tuple(n, FFTW_HC2R))
                    .first;
        return (*precomp_it).second;
    }

    // ============================================================================
    // Function Declarations
    // ============================================================================

    /**
     * @brief Placeholder for unimplemented inverse basis transformations
     * @param tab Array to transform (unused)
     * @note Aborts execution with error message if called
     */
    void coef_i_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("Coef_1d not implemented.");
    }

    // ============================================================================
    // Inverse Chebyshev Transformations
    // ============================================================================

    /**
     * @brief Inverse transform from Chebyshev coefficients to collocation values
     * @param tab Input: Chebyshev polynomial expansion coefficients;
     *            Output: function values at Chebyshev-Gauss-Lobatto points
     * @pre tab must be 1-dimensional with size > 3
     * @pre Input contains valid Chebyshev coefficients c_k for expansion Σ c_k T_k(x)
     *
     * Reconstructs function values at collocation points x_i = cos(πi/(n-1))
     * from Chebyshev coefficients using inverse FFT. This is the inverse of
     * coef_1d_cheb().
     *
     * Algorithm steps:
     * 1. Reconstruct odd coefficients from recurrence relation
     * 2. Prepare even coefficients for inverse FFT
     * 3. Execute HC2R inverse FFT
     * 4. Apply symmetry operations to obtain final collocation values
     *
     * @post tab contains function values at Chebyshev collocation points
     */
    void coef_i_1d_cheb(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            auto& fftw_data = coef_i_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            double* cf = coef_mem::get_mem(0, nbr);

            // Reconstruct odd coefficients
            double c1 = tab(1);
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = tab(i) - c1;
                somme += cf[i];
            }
            double fmoins0 = -(nbr - 1) / 2 * c1 - somme;

            // Prepare for inverse FFT
            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = -0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = tab(0);
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * tab(2 * i);
            tab_auxi[(nbr - 1) / 2] = tab(nbr - 1);

            fftw_execute(p);

            // Apply symmetry to reconstruct values
            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
                tab.set(i) = fp + fm;
                tab.set(nbr - i - 1) = fp - fm;
            }
            tab.set(0) = tab_auxi[0] + fmoins0;
            tab.set(nbr - 1) = tab_auxi[0] - fmoins0;
            tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
        }
    }

    /**
     * @brief Inverse transform for even-parity Chebyshev coefficients
     * @param tab Input: even Chebyshev polynomial coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     * @pre Input coefficients represent even function: only even T_k present
     *
     * Reconstructs symmetric function values from even Chebyshev coefficients.
     * Inverse of coef_1d_cheb_even().
     *
     * @post Output satisfies f(x_i) = f(x_{n-1-i}) (even symmetry)
     */
    void coef_i_1d_cheb_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        auto& fftw_data = coef_i_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        double* cf = coef_mem::get_mem(0, nbr);

        double c1 = tab(1);
        double somme = 0;
        cf[1] = 0;
        for (int i = 3; i < nbr; i += 2) {
            cf[i] = tab(i) - c1;
            somme += cf[i];
        }
        double fmoins0 = (nbr - 1) / 2 * c1 + somme;

        for (int i = 3; i < nbr; i += 2)
            tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
        tab_auxi[0] = tab(0);
        for (int i = 1; i < (nbr - 1) / 2; i++)
            tab_auxi[i] = 0.5 * tab(2 * i);
        tab_auxi[(nbr - 1) / 2] = tab(nbr - 1);

        fftw_execute(p);

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
            tab.set(nbr - 1 - i) = fp + fm;
            tab.set(i) = fp - fm;
        }
        tab.set(0) = tab_auxi[0] - fmoins0;
        tab.set(nbr - 1) = tab_auxi[0] + fmoins0;
        tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
    }

    /**
     * @brief Inverse transform for odd-parity Chebyshev coefficients
     * @param tab Input: odd Chebyshev polynomial coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     * @pre Input coefficients represent odd function: only odd T_k present
     *
     * Reconstructs antisymmetric function values from odd Chebyshev coefficients.
     * Includes recurrence relation application and sin-weighting removal.
     * Inverse of coef_1d_cheb_odd().
     *
     * @post Output satisfies f(x_i) = -f(x_{n-1-i}) (odd symmetry)
     * @post tab(0) = 0 (odd function vanishes at x=1)
     */
    void coef_i_1d_cheb_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        auto& fftw_data = coef_i_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        double* cf = coef_mem::get_mem(0, nbr);
        double* ti = coef_mem::get_mem(1, nbr);

        // Apply recurrence relation
        ti[0] = 0.5 * tab(0);
        for (int i = 1; i < nbr - 1; i++)
            ti[i] = 0.5 * (tab(i) + tab(i - 1));
        ti[nbr - 1] = 0.5 * tab(nbr - 2);

        double c1 = ti[1];
        double somme = 0;
        cf[1] = 0;
        for (int i = 3; i < nbr; i += 2) {
            cf[i] = ti[i] - c1;
            somme += cf[i];
        }
        double fmoins0 = (nbr - 1) / 2 * c1 + somme;

        for (int i = 3; i < nbr; i += 2)
            tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
        tab_auxi[0] = ti[0];
        for (int i = 1; i < (nbr - 1) / 2; i++)
            tab_auxi[i] = 0.5 * ti[2 * i];
        tab_auxi[(nbr - 1) / 2] = ti[nbr - 1];

        fftw_execute(p);

        // Remove sin-weighting applied in forward transform
        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
            tab.set(nbr - 1 - i) = (fp + fm) / sin(M_PI * (nbr - 1 - i) / 2 / (nbr - 1));
            tab.set(i) = (fp - fm) / sin(M_PI * i / 2 / (nbr - 1));
        }
        tab.set(0) = 0;
        tab.set(nbr - 1) = tab_auxi[0] + fmoins0;
        tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2] / sin(M_PI * (nbr - 1) / 4 / (nbr - 1));
    }

    // ============================================================================
    // Inverse Legendre Transformations
    // ============================================================================

    // Forward declarations for Legendre helper functions
    double coloc_leg(int, int);
    double summation_1d_leg(double xx, const Array<double>&);
    double coloc_leg_parity(int, int);
    double summation_1d_leg_even(double xx, const Array<double>&);
    double summation_1d_leg_odd(double xx, const Array<double>&);

    /**
     * @brief Inverse transform from Legendre coefficients to collocation values
     * @param tab Input: Legendre polynomial coefficients;
     *            Output: function values at Gauss-Legendre points
     * @pre tab must be 1-dimensional
     *
     * Evaluates the Legendre series f(x) = Σ_k c_k P_k(x) at Gauss-Legendre
     * collocation points. Uses direct summation (no FFT available for Legendre).
     * Inverse of coef_1d_leg().
     *
     * @note O(n²) complexity due to direct polynomial evaluation
     * @post Output contains function values at Gauss-Legendre quadrature points
     */
    void coef_i_1d_leg(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        for (int i = 0; i < nbr; i++)
            res.set(i) = summation_1d_leg(coloc_leg(i, nbr), tab);
        tab = res;
    }

    /**
     * @brief Inverse transform for even-parity Legendre coefficients
     * @param tab Input: even Legendre polynomial coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     * @pre Input contains coefficients for even Legendre polynomials only
     *
     * Evaluates f(x) = Σ_k c_k P_{2k}(x) at parity-adapted collocation points.
     * Inverse of coef_1d_leg_even().
     */
    void coef_i_1d_leg_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        for (int i = 0; i < nbr; i++)
            res.set(i) = summation_1d_leg_even(coloc_leg_parity(i, nbr), tab);
        tab = res;
    }

    /**
     * @brief Inverse transform for odd-parity Legendre coefficients
     * @param tab Input: odd Legendre polynomial coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     * @pre Input contains coefficients for odd Legendre polynomials only
     *
     * Evaluates f(x) = Σ_k c_k P_{2k+1}(x) at parity-adapted collocation points.
     * Inverse of coef_1d_leg_odd().
     */
    void coef_i_1d_leg_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        Array<double> res(nbr);
        for (int i = 0; i < nbr; i++)
            res.set(i) = summation_1d_leg_odd(coloc_leg_parity(i, nbr), tab);
        tab = res;
    }

    // ============================================================================
    // Inverse Fourier (Cos/Sin) Transformations
    // ============================================================================

    /**
     * @brief Inverse transform from combined cosine-sine coefficients to values
     * @param tab Input: interleaved cosine and sine coefficients;
     *            Output: function values at equally-spaced points
     * @pre tab must be 1-dimensional with size >= 3
     * @pre np = nbr-2 > 1
     *
     * Reconstructs f(x) = Σ [a_k cos(kx) + b_k sin(kx)] from interleaved
     * coefficients [a_0, b_0, a_1, b_1, ...]. Uses inverse FFT for efficiency.
     * Inverse of coef_1d_cossin().
     *
     * @post Output contains function values at equally-spaced points
     */
    void coef_i_1d_cossin(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);
        int np = nbr - 2;

        if (np > 1) {
            auto& fftw_data = coef_i_1d_fftw(np);
            double* cf = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            // Deinterleave coefficients for HC2R format
            cf[0] = tab(0);
            for (int i = 1; i < np / 2; i++) {
                cf[i] = 0.5 * tab(2 * i);           // Cosine coefficients
                cf[np - i] = -0.5 * tab(2 * i + 1); // Sine coefficients
            }
            cf[np / 2] = tab(np);

            fftw_execute(p);

            for (int i = 0; i < np; i++)
                tab.set(i) = cf[i];
        }
    }

    /**
     * @brief Inverse transform from pure cosine coefficients to values
     * @param tab Input: cosine series coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     *
     * Reconstructs f(x) = Σ a_k cos(kx) from cosine coefficients using
     * inverse FFT with symmetry exploitation. Inverse of coef_1d_cos().
     */
    void coef_i_1d_cos(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        auto& fftw_data = coef_i_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        double* cf = coef_mem::get_mem(0, nbr);

        double c1 = tab(1);
        double somme = 0;
        cf[1] = 0;
        for (int i = 3; i < nbr; i += 2) {
            cf[i] = tab(i) - c1;
            somme += cf[i];
        }
        double fmoins0 = (nbr - 1) / 2 * c1 + somme;

        for (int i = 3; i < nbr; i += 2)
            tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
        tab_auxi[0] = tab(0);
        for (int i = 1; i < (nbr - 1) / 2; i++)
            tab_auxi[i] = 0.5 * tab(2 * i);
        tab_auxi[(nbr - 1) / 2] = tab(nbr - 1);

        fftw_execute(p);

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
            tab.set(i) = fp + fm;
            tab.set(nbr - i - 1) = fp - fm;
        }
        tab.set(0) = tab_auxi[0] + fmoins0;
        tab.set(nbr - 1) = tab_auxi[0] - fmoins0;
        tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
    }

    /**
     * @brief Inverse transform from pure sine coefficients to values
     * @param tab Input: sine series coefficients;
     *            Output: function values at collocation points
     * @pre tab must be 1-dimensional
     *
     * Reconstructs f(x) = Σ b_k sin(kx) from sine coefficients using
     * inverse FFT with symmetry exploitation. Inverse of coef_1d_sin().
     *
     * @post tab(0) = 0 and tab(nbr-1) should be near zero (sine boundary values)
     */
    void coef_i_1d_sin(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        auto& fftw_data = coef_i_1d_fftw(nbr - 1);
        double* tab_auxi = fftw_data.buffer;
        fftw_plan p = fftw_data.plan;

        // Prepare sine coefficients for inverse FFT
        for (int i = 2; i < nbr - 1; i += 2)
            tab_auxi[nbr - 1 - i / 2] = -0.5 * tab(i);
        tab_auxi[0] = 0.5 * tab(1);
        for (int i = 3; i < nbr; i += 2)
            tab_auxi[i / 2] = 0.25 * (tab(i) - tab(i - 2));
        tab_auxi[(nbr - 1) / 2] = -0.5 * tab(nbr - 2);

        fftw_execute(p);

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]);
            tab.set(i) = fp + fm;
            tab.set(nbr - i - 1) = fp - fm;
        }
        tab.set(0) = 0;
        tab.set(nbr - 1) = -2 * tab_auxi[0];
        tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
    }

    /**
     * @brief Inverse transform for even-parity cosine coefficients
     * @param tab Input: cosine coefficients for even function;
     *            Output: symmetric function values
     * @pre tab must be 1-dimensional with size > 3
     *
     * Inverse of coef_1d_cos_even(). Reconstructs even function from
     * cosine expansion.
     */
    void coef_i_1d_cos_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double* cf = coef_mem::get_mem(0, nbr);

            auto& fftw_data = coef_i_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            double c1 = tab(1);
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = tab(i) - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;

            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = tab(0);
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * tab(2 * i);
            tab_auxi[(nbr - 1) / 2] = tab(nbr - 1);

            fftw_execute(p);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
                tab.set(i) = fp + fm;
                tab.set(nbr - i - 1) = fp - fm;
            }
            tab.set(0) = tab_auxi[0] + fmoins0;
            tab.set(nbr - 1) = tab_auxi[0] - fmoins0;
            tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
        }
    }

    /**
     * @brief Inverse transform for odd-parity cosine coefficients
     * @param tab Input: cosine coefficients for odd function;
     *            Output: antisymmetric function values
     * @pre tab must be 1-dimensional with size > 3
     *
     * Inverse of coef_1d_cos_odd(). Includes recurrence relation and
     * removal of sin-weighting used in forward transform.
     *
     * @post tab(0) and tab(nbr-1) reflect odd parity constraints
     */
    void coef_i_1d_cos_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double* cf = coef_mem::get_mem(0, nbr);
            double* ti = coef_mem::get_mem(1, nbr);

            auto& fftw_data = coef_i_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            ti[0] = 0.5 * tab(0);
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (tab(i) + tab(i - 1));
            ti[nbr - 1] = 0.5 * tab(nbr - 2);

            double c1 = ti[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = ti[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;

            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = ti[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * ti[2 * i];
            tab_auxi[(nbr - 1) / 2] = ti[nbr - 1];

            fftw_execute(p);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
                tab.set(i) = (fp + fm) / sin(M_PI * (nbr - 1 - i) / 2 / (nbr - 1));
                tab.set(nbr - i - 1) = (fp - fm) / sin(M_PI * i / 2 / (nbr - 1));
            }
            tab.set(0) = tab_auxi[0] + fmoins0;
            tab.set(nbr - 1) = 0;
            tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2] / sin(M_PI * (nbr - 1) / 4 / (nbr - 1));
        }
    }

    /**
     * @brief Inverse transform for even-parity sine coefficients
     * @param tab Input: sine coefficients for even function;
     *            Output: function values with even symmetry
     * @pre tab must be 1-dimensional with size > 3
     *
     * Inverse of coef_1d_sin_even(). Special handling for sine basis
     * representing an even function.
     */
    void coef_i_1d_sin_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            auto& fftw_data = coef_i_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            for (int i = 2; i < nbr - 1; i += 2)
                tab_auxi[nbr - 1 - i / 2] = -0.5 * tab(i);
            tab_auxi[0] = 0.5 * tab(1);
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.25 * (tab(2 * i + 1) - tab(2 * i - 1));
            tab_auxi[(nbr - 1) / 2] = -0.5 * tab(nbr - 2);

            fftw_execute(p);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]);
                tab.set(i) = fp + fm;
                tab.set(nbr - i - 1) = fp - fm;
            }
            tab.set(0) = 0;
            tab.set(nbr - 1) = -2 * tab_auxi[0];
            tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2];
        }
    }

    /**
     * @brief Inverse transform for odd-parity sine coefficients
     * @param tab Input: sine coefficients for odd function;
     *            Output: antisymmetric function values
     * @pre tab must be 1-dimensional with size > 3
     *
     * Inverse of coef_1d_sin_odd(). Natural pairing of odd functions with
     * sine basis. Includes recurrence relations and sin-weighting removal.
     *
     * @post tab(0) = 0 (odd function constraint at boundary)
     */
    void coef_i_1d_sin_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        if (nbr > 3) {
            double* cf = coef_mem::get_mem(0, nbr);
            double* ti = coef_mem::get_mem(1, nbr);

            auto& fftw_data = coef_i_1d_fftw(nbr - 1);
            double* tab_auxi = fftw_data.buffer;
            fftw_plan p = fftw_data.plan;

            ti[0] = 0.5 * tab(0);
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (tab(i) - tab(i - 1));
            ti[nbr - 1] = -0.5 * tab(nbr - 2);

            double c1 = ti[1];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = ti[i] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;

            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = ti[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * ti[2 * i];
            tab_auxi[(nbr - 1) / 2] = ti[nbr - 1];

            fftw_execute(p);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin(M_PI * i / (nbr - 1));
                tab.set(i) = (fp + fm) / sin(M_PI * i / 2 / (nbr - 1));
                tab.set(nbr - i - 1) = (fp - fm) / sin(M_PI * (nbr - 1 - i) / 2 / (nbr - 1));
            }
            tab.set(0) = 0;
            tab.set(nbr - 1) = tab_auxi[0] - fmoins0;
            tab.set((nbr - 1) / 2) = tab_auxi[(nbr - 1) / 2] / sin(M_PI * (nbr - 1) / 4 / (nbr - 1));
        }
    }

    /**
     * @brief Inverse transform for even-parity combined cosine-sine coefficients
     * @param tab Input: cossin coefficients for even function;
     *            Output: symmetric function values
     * @pre tab must be 1-dimensional
     *
     * Inverse of coef_1d_cossin_even(). Creates doubled array with appropriate
     * zero-padding, performs inverse cossin transform, then extracts first half.
     *
     * @post Last two values set to zero
     */
    void coef_i_1d_cossin_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Create double-sized array with zero-padding
        Array<double> tab2(nbr * 2 - 2);
        int conte = 0;
        for (int i = 0; i < nbr - 2; i += 2) {
            tab2.set(conte) = tab(i);
            tab2.set(conte + 1) = tab(i + 1);
            tab2.set(conte + 2) = 0.;
            tab2.set(conte + 3) = 0.;
            conte += 4;
        }
        tab2.set(conte) = tab(nbr - 2);
        tab2.set(conte + 1) = tab(nbr - 1);

        coef_i_1d_cossin(tab2);

        for (int i = 0; i < nbr - 2; i++)
            tab.set(i) = tab2(i);
        tab.set(nbr - 2) = 0;
        tab.set(nbr - 1) = 0;
    }

    /**
     * @brief Inverse transform for odd-parity combined cosine-sine coefficients
     * @param tab Input: cossin coefficients for odd function;
     *            Output: antisymmetric function values
     * @pre tab must be 1-dimensional
     *
     * Inverse of coef_1d_cossin_odd(). Creates doubled array with coefficients
     * shifted and zero-padded, performs inverse cossin transform, extracts result.
     *
     * @post Last two values set to zero
     */
    void coef_i_1d_cossin_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nbr = tab.get_size(0);

        // Create double-sized array with shifted coefficients
        Array<double> tab2(nbr * 2 - 2);
        int conte = 0;
        for (int i = 0; i < nbr - 2; i += 2) {
            tab2.set(conte) = 0.;
            tab2.set(conte + 1) = 0.;
            tab2.set(conte + 2) = tab(i);
            tab2.set(conte + 3) = tab(i + 1);
            conte += 4;
        }
        tab2.set(conte) = tab(nbr - 2);
        tab2.set(conte + 1) = tab(nbr - 1);

        coef_i_1d_cossin(tab2);

        for (int i = 0; i < nbr - 2; i++)
            tab.set(i) = tab2(i);
        tab.set(nbr - 2) = 0;
        tab.set(nbr - 1) = 0;
    }

    // ============================================================================
    // Dispatcher Function
    // ============================================================================

    /**
     * @brief Main dispatcher for 1D inverse spectral transformations
     * @param base Basis type identifier (integer constant from base_spectral.hpp)
     * @param tab Array to transform (modified in-place)
     *
     * This function dispatches to the appropriate inverse transformation routine
     * using a static function pointer table initialized once on first call.
     * Inverse transforms convert from spectral coefficients back to physical
     * space values at collocation points.
     *
     * Supported basis type constants:
     * - CHEB, CHEB_EVEN, CHEB_ODD: Chebyshev polynomials
     * - LEG, LEG_EVEN, LEG_ODD: Legendre polynomials
     * - COS, COS_EVEN, COS_ODD: Pure cosine bases
     * - SIN, SIN_EVEN, SIN_ODD: Pure sine bases
     * - COSSIN, COSSIN_EVEN, COSSIN_ODD: Combined cosine-sine bases
     *
     * Mathematical operation: Given spectral coefficients {c_k}, compute
     * f(x_i) = Σ_k c_k φ_k(x_i) where φ_k are the basis functions and x_i
     * are the appropriate collocation points for the chosen basis.
     *
     * @pre base must be a valid basis type constant (0 <= base < NBR_MAX_BASE)
     * @pre tab must contain valid spectral coefficients for the specified basis
     *
     * @post tab contains function values at collocation points
     * @note Thread-safe after first initialization
     * @note Unimplemented basis types call coef_i_1d_pasprevu which aborts
     *
     * @throws Aborts program if basis type is unimplemented
     */
    void coef_i_1d(int base, Array<double>& tab)
    {
        static void (*coef_i_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // First call: initialize function pointer table
        if (premier_appel) {
            premier_appel = false;

            // Default all entries to unimplemented handler
            for (int i = 0; i < NBR_MAX_BASE; i++)
                coef_i_1d[i] = coef_i_1d_pasprevu;

            // Register implemented inverse transformations
            coef_i_1d[CHEB] = coef_i_1d_cheb;
            coef_i_1d[CHEB_EVEN] = coef_i_1d_cheb_even;
            coef_i_1d[CHEB_ODD] = coef_i_1d_cheb_odd;
            coef_i_1d[COSSIN] = coef_i_1d_cossin;
            coef_i_1d[COS_EVEN] = coef_i_1d_cos_even;
            coef_i_1d[COS_ODD] = coef_i_1d_cos_odd;
            coef_i_1d[SIN_EVEN] = coef_i_1d_sin_even;
            coef_i_1d[SIN_ODD] = coef_i_1d_sin_odd;
            coef_i_1d[COS] = coef_i_1d_cos;
            coef_i_1d[SIN] = coef_i_1d_sin;
            coef_i_1d[LEG] = coef_i_1d_leg;
            coef_i_1d[LEG_EVEN] = coef_i_1d_leg_even;
            coef_i_1d[LEG_ODD] = coef_i_1d_leg_odd;
            coef_i_1d[COSSIN_EVEN] = coef_i_1d_cossin_even;
            coef_i_1d[COSSIN_ODD] = coef_i_1d_cossin_odd;
        }

        // Dispatch to appropriate inverse transformation
        coef_i_1d[base](tab);
    }

} // namespace Kadath