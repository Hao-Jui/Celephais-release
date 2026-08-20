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
 *   2026-08-04  Kernels read the source line and write the result line in
 *               place at the traversal stride. Every expression is unchanged;
 *               only the addresses its operands live at moved, so the results
 *               are bit-identical to the gathered contiguous form.
 *   2026-08-08  Route selected native N=6..20 CHEB/COSSIN lines through fused
 *               strided packing, transform, and unfolding codelets, retaining
 *               the buffer path for measured-slower N=14/16 COSSIN calls.
 *   2026-08-08  Reject raw transform sizes outside even N=2..32 before cache
 *               allocation and remove unreachable odd-length slot handling.
 *   2026-08-08  Replace the growing transform cache with fixed N-indexed slots
 *               and outline cache misses and invalid-size errors.
 *   2026-08-09  Guard the doubled COSSIN parity terminal pair at odd nbr: the
 *               4-slot packing loop already fills the doubled line there, so
 *               the two terminal stores were out-of-bounds dead writes.
 *   2026-08-09  Route native N=6..16 COS/SIN parity-family inverse lines through
 *               fused strided staging, transform, and unfolding codelets.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/base_r2hc.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#include <array>
#include <memory>

namespace Kadath
{
    // Keep each selected backend's transform state at its size so repeated line
    // transforms need only an indexed load.
    std::array<std::unique_ptr<r2hc_precomp_t>, 33> r2hc_precomp_i_by_size;

    [[gnu::cold, gnu::noinline]] static r2hc_precomp_t& initialize_r2hc_precomp_i(int n)
    {
        if (n < 2 || n > 32 || n % 2 != 0)
            KADATH_THROW("Inverse coefficient transform size must be even and in [2, 32].");
        auto& precomp = r2hc_precomp_i_by_size[static_cast<std::size_t>(n)];
        if (!precomp)
            precomp = std::make_unique<r2hc_precomp_t>(n, r2hc_direction::inverse);
        return *precomp;
    }

    // Get or create a half-complex transform workspace.
    r2hc_precomp_t& coef_i_1d_hc2r(int n)
    {
        const auto index = static_cast<unsigned int>(n);
        if (index - 2u > 30u || (index & 1u) != 0)
            return initialize_r2hc_precomp_i(n);
        auto* const precomp = r2hc_precomp_i_by_size[index].get();
        return precomp ? *precomp : initialize_r2hc_precomp_i(n);
    }

    /**
     * Ships the value the gathering driver would have left in an output slot no
     * kernel statement writes. Mirror of the forward helper of the same name in
     * src/Coef/coef_1d.cpp, where the reasoning is written out.
     */
    inline void copy_unwritten_line_slot(const double* src, double* dst, int k, int nbr_in, int stride)
    {
        dst[k * stride] = (k < nbr_in) ? src[k * stride] : 0.;
    }

    /// Ships a whole line the kernel declines to transform (its short-line guard).
    inline void copy_untransformed_line(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        for (int k = 0; k < nbr_out; k++)
            copy_unwritten_line_slot(src, dst, k, nbr_in, stride);
    }

    // `src` and `dst` are the same line of two distinct arrays, or - through the
    // Array overload of coef_i_1d below - one contiguous line read and written in
    // place. Both work because every kernel here reads all of its input before
    // its first output write.

    void coef_i_1d_pasprevu(const double*, double*, int, int, int)
    {
        KADATH_THROW("Coef_1d not implemented.");
    }

    void coef_i_1d_cheb(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        if (nbr > 3) {

            auto& transform_data = coef_i_1d_hc2r(nbr - 1);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::cheb,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();

            double* cf = coef_mem::get_mem(0, nbr);

            double c1 = src[stride];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = src[i * stride] - c1;
                somme += cf[i];
            }
            double fmoins0 = -(nbr - 1) / 2 * c1 - somme;
            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = -0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = src[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * src[(2 * i) * stride];
            tab_auxi[(nbr - 1) / 2] = src[(nbr - 1) * stride];

            transform_data.execute_hc2r();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
                dst[i * stride] = fp + fm;
                dst[(nbr - i - 1) * stride] = fp - fm;
            }
            dst[0] = tab_auxi[0] + fmoins0;
            dst[(nbr - 1) * stride] = tab_auxi[0] - fmoins0;
            dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_cheb_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        auto& transform_data = coef_i_1d_hc2r(nbr - 1);
        if (transform_data.try_execute_fused_inverse(native_spectral_family::cheb_even,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        double* cf = coef_mem::get_mem(0, nbr);

        double c1 = src[stride];
        double somme = 0;
        cf[1] = 0;
        for (int i = 3; i < nbr; i += 2) {
            cf[i] = src[i * stride] - c1;
            somme += cf[i];
        }
        double fmoins0 = (nbr - 1) / 2 * c1 + somme;
        for (int i = 3; i < nbr; i += 2)
            tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
        tab_auxi[0] = src[0];
        for (int i = 1; i < (nbr - 1) / 2; i++)
            tab_auxi[i] = 0.5 * src[(2 * i) * stride];
        tab_auxi[(nbr - 1) / 2] = src[(nbr - 1) * stride];

        transform_data.execute_hc2r();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
            dst[(nbr - 1 - i) * stride] = fp + fm;
            dst[i * stride] = fp - fm;
        }
        dst[0] = tab_auxi[0] - fmoins0;
        dst[(nbr - 1) * stride] = tab_auxi[0] + fmoins0;
        dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

    }

    void coef_i_1d_cheb_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        auto& transform_data = coef_i_1d_hc2r(nbr - 1);
        if (transform_data.try_execute_fused_inverse(native_spectral_family::cheb_odd,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();
        const double* sin_half_pi = transform_data.sin_pi_i_over_2n.data();

        double* cf = coef_mem::get_mem(0, nbr);
        double* ti = coef_mem::get_mem(1, nbr);

        ti[0] = 0.5 * src[0];
        for (int i = 1; i < nbr - 1; i++)
            ti[i] = 0.5 * (src[i * stride] + src[(i - 1) * stride]);
        ti[nbr - 1] = 0.5 * src[(nbr - 2) * stride];

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

        transform_data.execute_hc2r();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
            dst[(nbr - 1 - i) * stride] = (fp + fm) / sin_half_pi[nbr - 1 - i];
            dst[i * stride] = (fp - fm) / sin_half_pi[i];
        }
        dst[0] = 0;
        dst[(nbr - 1) * stride] = tab_auxi[0] + fmoins0;
        dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2] / transform_data.sin_pi_quarter;

    }

    double coloc_leg(int, int);
    double summation_1d_leg(double xx, const Array<double>&);
    double coloc_leg_parity(int, int);
    double summation_1d_leg_even(double xx, const Array<double>&);
    double summation_1d_leg_odd(double xx, const Array<double>&);

    // The three Legendre kernels evaluate a dense sum over the whole
    // coefficient line through an Array, so they keep a contiguous copy of it.
    // Only the address the copy is filled from, and the one its result is
    // written to, moved.

    void coef_i_1d_leg(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> coefs(nbr);
        for (int i = 0; i < nbr; i++)
            coefs.set(i) = src[i * stride];
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = summation_1d_leg(coloc_leg(i, nbr), coefs);
    }

    void coef_i_1d_leg_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> coefs(nbr);
        for (int i = 0; i < nbr; i++)
            coefs.set(i) = src[i * stride];
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = summation_1d_leg_even(coloc_leg_parity(i, nbr), coefs);
    }

    void coef_i_1d_leg_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> coefs(nbr);
        for (int i = 0; i < nbr; i++)
            coefs.set(i) = src[i * stride];
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = summation_1d_leg_odd(coloc_leg_parity(i, nbr), coefs);
    }

    // The only kernel whose two line extents differ: the phi axis carries
    // nbr_coefs = nbr_points + 2, so the inverse transform shrinks its axis. It
    // reads the whole coefficient line and writes exactly the np collocation
    // samples the driver scatters.
    void coef_i_1d_cossin(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        const int nbr = (nbr_in > nbr_out) ? nbr_in : nbr_out;
        int np = nbr - 2;
        if (np > 1) {

            auto& transform_data = coef_i_1d_hc2r(np);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::cossin,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = transform_data.buffer;

            cf[0] = src[0];
            for (int i = 1; i < np / 2; i++) {
                cf[i] = 0.5 * src[(2 * i) * stride];
                cf[np - i] = -0.5 * src[(2 * i + 1) * stride];
            }
            cf[np / 2] = src[np * stride];
            transform_data.execute_hc2r();

            // np is the collocation count; a destination line that is not
            // shorter than that keeps whatever the line scratch held above it.
            const int written = (np < nbr_out) ? np : nbr_out;
            for (int i = 0; i < written; i++)
                dst[i * stride] = cf[i];
            for (int i = written; i < nbr_out; i++)
                copy_unwritten_line_slot(src, dst, i, nbr_in, stride);
        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_cos(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        auto& transform_data = coef_i_1d_hc2r(nbr - 1);
        if (transform_data.try_execute_fused_inverse(native_spectral_family::cos,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        double* cf = coef_mem::get_mem(0, nbr);

        double c1 = src[stride];
        double somme = 0;
        cf[1] = 0;
        for (int i = 3; i < nbr; i += 2) {
            cf[i] = src[i * stride] - c1;
            somme += cf[i];
        }
        double fmoins0 = (nbr - 1) / 2 * c1 + somme;
        for (int i = 3; i < nbr; i += 2)
            tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
        tab_auxi[0] = src[0];
        for (int i = 1; i < (nbr - 1) / 2; i++)
            tab_auxi[i] = 0.5 * src[(2 * i) * stride];
        tab_auxi[(nbr - 1) / 2] = src[(nbr - 1) * stride];

        transform_data.execute_hc2r();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
            dst[i * stride] = fp + fm;
            dst[(nbr - i - 1) * stride] = fp - fm;
        }
        dst[0] = tab_auxi[0] + fmoins0;
        dst[(nbr - 1) * stride] = tab_auxi[0] - fmoins0;
        dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

    }

    void coef_i_1d_sin(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        auto& transform_data = coef_i_1d_hc2r(nbr - 1);
        if (transform_data.try_execute_fused_inverse(native_spectral_family::sin,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        for (int i = 2; i < nbr - 1; i += 2)
            tab_auxi[nbr - 1 - i / 2] = -0.5 * src[i * stride];
        tab_auxi[0] = 0.5 * src[stride];
        for (int i = 3; i < nbr; i += 2)
            tab_auxi[i / 2] = 0.25 * (src[i * stride] - src[(i - 2) * stride]);
        tab_auxi[(nbr - 1) / 2] = -0.5 * src[(nbr - 2) * stride];

        transform_data.execute_hc2r();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]) / sin_pi[i];
            double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]);
            dst[i * stride] = fp + fm;
            dst[(nbr - i - 1) * stride] = fp - fm;
        }
        dst[0] = 0;
        dst[(nbr - 1) * stride] = -2 * tab_auxi[0];
        dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

    }

    void coef_i_1d_cos_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            auto& transform_data = coef_i_1d_hc2r(nbr - 1);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::cos_even,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = coef_mem::get_mem(0, nbr);
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();

            double c1 = src[stride];
            double somme = 0;
            cf[1] = 0;
            for (int i = 3; i < nbr; i += 2) {
                cf[i] = src[i * stride] - c1;
                somme += cf[i];
            }
            double fmoins0 = (nbr - 1) / 2 * c1 + somme;
            for (int i = 3; i < nbr; i += 2)
                tab_auxi[nbr - 1 - i / 2] = 0.25 * (cf[i] - cf[i - 2]);
            tab_auxi[0] = src[0];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.5 * src[(2 * i) * stride];
            tab_auxi[(nbr - 1) / 2] = src[(nbr - 1) * stride];

            transform_data.execute_hc2r();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
                dst[i * stride] = fp + fm;
                dst[(nbr - i - 1) * stride] = fp - fm;
            }
            dst[0] = tab_auxi[0] + fmoins0;
            dst[(nbr - 1) * stride] = tab_auxi[0] - fmoins0;
            dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_cos_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            auto& transform_data = coef_i_1d_hc2r(nbr - 1);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::cos_odd,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = coef_mem::get_mem(0, nbr);
            double* ti = coef_mem::get_mem(1, nbr);
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();
            const double* sin_half_pi = transform_data.sin_pi_i_over_2n.data();

            ti[0] = 0.5 * src[0];
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (src[i * stride] + src[(i - 1) * stride]);
            ti[nbr - 1] = 0.5 * src[(nbr - 2) * stride];

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

            transform_data.execute_hc2r();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
                dst[i * stride] = (fp + fm) / sin_half_pi[nbr - 1 - i];
                dst[(nbr - i - 1) * stride] = (fp - fm) / sin_half_pi[i];
            }
            dst[0] = tab_auxi[0] + fmoins0;
            dst[(nbr - 1) * stride] = 0;
            dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2] / transform_data.sin_pi_quarter;

        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_sin_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {

            auto& transform_data = coef_i_1d_hc2r(nbr - 1);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::sin_even,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();

            for (int i = 2; i < nbr - 1; i += 2)
                tab_auxi[nbr - 1 - i / 2] = -0.5 * src[i * stride];
            tab_auxi[0] = 0.5 * src[stride];
            for (int i = 1; i < (nbr - 1) / 2; i++)
                tab_auxi[i] = 0.25 * (src[(2 * i + 1) * stride] - src[(2 * i - 1) * stride]);
            tab_auxi[(nbr - 1) / 2] = -0.5 * src[(nbr - 2) * stride];

            transform_data.execute_hc2r();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]) / sin_pi[i];
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]);
                dst[i * stride] = fp + fm;
                dst[(nbr - i - 1) * stride] = fp - fm;
            }
            dst[0] = 0;
            dst[(nbr - 1) * stride] = -2 * tab_auxi[0];
            dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2];

        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_sin_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            auto& transform_data = coef_i_1d_hc2r(nbr - 1);
            if (transform_data.try_execute_fused_inverse(native_spectral_family::sin_odd,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = coef_mem::get_mem(0, nbr);
            double* ti = coef_mem::get_mem(1, nbr);
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();
            const double* sin_half_pi = transform_data.sin_pi_i_over_2n.data();

            ti[0] = 0.5 * src[0];
            for (int i = 1; i < nbr - 1; i++)
                ti[i] = 0.5 * (src[i * stride] - src[(i - 1) * stride]);
            ti[nbr - 1] = -0.5 * src[(nbr - 2) * stride];

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

            transform_data.execute_hc2r();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (tab_auxi[i] + tab_auxi[nbr - 1 - i]);
                double fm = 0.5 * (tab_auxi[i] - tab_auxi[nbr - 1 - i]) / sin_pi[i];
                dst[i * stride] = (fp + fm) / sin_half_pi[i];
                dst[(nbr - i - 1) * stride] = (fp - fm) / sin_half_pi[nbr - 1 - i];
            }
            dst[0] = 0;
            dst[(nbr - 1) * stride] = tab_auxi[0] - fmoins0;
            dst[((nbr - 1) / 2) * stride] = tab_auxi[(nbr - 1) / 2] / transform_data.sin_pi_quarter;

        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_i_1d_cossin_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        // Double-sized array :
        Array<double> tab2(nbr * 2 - 2);
        int conte = 0;
        for (int i = 0; i < nbr - 2; i += 2) {
            tab2.set(conte) = src[i * stride];
            tab2.set(conte + 1) = src[(i + 1) * stride];
            tab2.set(conte + 2) = 0.;
            tab2.set(conte + 3) = 0.;
            conte += 4;
        }
        // At even nbr the loop stops at conte = 2*nbr-4 and the terminal pair
        // fills the last two doubled slots. At odd nbr the loop has already
        // filled the whole doubled line, so the terminal stores would land two
        // slots past it and nothing reads them: skip the dead writes.
        if (nbr % 2 == 0) {
            tab2.set(conte) = src[(nbr - 2) * stride];
            tab2.set(conte + 1) = src[(nbr - 1) * stride];
        }

        // The doubled line is contiguous, so the inner transform runs in place
        // on it at stride 1.
        coef_i_1d_cossin(tab2.get_data(), tab2.get_data(), nbr * 2 - 2, nbr * 2 - 2, 1);
        for (int i = 0; i < nbr - 2; i++)
            dst[i * stride] = tab2(i);
        dst[(nbr - 2) * stride] = 0;
        dst[(nbr - 1) * stride] = 0;
    }

    void coef_i_1d_cossin_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        // Double-sized array :
        Array<double> tab2(nbr * 2 - 2);
        int conte = 0;
        for (int i = 0; i < nbr - 2; i += 2) {
            tab2.set(conte) = 0.;
            tab2.set(conte + 1) = 0.;
            tab2.set(conte + 2) = src[i * stride];
            tab2.set(conte + 3) = src[(i + 1) * stride];
            conte += 4;
        }
        // Same terminal-pair guard as the even wrapper: at odd nbr the loop
        // has already filled the doubled line and the two stores would be
        // out-of-bounds dead writes.
        if (nbr % 2 == 0) {
            tab2.set(conte) = src[(nbr - 2) * stride];
            tab2.set(conte + 1) = src[(nbr - 1) * stride];
        }

        coef_i_1d_cossin(tab2.get_data(), tab2.get_data(), nbr * 2 - 2, nbr * 2 - 2, 1);
        for (int i = 0; i < nbr - 2; i++)
            dst[i * stride] = tab2(i);
        dst[(nbr - 2) * stride] = 0;
        dst[(nbr - 1) * stride] = 0;
    }

    using Coef_i_1d_kernel = void (*)(const double*, double*, int, int, int);

    Coef_i_1d_kernel coef_i_1d_kernel_for(int base)
    {
        static Coef_i_1d_kernel coef_i_1d[NBR_MAX_BASE];
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                coef_i_1d[i] = coef_i_1d_pasprevu;

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

        return coef_i_1d[base];
    }

    /**
     * Transforms one line in place at the traversal stride, or declines.
     *
     * Counterpart of the forward dispatcher in src/Coef/coef_1d.cpp: every
     * kernel but COSSIN is written for a line whose two extents match, and the
     * inverse transform only ever shrinks an axis - the COSSIN phi axis, whose
     * nbr_coefs is nbr_points + 2. Declined lines stay on the driver's gathering
     * path, which materialises the whole line in a contiguous scratch.
     */
    bool coef_i_1d(int base, const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        if (nbr_in != nbr_out && !(base == COSSIN && nbr_out < nbr_in))
            return false;

        coef_i_1d_kernel_for(base)(src, dst, nbr_in, nbr_out, stride);
        return true;
    }

    void coef_i_1d(int base, Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        const int nbr = tab.get_size(0);
        double* line = tab.get_data();

        // One contiguous line, read and written in place: every kernel reads
        // all of its input before its first output write, so src == dst is safe
        // and reproduces what the previous in-place kernels did.
        coef_i_1d_kernel_for(base)(line, line, nbr, nbr, 1);
    }
} // namespace Kadath
