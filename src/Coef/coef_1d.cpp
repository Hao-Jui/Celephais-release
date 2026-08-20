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
 *               strided fold, transform, and extraction codelets, retaining
 *               the buffer path for measured-slower N=14/16 COSSIN calls.
 *   2026-08-09  Extend forward fusion through COS/SIN and their even/odd
 *               parity families, initially gated at N=6..20.
 *   2026-08-08  Reject raw transform sizes outside even N=2..32 before cache
 *               allocation and remove unreachable odd-length slot handling.
 *   2026-08-08  Replace the growing transform cache with fixed N-indexed slots
 *               and outline cache misses and invalid-size errors.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/base_r2hc.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#include "math.h"
#include <array>
#include <memory>
#include "For_Kadath/Array/memory.hpp"

namespace Kadath
{
    // Keep each selected backend's transform state at its size so repeated line
    // transforms need only an indexed load.
    std::array<std::unique_ptr<r2hc_precomp_t>, 33> r2hc_precomp_by_size;

    [[gnu::cold, gnu::noinline]] static r2hc_precomp_t& initialize_r2hc_precomp(int n)
    {
        if (n < 2 || n > 32 || n % 2 != 0)
            KADATH_THROW("Forward coefficient transform size must be even and in [2, 32].");
        auto& precomp = r2hc_precomp_by_size[static_cast<std::size_t>(n)];
        if (!precomp)
            precomp = std::make_unique<r2hc_precomp_t>(n, r2hc_direction::forward);
        return *precomp;
    }

    // Get or create a half-complex transform workspace.
    r2hc_precomp_t& coef_1d_r2hc(int n)
    {
        const auto index = static_cast<unsigned int>(n);
        if (index - 2u > 30u || (index & 1u) != 0)
            return initialize_r2hc_precomp(n);
        auto* const precomp = r2hc_precomp_by_size[index].get();
        return precomp ? *precomp : initialize_r2hc_precomp(n);
    }

    std::array<std::unique_ptr<double[]>, 2> coef_mem::mem_ptrs;
    std::array<size_t, 2> coef_mem::lengths{};

    void coef_i_1d(int, Array<double>&);

    /**
     * Ships the value the gathering driver would have left in an output slot no
     * kernel statement writes.
     *
     * The gathering driver ran the kernel on a line scratch holding the
     * gathered samples with a zero tail above them (see src/Coef/coef.cpp), then
     * copied the whole scratch into the result, so a slot the kernel skips
     * shipped that scratch value: the source sample below \c nbr_in, zero above
     * it. The strided form writes the destination directly and has to reproduce
     * it. Deliberately reproduces the skipped slots rather than filling them —
     * they are a property of the transforms, not of the addressing.
     */
    inline void copy_unwritten_line_slot(const double* src, double* dst, int k, int nbr_in, int stride)
    {
        dst[k * stride] = (k < nbr_in) ? src[k * stride] : 0.;
    }

    /**
     * Ships a whole line the kernel declines to transform.
     *
     * The short-line guards (\c nbr > 3) return without writing anything, so the
     * gathering form shipped the line scratch exactly as gathered.
     */
    inline void copy_untransformed_line(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        for (int k = 0; k < nbr_out; k++)
            copy_unwritten_line_slot(src, dst, k, nbr_in, stride);
    }

    // `src` and `dst` are the same line of two distinct arrays, or - through the
    // Array overload of coef_1d below - one contiguous line read and written in
    // place. Both work because every kernel here reads all of its input before
    // its first output write.

    void coef_1d_pasprevu(const double*, double*, int, int, int)
    {
        KADATH_THROW("Coef_1d not implemented.");
    }

    void coef_1d_cheb(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nr = nbr_in;

        double fmoins0 = 0.5 * (src[0] - src[(nr - 1) * stride]);

        auto& transform_data = coef_1d_r2hc(nr - 1);
        if (transform_data.try_execute_fused_forward(native_spectral_family::cheb,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (src[i * stride] + src[(nr - 1 - i) * stride]);
            double fms = 0.5 * (src[i * stride] - src[(nr - 1 - i) * stride]) * sin_pi[i];
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (src[0] + src[(nr - 1) * stride]);
        tab_auxi[(nr - 1) / 2] = src[((nr - 1) / 2) * stride];

        transform_data.execute_r2hc();

        // Coefficient pairs :
        dst[0] = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            dst[i * stride] = 2 * tab_auxi[(i / 2)] / (nr - 1);
        dst[(nr - 1) * stride] = tab_auxi[(nr - 1) / 2] / (nr - 1);

        // Coefficients impairs :
        dst[stride] = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            dst[i * stride] = dst[(i - 2) * stride] - 4 * tab_auxi[nr - 1 - (i - 1) / 2] / (nr - 1);
            som += dst[i * stride];
        }

        // 2. Calcul de c_1 :
        double c1 = -(fmoins0 + som) / ((nr - 1) / 2);

        // 3. Coef. c_k avec k impair:
        dst[stride] = c1;
        for (int i = 3; i < nr; i += 2)
            dst[i * stride] += c1;
    }

    void coef_1d_cheb_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nr = nbr_in;

        double fmoins0 = -0.5 * (src[0] - src[(nr - 1) * stride]);

        auto& transform_data = coef_1d_r2hc(nr - 1);
        if (transform_data.try_execute_fused_forward(native_spectral_family::cheb_even,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (src[i * stride] + src[(nr - 1 - i) * stride]);
            double fms = 0.5 * (-src[i * stride] + src[(nr - 1 - i) * stride]) * sin_pi[i];
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (src[0] + src[(nr - 1) * stride]);
        tab_auxi[(nr - 1) / 2] = src[((nr - 1) / 2) * stride];

        transform_data.execute_r2hc();

        // Coefficient pairs :
        dst[0] = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            dst[i * stride] = 2 * tab_auxi[(i / 2)] / (nr - 1);
        dst[(nr - 1) * stride] = tab_auxi[(nr - 1) / 2] / (nr - 1);

        // Coefficients impairs :
        dst[stride] = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            dst[i * stride] = dst[(i - 2) * stride] + 4 * tab_auxi[nr - 1 - i / 2] / (nr - 1);
            som += dst[i * stride];
        }

        // 2. Calcul de c_1 :
        double c1 = (fmoins0 - som) / ((nr - 1) / 2);

        // 3. Coef. c_k avec k impair:
        dst[stride] = c1;
        for (int i = 3; i < nr; i += 2)
            dst[i * stride] += c1;
    }

    void coef_1d_cheb_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nr = nbr_in;

        auto& transform_data = coef_1d_r2hc(nr - 1);
        if (transform_data.try_execute_fused_forward(native_spectral_family::cheb_odd,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* cf = coef_mem::get_mem(0, nr);
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();
        const double* sin_half_pi = transform_data.sin_half_pi_i_over_n.data();

        for (int i = 0; i < nr; i++)
            cf[i] = src[i * stride] * sin_half_pi[i];

        double fmoins0 = -0.5 * (cf[0] - cf[nr - 1]);

        for (int i = 1; i < (nr - 1) / 2; i++) {
            double fp = 0.5 * (cf[i] + cf[nr - 1 - i]);
            double fms = 0.5 * (-cf[i] + cf[nr - 1 - i]) * sin_pi[i];
            tab_auxi[i] = fp + fms;
            tab_auxi[nr - 1 - i] = fp - fms;
        }
        tab_auxi[0] = 0.5 * (cf[0] + cf[nr - 1]);
        tab_auxi[(nr - 1) / 2] = cf[(nr - 1) / 2];

        transform_data.execute_r2hc();

        // Coefficient pairs :
        cf[0] = tab_auxi[0] / (nr - 1);
        for (int i = 2; i < nr - 1; i += 2)
            cf[i] = 2 * tab_auxi[(i / 2)] / (nr - 1);
        cf[nr - 1] = tab_auxi[(nr - 1) / 2] / (nr - 1);

        // Coefficients impairs :
        cf[1] = 0;
        double som = 0;
        for (int i = 3; i < nr; i += 2) {
            cf[i] = cf[i - 2] + 4 * tab_auxi[nr - 1 - i / 2] / (nr - 1);
            som += cf[i];
        }

        // 2. Calcul de c_1 :
        double c1 = (fmoins0 - som) / ((nr - 1) / 2);

        // 3. Coef. c_k avec k impair:
        cf[1] = c1;
        for (int i = 3; i < nr; i += 2)
            cf[i] += c1;

        cf[0] = 2 * cf[0];
        for (int i = 1; i < nr - 1; i++)
            cf[i] = 2 * cf[i] - cf[i - 1];
        cf[nr - 1] = 0;

        for (int i = 0; i < nr; i++)
            dst[i * stride] = cf[i];
    }

    double coloc_leg(int, int);
    double weight_leg(int, int);
    double gamma_leg(int, int);
    double leg(int, double);
    double coloc_leg_parity(int, int);
    double weight_leg_parity(int, int);
    double gamma_leg_even(int, int);
    double gamma_leg_odd(int, int);

    // The three Legendre kernels accumulate into a local line; only the address
    // their operand is loaded from moved, and the reduction is a serial
    // dependent chain either way. The local line stays: it is what lets the
    // in-place (src == dst) call read its input while the sums build.

    void coef_1d_leg(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) += src[j * stride] * leg(i, coloc_leg(j, nbr)) * weight_leg(j, nbr) / gamma_leg(i, nbr);
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = res(i);
    }

    void coef_1d_leg_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) += src[j * stride] * leg(i * 2, coloc_leg_parity(j, nbr)) * weight_leg_parity(j, nbr) /
                              gamma_leg_even(i, nbr);
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = res(i);
    }

    void coef_1d_leg_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        Array<double> res(nbr);
        res = 0;
        for (int i = 0; i < nbr - 1; i++)
            for (int j = 0; j < nbr; j++)
                res.set(i) += src[j * stride] * leg(i * 2 + 1, coloc_leg_parity(j, nbr)) * weight_leg_parity(j, nbr) /
                              gamma_leg_odd(i, nbr);
        for (int i = 0; i < nbr; i++)
            dst[i * stride] = res(i);
    }

    // The only kernel whose two line extents differ: the phi axis carries
    // nbr_coefs = nbr_points + 2, so the forward transform grows its axis. It
    // reads exactly the nbr-2 samples the driver gathered and writes the wider
    // coefficient line.
    void coef_1d_cossin(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        const int nbr = (nbr_in > nbr_out) ? nbr_in : nbr_out;
        if (nbr > 3) {

            auto& transform_data = coef_1d_r2hc(nbr - 2);
            if (transform_data.try_execute_fused_forward(native_spectral_family::cossin,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = transform_data.buffer;

            // The gathering form folded a line scratch whose tail above the
            // gathered count is zero, so a growing transform contributes zeros
            // there. Split the loop instead of branching per element - the
            // second half is empty whenever the extents are equal.
            const int gathered = (nbr - 2 < nbr_in) ? nbr - 2 : nbr_in;
            for (int i = 0; i < gathered; i++)
                cf[i] = src[i * stride];
            for (int i = gathered; i < nbr - 2; i++)
                cf[i] = 0.;
            transform_data.execute_r2hc();

            int index = 0;
            double* pcos = cf;
            double* psin = cf + nbr - 3;

            dst[index * stride] = *pcos / double(nbr - 2);
            index++;
            pcos++;
            dst[index * stride] = 0;
            index++;

            for (int i = 1; i < nbr / 2 - 1; i++) {
                dst[index * stride] = 2. * (*pcos) / double(nbr - 2); // the cosines
                index++;
                pcos++;
                dst[index * stride] = -2. * (*psin) / double(nbr - 2); // the sines
                index++;
                psin--;
            }

            dst[index * stride] = *pcos / double(nbr - 2);
            index++;
            dst[index * stride] = 0;

            // At odd nbr the packing stops one slot short of the line (the
            // cosine/sine pairs cover an even count), so the last slot keeps
            // what the line scratch held there. Empty at even nbr.
            for (int k = index + 1; k < nbr_out; k++)
                copy_unwritten_line_slot(src, dst, k, nbr_in, stride);
        } else {
            // nbr <= 3 is not a length this packing is defined for; the
            // gathering form zeroed slots 1 and 2 of the line scratch and
            // shipped slot 0 untouched. Reproduced here, except that slot 2 of
            // a two-slot line is left alone rather than written past the end.
            copy_unwritten_line_slot(src, dst, 0, nbr_in, stride);
            for (int k = 1; k < nbr_out && k < 3; k++)
                dst[k * stride] = 0;
        }
    }

    void coef_1d_cos(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        // Symetrie taken into account
        double fmoins0 = 0.5 * (src[0] - src[(nbr - 1) * stride]);

        auto& transform_data = coef_1d_r2hc(nbr - 1);
        if (transform_data.try_execute_fused_forward(native_spectral_family::cos,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]);
            double fms = 0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride]) * sin_pi[i];
            tab_auxi[i] = fp + fms;
            tab_auxi[nbr - 1 - i] = fp - fms;
        }

        tab_auxi[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
        tab_auxi[(nbr - 1) / 2] = src[((nbr - 1) / 2) * stride];

        transform_data.execute_r2hc();

        dst[0] = tab_auxi[0] / (nbr - 1);
        for (int i = 2; i < nbr - 1; i += 2)
            dst[i * stride] = 2 * tab_auxi[i / 2] / (nbr - 1);
        dst[(nbr - 1) * stride] = tab_auxi[(nbr - 1) / 2] / (nbr - 1);

        dst[stride] = 0;
        double som = 0;
        for (int i = 3; i < nbr; i += 2) {
            dst[i * stride] = dst[(i - 2) * stride] + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
            som += dst[i * stride];
        }

        // 2. Calcul de c_1 :
        double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

        // 3. Coef. c_k avec k impair:
        dst[stride] = c1;
        for (int i = 3; i < nbr; i += 2)
            dst[i * stride] += c1;
    }

    void coef_1d_sin(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        // Symetrie taken into account
        auto& transform_data = coef_1d_r2hc(nbr - 1);
        if (transform_data.try_execute_fused_forward(native_spectral_family::sin,
                                                     src, dst, nbr_in, nbr_out, stride))
            return;
        double* tab_auxi = transform_data.buffer;
        const double* sin_pi = transform_data.sin_pi_over_n.data();

        for (int i = 1; i < (nbr - 1) / 2; i++) {
            double fp = 0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]) * sin_pi[i];
            double fms = 0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride]);
            tab_auxi[i] = fp + fms;
            tab_auxi[nbr - 1 - i] = fp - fms;
        }

        tab_auxi[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
        tab_auxi[(nbr - 1) / 2] = src[((nbr - 1) / 2) * stride];

        transform_data.execute_r2hc();

        dst[0] = 0;
        for (int i = 2; i < nbr - 1; i += 2)
            dst[i * stride] = -2 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
        dst[(nbr - 1) * stride] = 0;

        dst[stride] = 2 * tab_auxi[0] / (nbr - 1);
        for (int i = 3; i < nbr; i += 2)
            dst[i * stride] = dst[(i - 2) * stride] + 4 * tab_auxi[i / 2] / (nbr - 1);
    }

    void coef_1d_cos_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            // Symetrie taken into account
            double fmoins0 = 0.5 * (src[0] - src[(nbr - 1) * stride]);

            auto& transform_data = coef_1d_r2hc(nbr - 1);
            if (transform_data.try_execute_fused_forward(native_spectral_family::cos_even,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]);
                double fms = 0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride]) * sin_pi[i];
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (src[0] + src[(nbr - 1) * stride]);
            tab_auxi[(nbr - 1) / 2] = src[((nbr - 1) / 2) * stride];

            transform_data.execute_r2hc();

            dst[0] = tab_auxi[0] / (nbr - 1);
            for (int i = 2; i < nbr - 1; i += 2)
                dst[i * stride] = 2 * tab_auxi[i / 2] / (nbr - 1);
            dst[(nbr - 1) * stride] = tab_auxi[(nbr - 1) / 2] / (nbr - 1);

            dst[stride] = 0;
            double som = 0;
            for (int i = 3; i < nbr; i += 2) {
                dst[i * stride] = dst[(i - 2) * stride] + 4 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
                som += dst[i * stride];
            }

            // 2. Calcul de c_1 :
            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            // 3. Coef. c_k avec k impair:
            dst[stride] = c1;
            for (int i = 3; i < nbr; i += 2)
                dst[i * stride] += c1;
        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_1d_cos_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            // Symetrie taken into account
            auto& transform_data = coef_1d_r2hc(nbr - 1);
            if (transform_data.try_execute_fused_forward(native_spectral_family::cos_odd,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = coef_mem::get_mem(0, nbr);
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();
            const double* sin_half_pi = transform_data.sin_pi_i_over_2n.data();

            for (int i = 0; i < nbr - 1; i++)
                cf[i] = src[i * stride] * sin_half_pi[nbr - 1 - i];
            cf[nbr - 1] = 0;
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin_pi[i];
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            tab_auxi[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            transform_data.execute_r2hc();

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

            // 2. Calcul de c_1 :
            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            // 3. Coef. c_k avec k impair:
            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] - cf[i - 1];
            cf[nbr - 1] = 0;
            for (int i = 0; i < nbr; i++)
                dst[i * stride] = cf[i];
        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_1d_sin_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            // Symetrie taken into account
            auto& transform_data = coef_1d_r2hc(nbr - 1);
            if (transform_data.try_execute_fused_forward(native_spectral_family::sin_even,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (src[i * stride] + src[(nbr - 1 - i) * stride]) * sin_pi[i];
                double fms = 0.5 * (src[i * stride] - src[(nbr - 1 - i) * stride]);
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (src[0] - src[(nbr - 1) * stride]);
            tab_auxi[(nbr - 1) / 2] = src[((nbr - 1) / 2) * stride];

            transform_data.execute_r2hc();

            dst[0] = 0.;
            for (int i = 2; i < nbr - 1; i += 2)
                dst[i * stride] = -2 * tab_auxi[nbr - 1 - i / 2] / (nbr - 1);
            dst[(nbr - 1) * stride] = 0;

            dst[stride] = 2 * tab_auxi[0] / (nbr - 1);
            for (int i = 3; i < nbr; i += 2)
                dst[i * stride] = dst[(i - 2) * stride] + 4 * tab_auxi[i / 2] / (nbr - 1);
        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_1d_sin_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;
        if (nbr > 3) {
            // Symetrie taken into account
            auto& transform_data = coef_1d_r2hc(nbr - 1);
            if (transform_data.try_execute_fused_forward(native_spectral_family::sin_odd,
                                                         src, dst, nbr_in, nbr_out, stride))
                return;
            double* cf = coef_mem::get_mem(0, nbr);
            cf[0] = 0;
            double* tab_auxi = transform_data.buffer;
            const double* sin_pi = transform_data.sin_pi_over_n.data();
            const double* sin_half_pi = transform_data.sin_pi_i_over_2n.data();

            for (int i = 1; i < nbr; i++)
                cf[i] = src[i * stride] * sin_half_pi[i];
            double fmoins0 = 0.5 * (cf[0] - cf[nbr - 1]);

            for (int i = 1; i < (nbr - 1) / 2; i++) {
                double fp = 0.5 * (cf[i] + cf[nbr - 1 - i]);
                double fms = 0.5 * (cf[i] - cf[nbr - 1 - i]) * sin_pi[i];
                tab_auxi[i] = fp + fms;
                tab_auxi[nbr - 1 - i] = fp - fms;
            }

            tab_auxi[0] = 0.5 * (cf[0] + cf[nbr - 1]);
            tab_auxi[(nbr - 1) / 2] = cf[(nbr - 1) / 2];

            transform_data.execute_r2hc();

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

            // 2. Calcul de c_1 :
            double c1 = (fmoins0 - som) / ((nbr - 1) / 2);

            // 3. Coef. c_k avec k impair:
            cf[1] = c1;
            for (int i = 3; i < nbr; i += 2)
                cf[i] += c1;

            cf[0] = 2 * cf[0];
            for (int i = 1; i < nbr - 1; i++)
                cf[i] = 2 * cf[i] + cf[i - 1];
            cf[nbr - 1] = 0;
            for (int i = 0; i < nbr; i++)
                dst[i * stride] = cf[i];
        } else
            copy_untransformed_line(src, dst, nbr_in, nbr_out, stride);
    }

    void coef_1d_cossin_even(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        // Copy values in double-size array :
        Array<double> tab2(2 * nbr - 2);
        for (int i = 0; i < nbr - 2; i++) {
            tab2.set(i) = src[i * stride];
            tab2.set(i + nbr - 2) = src[i * stride];
        }

        // The doubled line is contiguous, so the inner transform runs in place
        // on it at stride 1.
        coef_1d_cossin(tab2.get_data(), tab2.get_data(), 2 * nbr - 2, 2 * nbr - 2, 1);

        int conte = 0;
        int k = 0;
        for (; k < nbr - 1; k += 2) {
            dst[k * stride] = tab2(conte);
            dst[(k + 1) * stride] = tab2(conte + 1);
            conte += 4;
        }
        // At odd nbr the pairs stop one slot short of the line; that slot keeps
        // what the line scratch held there. Empty at even nbr.
        for (; k < nbr_out; k++)
            copy_unwritten_line_slot(src, dst, k, nbr_in, stride);
    }

    void coef_1d_cossin_odd(const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        assert(nbr_in == nbr_out);
        int nbr = nbr_in;

        // Copy values in double-size array :
        Array<double> tab2(2 * nbr - 2);
        tab2 = 0;
        for (int i = 0; i < nbr - 2; i++) {
            tab2.set(i) = src[i * stride];
            tab2.set(i + nbr - 2) = -src[i * stride];
        }
        coef_1d_cossin(tab2.get_data(), tab2.get_data(), 2 * nbr - 2, 2 * nbr - 2, 1);

        int conte = 2;
        int k = 0;
        for (; k < nbr - 3; k += 2) {
            dst[k * stride] = tab2(conte);
            dst[(k + 1) * stride] = tab2(conte + 1);
            conte += 4;
        }
        // Same one-slot shortfall at odd nbr, here below the two zeroed
        // endpoints rather than at the end of the line.
        for (; k < nbr - 2 && k < nbr_out; k++)
            copy_unwritten_line_slot(src, dst, k, nbr_in, stride);
        dst[(nbr - 2) * stride] = 0.;
        dst[(nbr - 1) * stride] = 0.;
    }

    using Coef_1d_kernel = void (*)(const double*, double*, int, int, int);

    Coef_1d_kernel coef_1d_kernel_for(int base)
    {
        static Coef_1d_kernel coef_1d[NBR_MAX_BASE];
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                coef_1d[i] = coef_1d_pasprevu;

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

        return coef_1d[base];
    }

    /**
     * Transforms one line in place at the traversal stride, or declines.
     *
     * Every kernel but COSSIN is written for a line whose two extents match: a
     * growing line would read past the gathered source, a shrinking one write
     * past the destination. Those combinations are declined and the driver keeps
     * them on its gathering path, which materialises the whole line - gathered
     * samples plus the zero tail above them - in a contiguous scratch. The
     * forward transform only ever grows an axis, and only the COSSIN phi axis
     * (nbr_coefs = nbr_points + 2), whose kernel is written for it.
     */
    bool coef_1d(int base, const double* src, double* dst, int nbr_in, int nbr_out, int stride)
    {
        if (nbr_in != nbr_out && !(base == COSSIN && nbr_out > nbr_in))
            return false;

        coef_1d_kernel_for(base)(src, dst, nbr_in, nbr_out, stride);
        return true;
    }

    void coef_1d(int base, Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        const int nbr = tab.get_size(0);
        double* line = tab.get_data();

        // One contiguous line, read and written in place: every kernel reads
        // all of its input before its first output write, so src == dst is safe
        // and reproduces what the previous in-place kernels did.
        coef_1d_kernel_for(base)(line, line, nbr, nbr, 1);
    }
} // namespace Kadath
