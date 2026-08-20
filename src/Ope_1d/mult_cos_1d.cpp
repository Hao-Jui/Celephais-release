/*
    Copyright 2017 Philippe Grandclement

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
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{
    // `src` and `dst` are the same line of two distinct arrays: Base_spectral::ope_1d
    // allocates its result before the traversal, so they never overlap. Each kernel
    // writes all nr entries of `dst`; nothing behind it fills a slot it skips.

    int mult_cos_1d_pasprevu(const double*, double*, int, int)
    {
        KADATH_THROW("mult_cos_1d not implemented.");
    }

    int mult_cos_1d_cossin(const double* src, double* dst, int nr, int stride)
    {
        assert(nr % 2 == 0);

        dst[0] = 0.5 * src[2 * stride];
        dst[stride] = 0;
        dst[2 * stride] = src[0] + 0.5 * src[4 * stride];
        dst[3 * stride] = 0.5 * src[5 * stride];
        for (int k = 4; k < nr - 2; k++)
            dst[k * stride] = 0.5 * (src[(2 + k) * stride] + src[(k - 2) * stride]);

        dst[(nr - 2) * stride] = 0.5 * src[(nr - 4) * stride];
        dst[(nr - 1) * stride] = 0.;

        return COSSIN;
    }

    int mult_cos_1d_cos(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = 0.5 * src[stride];
        dst[stride] = src[0] + 0.5 * src[2 * stride];
        for (int k = 2; k < nr - 1; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[(k - 1) * stride] + src[(k + 1) * stride]);
        // dst[(nr-1)*stride] = 0.5*src[(nr-2)*stride] ;
        dst[(nr - 1) * stride] = 0;

        return COS;
    }

    int mult_cos_1d_sin(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = 0.;
        dst[stride] = 0.5 * src[2 * stride];
        for (int k = 2; k < nr - 1; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[(k - 1) * stride] + src[(k + 1) * stride]);
        // dst[(nr-1)*stride] = 0.5*src[(nr-2)*stride] ;
        dst[(nr - 1) * stride] = 0;

        return SIN;
    }

    int mult_cos_1d_cos_even(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = src[0] + 0.5 * src[stride];
        for (int k = 1; k < nr - 1; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[k * stride] + src[(k + 1) * stride]);
        // dst[(nr-1)*stride] = 0.5*src[(nr-1)*stride] ;
        dst[(nr - 1) * stride] = 0.;

        return COS_ODD;
    }

    int mult_cos_1d_cos_odd(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = 0.5 * src[0];
        for (int k = 1; k < nr; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[(k - 1) * stride] + src[k * stride]);

        return COS_EVEN;
    }

    int mult_cos_1d_sin_even(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = 0.5 * src[stride];
        for (int k = 1; k < nr - 1; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[k * stride] + src[(k + 1) * stride]);
        dst[(nr - 1) * stride] = 0.;

        return SIN_ODD;
    }

    int mult_cos_1d_sin_odd(const double* src, double* dst, int nr, int stride)
    {
        dst[0] = 0.;
        for (int k = 1; k < nr; k++)
            // Sines
            dst[k * stride] = 0.5 * (src[(k - 1) * stride] + src[k * stride]);

        return SIN_EVEN;
    }

    int mult_cos_1d(int base, const double* src, double* dst, int nr, int stride)
    {
        static int (*mult_cos_1d[NBR_MAX_BASE])(const double*, double*, int, int);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                mult_cos_1d[i] = mult_cos_1d_pasprevu;

            mult_cos_1d[COSSIN] = mult_cos_1d_cossin;
            mult_cos_1d[COS_EVEN] = mult_cos_1d_cos_even;
            mult_cos_1d[COS_ODD] = mult_cos_1d_cos_odd;
            mult_cos_1d[SIN_EVEN] = mult_cos_1d_sin_even;
            mult_cos_1d[SIN_ODD] = mult_cos_1d_sin_odd;
            mult_cos_1d[COS] = mult_cos_1d_cos;
            mult_cos_1d[SIN] = mult_cos_1d_sin;
        }

        return mult_cos_1d[base](src, dst, nr, stride);
    }
} // namespace Kadath
