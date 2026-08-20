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
 */

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    double integral_1d_pasprevu(const Array<double>&)
    {
        KADATH_THROW("Integral_1d not implemented.");
    }

    double integral_1d_cos_even(const Array<double>& so)
    {
        return M_PI * so(0);
    }

    double integral_1d_sin_odd(const Array<double>& so)
    {
        int nn = so.get_size(0);
        double res = 0.;
        for (int i = 0; i < nn; i++)
            res += so(i) * 2. / (2. * i + 1);
        return res;
    }

    double integral_1d_sin_even(const Array<double>&)
    {
        return 0.;
    }

    double integral_1d_cos_odd(const Array<double>&)
    {
        return 0.;
    }

    double integral_1d_cheb_even(const Array<double>& so)
    {
        int nn = so.get_size(0);
        double res = 0.;
        for (int i = 0; i < nn; i++)
            res += -so(i) / (4. * i * i - 1);
        return res;
    }
    double integral_1d_cheb(const Array<double>& so)
    {
        int nn = so.get_size(0);
        double res = 0.;
        for (int i = 0; i < nn; i += 2)
            res += -2 * so(i) / (i * i - 1.);
        return res;
    }

    double integral_1d(int base, const Array<double>& tab)
    {

        static double (*integral_1d[NBR_MAX_BASE])(const Array<double>&);
        static bool premier_appel = true;
        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                integral_1d[i] = integral_1d_pasprevu;

            integral_1d[COS_EVEN] = integral_1d_cos_even;
            integral_1d[COS_ODD] = integral_1d_cos_odd;
            integral_1d[SIN_ODD] = integral_1d_sin_odd;
            integral_1d[SIN_EVEN] = integral_1d_sin_even;
            integral_1d[CHEB_EVEN] = integral_1d_cheb_even;
            integral_1d[CHEB] = integral_1d_cheb;
        }

        return integral_1d[base](tab);
    }
} // namespace Kadath
