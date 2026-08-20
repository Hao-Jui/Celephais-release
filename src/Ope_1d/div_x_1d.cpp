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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Matrice/matrice.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int div_x_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("div_x_1d not implemented.");
    }

    int div_x_1d_cheb_even(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double>& res = ope_1d_line_scratch(nr);

        int sgn = 1;
        res.set(nr - 1) = 0;
        double somme = 2 * sgn * so(nr - 1);
        res.set(nr - 2) = somme;
        for (int i = nr - 3; i >= 0; i--) {
            sgn *= -1;
            somme += 2 * sgn * so(i + 1);
            res.set(i) = somme;
        }
        for (int i = 0; i < nr; i += 2)
            res.set(i) *= -1;

        so = res;
        return CHEB_ODD;
    }

    int div_x_1d_cheb_odd(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double>& res = ope_1d_line_scratch(nr);

        int sgn = 1;
        res.set(nr - 1) = 0;
        double somme = 2 * sgn * so(nr - 2);
        res.set(nr - 2) = somme;
        for (int i = nr - 3; i >= 0; i--) {
            sgn *= -1;
            somme += 2 * sgn * so(i);
            res.set(i) = somme;
        }
        for (int i = 0; i < nr; i += 2)
            res.set(i) *= -1;
        res.set(0) *= 0.5;

        so = res;
        return CHEB_EVEN;
    }

    int div_x_1d_leg_odd(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double>& res = ope_1d_line_scratch(nr);

        double coef;
        for (int i = 0; i < nr; i++) {
            coef = double(4 * i + 1) / double(2 * i + 1);
            res.set(i) = coef * so(i);
            for (int j = i + 1; j < nr; j++) {
                coef *= -double(2 * j) / double(2 * j + 1);
                res.set(i) += coef * so(j);
            }
        }

        so = res;
        return LEG_EVEN;
    }

    int div_x_1d_leg_even(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double>& res = ope_1d_line_scratch(nr);
        res = 0;

        double coef;
        for (int i = 0; i < nr - 1; i++) {
            coef = double(4 * i + 3) / double(2 * i + 2);
            res.set(i) = coef * so(i + 1);
            for (int j = i + 2; j < nr; j++) {
                coef *= -double(2 * j - 1) / double(2 * j);
                res.set(i) += coef * so(j);
            }
        }

        so = res;
        return LEG_ODD;
    }

    int div_x_1d(int base, Array<double>& so)
    {
        static int (*div_x_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                div_x_1d[i] = div_x_1d_pasprevu;

            div_x_1d[CHEB_EVEN] = div_x_1d_cheb_even;
            div_x_1d[CHEB_ODD] = div_x_1d_cheb_odd;
            div_x_1d[LEG_EVEN] = div_x_1d_leg_even;
            div_x_1d[LEG_ODD] = div_x_1d_leg_odd;
        }

        return div_x_1d[base](so);
    }
} // namespace Kadath
