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

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Matrice/matrice.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int mult_x_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("mult_x_1d not implemented.");
    }

    int mult_x_1d_cheb(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        res.set(0) = 0.5 * so(1);
        res.set(1) = so(0) + 0.5 * so(2);
        for (int i = 2; i < nr - 1; i++)
            res.set(i) = 0.5 * (so(i + 1) + so(i - 1));
        res.set(nr - 1) = 0.5 * so(nr - 2);
        so = res;
        return CHEB;
    }

    int mult_x_1d_cheb_even(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);

        res.set(0) = so(0) + 0.5 * so(1);
        for (int i = 1; i < nr - 1; i++)
            res.set(i) = 0.5 * (so(i) + so(i + 1));
        res.set(nr - 1) = 0.;
        so = res;

        return CHEB_ODD;
    }

    int mult_x_1d_cheb_odd(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        res.set(0) = 0.5 * so(0);
        for (int i = 1; i < nr - 1; i++)
            res.set(i) = 0.5 * (so(i) + so(i - 1));
        res.set(nr - 1) = 0.5 * so(nr - 2);

        so = res;
        return CHEB_EVEN;
    }

    int mult_x_1d_leg(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        res.set(0) = 1. / 3. * so(1);
        for (int i = 1; i < nr - 1; i++)
            res.set(i) = (double(i) / double(2 * i - 1) * so(i - 1) + double(i + 1) / double(2 * i + 3) * so(i + 1));
        res.set(nr - 1) = double(nr - 1) / double(2 * nr - 3) * so(nr - 2);
        so = res;
        return LEG;
    }

    int mult_x_1d_leg_odd(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);

        res.set(0) = 1. / 3. * so(0);
        for (int i = 1; i < nr - 1; i++)
            res.set(i) =
                (double(2 * i) / double(4 * i - 1) * so(i - 1) + double(2 * i + 1) / double(4 * i + 3) * so(i));
        res.set(nr - 1) = double(2 * nr - 2) / double(4 * nr - 5) * so(nr - 2);

        so = res;
        return LEG_EVEN;
    }

    int mult_x_1d_leg_even(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        for (int i = 0; i < nr - 1; i++)
            res.set(i) =
                (double(2 * i + 1) / double(4 * i + 1) * so(i) + double(2 * i + 2) / double(4 * i + 5) * so(i + 1));
        res.set(nr - 1) = 0.;

        so = res;
        return LEG_ODD;
    }

    int mult_x_1d(int base, Array<double>& so)
    {
        static int (*mult_x_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                mult_x_1d[i] = mult_x_1d_pasprevu;

            mult_x_1d[CHEB_EVEN] = mult_x_1d_cheb_even;
            mult_x_1d[CHEB] = mult_x_1d_cheb;
            mult_x_1d[CHEB_ODD] = mult_x_1d_cheb_odd;
            mult_x_1d[LEG_EVEN] = mult_x_1d_leg_even;
            mult_x_1d[LEG_ODD] = mult_x_1d_leg_odd;
            mult_x_1d[LEG] = mult_x_1d_leg;
        }

        return mult_x_1d[base](so);
    }
} // namespace Kadath
