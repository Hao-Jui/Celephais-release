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
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int div_cos_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("div_cos_1d not implemented.");
    }

    int div_cos_1d_cos_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        Array<double> res(nr);

        double somme = 0;
        res.set(nr - 1) = 0.;
        int signe = 1;
        for (int k = nr - 2; k >= 0; k--) {
            somme -= 2 * signe * tab(k + 1);
            signe *= -1;
            res.set(k) = (k % 2 == 0) ? somme : -somme;
        }
        tab = res;
        return COS_ODD;
    }

    int div_cos_1d_sin_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        Array<double> res(nr);

        double somme = 0;
        res.set(nr - 1) = 0.;
        int signe = -1;
        for (int k = nr - 2; k > 0; k--) {
            somme -= 2 * signe * tab(k);
            signe *= -1;
            res.set(k) = (k % 2 == 0) ? -somme : +somme;
        }

        tab = res;
        return SIN_EVEN;
    }

    int div_cos_1d_cos_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        Array<double> res(nr);

        double somme = 0;
        res.set(nr - 1) = 0.;
        int signe = -1;
        for (int k = nr - 2; k >= 0; k--) {
            somme += 2 * tab(k) * signe;
            res.set(k) = (k % 2 == 0) ? somme : -somme;
            signe *= -1;
        }
        res.set(0) *= 0.5;
        tab = res;
        return COS_EVEN;
    }

    int div_cos_1d_sin_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        int nr = tab.get_size(0);

        Array<double> res(nr);

        double somme = 0;
        int signe = -1;
        res.set(nr - 1) = 0.;
        for (int k = nr - 2; k >= 0; k--) {
            somme -= 2 * tab(k + 1) * signe;
            signe *= -1;
            res.set(k) = (k % 2 == 0) ? -somme : somme;
        }
        tab = res;
        return SIN_ODD;
    }

    int div_cos_1d(int base, Array<double>& tab)
    {
        static int (*div_cos_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                div_cos_1d[i] = div_cos_1d_pasprevu;

            div_cos_1d[COS_EVEN] = div_cos_1d_cos_even;
            div_cos_1d[SIN_ODD] = div_cos_1d_sin_odd;
            div_cos_1d[COS_ODD] = div_cos_1d_cos_odd;
            div_cos_1d[SIN_EVEN] = div_cos_1d_sin_even;
        }

        return div_cos_1d[base](tab);
    }
} // namespace Kadath
