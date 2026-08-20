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
    int div_xp1_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("div_xp1_1d not implemented.");
    }

    int div_xp1_1d_cheb(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        res.set(nr - 1) = 0;
        res.set(nr - 2) = 2 * so(nr - 1);
        for (int i = nr - 3; i > 0; i--)
            res.set(i) = 2 * so(i + 1) - 2 * res(i + 1) - res(i + 2);

        double somme = 0;
        for (int i = 0; i < nr; i++)
            somme += (i % 2 == 0) ? so(i) : -so(i);

        res.set(0) = so(0) - res(1) / 2. - somme;

        so = res;
        return CHEB;
    }

    int div_xp1_1d(int base, Array<double>& so)
    {
        static int (*div_xp1_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                div_xp1_1d[i] = div_xp1_1d_pasprevu;

            div_xp1_1d[CHEB] = div_xp1_1d_cheb;
        }

        return div_xp1_1d[base](so);
    }
} // namespace Kadath
