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
    int div_xm1_1d_pasprevu(Array<double>&)
    {
        KADATH_THROW("div_xm1_1d not implemented.");
    }

    int div_xm1_1d_cheb(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);
        Array<double> res(nr);
        res = 0;
        for (int i = 0; i < nr - 1; i++)
            for (int j = i + 1; j < nr; j++)
                res.set(i) += 2 * (j - i) * so(j);
        res.set(0) *= 0.5;

        so = res;
        return CHEB;
    }

    double leg(int n, double x);
    int div_xm1_1d_leg(Array<double>& so)
    {
        assert(so.get_ndim() == 1);
        int nr = so.get_size(0);

        Array<double> res(nr);

        static Matrice* ope = nullptr;
        static int nr_done = 0;

        if (nr_done != nr) {
            // Compute the matrix :
            nr_done = nr;
            if (ope != nullptr)
                delete ope;
            ope = new Matrice(nr, nr);
            *ope = 0;
            ope->set(0, 0) = -1;
            ope->set(0, 1) = 1. / 3.;
            for (int i = 1; i < nr - 1; i++) {
                ope->set(i, i - 1) = double(i) / double(2 * i - 1);
                ope->set(i, i) = -1;
                ope->set(i, i + 1) = double(i + 1) / double(2 * i + 3);
            }
            ope->set(nr - 1, nr - 2) = double(nr - 1) / double(2 * nr - 3);
            ope->set(nr - 1, nr - 1) = -1;
            ope->set_band(1, 1);
            ope->set_lu();
        }

        Array<double> copie(so);
        copie.set(0) = 0;
        for (int i = 1; i < nr; i++)
            copie.set(0) -= so(i) * leg(i, 1.);
        res = ope->solve(copie);

        so = res;
        return LEG;
    }

    int div_xm1_1d(int base, Array<double>& so)
    {
        static int (*div_xm1_1d[NBR_MAX_BASE])(Array<double>&);
        static bool premier_appel = true;

        // Premier appel
        if (premier_appel) {
            premier_appel = false;

            for (int i = 0; i < NBR_MAX_BASE; i++)
                div_xm1_1d[i] = div_xm1_1d_pasprevu;

            div_xm1_1d[CHEB] = div_xm1_1d_cheb;
            div_xm1_1d[LEG] = div_xm1_1d_leg;
            ;
        }

        return div_xm1_1d[base](so);
    }
} // namespace Kadath
