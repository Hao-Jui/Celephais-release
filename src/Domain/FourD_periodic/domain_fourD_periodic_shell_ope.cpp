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

#include "For_Kadath/Domain/fourD_periodic.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
    int div_sin_1d(int, Array<double>&);
    int div_xm1_1d(int, Array<double>&);

    Val_domain Domain_fourD_periodic_shell::mult_cos_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_fourD_periodic_shell::mult_sin_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_fourD_periodic_shell::mult_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_fourD_periodic_shell::mult_sin_theta(const Val_domain& so) const
    {

        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_fourD_periodic_shell::div_sin_theta(const Val_domain& so) const
    {

        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_fourD_periodic_shell::div_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Array<double> Domain_fourD_periodic_shell::integ_outer_boundary(const Val_domain& so) const
    {

        int ntime = get_nbr_coefs()(3);
        Array<double> res(ntime);
        res = 0.;
        so.coef();

        for (int l = 0; l < ntime; l++) {

            int baset((*so.base.bases_1d[1])(0, l));
            if (baset == COS_EVEN) {

                // Loop on theta :
                Index pos(get_nbr_coefs());
                pos.set(2) = 0;
                pos.set(3) = l;
                for (int j(0); j < nbr_coefs(1); ++j) {
                    pos.set(1) = j;
                    double fact_tet(2.0 / (1.0 - 4.0 * j * j));
                    // Loop on r :
                    for (int i(0); i < nbr_coefs(0); ++i) {
                        pos.set(0) = i;
                        res.set(l) += fact_tet * (*so.cf)(pos);
                    }
                }
                res.set(l) *= 2.0 * M_PI;
            }
        }
        return res;
    }

} // namespace Kadath
