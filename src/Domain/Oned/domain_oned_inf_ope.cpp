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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    int mult_xm1_1d(int, Array<double>&);
    int div_xm1_1d(int, Array<double>&);

    Val_domain Domain_oned_inf::mult_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_oned_inf::der_partial_var(const Val_domain& so, int which_var) const
    {

        switch (which_var) {
            case 0:
                return -alpha * so.der_var(1).mult_xm1().mult_xm1();
                break;
            default:
                KADATH_THROW("Unknown variable in Domain_oned_inf::der_partial_var");
        }
    }

    Val_domain Domain_oned_inf::div_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_oned_inf::mult_x(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(div_xm1(so));
        res /= alpha;
        return res;
    }

    double Domain_oned_inf::integrale(const Val_domain& so) const
    {
        double res = 0;
        Val_domain integrant(div_xm1(div_xm1(so)));
        integrant.coef();
        int baser = (*integrant.get_base().bases_1d[0])(0);

        switch (baser) {
            case CHEB:
                for (int i = 0; i < nbr_coefs(0); i += 2)
                    res += integrant.get_coef()(i) * (1. / double(i + 1) - 1. / double(i - 1));
                break;
            default:
                cerr << "Case not implemented in Domain_oned_inf::integrale" << endl;
        }
        return -res / alpha;
    }
} // namespace Kadath
