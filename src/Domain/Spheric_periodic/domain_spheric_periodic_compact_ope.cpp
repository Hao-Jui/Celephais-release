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
#include "For_Kadath/Domain/spheric_periodic.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int div_xm1_1d(int, Array<double>&);
    int mult_xm1_1d(int, Array<double>&);

    Val_domain Domain_spheric_periodic_compact::mult_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_spheric_periodic_compact::div_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_spheric_periodic_compact::mult_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(div_xm1(so));
        res /= alpha;
        return res;
    }

    Val_domain Domain_spheric_periodic_compact::div_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(mult_xm1(so));
        res *= alpha;
        return res;
    }

    Val_domain Domain_spheric_periodic_compact::ddtime(const Val_domain& so) const
    {
        return (so.der_var(2).der_var(2) * ome * ome);
    }

    Val_domain Domain_spheric_periodic_compact::dtime(const Val_domain& so) const
    {
        return (so.der_var(2) * ome);
    }

    Val_domain Domain_spheric_periodic_compact::der_r(const Val_domain& so) const
    {
        return (-alpha * so.der_var(1).mult_xm1().mult_xm1());
    }

    Val_domain Domain_spheric_periodic_compact::laplacian(const Val_domain& so, int m) const
    {
        if (m != 0) {
            KADATH_THROW("Laplacian only defiend for m=0 for Domain_spheric_nucleus");
        }
        Val_domain derr(-alpha * so.der_var(1).mult_xm1().mult_xm1());
        Val_domain dderr(-alpha * derr.der_var(1).mult_xm1().mult_xm1());
        Val_domain res(dderr + div_r(2 * derr));
        return res;
    }
} // namespace Kadath
