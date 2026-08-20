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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{
    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
    int div_sin_1d(int, Array<double>&);
    int div_x_1d(int, Array<double>&);
    int mult_x_1d(int, Array<double>&);

    Val_domain Domain_polar_periodic_nucleus::mult_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::mult_sin_theta(const Val_domain& so) const
    {

        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::div_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::div_x(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::mult_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_x_1d, 0, *so.cf, res.base));
        *res.cf *= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::div_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        *res.cf /= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::srdr(const Val_domain& so) const
    {
        return (div_x(so.der_var(1)) / alpha / alpha);
    }

    Val_domain Domain_polar_periodic_nucleus::der_r(const Val_domain& so) const
    {
        return (so.der_var(1) / alpha);
    }

    Val_domain Domain_polar_periodic_nucleus::dt(const Val_domain& so) const
    {
        return (so.der_var(2));
    }

    Val_domain Domain_polar_periodic_nucleus::dtime(const Val_domain& so) const
    {
        return (so.der_var(3) * ome);
    }

    Val_domain Domain_polar_periodic_nucleus::mult_cos_time(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_periodic_nucleus::mult_sin_time(const Val_domain& so) const
    {

        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }
} // namespace Kadath
