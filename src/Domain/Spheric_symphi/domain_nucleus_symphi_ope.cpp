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
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath
{
    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
    int div_sin_1d(int, Array<double>&);
    int div_cos_1d(int, Array<double>&);
    int div_x_1d(int, Array<double>&);
    int mult_x_1d(int, Array<double>&);
    int div_1mx2_1d(int, Array<double>&);

    Val_domain Domain_nucleus_symphi::mult_cos_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::mult_sin_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::mult_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::mult_sin_theta(const Val_domain& so) const
    {

        so.coef();
        Val_domain res(this);

        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::div_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::div_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::div_x(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::div_1mx2(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_1mx2_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::mult_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_x_1d, 0, *so.cf, res.base));
        *res.cf *= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::div_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        *res.cf /= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_nucleus_symphi::der_r(const Val_domain& so) const
    {
        return (so.der_var(1) / alpha);
    }

    Val_domain Domain_nucleus_symphi::ddp(const Val_domain& so) const
    {
        return (so.der_var(3).der_var(3));
    }

    Val_domain Domain_nucleus_symphi::srdr(const Val_domain& so) const
    {
        return (div_x(so.der_var(1)) / alpha / alpha);
    }

    double integral_1d(int, const Array<double>&);
    double Domain_nucleus_symphi::integ_volume(const Val_domain& so) const
    { // compute int (so*r*r*sin(theta) dr dtheta dphi)

        if (so.check_if_zero())
            return 0;
        else {

            Val_domain integrant(mult_r(mult_r(mult_sin_theta(so))) * alpha);
            integrant.coef();

            double val = 0;
            int basep = (*integrant.get_base().bases_1d[2])(0);
            if (basep == COS_EVEN) {
                // Only k = 0
                assert((*integrant.get_base().bases_1d[1])(0) == SIN_ODD);
                Index pos(nbr_coefs);
                for (int j = 0; j < nbr_coefs(1); j++) {
                    pos.set(1) = j;
                    assert((*integrant.get_base().bases_1d[0])(j, 0) == CHEB_EVEN);

                    Array<double> cf(nbr_coefs(0));
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        pos.set(0) = i;
                        cf.set(i) = integrant.get_coef(pos);
                    }
                    val += 2. / (2. * j + 1) * integral_1d(CHEB_EVEN, cf);
                }
                return (val * 2 * M_PI);
            } else
                return 0;
        }
    }
} // namespace Kadath
