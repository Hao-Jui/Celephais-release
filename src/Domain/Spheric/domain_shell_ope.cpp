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

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
    int div_sin_1d(int, Array<double>&);
    int div_cos_1d(int, Array<double>&);
    int mult_x_1d(int, Array<double>&);
    int div_xp1_1d(int, Array<double>&);
    int div_xm1_1d(int, Array<double>&);
    int mult_xm1_1d(int, Array<double>&);
    int div_1mx2_1d(int, Array<double>&);

    Val_domain Domain_shell::mult_cos_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base.allocate(nbr_coefs);
        *res.base.bases_1d[2] = *so.base.bases_1d[2];

        Index index_t(res.base.bases_1d[1]->get_dimensions());
        // Inversion in theta :
        do {
            switch ((*so.base.bases_1d[1])(index_t)) {
                case COS_EVEN:
                    res.base.bases_1d[1]->set(index_t) = SIN_ODD;
                    break;
                case COS_ODD:
                    res.base.bases_1d[1]->set(index_t) = SIN_EVEN;
                    break;
                case SIN_EVEN:
                    res.base.bases_1d[1]->set(index_t) = COS_ODD;
                    break;
                case SIN_ODD:
                    res.base.bases_1d[1]->set(index_t) = COS_EVEN;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_shell::mult_cos_phi");
            }
        } while (index_t.inc());
        *res.base.bases_1d[0] = *so.base.bases_1d[0];

        res.base.def = true;

        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::mult_sin_phi(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base.allocate(nbr_coefs);
        *res.base.bases_1d[2] = *so.base.bases_1d[2];

        Index index_t(res.base.bases_1d[1]->get_dimensions());
        // Inversion in theta :
        do {
            switch ((*so.base.bases_1d[1])(index_t)) {
                case COS_EVEN:
                    res.base.bases_1d[1]->set(index_t) = SIN_ODD;
                    break;
                case COS_ODD:
                    res.base.bases_1d[1]->set(index_t) = SIN_EVEN;
                    break;
                case SIN_EVEN:
                    res.base.bases_1d[1]->set(index_t) = COS_ODD;
                    break;
                case SIN_ODD:
                    res.base.bases_1d[1]->set(index_t) = COS_EVEN;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_shell::mult_sin_phi");
            }
        } while (index_t.inc());
        *res.base.bases_1d[0] = *so.base.bases_1d[0];

        res.base.def = true;

        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::mult_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::mult_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::mult_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_x_1d, 0, *so.cf, res.base) * alpha + (*so.cf) * beta);
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::mult_1mrsL(const Val_domain& so) const
    {
        if (so.check_if_zero())
            return so;
        Val_domain res(so * (1. - get_radius() / (alpha + beta)));

        res.base = so.base;
        return res;
    }

    Val_domain Domain_shell::div_1mrsL(const Val_domain& so) const
    {
        if (so.check_if_zero())
            return so;
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(-(alpha + beta) / alpha * so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_xp1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_xp1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_1mx2(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_1mx2_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_r(const Val_domain& so) const
    {
        Val_domain res(so / get_radius());
        res.base = so.base;
        return (res);
    }

    Val_domain Domain_shell::der_r(const Val_domain& so) const
    {
        return (so.der_var(1) / alpha);
    }

    Val_domain Domain_shell::ddp(const Val_domain& so) const
    {
        return (so.der_var(3).der_var(3));
    }

    Val_domain Domain_shell::mult_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(mult_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::div_xm1(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);

        res.base = so.base;

        res.cf = new Array<double>(so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_shell::dt(const Val_domain& so) const
    {
        return (so.der_var(2));
    }

    Val_domain Domain_shell::der_partial_var(const Val_domain& so, int which_var) const
    {

        switch (which_var) {
            case 0:
                return so.der_r();
                break;
            case 1:
                return so.der_var(2);
                break;
            case 2:
                return so.der_var(3);
            default:
                KADATH_THROW("Unknown variable in Domain_shell::der_partial_var");
        }
    }

    Val_domain Domain_shell::laplacian2(const Val_domain& so, int m) const
    {
        Val_domain derr(so.der_var(1) / alpha);
        Val_domain dert(so.der_var(2));
        Val_domain res(derr.der_var(1) / alpha + div_r(derr + div_r(dert.der_var(2))));
        if (m != 0)
            res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
        return res;
    }

    double integral_1d(int, const Array<double>&);
    double Domain_shell::integ_volume(const Val_domain& so) const
    {

        if (so.check_if_zero())
            return 0;
        else {

            Val_domain integrant(mult_r(mult_r(mult_sin_theta(so))) * alpha);
            integrant.coef();

            double val = 0;
            // Only k = 0
            [[maybe_unused]] int baset = (*integrant.get_base().bases_1d[1])(0);
            assert(baset == SIN_ODD);
            Index pos(nbr_coefs);
            for (int j = 0; j < nbr_coefs(1); j++) {
                pos.set(1) = j;
                [[maybe_unused]] int baser = (*integrant.get_base().bases_1d[0])(j, 0);
                assert(baser == CHEB);

                Array<double> cf(nbr_coefs(0));
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos.set(0) = i;
                    cf.set(i) = integrant.get_coef(pos);
                }
                val += 2. / (2. * j + 1) * integral_1d(CHEB, cf);
            }

            return val * 2 * M_PI;
        }
    }
} // namespace Kadath
