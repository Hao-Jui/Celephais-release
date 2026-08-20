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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{

    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
    int div_sin_1d(int, Array<double>&);
    int div_x_1d(int, Array<double>&);
    int mult_x_1d(int, Array<double>&);

    Val_domain Domain_polar_nucleus::mult_cos_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::mult_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::div_sin_theta(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::div_x(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::mult_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(mult_x_1d, 0, *so.cf, res.base));
        *res.cf *= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::div_r(const Val_domain& so) const
    {
        so.coef();
        Val_domain res(this);
        res.base = so.base;
        res.cf = new Array<double>(so.base.ope_1d(div_x_1d, 0, *so.cf, res.base));
        *res.cf /= alpha;
        res.in_coef = true;
        return res;
    }

    Val_domain Domain_polar_nucleus::srdr(const Val_domain& so) const
    {
        return (div_x(so.der_var(1)) / alpha / alpha);
    }

    Val_domain Domain_polar_nucleus::laplacian2(const Val_domain& so, int m) const
    {
        Val_domain derr(so.der_var(1) / alpha);
        Val_domain dert(so.der_var(2));
        Val_domain res(derr.der_var(1) / alpha + div_r(derr + div_r(dert.der_var(2))));
        if (m != 0) {
            res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
        }
        return res;
    }

    Val_domain Domain_polar_nucleus::laplacian(const Val_domain& so, int m) const
    {
        Val_domain derr(so.der_var(1) / alpha);
        Val_domain dert(so.der_var(2));
        Val_domain res(derr.der_var(1) / alpha +
                       div_r(2 * derr + div_r(dert.der_var(2) + dert.mult_cos_theta().div_sin_theta())));
        if (m != 0) {
            res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta()));
        }
        return res;
    }

    Val_domain Domain_polar_nucleus::der_r(const Val_domain& so) const
    {
        return (so.der_var(1) / alpha);
    }

    Val_domain Domain_polar_nucleus::dt(const Val_domain& so) const
    {
        return (so.der_var(2));
    }

    double Domain_polar_nucleus::integrale(const Val_domain& so) const
    {
        double res = 0.0;
        Val_domain integrant(mult_r(mult_r(mult_sin_theta(so))) * alpha);
        const Array<double>& cf = integrant.get_coef_ref();

        int baset = (*integrant.get_base().bases_1d[1])(0);

        switch (baset) {
            case COS_ODD:
                break;

            case SIN_EVEN:
                break;

            case COS_EVEN: {
                double facttheta = M_PI;
                int baser = (*integrant.get_base().bases_1d[0])(0);

                switch (baser) {
                    case CHEB_EVEN:
                        for (int i = 0; i < nbr_coefs(0); i++) {
                            res += -facttheta / (4 * i * i - 1) * cf(i, 0);
                        }
                        break;

                    case CHEB_ODD:
                        for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                            res += (i % 2 == 0) ? facttheta / double(2 * i + 2) * cf(i, 0)
                                                : -facttheta / double(2 * i) * cf(i, 0);
                        }
                        break;

                    case LEG_EVEN:
                        res += facttheta * cf(0, 0);
                        break;

                    case LEG_ODD: {
                        double pm1 = 1.0;
                        double pp1 = -0.5;
                        for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                            res += facttheta * (pm1 - pp1) / double(4 * i + 3);
                            pm1 = pp1;
                            pp1 *= -double(2 * i + 3) / double(2 * i + 4);
                        }
                        break;
                    }

                    default:
                        KADATH_THROW("Case not yet implemented in Domain_polar_nucleus::integrale");
                }
                break;
            }

            case SIN_ODD: {
                for (int j = 0; j < nbr_coefs(1); j++) {
                    double facttheta = 2.0 / double(2 * j + 1);
                    int baser = (*integrant.get_base().bases_1d[0])(0);

                    switch (baser) {
                        case CHEB_EVEN:
                            for (int i = 0; i < nbr_coefs(0); i++) {
                                res += -facttheta / double(4 * i * i - 1) * cf(i, j);
                            }
                            break;

                        case CHEB_ODD: {
                            res += facttheta / 2.0 * cf(0, j);
                            int signe = -1;
                            for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                                res += facttheta * (-1.0 + signe * double(2 * i + 1)) /
                                       ((2 * i + 1) * (2 * i + 1) - 1) * cf(i, j);
                                signe *= -1;
                            }
                            break;
                        }

                        case LEG_EVEN:
                            res += facttheta * cf(0, j);
                            break;

                        case LEG_ODD: {
                            double pm1 = 1.0;
                            double pp1 = -0.5;
                            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                                res += facttheta * (pm1 - pp1) / double(4 * i + 3) * cf(i, j);
                                pm1 = pp1;
                                pp1 *= -double(2 * i + 3) / double(2 * i + 4);
                            }
                            break;
                        }

                        default:
                            KADATH_THROW("Case not yet implemented in Domain_polar_nucleus::integrale");
                    }
                }
                break;
            }

            default:
                KADATH_THROW("Case not yet implemented in Domain_polar_nucleus::integrale");
        }

        return res;
    }

    double Domain_polar_nucleus::integ_volume(const Val_domain& so) const
    {
        return integrale(so);
    }

} // namespace Kadath
