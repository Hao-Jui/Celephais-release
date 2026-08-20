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
#include "For_Kadath/Domain/homothetic_polar.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"

namespace Kadath
{
    Term_eq Domain_polar_shell_inner_homothetic::integ_term_eq(const Term_eq& so, int bound) const
    {
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("integ_term_eq only defined with respect for a tensor");
        }

        if (so.get_val_t().get_n_comp() != 1) {
            KADATH_THROW("integ_term_eq only defined with respect to a scalar");
        }

        assert(so.get_dom() == num_dom);

        Term_eq rrso(mult_r_term_eq(mult_r_term_eq(so)));

        double resval = 0;
        {
            Array<int> ind(so.get_val_t().indices(0));
            Val_domain value(mult_sin_theta(rrso.get_val_t()(ind)(num_dom)));
            if (value.check_if_zero()) {
                resval = 0.;
            } else {
                int baset = (*value.get_base().bases_1d[1])(0);
                Index pcf(nbr_coefs);
                switch (baset) {
                    case COS_ODD:
                        break;
                    case SIN_EVEN:
                        break;
                    case COS_EVEN: {
                        resval += M_PI * val_boundary(bound, value, pcf);
                        break;
                    }
                    case SIN_ODD: {
                        for (int j = 0; j < nbr_coefs(1); j++) {
                            pcf.set(1) = j;
                            resval += 2. / (2 * double(j) + 1) * val_boundary(bound, value, pcf);
                        }
                        break;
                    }

                    default:
                        KADATH_THROW("Case not yet implemented in Domain_polar_shell_inner_homothetic::integrale");
                }
            }
            resval *= 2 * M_PI;
        }

        Term_eq result(num_dom, resval);
        result.set_derivative_lane_count(rrso.get_derivative_lane_count());
        for (int lane = 0; lane < rrso.get_derivative_lane_count(); ++lane) {
            if (!rrso.has_der_t(lane))
                continue;
            Array<int> ind(so.get_val_t().indices(0));
            Val_domain valueder(mult_sin_theta(rrso.get_der_t(lane)(ind)(num_dom)));
            double resder = 0;
            if (valueder.check_if_zero()) {
                resder = 0.;
            } else {

                int baset = (*valueder.get_base().bases_1d[1])(0);
                Index pcf(nbr_coefs);
                switch (baset) {
                    case COS_ODD:
                        break;
                    case SIN_EVEN:
                        break;
                    case COS_EVEN: {
                        resder += M_PI * val_boundary(bound, valueder, pcf);
                        break;
                    }
                    case SIN_ODD: {
                        for (int j = 0; j < nbr_coefs(1); j++) {
                            pcf.set(1) = j;
                            resder += 2. / (2 * double(j) + 1) * val_boundary(bound, valueder, pcf);
                        }
                        break;
                    }
                    default:
                        KADATH_THROW("Case not yet implemented in Domain_polar_shell_inner_homothetic::integrale");
                }
                resder *= 2 * M_PI;
            }
            result.set_der_d(lane, resder);
        }
        return result;
    }

} // namespace Kadath
