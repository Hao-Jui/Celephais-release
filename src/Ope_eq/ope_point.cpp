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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
namespace Kadath
{
    Ope_point::Ope_point(const System_of_eqs* zesys, const Point& MM, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1), num(MM)
    {

        parts[0].reset(target);
        // num = zesys->get_space().get_domain(dom)->absol_to_num(MM) ;
    }

    Ope_point::~Ope_point() {}

    Term_eq Ope_point::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        Term_eq target(parts[0]->action());

        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_point only defined with respect for a tensor");
        }

        if (target.val_t->get_n_comp() != 1) {
            KADATH_THROW("Ope_point only defined with respect to a scalar (yet)");
        }

        // The value
        Array<int> ind(target.val_t->indices(0));
        Val_domain val((*target.val_t)(ind)(dom));

        double resval;
        if (val.check_if_zero())
            resval = 0.;
        else
            resval = val.get_base().summation(num, val.get_coef_ref());
        auto evaluate_derivative = [&](int lane) {
            Val_domain valder(target.get_der_t(lane)(ind)(dom));
            double resder;
            if (valder.check_if_zero())
                resder = 0.;
            else
                resder = valder.get_base().summation(num, valder.get_coef_ref());
            return resder;
        };

        Term_eq res(dom, resval);
        if (target.has_der_t(0))
            res.set_der_d(evaluate_derivative(0));
        res.set_derivative_lane_count(target.get_derivative_lane_count());
        for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane)
            if (target.has_der_t(lane))
                res.set_der_d(lane, evaluate_derivative(lane));
        return res;
    }
} // namespace Kadath
