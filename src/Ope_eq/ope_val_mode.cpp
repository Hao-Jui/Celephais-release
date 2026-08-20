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
#include "ope_tensor_scalar_lane_helpers.hpp"
namespace Kadath
{
    Ope_val_mode::Ope_val_mode(const System_of_eqs* zesys, const Index& pp, double vv, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1), pos_cf(pp), value(vv)
    {
        parts[0].reset(target);
    }

    Ope_val_mode::~Ope_val_mode() {}

    Term_eq Ope_val_mode::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        Term_eq target(parts[0]->action());
        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_int only defined with respect for a tensor");
        }

        if (target.val_t->get_n_comp() != 1) {
            KADATH_THROW("Ope_mode only defined with respect to a scalar (yet)");
        }

        // The value
        Array<int> ind(target.val_t->indices(0));
        Val_domain val((*target.val_t)(ind)(dom));
        double resval;
        if (val.check_if_zero())
            resval = -value;
        else {
            val.coef();
            resval = val.get_coef(pos_cf) - value;
        }
        auto evaluate_derivative = [&](const Tensor& derivative) {
            Val_domain derivative_value(derivative(ind)(dom));
            if (derivative_value.check_if_zero())
                return 0.0;
            derivative_value.coef();
            return derivative_value.get_coef(pos_cf);
        };
        return ope_tensor_scalar_lane_detail::make_double_term_from_tensor_lanes(
            dom, resval, target, evaluate_derivative);
    }
} // namespace Kadath
