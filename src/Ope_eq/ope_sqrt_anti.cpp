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
#include "ope_scalar_unary_operator.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Ope_sqrt_anti::Ope_sqrt_anti(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_sqrt_anti::~Ope_sqrt_anti() {}

    Term_eq Ope_sqrt_anti::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());
        switch (target.type_data) {
            case TERM_T: {

                // Check it is a scalar
                if (target.val_t->get_valence() != 0) {
                    KADATH_THROW("Ope_sqrt_anti only defined with respect for a scalar");
                }

                // The value
                Tensor resval(
                    one_domain_storage, dom, *target.val_t, false);

                for (int i = 0; i < target.val_t->get_n_comp(); i++) {
                    Array<int> ind(target.val_t->indices(i));
                    const Val_domain& value((*target.val_t)(ind)(dom));
                    if (value.check_if_zero())
                        resval.set(ind).set_domain(dom).set_zero();
                    else {
                        resval.set(ind).set_domain(dom) = sqrt(value);
                        // Force the base
                        resval.set(ind).set_domain(dom).std_anti_base();
                    }
                }

                Term_eq res(dom, resval);
                if (target.has_der_t(0)) {
                    Tensor resder(
                        one_domain_storage, dom, target.get_der_t(0), false);
                    for (int i = 0; i < target.der_t->get_n_comp(); i++) {
                        Array<int> ind(target.der_t->indices(i));
                        Val_domain valder((*target.der_t)(ind)(dom));
                        const Val_domain& value((*target.val_t)(ind)(dom));
                        if (valder.check_if_zero())
                            resder.set(ind).set_domain(dom).set_zero();
                        else {
                            resder.set(ind).set_domain(dom) = valder / 2. / sqrt(value);
                        }
                    }
                    res.set_der_t(resder);
                }
                res.set_derivative_lane_count(target.get_derivative_lane_count());
                    for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                        if (target.has_der_t(lane)) {
                            res.set_der_t(
                                lane,
                                detail::apply_scalar_tensor_derivative_lane(
                                    dom,
                                    target,
                                    lane,
                                    [](Val_domain derivative, const Val_domain& value) {
                                        if (derivative.check_if_zero()) {
                                            derivative.set_zero();
                                            return derivative;
                                        }
                                        return derivative / 2. / sqrt(value);
                                    }));
                        }
                    }
                    return res;
            } break;
            case TERM_D: {
                Term_eq res(dom, sqrt(*target.val_d));
                if (target.has_der_d(0))
                    res.set_der_d((*target.der_d) / 2. / sqrt(*target.val_d));
                res.set_derivative_lane_count(target.get_derivative_lane_count());
                for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                    if (target.has_der_d(lane))
                        res.set_der_d(lane, target.get_der_d(lane) / 2. / sqrt(target.get_val_d()));
                }
                return res;
            } break;
            default: {
                KADATH_THROW("Unknown storage in Term_eq...");
            }
        }
        KADATH_THROW("Warning should not be here in Ope_sqrt_anti::action...");
    }
} // namespace Kadath
