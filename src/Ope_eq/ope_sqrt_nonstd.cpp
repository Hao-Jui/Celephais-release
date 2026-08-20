/*
    Copyright 2018 Philippe Grandclement

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
    Ope_sqrt_nonstd::Ope_sqrt_nonstd(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_sqrt_nonstd::~Ope_sqrt_nonstd() {}

    Term_eq Ope_sqrt_nonstd::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());
        switch (target.type_data) {
            case TERM_T: {

                // Check it is a scalar
                if (target.val_t->get_valence() != 0) {
                    KADATH_THROW("Ope_sqrt_nonstd only defined with respect for a scalar");
                }

                // To get the base
                Val_domain rho(target.val_t->get_space().get_domain(dom));
                rho = 1;
                rho.std_base();
                rho = target.val_t->get_space().get_domain(dom)->mult_r(
                    target.val_t->get_space().get_domain(dom)->mult_sin_theta(rho));

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
                        resval.set(ind).set_domain(dom).set_base() = rho.get_base();
                    }
                }

                Term_eq res(dom, resval);
                if (target.has_der_t(0)) {
                    Tensor resder(
                        one_domain_storage, dom, target.get_der_t(0), false);
                    for (int i = 0; i < target.der_t->get_n_comp(); i++) {
                        Array<int> ind(target.der_t->indices(i));
                        Val_domain valder((*target.der_t)(ind)(dom));
                        // derivative of sqrt: f' / (2*resval), where resval = sqrt(f)
                        if (valder.check_if_zero())
                            resder.set(ind).set_domain(dom).set_zero();
                        else {
                            resder.set(ind).set_domain(dom) =
                                target.val_t->get_space().get_domain(dom)->div_sin_theta(valder) / 2. /
                                target.val_t->get_space().get_domain(dom)->div_sin_theta(resval(ind)(dom));
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
                                    [&](Val_domain derivative, const Val_domain&) {
                                        if (derivative.check_if_zero()) {
                                            derivative.set_zero();
                                            return derivative;
                                        }
                                        return target.val_t->get_space().get_domain(dom)->div_sin_theta(derivative) /
                                               2. /
                                               target.val_t->get_space().get_domain(dom)->div_sin_theta(resval()(dom));
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
        KADATH_THROW("Warning should not be here in Ope_sqrt_nonstd::action...");
    }
} // namespace Kadath
