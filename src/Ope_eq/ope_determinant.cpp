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

#include <sstream>
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Ope_determinant::Ope_determinant(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_determinant::~Ope_determinant() {}

    Term_eq Ope_determinant::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());

        // Various checks
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_determinant only defined with respect for a tensor");
        }
        if (target.val_t->get_valence() != 2) {
            KADATH_THROW("Ope_determinant only defined with respect to a second order tensor");
        }
        if (target.val_t->get_index_type(0) != target.val_t->get_index_type(1)) {
            KADATH_THROW("Ope_determinant only defined with respect to a tensor which indices are of the same type");
        }

        int ndim = target.val_t->get_space().get_ndim();

        switch (ndim) {
            case 2: {

                Val_domain resval((*target.val_t)(1, 1)(dom) * (*target.val_t)(2, 2)(dom) -
                                  (*target.val_t)(1, 2)(dom) * (*target.val_t)(2, 1)(dom));
                Scalar res(target.val_t->get_space());
                res.set_domain(dom) = resval;

                auto determinant_derivative = [&](const Tensor& derivative) {
                    Val_domain derval((*target.val_t)(1, 1)(dom) * derivative(2, 2)(dom) +
                                      derivative(1, 1)(dom) * (*target.val_t)(2, 2)(dom) -
                                      (*target.val_t)(1, 2)(dom) * derivative(2, 1)(dom) -
                                      derivative(1, 2)(dom) * (*target.val_t)(2, 1)(dom));

                    Scalar der(target.val_t->get_space());
                    der.set_domain(dom) = derval;
                    return der;
                };

                Term_eq result(dom, res);
                if (target.has_der_t(0))
                    result.set_der_t(determinant_derivative(target.get_der_t(0)));
                result.set_derivative_lane_count(target.get_derivative_lane_count());
                for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane)
                    if (target.has_der_t(lane))
                        result.set_der_t(lane, determinant_derivative(target.get_der_t(lane)));
                return result;
            }
            case 3: {

                Val_domain resval((*target.val_t)(1, 1)(dom) * (*target.val_t)(2, 2)(dom) * (*target.val_t)(3, 3)(dom) +
                                  (*target.val_t)(2, 1)(dom) * (*target.val_t)(3, 2)(dom) * (*target.val_t)(1, 3)(dom) +
                                  (*target.val_t)(3, 1)(dom) * (*target.val_t)(1, 2)(dom) * (*target.val_t)(2, 3)(dom) -
                                  (*target.val_t)(1, 3)(dom) * (*target.val_t)(2, 2)(dom) * (*target.val_t)(3, 1)(dom) -
                                  (*target.val_t)(2, 3)(dom) * (*target.val_t)(3, 2)(dom) * (*target.val_t)(1, 1)(dom) -
                                  (*target.val_t)(3, 3)(dom) * (*target.val_t)(1, 2)(dom) * (*target.val_t)(2, 1)(dom));
                Scalar res(target.val_t->get_space());
                res.set_domain(dom) = resval;

                // if (target.val_t->get_basis().get_basis(dom)==SPHERICAL_BASIS) {
                //	res.affect_parameters() ;
                //	res.set_parameters()->set_m_order() = 2 ;
                // }

                auto determinant_derivative = [&](const Tensor& derivative) {
                    Val_domain derval(
                        derivative(1, 1)(dom) * (*target.val_t)(2, 2)(dom) * (*target.val_t)(3, 3)(dom) +
                        (*target.val_t)(1, 1)(dom) * derivative(2, 2)(dom) * (*target.val_t)(3, 3)(dom) +
                        (*target.val_t)(1, 1)(dom) * (*target.val_t)(2, 2)(dom) * derivative(3, 3)(dom) +
                        derivative(2, 1)(dom) * (*target.val_t)(3, 2)(dom) * (*target.val_t)(1, 3)(dom) +
                        (*target.val_t)(2, 1)(dom) * derivative(3, 2)(dom) * (*target.val_t)(1, 3)(dom) +
                        (*target.val_t)(2, 1)(dom) * (*target.val_t)(3, 2)(dom) * derivative(1, 3)(dom) +
                        derivative(3, 1)(dom) * (*target.val_t)(1, 2)(dom) * (*target.val_t)(2, 3)(dom) +
                        (*target.val_t)(3, 1)(dom) * derivative(1, 2)(dom) * (*target.val_t)(2, 3)(dom) +
                        (*target.val_t)(3, 1)(dom) * (*target.val_t)(1, 2)(dom) * derivative(2, 3)(dom) -
                        derivative(1, 3)(dom) * (*target.val_t)(2, 2)(dom) * (*target.val_t)(3, 1)(dom) -
                        (*target.val_t)(1, 3)(dom) * derivative(2, 2)(dom) * (*target.val_t)(3, 1)(dom) -
                        (*target.val_t)(1, 3)(dom) * (*target.val_t)(2, 2)(dom) * derivative(3, 1)(dom) -
                        derivative(2, 3)(dom) * (*target.val_t)(3, 2)(dom) * (*target.val_t)(1, 1)(dom) -
                        (*target.val_t)(2, 3)(dom) * derivative(3, 2)(dom) * (*target.val_t)(1, 1)(dom) -
                        (*target.val_t)(2, 3)(dom) * (*target.val_t)(3, 2)(dom) * derivative(1, 1)(dom) -
                        derivative(3, 3)(dom) * (*target.val_t)(1, 2)(dom) * (*target.val_t)(2, 1)(dom) -
                        (*target.val_t)(3, 3)(dom) * derivative(1, 2)(dom) * (*target.val_t)(2, 1)(dom) -
                        (*target.val_t)(3, 3)(dom) * (*target.val_t)(1, 2)(dom) * derivative(2, 1)(dom));

                    Scalar der(target.val_t->get_space());
                    der.set_domain(dom) = derval;

                    // if (target.val_t->get_basis().get_basis(dom)==SPHERICAL_BASIS) {
                    //		der.affect_parameters() ;
                    //	der.set_parameters()->set_m_order() = 2 ;
                    ///}
                    return der;
                };

                Term_eq result(dom, res);
                if (target.has_der_t(0))
                    result.set_der_t(determinant_derivative(target.get_der_t(0)));
                result.set_derivative_lane_count(target.get_derivative_lane_count());
                for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane)
                    if (target.has_der_t(lane))
                        result.set_der_t(lane, determinant_derivative(target.get_der_t(lane)));
                return result;
            }
            default:
                std::ostringstream oss;
                oss << "Ope_determinant not implemented in " << ndim << " dimensions" << endl;
                KADATH_THROW(oss.str());
        }
    }
} // namespace Kadath
