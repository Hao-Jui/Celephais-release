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
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "adapted_spherical_term_eq_lanes.hpp"

#include <algorithm>

namespace Kadath
{
    Term_eq Domain_shell_inner_adapted::derive_flat_spher(int type_der, char ind_der, const Term_eq& so,
                                                          const Metric* manipulator) const
    {

        assert((type_der == COV) || (type_der == CON));
        int val_res = so.val_t->get_valence() + 1;

        bool donames = (ind_der == ' ') ? false : true;
        bool need_sum = false;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        if (donames) {
            if (so.val_t->get_valence() > 0)
                assert(so.val_t->is_name_affected());
            // Need for summation ?
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;
        }

        // Flat grad space
        Term_eq fgrad(flat_grad_spher(so));
        bool doder = (fgrad.der_t == nullptr) ? false : true;

        // Connexions parts
        Base_tensor basis(so.val_t->get_space(), SPHERICAL_BASIS);
        Tensor auxi_val(so.val_t->get_space(), val_res, type_ind, basis);
        auxi_val = 0;

        if (donames) {
            // Set the names of the indices :
            auxi_val.set_name_affected();
            auxi_val.set_name_ind(0, ind_der);
            fgrad.val_t->set_name_affected();
            fgrad.val_t->set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++) {
                auxi_val.set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
                fgrad.val_t->set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
            }
        }

        // Part derivative
        // Loop on the components :
        Index pos_auxi_bis(auxi_val);
        Index pos_so_bis(*so.val_t);

        // Loop indice summation on connection symbols
        for (int ind_sum = 0; ind_sum < val_res - 1; ind_sum++) {

            // Loop on the components :
            Index pos_auxi(auxi_val);
            Index pos_so(*so.val_t);

            do {
                for (int i = 0; i < val_res - 1; i++)
                    pos_so.set(i) = pos_auxi(i + 1);
                // Different cases of the derivative index :
                switch (pos_auxi(0)) {
                    case 0:
                        // Dr nothing
                        break;
                    case 1:
                        // Dtheta
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dtheta S_r
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(num_dom) -= (*so.val_t)(pos_so)(num_dom);
                                break;
                            case 1:
                                // Dtheta S_theta
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(num_dom) += (*so.val_t)(pos_so)(num_dom);
                                break;
                            case 2:
                                // Dtheta S_phi
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                        }
                        break;
                    case 2:
                        // Dphi
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dphi S_r
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(num_dom) -= (*so.val_t)(pos_so)(num_dom);
                                break;
                            case 1:
                                // Dphi S_theta
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(num_dom) -=
                                    (*so.val_t)(pos_so)(num_dom).mult_cos_theta().div_sin_theta();
                                break;
                            case 2:
                                // Dphi S_phi
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(num_dom) += (*so.val_t)(pos_so)(num_dom);
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(num_dom) +=
                                    (*so.val_t)(pos_so)(num_dom).mult_cos_theta().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                }
            } while (pos_auxi.inc());
        }

        if (!doder) {
            // No need for derivative :
            Term_eq occi(num_dom, auxi_val);

            Term_eq auxi(occi / (*rad_term_eq) + fgrad);

            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return adapted_spherical_detail::sum_one_domain_term_eq(num_dom, auxi);
        } else {
            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.val_t->get_space(), val_res, type_ind, basis);
            auxi_der = 0;

            if (donames) {
                // Set the names of the indices :
                auxi_der.set_name_affected();
                auxi_der.set_name_ind(0, ind_der);
                fgrad.der_t->set_name_affected();
                fgrad.der_t->set_name_ind(0, ind_der);
                for (int i = 1; i < val_res; i++) {
                    auxi_der.set_name_ind(i, so.der_t->get_name_ind()[i - 1]);
                    fgrad.der_t->set_name_ind(i, so.der_t->get_name_ind()[i - 1]);
                }
                for (int lane = 1; lane < fgrad.get_derivative_lane_count(); ++lane) {
                    if (!fgrad.has_der_t(lane))
                        continue;
                    Tensor* lane_derivative = fgrad.set_der_t(lane);
                    lane_derivative->set_name_affected();
                    lane_derivative->set_name_ind(0, ind_der);
                    for (int i = 1; i < val_res; i++)
                        lane_derivative->set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
                }
            }

            // Part derivative
            // Loop on the components :
            Index pos_auxi_der_bis(auxi_der);

            // Loop indice summation on connection symbols
            for (int ind_sum = 0; ind_sum < val_res - 1; ind_sum++) {

                // Loop on the components :
                Index pos_auxi_der(auxi_val);
                Index pos_so(*so.val_t);

                do {
                    for (int i = 0; i < val_res - 1; i++)
                        pos_so.set(i) = pos_auxi_der(i + 1);
                    // Different cases of the derivative index :
                    switch (pos_auxi_der(0)) {
                        case 0:
                            // Dr nothing
                            break;
                        case 1:
                            // Dtheta
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dtheta S_r
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -= (*so.der_t)(pos_so)(num_dom);
                                    break;
                                case 1:
                                    // Dtheta S_theta
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) += (*so.der_t)(pos_so)(num_dom);
                                    break;
                                case 2:
                                    // Dtheta S_phi
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                            }
                            break;
                        case 2:
                            // Dphi
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dphi S_r
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -= (*so.der_t)(pos_so)(num_dom);
                                    break;
                                case 1:
                                    // Dphi S_theta
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -=
                                        (*so.der_t)(pos_so)(num_dom).mult_cos_theta().div_sin_theta();
                                    break;
                                case 2:
                                    // Dphi S_phi
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) += (*so.der_t)(pos_so)(num_dom);
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) +=
                                        (*so.der_t)(pos_so)(num_dom).mult_cos_theta().div_sin_theta();
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                            }
                            break;
                        default:
                            KADATH_THROW("Bad indice in Domain_shell_inner_adapted::derive_flat_spher");
                    }
                } while (pos_auxi_der.inc());
            }

            // Need for derivative :
            Term_eq occi(num_dom, auxi_val, auxi_der);
            Term_eq auxi(occi / (*rad_term_eq) + fgrad);

            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);
            if (!need_sum)
                return auxi;
            else
                return adapted_spherical_detail::sum_one_domain_term_eq(num_dom, auxi);
        }
    }

    Term_eq Domain_shell_inner_adapted::derive_flat_cart(int type_der, char ind_der, const Term_eq& so,
                                                         const Metric* manipulator) const
    {

        bool doder = ((so.der_t == nullptr) || (rad_term_eq->der_t == nullptr)) ? false : true;
        if ((type_der == CON) && (manipulator->p_met_con[num_dom]->der_t == nullptr))
            doder = false;

        assert((type_der == COV) || (type_der == CON));
        int val_res = so.val_t->get_valence() + 1;

        bool doname = true;
        if (so.val_t->get_valence() > 0)
            if (!so.val_t->is_name_affected())
                doname = false;

        // Need for summation ?
        bool need_sum = false;
        int contraction_index = -1;
        if (doname)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1]) {
                    need_sum = true;
                    contraction_index = i - 1;
                }

        if (doder && need_sum && contraction_index >= 0 && so.val_t->get_valence() <= 2 &&
            so.val_t->get_index_type(contraction_index) != type_der) {

            Term_eq comp_r(derive_r(so));
            Term_eq theta_derivative(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 2));
            Term_eq phi_derivative(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 3));
            Term_eq comp_t((theta_derivative - (*dt_rad_term_eq) * comp_r) / (*rad_term_eq));
            Term_eq comp_p(
                do_comp_by_comp(((phi_derivative - (*dp_rad_term_eq) * comp_r) / (*rad_term_eq)), &Domain::div_sin_theta));

            auto cartesian_component = [this](const Tensor& radial, const Tensor& theta, const Tensor& phi,
                                              const Index& source_index, int axis) {
                const Val_domain& radial_component = radial(source_index)(num_dom);
                const Val_domain& theta_component = theta(source_index)(num_dom);
                switch (axis) {
                    case 0: {
                        Val_domain xy_radial(mult_sin_theta(radial_component) + mult_cos_theta(theta_component));
                        return mult_cos_phi(xy_radial) - mult_sin_phi(phi(source_index)(num_dom));
                    }
                    case 1: {
                        Val_domain xy_radial(mult_sin_theta(radial_component) + mult_cos_theta(theta_component));
                        return mult_sin_phi(xy_radial) + mult_cos_phi(phi(source_index)(num_dom));
                    }
                    case 2:
                        return mult_cos_theta(radial_component) - mult_sin_theta(theta_component);
                    default:
                        KADATH_THROW("Bad Cartesian axis in Domain_shell_inner_adapted::derive_flat_cart");
                }
            };

            auto build_contracted_value = [&](const Tensor& radial, const Tensor& theta, const Tensor& phi) {
                const int source_valence = so.val_t->get_valence();
                if (source_valence == 1) {
                    Scalar result(so.val_t->get_space());
                    Index source_index(*so.val_t);
                    for (int axis = 0; axis < ndim; axis++) {
                        source_index.set(contraction_index) = axis;
                        if (axis == 0)
                            result.set_domain(num_dom) =
                                cartesian_component(radial, theta, phi, source_index, axis);
                        else
                            result.set_domain(num_dom) +=
                                cartesian_component(radial, theta, phi, source_index, axis);
                    }
                    return Tensor(result);
                }

                Array<int> result_type(source_valence - 1);
                Base_tensor result_basis(so.val_t->get_space());
                result_basis.set_basis(num_dom) = CARTESIAN_BASIS;
                const int free_source_index = (contraction_index == 0) ? 1 : 0;
                result_type.set(0) = so.val_t->get_index_type(free_source_index);
                Tensor result(so.val_t->get_space(), source_valence - 1, result_type, result_basis);
                result.set_name_affected();
                result.set_name_ind(0, so.val_t->get_name_ind()[free_source_index]);

                Index result_index(result);
                Index source_index(*so.val_t);
                do {
                    source_index.set(free_source_index) = result_index(0);
                    for (int axis = 0; axis < ndim; axis++) {
                        source_index.set(contraction_index) = axis;
                        if (axis == 0)
                            result.set(result_index).set_domain(num_dom) =
                                cartesian_component(radial, theta, phi, source_index, axis);
                        else
                            result.set(result_index).set_domain(num_dom) +=
                                cartesian_component(radial, theta, phi, source_index, axis);
                    }
                } while (result_index.inc());

                return result;
            };

            Tensor val_result(build_contracted_value(*comp_r.val_t, *comp_t.val_t, *comp_p.val_t));
            Tensor der_result(build_contracted_value(*comp_r.der_t, *comp_t.der_t, *comp_p.der_t));
            Term_eq result(num_dom, val_result, der_result);
            const int lanes =
                std::max(comp_r.get_derivative_lane_count(),
                         std::max(comp_t.get_derivative_lane_count(), comp_p.get_derivative_lane_count()));
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                if (comp_r.has_der_t(lane) && comp_t.has_der_t(lane) && comp_p.has_der_t(lane))
                    result.set_der_t(lane,
                                     build_contracted_value(comp_r.get_der_t(lane),
                                                            comp_t.get_der_t(lane),
                                                            comp_p.get_der_t(lane)));
            }
            return result;
        }

        Term_eq fgrad(partial_cart(so));
        if (fgrad.der_t == nullptr)
            doder = false;

        // Set the names of the indices :
        if (doname) {
            fgrad.val_t->set_name_affected();
            fgrad.val_t->set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                fgrad.val_t->set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
        }

        if (!doder) {
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(fgrad, 0);

            if (!need_sum)
                return fgrad;
            else
                return adapted_spherical_detail::sum_one_domain_term_eq(num_dom, fgrad);
        } else {
            // Need to compute the derivative :

            fgrad.der_t->set_name_affected();
            fgrad.der_t->set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                fgrad.der_t->set_name_ind(i, so.der_t->get_name_ind()[i - 1]);
            for (int lane = 1; lane < fgrad.get_derivative_lane_count(); ++lane) {
                if (!fgrad.has_der_t(lane))
                    continue;
                Tensor* lane_derivative = fgrad.set_der_t(lane);
                lane_derivative->set_name_affected();
                lane_derivative->set_name_ind(0, ind_der);
                for (int i = 1; i < val_res; i++)
                    lane_derivative->set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
            }

            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(fgrad, 0);

            if (!need_sum)
                return fgrad;
            else
                return adapted_spherical_detail::sum_one_domain_term_eq(num_dom, fgrad);
        }
    }
} // namespace Kadath
