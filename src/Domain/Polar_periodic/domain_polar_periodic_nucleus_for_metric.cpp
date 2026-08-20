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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{

    Term_eq Domain_polar_periodic_nucleus::partial_spher(const Term_eq& so) const
    {

        int dom = so.get_dom();
        int val_res = so.val_t->get_valence() + 1;
        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        // Tensor for val
        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis, 3);

        // Loop on the components :
        Index pos_auxi(auxi_val);
        Index pos_so(*so.val_t);
        do {
            for (int i = 0; i < val_res - 1; i++)
                pos_so.set(i) = pos_auxi(i + 1);
            switch (pos_auxi(0)) {
                case 0:
                    // d/dr :
                    auxi_val.set(pos_auxi).set_domain(dom) = (*so.val_t)(pos_so)(dom).der_r();
                    break;
                case 1:
                    // 1/r dtheta
                    auxi_val.set(pos_auxi).set_domain(dom) = (*so.val_t)(pos_so)(dom).der_var(2).div_r();
                    break;
                case 2:
                    // 1/r sint d/dphi
                    auxi_val.set(pos_auxi).set_domain(dom) = 0.;
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_partial_spher");
            }
        } while (pos_auxi.inc());

        if (so.der_t == nullptr) {
            return Term_eq(dom, auxi_val);
        } else {
            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis);

            // Loop on the components :
            Index pos_auxi_der(auxi_der);
            do {
                for (int i = 0; i < val_res - 1; i++)
                    pos_so.set(i) = pos_auxi_der(i + 1);
                switch (pos_auxi_der(0)) {
                    case 0:
                        // d/dr :
                        auxi_der.set(pos_auxi_der).set_domain(dom) = (*so.der_t)(pos_so)(dom).der_r();
                        break;
                    case 1:
                        // 1/r dtheta
                        auxi_der.set(pos_auxi_der).set_domain(dom) = (*so.der_t)(pos_so)(dom).der_var(2).div_r();
                        break;
                    case 2:
                        // 1/r sint d/dphi
                        auxi_der.set(pos_auxi_der).set_domain(dom) = 0.;
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_partial_spher");
                }
            } while (pos_auxi_der.inc());

            return Term_eq(dom, auxi_val, auxi_der);
        }
    }

    Term_eq Domain_polar_periodic_nucleus::derive_flat_spher(int type_der, char ind_der, const Term_eq& so,
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

        // Tensor for val
        Base_tensor basis(so.val_t->get_space(), SPHERICAL_BASIS);
        Tensor auxi_val(so.val_t->get_space(), val_res, type_ind, basis, 3);

        if (donames) {
            // Set the names of the indices :
            auxi_val.set_name_affected();
            auxi_val.set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                auxi_val.set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
        }

        // Part derivative
        // Loop on the components :
        Index pos_auxi_bis(auxi_val);
        Index pos_so_bis(*so.val_t);
        do {
            for (int i = 0; i < val_res - 1; i++)
                pos_so_bis.set(i) = pos_auxi_bis(i + 1);
            switch (pos_auxi_bis(0)) {
                case 0:
                    // d/dr :
                    auxi_val.set(pos_auxi_bis).set_domain(num_dom) = (*so.val_t)(pos_so_bis)(num_dom).der_r();
                    break;
                case 1:
                    // 1/r dtheta
                    auxi_val.set(pos_auxi_bis).set_domain(num_dom) =
                        (*so.val_t)(pos_so_bis)(num_dom).der_var(2).div_r();
                    break;
                case 2:
                    // 1/r sint d/dphi
                    auxi_val.set(pos_auxi_bis).set_domain(num_dom) = 0;
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_flat_spher");
            }
        } while (pos_auxi_bis.inc());

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
                                auxi_val.set(pos_auxi).set_domain(num_dom) -= (*so.val_t)(pos_so)(num_dom).div_r();
                                break;
                            case 1:
                                // Dtheta S_theta
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(num_dom) += (*so.val_t)(pos_so)(num_dom).div_r();
                                break;
                            case 2:
                                // Dtheta S_phi
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_flat_spher");
                        }
                        break;
                    case 2:
                        // Dphi
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dphi S_r
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(num_dom) -= (*so.val_t)(pos_so)(num_dom).div_r();
                                break;
                            case 1:
                                // Dphi S_theta
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(num_dom) -=
                                    (*so.val_t)(pos_so)(num_dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            case 2:
                                // Dphi S_phi
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(num_dom) += (*so.val_t)(pos_so)(num_dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(num_dom) +=
                                    (*so.val_t)(pos_so)(num_dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_nucleus::derive_flat_spher");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain_nucleus::derive_flat_spher");
                }
            } while (pos_auxi.inc());
        }

        if (so.der_t == nullptr) {

            // No need for derivative :
            Term_eq auxi(num_dom, auxi_val);
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return Term_eq(num_dom, auxi.val_t->do_summation_one_dom(num_dom));
        } else {

            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.val_t->get_space(), val_res, type_ind, basis);

            if (donames) {
                // Set the names of the indices :
                auxi_der.set_name_affected();
                auxi_der.set_name_ind(0, ind_der);
                for (int i = 1; i < val_res; i++)
                    auxi_der.set_name_ind(i, so.der_t->get_name_ind()[i - 1]);
            }

            // Part derivative
            // Loop on the components :
            Index pos_auxi_der_bis(auxi_der);
            do {
                for (int i = 0; i < val_res - 1; i++)
                    pos_so_bis.set(i) = pos_auxi_der_bis(i + 1);
                switch (pos_auxi_der_bis(0)) {
                    case 0:
                        // d/dr :
                        auxi_der.set(pos_auxi_der_bis).set_domain(num_dom) = (*so.der_t)(pos_so_bis)(num_dom).der_r();
                        break;
                    case 1:
                        // 1/r dtheta
                        auxi_der.set(pos_auxi_der_bis).set_domain(num_dom) =
                            (*so.der_t)(pos_so_bis)(num_dom).der_var(2).div_r();
                        break;
                    case 2:
                        // 1/r sint d/dphi
                        auxi_der.set(pos_auxi_der_bis).set_domain(num_dom) = 0.;
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_flat_spher");
                }
            } while (pos_auxi_der_bis.inc());

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
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -=
                                        (*so.der_t)(pos_so)(num_dom).div_r();
                                    break;
                                case 1:
                                    // Dtheta S_theta
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) +=
                                        (*so.der_t)(pos_so)(num_dom).div_r();
                                    break;
                                case 2:
                                    // Dtheta S_phi
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_nucleus::derive_flat_spher");
                            }
                            break;
                        case 2:
                            // Dphi
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dphi S_r
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -=
                                        (*so.der_t)(pos_so)(num_dom).div_r();
                                    break;
                                case 1:
                                    // Dphi S_theta
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) -=
                                        (*so.der_t)(pos_so)(num_dom).div_r().mult_cos_theta().div_sin_theta();
                                    break;
                                case 2:
                                    // Dphi S_phi
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) +=
                                        (*so.der_t)(pos_so)(num_dom).div_r();
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(num_dom) +=
                                        (*so.der_t)(pos_so)(num_dom).div_r().mult_cos_theta().div_sin_theta();
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_flat_spher");
                            }
                            break;
                        default:
                            KADATH_THROW("Bad indice in Domain_polar_periodic_nucleus::derive_flat_spher");
                    }
                } while (pos_auxi_der.inc());
            }

            // Need for derivative :
            Term_eq auxi(num_dom, auxi_val, auxi_der);
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                // Might be in the case where der_t has been destroyed
                if (auxi.der_t == nullptr)
                    return Term_eq(num_dom, auxi.val_t->do_summation_one_dom(num_dom));
                else
                    return Term_eq(num_dom, auxi.val_t->do_summation_one_dom(num_dom),
                                   auxi.der_t->do_summation_one_dom(num_dom));
        }
    }
} // namespace Kadath
