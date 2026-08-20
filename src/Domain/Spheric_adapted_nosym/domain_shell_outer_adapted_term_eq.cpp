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
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "adapted_nosym_spherical_term_eq_lanes.hpp"

#include <utility>

namespace Kadath
{
    Term_eq Domain_shell_outer_adapted_nosym::dr_term_eq(const Term_eq& so) const
    {
        return derive_r(so);
    }

    Term_eq Domain_shell_outer_adapted_nosym::derive_r(const Term_eq& so) const
    {

        assert(so.get_dom() == num_dom);
        return adapted_spherical_detail::radial_derivative_term_eq(num_dom, so, *der_rad_term_eq);
    }

    Term_eq Domain_shell_outer_adapted_nosym::derive_t(const Term_eq& so) const
    {

        assert(so.get_dom() == num_dom);
        Term_eq dtprime(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 2));
        Term_eq res(dtprime - (*dt_rad_term_eq) * derive_r(so));
        return res;
    }

    Term_eq Domain_shell_outer_adapted_nosym::derive_p(const Term_eq& so) const
    {

        Term_eq dpprime(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 3));
        Term_eq res(dpprime - (*dp_rad_term_eq) * derive_r(so));
        return res;
    }

    Term_eq Domain_shell_outer_adapted_nosym::flat_grad_spher(const Term_eq& so) const
    {

        assert(so.get_dom() == num_dom);

        int valso = so.get_val_t().get_valence();
        Array<int> type_ind(valso + 1);
        type_ind.set(0) = COV;
        for (int i = 0; i < valso; i++)
            type_ind.set(i + 1) = so.get_val_t().get_index_type(i);

        Base_tensor basis(sp);
        basis.set_basis(num_dom) = SPHERICAL_BASIS;
        Tensor res(sp, valso + 1, type_ind, basis);

        Term_eq comp_r(derive_r(so));
        Term_eq comp_t(derive_t(so) / (*rad_term_eq));
        Term_eq comp_p(do_comp_by_comp(derive_p(so), &Domain::div_sin_theta) / (*rad_term_eq));

        // Loop on cmp :
        for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {

            Array<int> ind(so.get_val_t().indices(nc));
            Array<int> indtarget(valso + 1);
            for (int i = 0; i < valso; i++)
                indtarget.set(i + 1) = ind(i);

            // R comp :
            indtarget.set(0) = 1;
            res.set(indtarget).set_domain(num_dom) = comp_r.get_val_t()(ind)(num_dom);
            // theta comp :
            indtarget.set(0) = 2;
            res.set(indtarget).set_domain(num_dom) = comp_t.get_val_t()(ind)(num_dom);
            // Phi comp :
            indtarget.set(0) = 3;
            res.set(indtarget).set_domain(num_dom) = comp_p.get_val_t()(ind)(num_dom);
        }
        auto build_gradient_derivative = [&](int lane) {
            Tensor derivative(sp, valso + 1, type_ind, basis);
            const Tensor* radial_derivative = comp_r.get_p_der_t(lane);
            const Tensor* theta_derivative = comp_t.get_p_der_t(lane);
            const Tensor* phi_derivative = comp_p.get_p_der_t(lane);
            for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {
                Array<int> ind(so.get_val_t().indices(nc));
                Array<int> indtarget(valso + 1);
                for (int i = 0; i < valso; i++)
                    indtarget.set(i + 1) = ind(i);

                indtarget.set(0) = 1;
                if (radial_derivative != nullptr)
                    derivative.set(indtarget).set_domain(num_dom) = (*radial_derivative)(ind)(num_dom);
                else
                    derivative.set(indtarget).set_domain(num_dom).set_zero();

                indtarget.set(0) = 2;
                if (theta_derivative != nullptr)
                    derivative.set(indtarget).set_domain(num_dom) = (*theta_derivative)(ind)(num_dom);
                else
                    derivative.set(indtarget).set_domain(num_dom).set_zero();

                indtarget.set(0) = 3;
                if (phi_derivative != nullptr)
                    derivative.set(indtarget).set_domain(num_dom) = (*phi_derivative)(ind)(num_dom);
                else
                    derivative.set(indtarget).set_domain(num_dom).set_zero();
            }
            return derivative;
        };

        const bool has_primary_derivative =
            comp_r.get_p_der_t() != nullptr || comp_t.get_p_der_t() != nullptr || comp_p.get_p_der_t() != nullptr;

        if (!has_primary_derivative) {
            return Term_eq(num_dom, res);
        } else {
            Term_eq result(num_dom, res, build_gradient_derivative(0));
            const int lanes = std::max(comp_r.get_derivative_lane_count(),
                                       std::max(comp_t.get_derivative_lane_count(),
                                                comp_p.get_derivative_lane_count()));
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; lane++) {
                if (comp_r.get_p_der_t(lane) != nullptr || comp_t.get_p_der_t(lane) != nullptr ||
                    comp_p.get_p_der_t(lane) != nullptr)
                    result.set_der_t(lane, build_gradient_derivative(lane));
            }
            return result;
        }
    }

    void Domain_shell_outer_adapted_nosym::do_normal_spher() const
    {

        Term_eq grad(flat_grad_spher(*rad_term_eq - *outer_radius_term_eq));

        Scalar val_norme(sp);
        val_norme.set_domain(num_dom) = sqrt((*grad.val_t)(1)(num_dom) * (*grad.val_t)(1)(num_dom) +
                                             (*grad.val_t)(2)(num_dom) * (*grad.val_t)(2)(num_dom) +
                                             (*grad.val_t)(3)(num_dom) * (*grad.val_t)(3)(num_dom));
        auto norm_derivative = [&](int lane) {
            Scalar result(sp);
            const Tensor& derivative = grad.get_der_t(lane);
            result.set_domain(num_dom) =
                (derivative(1)(num_dom) * (*grad.val_t)(1)(num_dom) +
                 derivative(2)(num_dom) * (*grad.val_t)(2)(num_dom) +
                 derivative(3)(num_dom) * (*grad.val_t)(3)(num_dom)) /
                val_norme(num_dom);
            return result;
        };
        Term_eq norme = grad.has_der_t(0)
            ? Term_eq(num_dom, val_norme, norm_derivative(0))
            : Term_eq(num_dom, val_norme);
        norme.set_derivative_lane_count(grad.get_derivative_lane_count());
        for (int lane = 1; lane < grad.get_derivative_lane_count(); ++lane)
            if (grad.has_der_t(lane))
                norme.set_der_t(lane, norm_derivative(lane));
        if (normal_spher == nullptr)
            normal_spher = new Term_eq(grad / norme);
        else
            *normal_spher = Term_eq(grad / norme);
    }

    void Domain_shell_outer_adapted_nosym::do_normal_cart() const
    {

        do_normal_spher();

        Base_tensor basis(sp);
        basis.set_basis(num_dom) = CARTESIAN_BASIS;
        Vector val(sp, CON, basis);

        val.set(1).set_domain(num_dom) = mult_cos_phi(mult_sin_theta((*normal_spher->val_t)(1)(num_dom)) +
                                                      mult_cos_theta((*normal_spher->val_t)(2)(num_dom))) -
                                         mult_sin_phi((*normal_spher->val_t)(3)(num_dom));
        val.set(2).set_domain(num_dom) = mult_sin_phi(mult_sin_theta((*normal_spher->val_t)(1)(num_dom)) +
                                                      mult_cos_theta((*normal_spher->val_t)(2)(num_dom))) +
                                         mult_cos_phi((*normal_spher->val_t)(3)(num_dom));
        val.set(3).set_domain(num_dom) =
            mult_cos_theta((*normal_spher->val_t)(1)(num_dom)) - mult_sin_theta((*normal_spher->val_t)(2)(num_dom));

        auto cartesian_derivative = [&](int lane) {
            Vector der(sp, CON, basis);
            const Tensor& derivative = normal_spher->get_der_t(lane);
            der.set(1).set_domain(num_dom) =
                mult_cos_phi(mult_sin_theta(derivative(1)(num_dom)) +
                             mult_cos_theta(derivative(2)(num_dom))) -
                mult_sin_phi(derivative(3)(num_dom));
            der.set(2).set_domain(num_dom) =
                mult_sin_phi(mult_sin_theta(derivative(1)(num_dom)) +
                             mult_cos_theta(derivative(2)(num_dom))) +
                mult_cos_phi(derivative(3)(num_dom));
            der.set(3).set_domain(num_dom) =
                mult_cos_theta(derivative(1)(num_dom)) - mult_sin_theta(derivative(2)(num_dom));
            return der;
        };
        if (normal_spher->has_der_t(0)) {
            const Vector der(cartesian_derivative(0));
            if (normal_cart == nullptr)
                normal_cart = new Term_eq(num_dom, val, der);
            else
                *normal_cart = Term_eq(num_dom, val, der);
        } else if (normal_cart == nullptr) {
            normal_cart = new Term_eq(num_dom, val);
        } else {
            *normal_cart = Term_eq(num_dom, val);
        }
        normal_cart->set_derivative_lane_count(normal_spher->get_derivative_lane_count());
        for (int lane = 1; lane < normal_spher->get_derivative_lane_count(); ++lane)
            if (normal_spher->has_der_t(lane))
                normal_cart->set_der_t(lane, cartesian_derivative(lane));
    }

    Term_eq Domain_shell_outer_adapted_nosym::der_normal_term_eq(const Term_eq& so, int bound) const
    {

        switch (bound) {
            case OUTER_BC: {
                // Deformed surface
                if (normal_spher == nullptr)
                    do_normal_spher();

                int valso = so.get_val_t().get_valence();

                Term_eq grad(flat_grad_spher(so));

                Tensor res(
                    one_domain_storage, num_dom, so.get_val_t(), false);
                Array<int> indgrad(valso + 1);

                // Loop on cmp :
                for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {

                    Array<int> ind(so.get_val_t().indices(nc));
                    for (int i = 0; i < valso; i++)
                        indgrad.set(i + 1) = ind(i);

                    indgrad.set(0) = 1;
                    res.set(ind).set_domain(num_dom) =
                        (*grad.val_t)(indgrad)(num_dom) * (*normal_spher->val_t)(1)(num_dom);
                    indgrad.set(0) = 2;
                    res.set(ind).set_domain(num_dom) +=
                        (*grad.val_t)(indgrad)(num_dom) * (*normal_spher->val_t)(2)(num_dom);
                    indgrad.set(0) = 3;
                    res.set(ind).set_domain(num_dom) +=
                        (*grad.val_t)(indgrad)(num_dom) * (*normal_spher->val_t)(3)(num_dom);
                }

                auto build_normal_derivative = [&](int lane) {
                    Tensor derivative(
                        one_domain_storage, num_dom, so.get_val_t(), false);
                    const Tensor* gradient_derivative = grad.get_p_der_t(lane);
                    const Tensor* normal_derivative = normal_spher->get_p_der_t(lane);
                    Array<int> indgrad(valso + 1);

                    // Loop on cmp :
                    for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {

                        Array<int> ind(so.get_val_t().indices(nc));
                        for (int i = 0; i < valso; i++)
                            indgrad.set(i + 1) = ind(i);

                        indgrad.set(0) = 1;
                        Val_domain component((*normal_spher->val_t)(1)(num_dom));
                        component.set_zero();
                        if (gradient_derivative != nullptr)
                            component += (*gradient_derivative)(indgrad)(num_dom) * (*normal_spher->val_t)(1)(num_dom);
                        if (normal_derivative != nullptr)
                            component += (*grad.val_t)(indgrad)(num_dom) * (*normal_derivative)(1)(num_dom);

                        indgrad.set(0) = 2;
                        if (gradient_derivative != nullptr)
                            component += (*gradient_derivative)(indgrad)(num_dom) * (*normal_spher->val_t)(2)(num_dom);
                        if (normal_derivative != nullptr)
                            component += (*grad.val_t)(indgrad)(num_dom) * (*normal_derivative)(2)(num_dom);

                        indgrad.set(0) = 3;
                        if (gradient_derivative != nullptr)
                            component += (*gradient_derivative)(indgrad)(num_dom) * (*normal_spher->val_t)(3)(num_dom);
                        if (normal_derivative != nullptr)
                            component += (*grad.val_t)(indgrad)(num_dom) * (*normal_derivative)(3)(num_dom);

                        derivative.set(ind).set_domain(num_dom) = component;
                    }
                    return derivative;
                };

                const bool has_primary_derivative =
                    grad.get_p_der_t() != nullptr || normal_spher->get_p_der_t() != nullptr;
                if (has_primary_derivative) {
                    Term_eq result(num_dom, res, build_normal_derivative(0));
                    const int lanes =
                        std::max(grad.get_derivative_lane_count(), normal_spher->get_derivative_lane_count());
                    result.set_derivative_lane_count(lanes);
                    for (int lane = 1; lane < lanes; lane++) {
                        if (grad.get_p_der_t(lane) != nullptr || normal_spher->get_p_der_t(lane) != nullptr)
                            result.set_der_t(lane, build_normal_derivative(lane));
                    }
                    return result;
                } else
                    return Term_eq(num_dom, res);
            }
            case INNER_BC: {
                return derive_r(so);
            }
            default:
                KADATH_THROW("Unknown boundary in Domain_shell_outer_adapted_nosym::der_normal");
        }
    }

    Term_eq Domain_shell_outer_adapted_nosym::lap_term_eq(const Term_eq& so, int mm) const
    {
        if (mm != 0) {
            KADATH_THROW("Domain_shell_outer_adapted_nosym::lap_term_eq not defined for m != 0 (for now)");
        }

        if (so.get_val_t().get_valence() != 0) {
            KADATH_THROW("Domain_shell_outer_adapted_nosym::lap_term_eq only defined for scalars");
        }

        // Angular part :
        Term_eq dert(derive_t(so));
        Term_eq p1(derive_t(dert));
        Term_eq p2(do_comp_by_comp(do_comp_by_comp(dert, &Domain::mult_cos_theta), &Domain::div_sin_theta));
        Term_eq der2p(derive_p(derive_p(so)));
        Term_eq p3(do_comp_by_comp(do_comp_by_comp(der2p, &Domain::div_sin_theta), &Domain::div_sin_theta));

        Term_eq dr(derive_r(so));

        Term_eq res(derive_r(dr) + 2 * dr / (*rad_term_eq) + (p1 + p2 + p3) / (*rad_term_eq) / (*rad_term_eq));

        return res;
    }

    Term_eq Domain_shell_outer_adapted_nosym::mult_r_term_eq(const Term_eq& so) const
    {

        return so * (*rad_term_eq);
    }

    void Domain_shell_outer_adapted_nosym::update_term_eq(Term_eq* so) const
    {
        update_term_eq_impl(so, false);
    }

    void Domain_shell_outer_adapted_nosym::accumulate_term_eq_mapping_derivative(
        Term_eq* so) const
    {
        update_term_eq_impl(so, true);
    }

    void Domain_shell_outer_adapted_nosym::update_term_eq_impl(
        Term_eq* so, bool accumulate) const
    {
        auto mapping_derivative = [&](int lane) {
            Tensor der(
                one_domain_storage, num_dom, *so->val_t, false);
            const Tensor& radius_derivative = outer_radius_term_eq->get_der_t(lane);
            const Val_domain& radius_lane = (*radius_derivative.cmp[0])(num_dom);
            if (radius_lane.check_if_zero()) {
                for (int cmp = 0; cmp < so->val_t->get_n_comp(); cmp++)
                    der.cmp[cmp]->set_domain(num_dom).set_zero();
                return der;
            }
            for (int cmp = 0; cmp < so->val_t->get_n_comp(); cmp++) {
                Val_domain derr((*(*so->val_t).cmp[cmp])(num_dom).der_var(1) /
                                (*der_rad_term_eq->val_t)()(num_dom));
                if (!derr.check_if_zero()) {
                    Val_domain res(this);
                    res.allocate_conf();
                    Index pos(nbr_points);
                    do {
                        res.set(pos) = radius_lane(pos) / 2. *
                                       (1 + ((*coloc[0])(pos(0)))) * derr(pos);
                    } while (pos.inc());
                    res.set_base() = (*(*so->val_t).cmp[cmp])(num_dom).get_base();
                    der.cmp[cmp]->set_domain(num_dom) = res;
                } else {
                    der.cmp[cmp]->set_domain(num_dom).set_zero();
                }
            }
            return der;
        };
        const int lane_count = outer_radius_term_eq->get_derivative_lane_count();
        so->set_derivative_lane_count(lane_count);
        for (int lane = 0; lane < lane_count; ++lane) {
            if (!outer_radius_term_eq->has_der_t(lane))
                continue;
            Tensor mapping(mapping_derivative(lane));
            // Matrix-free do_JX seeds lane 0. Preserve the established replace
            // semantics for any retained packed lanes, exactly as the former
            // snapshot/add-back path did.
            if (!accumulate || lane != 0) {
                so->set_der_t(lane, mapping);
                continue;
            }

            if (!so->try_accumulate_der_t(lane, std::move(mapping))) {
                if (so->has_der_t(lane)) {
                    so->set_der_t(
                        lane, add_one_dom(num_dom, mapping, so->get_der_t(lane)));
                } else {
                    so->set_der_t(lane, mapping);
                }
            }
        }
    }

    Term_eq Domain_shell_outer_adapted_nosym::partial_spher(const Term_eq& so) const
    {
        int dom = so.get_dom();
        assert(dom == num_dom);

        int valence = so.val_t->get_valence();

        Array<int> type_ind(valence + 1);
        type_ind.set(0) = COV;
        for (int i = 1; i < valence + 1; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        Term_eq comp_r(derive_r(so));
        Term_eq comp_t(derive_t(so) / (*rad_term_eq));
        Term_eq comp_p(do_comp_by_comp((derive_p(so)), &Domain::div_sin_theta) / (*rad_term_eq));

        Tensor val_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
        {
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);
                // R part
                pos_res.set(0) = 0;
                val_res.set(pos_res).set_domain(num_dom) = (*comp_r.val_t)(pos_so)(num_dom);
                // Theta part
                pos_res.set(0) = 1;
                val_res.set(pos_res).set_domain(num_dom) = (*comp_t.val_t)(pos_so)(num_dom);
                // Phi part
                pos_res.set(0) = 2;
                val_res.set(pos_res).set_domain(num_dom) = (*comp_p.val_t)(pos_so)(num_dom);
            } while (pos_so.inc());
        }

        auto build_spherical_partial_derivative = [&](int lane) {
            Tensor derivative(so.get_val_t().get_space(), valence + 1, type_ind, basis);
            const Tensor* radial_derivative = comp_r.get_p_der_t(lane);
            const Tensor* theta_derivative = comp_t.get_p_der_t(lane);
            const Tensor* phi_derivative = comp_p.get_p_der_t(lane);
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);
                // R part
                pos_res.set(0) = 0;
                if (radial_derivative != nullptr)
                    derivative.set(pos_res).set_domain(num_dom) = (*radial_derivative)(pos_so)(num_dom);
                else
                    derivative.set(pos_res).set_domain(num_dom).set_zero();
                // Theta part
                pos_res.set(0) = 1;
                if (theta_derivative != nullptr)
                    derivative.set(pos_res).set_domain(num_dom) = (*theta_derivative)(pos_so)(num_dom);
                else
                    derivative.set(pos_res).set_domain(num_dom).set_zero();
                // Phi part
                pos_res.set(0) = 2;
                if (phi_derivative != nullptr)
                    derivative.set(pos_res).set_domain(num_dom) = (*phi_derivative)(pos_so)(num_dom);
                else
                    derivative.set(pos_res).set_domain(num_dom).set_zero();
            } while (pos_so.inc());
            return derivative;
        };

        const bool has_primary_derivative =
            comp_r.get_p_der_t() != nullptr || comp_t.get_p_der_t() != nullptr || comp_p.get_p_der_t() != nullptr;
        if (has_primary_derivative) {
            Term_eq result(num_dom, val_res, build_spherical_partial_derivative(0));
            const int lanes = std::max(comp_r.get_derivative_lane_count(),
                                       std::max(comp_t.get_derivative_lane_count(),
                                                comp_p.get_derivative_lane_count()));
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; lane++) {
                if (comp_r.get_p_der_t(lane) != nullptr || comp_t.get_p_der_t(lane) != nullptr ||
                    comp_p.get_p_der_t(lane) != nullptr)
                    result.set_der_t(lane, build_spherical_partial_derivative(lane));
            }
            return result;
        } else
            return Term_eq(num_dom, val_res);
    }

    Term_eq Domain_shell_outer_adapted_nosym::connection_spher(const Term_eq& so) const
    {

        int dom = so.get_dom();
        assert(dom == num_dom);

        int valence = so.val_t->get_valence();
        int val_res = so.val_t->get_valence() + 1;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis);
        for (int cmp = 0; cmp < auxi_val.get_n_comp(); cmp++)
            auxi_val.set(auxi_val.indices(cmp)) = 0;

        for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

            // Loop on the components :
            Index pos_auxi(auxi_val);
            Index pos_so(*so.val_t);

            do {
                for (int i = 0; i < valence; i++)
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
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dtheta S_theta
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 2:
                                // Dtheta S_phi
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                        }
                        break;
                    case 2:
                        // Dphi
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dphi S_r
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dphi S_theta
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -=
                                    (*so.val_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            case 2:
                                // Dphi S_phi
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(dom) +=
                                    (*so.val_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                }
            } while (pos_auxi.inc());
        }

        if ((so.der_t == nullptr) || (outer_radius_term_eq->der_t == nullptr)) {
            // No need for derivative :
            return Term_eq(dom, auxi_val);
        } else {

            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis);
            for (int cmp = 0; cmp < auxi_der.get_n_comp(); cmp++)
                auxi_der.set(auxi_der.indices(cmp)) = 0;

            // Loop indice summation on connection symbols
            for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

                // Loop on the components :
                Index pos_auxi_der(auxi_der);
                Index pos_so(*so.der_t);

                do {
                    for (int i = 0; i < valence; i++)
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
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r();
                                    break;
                                case 1:
                                    // Dtheta S_theta
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r();
                                    break;
                                case 2:
                                    // Dtheta S_phi
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                            }
                            break;
                        case 2:
                            // Dphi
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dphi S_r
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r();
                                    break;
                                case 1:
                                    // Dphi S_theta
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r()
                                            .mult_cos_theta()
                                            .div_sin_theta();
                                    break;
                                case 2:
                                    // Dphi S_phi
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r();
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                        ((*so.der_t)(pos_so)(dom) - (*so.val_t)(pos_so)(dom) *
                                                                        (*rad_term_eq->der_t)()(dom) /
                                                                        (*rad_term_eq->val_t)()(dom))
                                            .div_r()
                                            .mult_cos_theta()
                                            .div_sin_theta();
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                            }
                            break;
                        default:
                            KADATH_THROW("Bad indice in Domain_shell_outer_adapted_nosym::connection_spher");
                    }
                } while (pos_auxi_der.inc());
            }

            return Term_eq(dom, auxi_val, auxi_der);
        }
    }

    Term_eq Domain_shell_outer_adapted_nosym::partial_cart(const Term_eq& so) const
    {
        int dom = so.get_dom();
        assert(dom == num_dom);

        int valence = so.val_t->get_valence();

        Array<int> type_ind(valence + 1);
        type_ind.set(0) = COV;
        for (int i = 1; i < valence + 1; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = CARTESIAN_BASIS;

        Term_eq comp_r(derive_r(so));
        Term_eq theta_derivative(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 2));
        Term_eq phi_derivative(adapted_spherical_detail::coordinate_derivative_term_eq(num_dom, so, 3));
        Term_eq comp_t((theta_derivative - (*dt_rad_term_eq) * comp_r) / (*rad_term_eq));
        Term_eq comp_p(
            do_comp_by_comp(((phi_derivative - (*dp_rad_term_eq) * comp_r) / (*rad_term_eq)), &Domain::div_sin_theta));

        Tensor val_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
        {
            const Tensor& comp_r_value = *comp_r.val_t;
            const Tensor& comp_t_value = *comp_t.val_t;
            const Tensor& comp_p_value = *comp_p.val_t;
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);

                const int source_component = comp_r_value.position(pos_so);
                const Val_domain& comp_r_domain = (*comp_r_value.cmp[source_component])(num_dom);
                const Val_domain& comp_t_domain = (*comp_t_value.cmp[source_component])(num_dom);
                const Val_domain& comp_p_domain = (*comp_p_value.cmp[source_component])(num_dom);

                Val_domain auxi(mult_sin_theta(comp_r_domain) + mult_cos_theta(comp_t_domain));
                // X part
                pos_res.set(0) = 0;
                val_res.cmp[val_res.position(pos_res)]->set_domain(num_dom) =
                    mult_cos_phi(auxi) - mult_sin_phi(comp_p_domain);
                // Y part
                pos_res.set(0) = 1;
                val_res.cmp[val_res.position(pos_res)]->set_domain(num_dom) =
                    mult_sin_phi(auxi) + mult_cos_phi(comp_p_domain);
                // Z part
                pos_res.set(0) = 2;
                val_res.cmp[val_res.position(pos_res)]->set_domain(num_dom) =
                    mult_cos_theta(comp_r_domain) - mult_sin_theta(comp_t_domain);
            } while (pos_so.inc());
        }

        auto build_cartesian_partial_derivative = [&](int lane) {
            Tensor derivative(so.get_val_t().get_space(), valence + 1, type_ind, basis);
            const Tensor* comp_r_derivative = comp_r.get_p_der_t(lane);
            const Tensor* comp_t_derivative = comp_t.get_p_der_t(lane);
            const Tensor* comp_p_derivative = comp_p.get_p_der_t(lane);
            const Tensor& comp_r_value = *comp_r.val_t;
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);

                const int source_component = comp_r_value.position(pos_so);
                Val_domain comp_r_domain((*comp_r_value.cmp[source_component])(num_dom));
                Val_domain comp_t_domain((*comp_r_value.cmp[source_component])(num_dom));
                Val_domain comp_p_domain((*comp_r_value.cmp[source_component])(num_dom));
                comp_r_domain.set_zero();
                comp_t_domain.set_zero();
                comp_p_domain.set_zero();
                if (comp_r_derivative != nullptr)
                    comp_r_domain = (*comp_r_derivative->cmp[source_component])(num_dom);
                if (comp_t_derivative != nullptr)
                    comp_t_domain = (*comp_t_derivative->cmp[source_component])(num_dom);
                if (comp_p_derivative != nullptr)
                    comp_p_domain = (*comp_p_derivative->cmp[source_component])(num_dom);

                Val_domain auxi(mult_sin_theta(comp_r_domain) + mult_cos_theta(comp_t_domain));
                // X part
                pos_res.set(0) = 0;
                derivative.cmp[derivative.position(pos_res)]->set_domain(num_dom) =
                    mult_cos_phi(auxi) - mult_sin_phi(comp_p_domain);
                // Y part
                pos_res.set(0) = 1;
                derivative.cmp[derivative.position(pos_res)]->set_domain(num_dom) =
                    mult_sin_phi(auxi) + mult_cos_phi(comp_p_domain);
                // Z part
                pos_res.set(0) = 2;
                derivative.cmp[derivative.position(pos_res)]->set_domain(num_dom) =
                    mult_cos_theta(comp_r_domain) - mult_sin_theta(comp_t_domain);
            } while (pos_so.inc());
            return derivative;
        };

        const bool has_cartesian_derivative =
            comp_r.get_p_der_t() != nullptr || comp_t.get_p_der_t() != nullptr || comp_p.get_p_der_t() != nullptr;
        if (has_cartesian_derivative) {
            Term_eq result(num_dom, val_res, build_cartesian_partial_derivative(0));
            const int lanes = std::max(comp_r.get_derivative_lane_count(),
                                       std::max(comp_t.get_derivative_lane_count(),
                                                comp_p.get_derivative_lane_count()));
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; lane++) {
                if (comp_r.get_p_der_t(lane) != nullptr || comp_t.get_p_der_t(lane) != nullptr ||
                    comp_p.get_p_der_t(lane) != nullptr)
                    result.set_der_t(lane, build_cartesian_partial_derivative(lane));
            }
            return result;
        } else
            return Term_eq(num_dom, val_res);
    }

    const Term_eq* Domain_shell_outer_adapted_nosym::give_normal(int bound, int tipe) const
    {
        assert(bound == OUTER_BC);
        switch (tipe) {
            case CARTESIAN_BASIS:
                if (normal_cart == nullptr)
                    do_normal_cart();
                return normal_cart;

            case SPHERICAL_BASIS:
                if (normal_spher == nullptr)
                    do_normal_spher();
                return normal_spher;

            default:
                KADATH_THROW("Unknown type of tensorial basis in Domain_shell_inner_adapted_nosym::give_normal");
        }
    }

    double integral_1d(int, const Array<double>&);
    Term_eq Domain_shell_outer_adapted_nosym::integ_volume_term_eq(const Term_eq& target) const
    {

        int dom = target.get_dom();
        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_int_volume only defined with respect for a tensor");
        }

        if (target.val_t->get_n_comp() != 1) {
            KADATH_THROW("Ope_int_volume only defined with respect to a scalar");
        }

        Term_eq integrant(
            do_comp_by_comp(mult_r_term_eq(mult_r_term_eq(target)) * (*der_rad_term_eq), &Domain::mult_sin_theta));

        Array<int> ind(target.val_t->indices(0));
        auto integrate_adapted_volume = [&](const Val_domain& value) {
            double integral = 0.;
            if (value.check_if_zero())
                return integral;
            [[maybe_unused]] int baset = (*value.get_base().bases_1d[1])(0);
            assert(baset == SIN);
            Index pos(nbr_coefs);
            // NONSYM SIN basis: l_quant = j. Only odd-j modes contribute;
            // even-j modes integrate to zero.
            for (int j = 1; j < nbr_coefs(1); j += 2) {
                pos.set(1) = j;
                if (j % 2 == 0)
                    continue;  // even-j SIN modes are z-odd: ∫₀^π sin(jθ)dθ = 0
                [[maybe_unused]] int baser = (*value.get_base().bases_1d[0])(j, 0);
                assert(baser == CHEB);

                Array<double> cf(nbr_coefs(0));
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos.set(0) = i;
                    cf.set(i) = value.get_coef(pos);
                }
                integral += 2. / double(j) * integral_1d(CHEB, cf);
            }
            return integral * 2 * M_PI;
        };

        const Val_domain value((*integrant.val_t)(ind)(dom));
        const double resval = integrate_adapted_volume(value);

        if (integrant.get_p_der_t() == nullptr)
            return Term_eq(dom, resval);

        auto integrate_derivative_lane = [&](int lane) {
            const Tensor& derivative = integrant.get_der_t(lane);
            const Val_domain derivative_value(derivative(ind)(dom));
            return integrate_adapted_volume(derivative_value);
        };

        Term_eq result(dom, resval, integrate_derivative_lane(0));
        result.set_derivative_lane_count(integrant.get_derivative_lane_count());
        for (int lane = 1; lane < integrant.get_derivative_lane_count(); ++lane) {
            if (integrant.has_der_t(lane))
                result.set_der_d(lane, integrate_derivative_lane(lane));
        }
        return result;
    }

    Term_eq Domain_shell_outer_adapted_nosym::integ_term_eq(const Term_eq&, int) const
    {
        KADATH_THROW("Domain_shell_outer_adapted_nosym::integ_term_eq not implemented yet");
    }
} // namespace Kadath
