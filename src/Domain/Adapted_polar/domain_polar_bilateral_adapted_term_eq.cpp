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
#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "adapted_polar_term_eq_lanes.hpp"

namespace Kadath
{

    Term_eq Domain_polar_shell_bilateral_adapted::dr_term_eq(const Term_eq& so) const
    {
        return derive_r(so);
    }

    Term_eq Domain_polar_shell_bilateral_adapted::derive_r(const Term_eq& so) const
    {

        assert(so.get_dom() == num_dom);
        return adapted_polar_detail::radial_derivative_term_eq(num_dom, so, *der_rad_term_eq);
    }

    Term_eq Domain_polar_shell_bilateral_adapted::derive_t(const Term_eq& so) const
    {

        assert(so.get_dom() == num_dom);
        Term_eq dtprime(adapted_polar_detail::theta_prime_derivative_term_eq(num_dom, so));
        Term_eq res(dtprime - (*dt_rad_term_eq) * derive_r(so));
        return res;
    }

    Term_eq Domain_polar_shell_bilateral_adapted::flat_grad_spher(const Term_eq& so) const
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
        Term_eq comp_t(
            (adapted_polar_detail::theta_prime_derivative_term_eq(num_dom, so) - (*dt_rad_term_eq) * comp_r) /
            (*rad_term_eq));

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
        }
        Term_eq result(num_dom, res);
        const int derivative_lanes = adapted_polar_detail::combined_derivative_lane_count(comp_r, comp_t);
        result.set_derivative_lane_count(derivative_lanes);
        for (int lane = 0; lane < derivative_lanes; ++lane) {
            if (!adapted_polar_detail::has_tensor_derivative_lane(comp_r, lane) &&
                !adapted_polar_detail::has_tensor_derivative_lane(comp_t, lane)) {
                continue;
            }
            Tensor resder(sp, valso + 1, type_ind, basis);
            // Loop on cmp :
            for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {

                Array<int> ind(so.get_val_t().indices(nc));
                Array<int> indtarget(valso + 1);
                for (int i = 0; i < valso; i++)
                    indtarget.set(i + 1) = ind(i);

                // R comp :
                indtarget.set(0) = 1;
                if (comp_r.has_der_t(lane))
                    resder.set(indtarget).set_domain(num_dom) = comp_r.get_der_t(lane)(ind)(num_dom);
                else
                    resder.set(indtarget).set_domain(num_dom).set_zero();
                // theta comp :
                indtarget.set(0) = 2;
                if (comp_t.has_der_t(lane))
                    resder.set(indtarget).set_domain(num_dom) = comp_t.get_der_t(lane)(ind)(num_dom);
                else
                    resder.set(indtarget).set_domain(num_dom).set_zero();
            }
            result.set_der_t(lane, resder);
        }
        return result;
    }

    void Domain_polar_shell_bilateral_adapted::do_normal_spher() const
    {

        Term_eq grad(flat_grad_spher(*rad_term_eq - *outer_radius_term_eq));

        Scalar val_norme(sp);
        val_norme.set_domain(num_dom) = sqrt((*grad.val_t)(1)(num_dom) * (*grad.val_t)(1)(num_dom) +
                                             (*grad.val_t)(2)(num_dom) * (*grad.val_t)(2)(num_dom));
        Term_eq norme(num_dom, val_norme);
        norme.set_derivative_lane_count(grad.get_derivative_lane_count());
        for (int lane = 0; lane < grad.get_derivative_lane_count(); ++lane) {
            if (!grad.has_der_t(lane))
                continue;
            Scalar der_norme(sp);
            der_norme.set_domain(num_dom) = (grad.get_der_t(lane)(1)(num_dom) * (*grad.val_t)(1)(num_dom) +
                                             grad.get_der_t(lane)(2)(num_dom) * (*grad.val_t)(2)(num_dom)) /
                                            val_norme(num_dom);
            norme.set_der_t(lane, der_norme);
        }
        if (normal_spher == nullptr)
            normal_spher = new Term_eq(grad / norme);
        else
            *normal_spher = Term_eq(grad / norme);
    }

    void Domain_polar_shell_bilateral_adapted::do_normal_cart() const
    {

        do_normal_spher();

        Base_tensor basis(sp);
        basis.set_basis(num_dom) = CARTESIAN_BASIS;
        Vector val(sp, CON, basis);

        val.set(1).set_domain(num_dom) =
            mult_sin_theta((*normal_spher->val_t)(1)(num_dom)) + mult_cos_theta((*normal_spher->val_t)(2)(num_dom));
        val.set(2).set_domain(num_dom) =
            mult_cos_theta((*normal_spher->val_t)(1)(num_dom)) - mult_sin_theta((*normal_spher->val_t)(2)(num_dom));

        Term_eq result(num_dom, val);
        result.set_derivative_lane_count(normal_spher->get_derivative_lane_count());
        for (int lane = 0; lane < normal_spher->get_derivative_lane_count(); ++lane) {
            if (!normal_spher->has_der_t(lane))
                continue;
            Vector der(sp, CON, basis);

            der.set(1).set_domain(num_dom) =
                mult_sin_theta(normal_spher->get_der_t(lane)(1)(num_dom)) +
                mult_cos_theta(normal_spher->get_der_t(lane)(2)(num_dom));
            der.set(2).set_domain(num_dom) =
                mult_cos_theta(normal_spher->get_der_t(lane)(1)(num_dom)) -
                mult_sin_theta(normal_spher->get_der_t(lane)(2)(num_dom));
            result.set_der_t(lane, der);
        }
        if (normal_cart == nullptr)
            normal_cart = new Term_eq(result);
        else
            *normal_cart = result;
    }

    Term_eq Domain_polar_shell_bilateral_adapted::der_normal_term_eq(const Term_eq& so, int bound) const
    {

        switch (bound) {
            case OUTER_BC: {
                // Deformed surface
                if (normal_spher == nullptr)
                    do_normal_spher();

                int valso = so.get_val_t().get_valence();

                Term_eq grad(flat_grad_spher(so));

                Tensor res(so.get_val_t(), false);
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
                }

                Term_eq result(num_dom, res);
                const int derivative_lanes =
                    adapted_polar_detail::combined_derivative_lane_count(grad, *normal_spher);
                result.set_derivative_lane_count(derivative_lanes);
                for (int lane = 0; lane < derivative_lanes; ++lane) {
                    if (!grad.has_der_t(lane) && !normal_spher->has_der_t(lane))
                        continue;
                    Tensor der(so.get_val_t(), false);
                    Array<int> indgrad(valso + 1);

                    // Loop on cmp :
                    for (int nc = 0; nc < so.get_val_t().get_n_comp(); nc++) {

                        Array<int> ind(so.get_val_t().indices(nc));
                        for (int i = 0; i < valso; i++)
                            indgrad.set(i + 1) = ind(i);

                        indgrad.set(0) = 1;
                        der.set(ind).set_domain(num_dom).set_zero();
                        if (grad.has_der_t(lane))
                            der.set(ind).set_domain(num_dom) +=
                                grad.get_der_t(lane)(indgrad)(num_dom) * (*normal_spher->val_t)(1)(num_dom);
                        if (normal_spher->has_der_t(lane))
                            der.set(ind).set_domain(num_dom) +=
                                (*grad.val_t)(indgrad)(num_dom) * normal_spher->get_der_t(lane)(1)(num_dom);
                        indgrad.set(0) = 2;
                        if (grad.has_der_t(lane))
                            der.set(ind).set_domain(num_dom) +=
                                grad.get_der_t(lane)(indgrad)(num_dom) * (*normal_spher->val_t)(2)(num_dom);
                        if (normal_spher->has_der_t(lane))
                            der.set(ind).set_domain(num_dom) +=
                                (*grad.val_t)(indgrad)(num_dom) * normal_spher->get_der_t(lane)(2)(num_dom);
                    }

                    result.set_der_t(lane, der);
                }
                return result;
            }
            case INNER_BC: {
                return derive_r(so);
            }
            default:
                KADATH_THROW("Unknown boundary in Domain_polar_shell_bilateral_adapted::der_normal");
        }
    }

    Term_eq Domain_polar_shell_bilateral_adapted::lap_term_eq(const Term_eq& so, int mm) const
    {

        if (so.get_val_t().get_valence() != 0) {
            KADATH_THROW("Domain_polar_shell_bilateral_adapted::lap_term_eq only defined for scalars");
        }

        return adapted_polar_detail::scalar_laplacian_term_eq(
            so,
            mm,
            *rad_term_eq,
            [this](const Term_eq& value) { return derive_r(value); },
            [this](const Term_eq& value) { return derive_t(value); },
            [this](const Term_eq& value) { return do_comp_by_comp(value, &Domain::div_sin_theta); },
            [this](const Term_eq& value) { return do_comp_by_comp(value, &Domain::mult_cos_theta); });
    }

    Term_eq Domain_polar_shell_bilateral_adapted::lap2_term_eq(const Term_eq& so, int mm) const
    {

        if (mm != 0) {
            KADATH_THROW("Domain_polar_shell_bilateral_adapted::lap2_term_eq not defined for m != 0 (for now)");
        }

        if (so.get_val_t().get_valence() != 0) {
            KADATH_THROW("Domain_polar_shell_bilateral_adapted::lap2_term_eq only defined for scalars");
        }

        // Angular part :
        Term_eq dert(derive_t(so));
        Term_eq p1(derive_t(dert));

        Term_eq dr(derive_r(so));

        Term_eq res(derive_r(dr) + dr / (*rad_term_eq) + p1 / (*rad_term_eq) / (*rad_term_eq));

        return res;
    }

    Term_eq Domain_polar_shell_bilateral_adapted::mult_r_term_eq(const Term_eq& so) const
    {

        return so * (*rad_term_eq);
    }

    Term_eq Domain_polar_shell_bilateral_adapted::div_r_term_eq(const Term_eq& so) const
    {

        return so / (*rad_term_eq);
    }

    void Domain_polar_shell_bilateral_adapted::update_term_eq(Term_eq* so) const
    {

        Tensor der(*so->val_t, false);
        for (int cmp = 0; cmp < so->val_t->get_n_comp(); cmp++) {

            Val_domain derr((*(*so->val_t).cmp[cmp])(num_dom).der_var(1) / (*der_rad_term_eq->val_t)()(num_dom));

            if (!derr.check_if_zero()) {
                Val_domain res(this);
                res.allocate_conf();
                Index pos(nbr_points);
                do {
                    double inner = 0.0;
                    double outer = 0.0;
                    if (inner_radius_term_eq->der_t != nullptr)
                        inner = (*(*inner_radius_term_eq->der_t).cmp[0])(num_dom)(pos);
                    if (outer_radius_term_eq->der_t != nullptr)
                        outer = (*(*outer_radius_term_eq->der_t).cmp[0])(num_dom)(pos);
                    res.set(pos) =
                        (outer * (1 + ((*coloc[0])(pos(0)))) + inner * (1 - ((*coloc[0])(pos(0))))) / 2. * derr(pos);
                } while (pos.inc());
                res.set_base() = (*(*so->val_t).cmp[cmp])(num_dom).get_base();
                der.cmp[cmp]->set_domain(num_dom) = res;
            } else
                der.cmp[cmp]->set_domain(num_dom).set_zero();
        }

        so->set_der_t(der);
    }

    Term_eq Domain_polar_shell_bilateral_adapted::partial_spher(const Term_eq& so) const
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
        Term_eq comp_t(
            (adapted_polar_detail::theta_prime_derivative_term_eq(num_dom, so) - (*dt_rad_term_eq) * comp_r) /
            (*rad_term_eq));

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
            } while (pos_so.inc());
        }

        Term_eq result(num_dom, val_res);
        const int derivative_lanes = adapted_polar_detail::combined_derivative_lane_count(comp_r, comp_t);
        result.set_derivative_lane_count(derivative_lanes);
        for (int lane = 0; lane < derivative_lanes; ++lane) {
            if (!comp_r.has_der_t(lane) && !comp_t.has_der_t(lane))
                continue;
            Tensor der_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);
                // R part
                pos_res.set(0) = 0;
                if (comp_r.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) = comp_r.get_der_t(lane)(pos_so)(num_dom);
                else
                    der_res.set(pos_res).set_domain(num_dom).set_zero();
                // Theta part
                pos_res.set(0) = 1;
                if (comp_t.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) = comp_t.get_der_t(lane)(pos_so)(num_dom);
                else
                    der_res.set(pos_res).set_domain(num_dom).set_zero();
            } while (pos_so.inc());

            result.set_der_t(lane, der_res);
        }
        return result;
    }

    Term_eq Domain_polar_shell_bilateral_adapted::partial_cart(const Term_eq& so) const
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
        Term_eq comp_t(
            (adapted_polar_detail::theta_prime_derivative_term_eq(num_dom, so) - (*dt_rad_term_eq) * comp_r) /
            (*rad_term_eq));

        Tensor val_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
        if (valence == 0) {
            const Val_domain& radial_component = (*comp_r.val_t->cmp[0])(num_dom);
            const Val_domain& angular_component = (*comp_t.val_t->cmp[0])(num_dom);
            val_res.cmp[0]->set_domain(num_dom) =
                mult_sin_theta(radial_component) + mult_cos_theta(angular_component);
            val_res.cmp[1]->set_domain(num_dom) =
                mult_cos_theta(radial_component) - mult_sin_theta(angular_component);

            Term_eq result(num_dom, val_res);
            const int derivative_lanes = adapted_polar_detail::combined_derivative_lane_count(comp_r, comp_t);
            result.set_derivative_lane_count(derivative_lanes);
            for (int lane = 0; lane < derivative_lanes; ++lane) {
                if (!comp_r.has_der_t(lane) && !comp_t.has_der_t(lane))
                    continue;
                Tensor der_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
                Val_domain radial_tangent(this);
                radial_tangent = 0.0;
                if (comp_r.has_der_t(lane))
                    radial_tangent = comp_r.get_der_t(lane)()(num_dom);
                Val_domain angular_tangent(this);
                angular_tangent = 0.0;
                if (comp_t.has_der_t(lane))
                    angular_tangent = comp_t.get_der_t(lane)()(num_dom);
                der_res.cmp[0]->set_domain(num_dom) =
                    mult_sin_theta(radial_tangent) + mult_cos_theta(angular_tangent);
                der_res.cmp[1]->set_domain(num_dom) =
                    mult_cos_theta(radial_tangent) - mult_sin_theta(angular_tangent);
                result.set_der_t(lane, der_res);
            }
            return result;
        }

        {
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);

                // X part
                pos_res.set(0) = 0;
                val_res.set(pos_res).set_domain(num_dom) =
                    mult_sin_theta((*comp_r.val_t)(pos_so)(num_dom)) + mult_cos_theta((*comp_t.val_t)(pos_so)(num_dom));
                // Z part
                pos_res.set(0) = 1;
                val_res.set(pos_res).set_domain(num_dom) =
                    mult_cos_theta((*comp_r.val_t)(pos_so)(num_dom)) - mult_sin_theta((*comp_t.val_t)(pos_so)(num_dom));
            } while (pos_so.inc());
        }

        Term_eq result(num_dom, val_res);
        const int derivative_lanes = adapted_polar_detail::combined_derivative_lane_count(comp_r, comp_t);
        result.set_derivative_lane_count(derivative_lanes);
        for (int lane = 0; lane < derivative_lanes; ++lane) {
            if (!comp_r.has_der_t(lane) && !comp_t.has_der_t(lane))
                continue;
            Tensor der_res(so.get_val_t().get_space(), valence + 1, type_ind, basis);
            Index pos_so(*so.val_t);
            Index pos_res(val_res);
            do {
                for (int i = 1; i < valence + 1; i++)
                    pos_res.set(i) = pos_so(i - 1);

                // X part
                pos_res.set(0) = 0;
                der_res.set(pos_res).set_domain(num_dom).set_zero();
                if (comp_r.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) +=
                        mult_sin_theta(comp_r.get_der_t(lane)(pos_so)(num_dom));
                if (comp_t.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) +=
                        mult_cos_theta(comp_t.get_der_t(lane)(pos_so)(num_dom));
                // Z part
                pos_res.set(0) = 1;
                der_res.set(pos_res).set_domain(num_dom).set_zero();
                if (comp_r.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) +=
                        mult_cos_theta(comp_r.get_der_t(lane)(pos_so)(num_dom));
                if (comp_t.has_der_t(lane))
                    der_res.set(pos_res).set_domain(num_dom) -=
                        mult_sin_theta(comp_t.get_der_t(lane)(pos_so)(num_dom));
            } while (pos_so.inc());

            result.set_der_t(lane, der_res);
        }
        return result;
    }

    Term_eq Domain_polar_shell_bilateral_adapted::grad_term_eq(const Term_eq& so) const
    {
        return partial_cart(so);
    }

    const Term_eq* Domain_polar_shell_bilateral_adapted::give_normal(int bound, int tipe) const
    {
        assert(bound == INNER_BC);
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
                KADATH_THROW("Unknown type of tensorial basis in Domain_polar_shell_bilateral_adapted::give_normal");
        }
    }

    Term_eq Domain_polar_shell_bilateral_adapted::integ_term_eq(const Term_eq&, int) const
    {
        KADATH_THROW("Domain_polar_shell_bilateral_adapted::integ_term_eq not implemented yet");
    }
} // namespace Kadath
