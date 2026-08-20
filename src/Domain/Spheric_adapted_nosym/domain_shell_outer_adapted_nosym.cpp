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
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace Kadath
{
    namespace
    {
        template <typename Function>
        void for_each_physical_point(const Dim_array& dimensions, Function&& function)
        {
            assert(dimensions.get_ndim() == 3);
            std::size_t offset = 0;
            for (int i = 0; i < dimensions(0); ++i)
                for (int j = 0; j < dimensions(1); ++j)
                    for (int k = 0; k < dimensions(2); ++k)
                        function(i, j, k, offset++);
        }
    }

    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_shell_outer_adapted_nosym::Domain_shell_outer_adapted_nosym(const Space& sss, int num, int ttype, double rin,
                                                           double rout, const Point& cr, const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), inner_radius(rin),
          center(cr)
    {
        assert(nbr.get_ndim() == 3);
        assert(cr.get_ndim() == 3);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        outer_radius = new Val_domain(this);
        *outer_radius = rout;
        outer_radius->std_base();
    }

    Domain_shell_outer_adapted_nosym::Domain_shell_outer_adapted_nosym(const Space& sss, int num, int ttype, double rin,
                                                           const Val_domain& rout, const Point& cr,
                                                           const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), inner_radius(rin),
          center(cr)
    {
        assert(nbr.get_ndim() == 3);
        assert(cr.get_ndim() == 3);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        outer_radius = new Val_domain(rout);
    }

    // Constructor by copy
    Domain_shell_outer_adapted_nosym::Domain_shell_outer_adapted_nosym(const Domain_shell_outer_adapted_nosym& so)
        : Domain(so), sp(so.sp), inner_radius(so.inner_radius),
          center(so.center)
    {

        outer_radius = new Val_domain(*so.outer_radius);
        if (so.outer_radius_term_eq != nullptr)
            outer_radius_term_eq = new Term_eq(*so.outer_radius_term_eq);
        if (so.rad_term_eq != nullptr)
            rad_term_eq = new Term_eq(*so.rad_term_eq);
        if (so.der_rad_term_eq != nullptr)
            der_rad_term_eq = new Term_eq(*so.der_rad_term_eq);
        if (so.dt_rad_term_eq != nullptr)
            dt_rad_term_eq = new Term_eq(*so.dt_rad_term_eq);
        if (so.dp_rad_term_eq != nullptr)
            dp_rad_term_eq = new Term_eq(*so.dp_rad_term_eq);
        if (so.normal_spher != nullptr)
            normal_spher = new Term_eq(*so.normal_spher);
        if (so.normal_cart != nullptr)
            normal_cart = new Term_eq(*so.normal_cart);
    }

    Domain_shell_outer_adapted_nosym::Domain_shell_outer_adapted_nosym(const Space& sss, int num, BinarySource& source)
        : Domain(num, source), sp(sss), center(source)
    {
        inner_radius = source.read<double>();
        outer_radius = new Val_domain(this, source);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
    }

    void Domain_shell_outer_adapted_nosym::do_radius() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        outer_radius->coef_i();
        const double* const outer_data = outer_radius->c->get_data();
        double* const radius_data = radius->c->get_data();
        for_each_physical_point(nbr_points, [&](int i, int, int, std::size_t offset) {
            radius_data[offset] = (outer_data[offset]-inner_radius) / 2. * (*coloc[0])(i) +
                                  (outer_data[offset] + inner_radius) / 2.;
        });
        radius->std_r_base();
    }

    // Destructor
    Domain_shell_outer_adapted_nosym::~Domain_shell_outer_adapted_nosym()
    {

        delete outer_radius;
        if (outer_radius_term_eq != nullptr)
            delete outer_radius_term_eq;
        if (rad_term_eq != nullptr)
            delete rad_term_eq;
        if (der_rad_term_eq != nullptr)
            delete der_rad_term_eq;
        if (normal_spher != nullptr)
            delete normal_spher;
        if (normal_cart != nullptr)
            delete normal_cart;
        if (dt_rad_term_eq != nullptr)
            delete dt_rad_term_eq;
        if (dp_rad_term_eq != nullptr)
            delete dp_rad_term_eq;
    }

    void Domain_shell_outer_adapted_nosym::del_deriv() const
    {

        if (outer_radius_term_eq != nullptr)
            delete outer_radius_term_eq;
        outer_radius_term_eq = nullptr;
        if (rad_term_eq != nullptr)
            delete rad_term_eq;
        rad_term_eq = nullptr;
        if (der_rad_term_eq != nullptr)
            delete der_rad_term_eq;
        der_rad_term_eq = nullptr;
        if (normal_spher != nullptr)
            delete normal_spher;
        normal_spher = nullptr;
        if (normal_cart != nullptr)
            delete normal_cart;
        normal_cart = nullptr;
        if (dt_rad_term_eq != nullptr)
            delete dt_rad_term_eq;
        dt_rad_term_eq = nullptr;
        if (dp_rad_term_eq != nullptr)
            delete dp_rad_term_eq;
        dp_rad_term_eq = nullptr;
    }
    int Domain_shell_outer_adapted_nosym::nbr_unknowns_from_adapted() const
    {

        int res = 0;
        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int baset = (mm % 2 == 0) ? COS : SIN;
                for (int j = 0; j < nbr_coefs(1); j++)
                    if (detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)))
                        res++;
            }
        return res;
    }

    void Domain_shell_outer_adapted_nosym::vars_to_terms() const
    {

        if (outer_radius_term_eq != nullptr)
            delete outer_radius_term_eq;
        Scalar val(sp);
        val.set_domain(num_dom) = *outer_radius;

        outer_radius_term_eq = new Term_eq(num_dom, val);
        update();
    }

    void Domain_shell_outer_adapted_nosym::affecte_coef(int& conte, int cc, bool& found) const
    {

        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        found = false;

        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int baset = (mm % 2 == 0) ? COS : SIN;
                for (int j = 0; j < nbr_coefs(1); j++) {
                    if (!detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)))
                        continue;
                    if (conte == cc) {
                        Index pos_cf(nbr_coefs);
                        pos_cf.set(0) = 0;
                        pos_cf.set(1) = j;
                        pos_cf.set(2) = k;
                        auxi.cf->set(pos_cf) = 1;
                        if (detail::spheric_nosym_uses_theta_galerkin(baset, k, 2)) {
                            pos_cf.set(1) = detail::spheric_nosym_theta_anchor(baset, j);
                            auxi.cf->set(pos_cf) = -detail::spheric_nosym_basis_anchor_weight(baset, j);
                        }
                        found = true;
                    }
                    conte++;
                }
            }

        if (found) {
            Scalar auxi_scal(sp);
            auxi_scal.set_domain(num_dom) = auxi;
            outer_radius_term_eq->set_der_t(auxi_scal);
        } else {
            outer_radius_term_eq->set_der_zero();
        }
        update();
    }

    void Domain_shell_outer_adapted_nosym::affecte_coef_lane(
        int& conte, int cc, int lane, int lane_count, bool& found) const
    {
        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        found = false;
        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                const int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                const int baset = (mm % 2 == 0) ? COS : SIN;
                for (int j = 0; j < nbr_coefs(1); j++) {
                    if (!detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)))
                        continue;
                    if (conte == cc) {
                        Index pos_cf(nbr_coefs);
                        pos_cf.set(0) = 0;
                        pos_cf.set(1) = j;
                        pos_cf.set(2) = k;
                        auxi.cf->set(pos_cf) = 1;
                        if (detail::spheric_nosym_uses_theta_galerkin(baset, k, 2)) {
                            pos_cf.set(1) = detail::spheric_nosym_theta_anchor(baset, j);
                            auxi.cf->set(pos_cf) =
                                -detail::spheric_nosym_basis_anchor_weight(baset, j);
                        }
                        found = true;
                    }
                    conte++;
                }
            }

        if (lane == 0)
            outer_radius_term_eq->reset_derivative_tile(lane_count);
        if (found) {
            Scalar auxi_scal(sp);
            auxi_scal.set_domain(num_dom) = auxi;
            outer_radius_term_eq->set_der_t(lane, auxi_scal);
        } else {
            outer_radius_term_eq->set_der_zero(lane);
        }
        update();
    }

    void Domain_shell_outer_adapted_nosym::restore_scalar_derivative_lanes() const
    {
        outer_radius_term_eq->reset_derivative_tile(1);
        update();
    }

    void Domain_shell_outer_adapted_nosym::xx_to_vars_from_adapted(Val_domain& new_outer_radius, const Array<double>& xx,
                                                             int& pos) const
    {

        new_outer_radius.allocate_coef();
        *new_outer_radius.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                pos_cf.set(2) = k;
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int baset = (mm % 2 == 0) ? COS : SIN;
                for (int j = 0; j < nbr_coefs(1); j++) {
                    if (!detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)))
                        continue;
                    pos_cf.set(1) = j;
                    new_outer_radius.cf->set(pos_cf) -= xx(pos);
                    if (detail::spheric_nosym_uses_theta_galerkin(baset, k, 2)) {
                        pos_cf.set(1) = detail::spheric_nosym_theta_anchor(baset, j);
                        new_outer_radius.cf->set(pos_cf) +=
                            detail::spheric_nosym_basis_anchor_weight(baset, j) * xx(pos);
                    }

                    pos++;
                }
            }

        new_outer_radius.set_base() = outer_radius->get_base();
    }

    void Domain_shell_outer_adapted_nosym::update_mapping(const Val_domain& cor)
    {

        *outer_radius += cor;
        clear_import_plan_cache();
        for (int l = 0; l < ndim; l++) {
            if (absol[l] != nullptr)
                delete absol[l];
            if (cart[l] != nullptr)
                delete cart[l];
            if (cart_surr[l] != nullptr)
                delete cart_surr[l];
            absol[l] = nullptr;
            cart_surr[l] = nullptr;
            cart[l] = nullptr;
        }
        if (radius != nullptr)
            delete radius;
        radius = nullptr;
        update();
    }

    void Domain_shell_outer_adapted_nosym::set_mapping(const Val_domain& cor) const
    {
        *outer_radius = cor;
        clear_import_plan_cache();
        for (int l = 0; l < ndim; l++) {
            if (absol[l] != nullptr)
                delete absol[l];
            if (cart[l] != nullptr)
                delete cart[l];
            if (cart_surr[l] != nullptr)
                delete cart_surr[l];
            absol[l] = nullptr;
            cart_surr[l] = nullptr;
            cart[l] = nullptr;
        }
        if (radius != nullptr)
            delete radius;
        radius = nullptr;
        vars_to_terms();
    }

    void Domain_shell_outer_adapted_nosym::update_variable(const Val_domain& cor_outer_radius, const Scalar& old,
                                                     Scalar& res) const
    {

        Val_domain dr(old(num_dom).der_r());
        if (dr.check_if_zero())
            res.set_domain(num_dom) = 0;
        else {
            res.set_domain(num_dom).allocate_conf();
            dr.coef_i();
            const double* const derivative_data = dr.c->get_data();
            double* const result_data = res.set_domain(num_dom).c->get_data();
            for_each_physical_point(nbr_points, [&](int i, int, int, std::size_t offset) {
                result_data[offset] = derivative_data[offset] * (1 + (*coloc[0])(i)) / 2.;
            });

            res.set_domain(num_dom) = cor_outer_radius * res(num_dom) + old(num_dom);
            res.set_domain(num_dom).set_base() = old(num_dom).get_base();
        }
    }

    void Domain_shell_outer_adapted_nosym::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const
    {

        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                pos_cf.set(2) = k;
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int baset = (mm % 2 == 0) ? COS : SIN;
                for (int j = 0; j < nbr_coefs(1); j++) {
                    if (!detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)))
                        continue;
                    pos_cf.set(1) = j;
                    auxi.cf->set(pos_cf) = xx(pos);

                    if (detail::spheric_nosym_uses_theta_galerkin(baset, k, 2)) {
                        pos_cf.set(1) = detail::spheric_nosym_theta_anchor(baset, j);
                        auxi.cf->set(pos_cf) += -detail::spheric_nosym_basis_anchor_weight(baset, j) * xx(pos);
                    }
                    pos++;
                }
            }
        Scalar auxi_scal(sp);
        auxi_scal.set_domain(num_dom) = auxi;
        outer_radius_term_eq->set_der_t(auxi_scal);
        update();
    }

    void Domain_shell_outer_adapted_nosym::update() const
    {
        // Computation of rad_term_eq
        Scalar val_res(sp);
        val_res.set_domain(num_dom).allocate_conf();
        Index index(nbr_points);
        do {
            val_res.set_domain(num_dom).set(index) =
                (((*outer_radius_term_eq->val_t)()(num_dom))(index)-inner_radius) / 2. * ((*coloc[0])(index(0))) +
                (((*outer_radius_term_eq->val_t)()(num_dom))(index) + inner_radius) / 2.;
        } while (index.inc());
        val_res.set_domain(num_dom).std_r_base();

        const int lane_count = outer_radius_term_eq->get_derivative_lane_count();
        auto mapping_derivative = [&](int lane) {
            Scalar result(sp);
            const Val_domain& radius_derivative = outer_radius_term_eq->get_der_t(lane)()(num_dom);
            if (radius_derivative.check_if_zero()) {
                result.set_domain(num_dom).set_zero();
            } else {
                result.set_domain(num_dom).allocate_conf();
                index.set_start();
                do {
                    result.set_domain(num_dom).set(index) =
                        radius_derivative(index) / 2. * ((*coloc[0])(index(0))) +
                        radius_derivative(index) / 2.;
                } while (index.inc());
                result.set_domain(num_dom).std_r_base();
            }
            return result;
        };
        std::unique_ptr<Term_eq> next_rad_term_eq;
        if (outer_radius_term_eq->has_der_t(0))
            next_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res, mapping_derivative(0));
        else
            next_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res);
        next_rad_term_eq->set_derivative_lane_count(lane_count);
        for (int lane = 1; lane < lane_count; ++lane)
            if (outer_radius_term_eq->has_der_t(lane))
                next_rad_term_eq->set_der_t(lane, mapping_derivative(lane));

        // Computation of der_rad_term_eq which is dr / dxi
        val_res.set_domain(num_dom) = ((*outer_radius_term_eq->val_t)()(num_dom)-inner_radius) / 2.;
        std::unique_ptr<Term_eq> next_der_rad_term_eq;
        if (outer_radius_term_eq->has_der_t(0)) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = outer_radius_term_eq->get_der_t(0)()(num_dom) / 2.;
            next_der_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res, der_res);
        } else
            next_der_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res);
        next_der_rad_term_eq->set_derivative_lane_count(lane_count);
        for (int lane = 1; lane < lane_count; ++lane) {
            if (!outer_radius_term_eq->has_der_t(lane))
                continue;
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = outer_radius_term_eq->get_der_t(lane)()(num_dom) / 2.;
            next_der_rad_term_eq->set_der_t(lane, der_res);
        }

        // Computation of dt_rad_term_eq which is dr / d theta
        val_res.set_domain(num_dom) = (*next_rad_term_eq->val_t)()(num_dom).der_var(2);
        std::unique_ptr<Term_eq> next_dt_rad_term_eq;
        if (next_rad_term_eq->has_der_t(0)) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = next_rad_term_eq->get_der_t(0)()(num_dom).der_var(2);
            next_dt_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res, der_res);
        } else
            next_dt_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res);
        next_dt_rad_term_eq->set_derivative_lane_count(lane_count);
        for (int lane = 1; lane < lane_count; ++lane) {
            if (!next_rad_term_eq->has_der_t(lane))
                continue;
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = next_rad_term_eq->get_der_t(lane)()(num_dom).der_var(2);
            next_dt_rad_term_eq->set_der_t(lane, der_res);
        }

        // Computation of dp_rad_term_eq which is dr / d phi
        val_res.set_domain(num_dom) = (*next_rad_term_eq->val_t)()(num_dom).der_var(3);
        std::unique_ptr<Term_eq> next_dp_rad_term_eq;
        if (next_rad_term_eq->has_der_t(0)) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = next_rad_term_eq->get_der_t(0)()(num_dom).der_var(3);
            next_dp_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res, der_res);
        } else
            next_dp_rad_term_eq = std::make_unique<Term_eq>(num_dom, val_res);
        next_dp_rad_term_eq->set_derivative_lane_count(lane_count);
        for (int lane = 1; lane < lane_count; ++lane) {
            if (!next_rad_term_eq->has_der_t(lane))
                continue;
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = next_rad_term_eq->get_der_t(lane)()(num_dom).der_var(3);
            next_dp_rad_term_eq->set_der_t(lane, der_res);
        }

        // Commit only after every replacement has been constructed. If any
        // allocation or spectral operation above throws, the existing caches
        // remain owned and valid for the caller's recovery path.
        delete rad_term_eq;
        rad_term_eq = next_rad_term_eq.release();
        delete der_rad_term_eq;
        der_rad_term_eq = next_der_rad_term_eq.release();
        delete dt_rad_term_eq;
        dt_rad_term_eq = next_dt_rad_term_eq.release();
        delete dp_rad_term_eq;
        dp_rad_term_eq = next_dp_rad_term_eq.release();

        do_normal_cart();
    }

    void Domain_shell_outer_adapted_nosym::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(inner_radius);
        outer_radius->save(sink);
    }

    // ostream& operator<< (ostream& o, const Domain_shell_outer_adapted_nosym& so) {
    ostream& Domain_shell_outer_adapted_nosym::print(ostream& o) const
    {
        o << "Adapted shell on the outside boundary" << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << "Inner radius " << inner_radius << endl;
        o << "Outer radius " << endl;
        o << *outer_radius << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_shell_outer_adapted_nosym::der_normal(const Val_domain&, int) const
    {
        KADATH_THROW("Domain_shell_outer_adapted_nosym::der_normal not implemeted");
    }

    // Computes the cartesian coordinates
    void Domain_shell_outer_adapted_nosym::do_absol() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        outer_radius->coef_i();
        const double* const outer_data = outer_radius->c->get_data();
        double* const x_data = absol[0]->c->get_data();
        double* const y_data = absol[1]->c->get_data();
        double* const z_data = absol[2]->c->get_data();
        for_each_physical_point(nbr_points, [&](int i, int j, int k, std::size_t offset) {
            const double rr = (outer_data[offset]-inner_radius) / 2. * (*coloc[0])(i) +
                              (outer_data[offset] + inner_radius) / 2.;
            x_data[offset] = rr * sin((*coloc[1])(j)) * cos((*coloc[2])(k)) + center(1);
            y_data[offset] = rr * sin((*coloc[1])(j)) * sin((*coloc[2])(k)) + center(2);
            z_data[offset] = rr * cos((*coloc[1])(j)) + center(3);
        });
        absol[0]->std_base();
        absol[1]->std_base();
        absol[2]->std_anti_base();
    }

    // Computes the cartesian coordinates
    void Domain_shell_outer_adapted_nosym::do_cart() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(cart[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            cart[i] = new Val_domain(this);
            cart[i]->allocate_conf();
        }
        outer_radius->coef_i();
        const double* const outer_data = outer_radius->c->get_data();
        double* const x_data = cart[0]->c->get_data();
        double* const y_data = cart[1]->c->get_data();
        double* const z_data = cart[2]->c->get_data();
        for_each_physical_point(nbr_points, [&](int i, int j, int k, std::size_t offset) {
            const double rr = (outer_data[offset]-inner_radius) / 2. * (*coloc[0])(i) +
                              (outer_data[offset] + inner_radius) / 2.;
            x_data[offset] = rr * sin((*coloc[1])(j)) * cos((*coloc[2])(k)) + center(1);
            y_data[offset] = rr * sin((*coloc[1])(j)) * sin((*coloc[2])(k)) + center(2);
            z_data[offset] = rr * cos((*coloc[1])(j)) + center(3);
        });
        cart[0]->std_base();
        cart[1]->std_base();
        cart[2]->std_anti_base();
    }

    // Computes the cartesian coordinates over the radius
    void Domain_shell_outer_adapted_nosym::do_cart_surr() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(cart_surr[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            cart_surr[i] = new Val_domain(this);
            cart_surr[i]->allocate_conf();
        }
        double* const x_data = cart_surr[0]->c->get_data();
        double* const y_data = cart_surr[1]->c->get_data();
        double* const z_data = cart_surr[2]->c->get_data();
        for_each_physical_point(nbr_points, [&](int, int j, int k, std::size_t offset) {
            x_data[offset] = sin((*coloc[1])(j)) * cos((*coloc[2])(k));
            y_data[offset] = sin((*coloc[1])(j)) * sin((*coloc[2])(k));
            z_data[offset] = cos((*coloc[1])(j));
        });
        cart_surr[0]->std_base();
        cart_surr[1]->std_base();
        cart_surr[2]->std_anti_base();
    }

    // Is a point inside this domain ?
    bool Domain_shell_outer_adapted_nosym::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 3);
        Point num(absol_to_num(xx));
        bool res = ((num(1) > -1 - prec) && (num(1) < 1 + prec)) ? true : false;
        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_shell_outer_adapted_nosym::absol_to_num(const Point& abs) const
    {

        Point num(3);

        double x_loc = abs(1) - center(1);
        double y_loc = abs(2) - center(2);
        double z_loc = abs(3) - center(3);
        double air = sqrt(x_loc * x_loc + y_loc * y_loc + z_loc * z_loc);
        double rho = sqrt(x_loc * x_loc + y_loc * y_loc);

        if (rho == 0) {
            // Sur l'axe
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
            num.set(3) = 0;
        } else {
            num.set(2) = atan(rho / z_loc);
            num.set(3) = atan2(y_loc, x_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        // Get the boundary for those angles
        num.set(1) = 1;
        outer_radius->coef();
        double outer = outer_radius->get_base().summation(num, outer_radius->get_coef_ref());
        num.set(1) = (2. / (outer - inner_radius)) * (air - (outer + inner_radius) / 2.);

        return num;
    }

    const Point Domain_shell_outer_adapted_nosym::absol_to_num_bound(const Point& abs, int bound) const
    {

        Point num(3);

        double x_loc = abs(1) - center(1);
        double y_loc = abs(2) - center(2);
        double z_loc = abs(3) - center(3);
        double rho = sqrt(x_loc * x_loc + y_loc * y_loc);

        if (rho == 0) {
            // Sur l'axe
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
            num.set(3) = 0;
        } else {
            num.set(2) = atan(rho / z_loc);
            num.set(3) = atan2(y_loc, x_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        // Get the boundary for those angles
        switch (bound) {
            case INNER_BC:
                num.set(1) = -1;
                break;
            case OUTER_BC:
                num.set(1) = 1;
                break;
            default:
                KADATH_THROW("Unknown case in Domain_shell_inner_adapted_nosym::absol_to_num_bound");
        }
        return num;
    }

    double coloc_leg(int, int);
    void Domain_shell_outer_adapted_nosym::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                nbr_coefs.set(2) += 2;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = -cos(M_PI * i / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI * j / (nbr_points(1) - 1);  // NONSYM: full theta range [0, pi]
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * 2. * k / nbr_points(2);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                nbr_coefs.set(2) += 2;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI * j / (nbr_points(1) - 1);  // NONSYM: full theta range [0, pi]
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * 2. * k / nbr_points(2);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_shell_outer_adapted_nosym::do_coloc");
        }
    }

    // standard base for a symetric function in z, using Chebyshev
    void Domain_shell_outer_adapted_nosym::set_cheb_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_cheb_r_base(Base_spectral& base) const
    {
        set_cheb_base(base);
    }

    void Domain_shell_outer_adapted_nosym::set_legendre_r_base(Base_spectral& base) const
    {
        set_legendre_base(base);
    }

    // standard base for a anti-symetric function in z, using Chebyshev
    void Domain_shell_outer_adapted_nosym::set_anti_cheb_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_cheb_base_r_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_cheb_base_t_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN : COS;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_cheb_base_p_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN : COS;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // standard base for a symetric function in z, using Legendre
    void Domain_shell_outer_adapted_nosym::set_legendre_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // standard base for a anti-symetric function in z, using Legendre
    void Domain_shell_outer_adapted_nosym::set_anti_legendre_base(Base_spectral& base) const
    {

        int m;
        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_legendre_base_r_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS : SIN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_legendre_base_t_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN : COS;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_outer_adapted_nosym::set_legendre_base_p_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN : COS;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to XYZ as a function of the numerical ones.
    void Domain_shell_outer_adapted_nosym::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {

        Val_domain rr(get_radius());
        Val_domain dr(*der_var[0] / (*der_rad_term_eq->val_t)()(num_dom));
        Val_domain dtsr((*der_var[1] - rr.der_var(2) * dr) / rr);
        dtsr.set_base() = der_var[1]->get_base();
        Val_domain dpsr((*der_var[2] - rr.der_var(3) * dr) / rr);
        dpsr.set_base() = der_var[2]->get_base();

        // d/dx :
        Val_domain sintdr(dr.mult_sin_theta());
        Val_domain costdtsr(dtsr.mult_cos_theta());

        Val_domain dpsrssint(dpsr.div_sin_theta());
        der_abs[0] = new Val_domain((sintdr + costdtsr).mult_cos_phi() - dpsrssint.mult_sin_phi());

        // d/dy :
        der_abs[1] = new Val_domain((sintdr + costdtsr).mult_sin_phi() + dpsrssint.mult_cos_phi());
        // d/dz :
        der_abs[2] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_shell_outer_adapted_nosym::mult(const Base_spectral& a, const Base_spectral& b) const
    {

        assert(a.ndim == 3);
        assert(b.ndim == 3);

        Base_spectral res(3);
        bool res_def = true;

        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;

        if (res_def) {

            // Base in phi :
            res.bases_1d[2] = std::make_unique<Array<int>>(a.bases_1d[2]->get_dimensions());
            switch ((*a.bases_1d[2])(0)) {
                case COSSIN:
                    switch ((*b.bases_1d[2])(0)) {
                        case COSSIN:
                            res.bases_1d[2]->set(0) = COSSIN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                default:
                    res_def = false;
                    break;
            }

            // Bases in theta :
            // On check l'alternance :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                switch ((*a.bases_1d[1])(index_1)) {
                    case COS:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS : SIN;
                                break;
                            case SIN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN : COS;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN : COS;
                                break;
                            case SIN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS : SIN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    default:
                        res_def = false;
                        break;
                }
            } while (index_1.inc());

            // Base in r :
            Index index_0(a.bases_1d[0]->get_dimensions());
            res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions());
            do {
                switch ((*a.bases_1d[0])(index_0)) {
                    case CHEB:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB:
                                res.bases_1d[0]->set(index_0) = CHEB;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG:
                                res.bases_1d[0]->set(index_0) = LEG;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    default:
                        res_def = false;
                        break;
                }
            } while (index_0.inc());
        }

        if (!res_def)
            for (int dim = 0; dim < a.ndim; dim++)
                if (res.bases_1d[dim] != nullptr) {
                    res.bases_1d[dim].reset();
                    res.bases_1d[dim] = nullptr;
                }
        res.def = res_def;

        return res;
    }

    int Domain_shell_outer_adapted_nosym::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "R ") == 0)
            res = 0;
        if (strcmp(p, "T ") == 0)
            res = 1;
        if (strcmp(p, "P ") == 0)
            res = 2;
        return res;
    }

    void Domain_shell_outer_adapted_nosym::update_constante(const Val_domain& cor_outer_radius, const Scalar& old,
                                                      Scalar& res) const
    {

        update_variable(cor_outer_radius, old, res);
        return;

        Point MM(3);
        Index pos(nbr_points);
        res.set_domain(num_dom).allocate_conf();
        do {

            double rr = ((*outer_radius)(pos) + cor_outer_radius(pos) - inner_radius) * (*coloc[0])(pos(0)) / 2. +
                        ((*outer_radius)(pos) + cor_outer_radius(pos) + inner_radius) / 2.;
            double theta = (*coloc[1])(pos(1));
            double phi = (*coloc[2])(pos(2));

            MM.set(1) = rr * sin(theta) * cos(phi) + center(1);
            MM.set(2) = rr * sin(theta) * sin(phi) + center(2);
            MM.set(3) = rr * cos(theta) + center(3);

            res.set_domain(num_dom).set(pos) = old.val_point(MM, -1);

        } while (pos.inc());

        res.set_domain(num_dom).set_base() = old(num_dom).get_base();
    }
} // namespace Kadath
