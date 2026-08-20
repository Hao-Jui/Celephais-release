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
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_shell_inner_adapted::Domain_shell_inner_adapted(const Space& sss, int num, int ttype, double rin,
                                                           double rout, const Point& cr, const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), outer_radius(rout), center(cr)
    {
        assert(nbr.get_ndim() == 3);
        assert(cr.get_ndim() == 3);

        inner_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        inner_radius = new Val_domain(this);
        *inner_radius = rin;
        inner_radius->std_base();
    }

    Domain_shell_inner_adapted::Domain_shell_inner_adapted(const Space& sss, int num, int ttype, const Val_domain& rin,
                                                           double rout, const Point& cr, const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), outer_radius(rout), center(cr)
    {
        assert(nbr.get_ndim() == 3);
        assert(cr.get_ndim() == 3);

        inner_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        inner_radius = new Val_domain(rin);
    }

    // Constructor by copy
    Domain_shell_inner_adapted::Domain_shell_inner_adapted(const Domain_shell_inner_adapted& so)
        : Domain(so), sp(so.sp), outer_radius(so.outer_radius), center(so.center)
    {

        inner_radius = new Val_domain(*so.inner_radius);
        if (so.inner_radius_term_eq != nullptr)
            inner_radius_term_eq = new Term_eq(*so.inner_radius_term_eq);
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

    Domain_shell_inner_adapted::Domain_shell_inner_adapted(const Space& sss, int num, BinarySource& source)
        : Domain(num, source), sp(sss), center(source)
    {
        outer_radius = source.read<double>();
        inner_radius = new Val_domain(this, source);

        inner_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        dp_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
    }

    void Domain_shell_inner_adapted::do_radius() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = (outer_radius - (*inner_radius)(index)) / 2. * ((*coloc[0])(index(0))) +
                                 (outer_radius + (*inner_radius)(index)) / 2.;
        while (index.inc());
        radius->std_r_base();
    }

    // Destructor
    Domain_shell_inner_adapted::~Domain_shell_inner_adapted()
    {

        delete inner_radius;
        if (inner_radius_term_eq != nullptr)
            delete inner_radius_term_eq;
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

    void Domain_shell_inner_adapted::del_deriv() const
    {

        if (inner_radius_term_eq != nullptr)
            delete inner_radius_term_eq;
        inner_radius_term_eq = nullptr;
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

    int Domain_shell_inner_adapted::nbr_unknowns_from_adapted() const
    {

        int res = 0;
        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int jmax = (mm % 2 == 0) ? nbr_coefs(1) : nbr_coefs(1) - 1;
                int jmin = (k >= 4) ? 1 : 0;
                for (int j = jmin; j < jmax; j++)
                    res++;
            }
        return res;
    }

    void Domain_shell_inner_adapted::vars_to_terms() const
    {

        if (inner_radius_term_eq != nullptr)
            delete inner_radius_term_eq;
        Scalar val(sp);
        val.set_domain(num_dom) = *inner_radius;

        inner_radius_term_eq = new Term_eq(num_dom, val);
        update();
    }

    void Domain_shell_inner_adapted::affecte_coef(int& conte, int cc, bool& found) const
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
                int jmax = (mm % 2 == 0) ? nbr_coefs(1) : nbr_coefs(1) - 1;
                int jmin = (k >= 4) ? 1 : 0;
                for (int j = jmin; j < jmax; j++) {
                    if (conte == cc) {
                        Index pos_cf(nbr_coefs);
                        pos_cf.set(0) = 0;
                        pos_cf.set(1) = j;
                        pos_cf.set(2) = k;
                        auxi.cf->set(pos_cf) = 1;
                        if ((jmin == 1) && (mm % 2 == 0)) {
                            // Galerkin base COS EVEN
                            pos_cf.set(1) = 0;
                            auxi.cf->set(pos_cf) = -1;
                        }
                        if ((jmin == 1) && (mm % 2 == 1)) {
                            // Galerkin base SIN ODD
                            pos_cf.set(1) = 0;
                            auxi.cf->set(pos_cf) = -(2 * j + 1);
                        }
                        found = true;
                    }
                    conte++;
                }
            }

        if (found) {
            Scalar auxi_scal(sp);
            auxi_scal.set_domain(num_dom) = auxi;
            inner_radius_term_eq->set_der_t(auxi_scal);
        } else {
            inner_radius_term_eq->set_der_zero();
        }
        update();
    }

    void Domain_shell_inner_adapted::xx_to_vars_from_adapted(Val_domain& new_inner_radius, const Array<double>& xx,
                                                             int& pos) const
    {

        new_inner_radius.allocate_coef();
        *new_inner_radius.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        for (int k = 0; k < nbr_coefs(2); k++)
            if ((k != 1) && (k != nbr_coefs(2) - 1)) {
                pos_cf.set(2) = k;
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                int jmax = (mm % 2 == 0) ? nbr_coefs(1) : nbr_coefs(1) - 1;
                int jmin = (k >= 4) ? 1 : 0;
                for (int j = jmin; j < jmax; j++) {
                    pos_cf.set(1) = j;
                    new_inner_radius.cf->set(pos_cf) -= xx(pos);
                    if ((jmin == 1) && (mm % 2 == 0)) {
                        // Galerkin basis COS EVEN
                        pos_cf.set(1) = 0;
                        new_inner_radius.cf->set(pos_cf) += xx(pos);
                    }
                    if ((jmin == 1) && (mm % 2 == 1)) {
                        // Galerkin basis SIN ODD
                        pos_cf.set(1) = 0;
                        new_inner_radius.cf->set(pos_cf) += (2 * j + 1) * xx(pos);
                    }

                    pos++;
                }
            }

        new_inner_radius.set_base() = inner_radius->get_base();
    }

    void Domain_shell_inner_adapted::update_mapping(const Val_domain& cor)
    {

        *inner_radius += cor;
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

    void Domain_shell_inner_adapted::set_mapping(const Val_domain& cor) const
    {
        *inner_radius = cor;
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

    void Domain_shell_inner_adapted::update_variable(const Val_domain& cor_inner_radius, const Scalar& old,
                                                     Scalar& res) const
    {

        Val_domain dr(old(num_dom).der_r());
        if (dr.check_if_zero())
            res.set_domain(num_dom) = 0;
        else {

            Index pos(nbr_points);
            res.set_domain(num_dom).allocate_conf();
            do {
                res.set_domain(num_dom).set(pos) = dr(pos) * (1 - (*coloc[0])(pos(0))) / 2.;
            } while (pos.inc());

            res.set_domain(num_dom) = cor_inner_radius * res(num_dom) + old(num_dom);
            res.set_domain(num_dom).set_base() = old(num_dom).get_base();
        }
    }

    void Domain_shell_inner_adapted::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const
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
                int jmax = (mm % 2 == 0) ? nbr_coefs(1) : nbr_coefs(1) - 1;
                int jmin = (k >= 4) ? 1 : 0;
                for (int j = jmin; j < jmax; j++) {
                    pos_cf.set(1) = j;
                    auxi.cf->set(pos_cf) = xx(pos);

                    if ((jmin == 1) && (mm % 2 == 0)) {
                        // Galerkin basis COS EVEN
                        pos_cf.set(1) = 0;
                        auxi.cf->set(pos_cf) += -xx(pos);
                    }
                    if ((jmin == 1) && (mm % 2 == 1)) {
                        // Galerkin basis SIN ODD
                        pos_cf.set(1) = 0;
                        auxi.cf->set(pos_cf) += -(2 * j + 1) * xx(pos);
                    }
                    pos++;
                }
            }

        Scalar auxi_scal(sp);
        auxi_scal.set_domain(num_dom) = auxi;
        inner_radius_term_eq->set_der_t(auxi_scal);
        update();
    }

    void Domain_shell_inner_adapted::update() const
    {

        if (rad_term_eq != nullptr)
            delete rad_term_eq;
        if (der_rad_term_eq != nullptr)
            delete der_rad_term_eq;
        if (dt_rad_term_eq != nullptr)
            delete dt_rad_term_eq;
        if (dp_rad_term_eq != nullptr)
            delete dp_rad_term_eq;

        // Computation of rad_term_eq
        Scalar val_res(sp);
        val_res.set_domain(num_dom).allocate_conf();
        Index index(nbr_points);
        do {
            val_res.set_domain(num_dom).set(index) =
                (outer_radius - ((*inner_radius_term_eq->val_t)()(num_dom))(index)) / 2. * ((*coloc[0])(index(0))) +
                (outer_radius + ((*inner_radius_term_eq->val_t)()(num_dom))(index)) / 2.;
        } while (index.inc());
        val_res.set_domain(num_dom).std_r_base();

        bool doder = (inner_radius_term_eq->der_t == nullptr) ? false : true;
        if (doder) {
            Scalar der_res(sp);
            if ((*inner_radius_term_eq->der_t)()(num_dom).check_if_zero()) {
                der_res.set_domain(num_dom).set_zero();
            } else {
                der_res.set_domain(num_dom).allocate_conf();
                index.set_start();
                do {
                    der_res.set_domain(num_dom).set(index) =
                        -((*inner_radius_term_eq->der_t)()(num_dom))(index) / 2. * ((*coloc[0])(index(0))) +
                        ((*inner_radius_term_eq->der_t)()(num_dom))(index) / 2.;
                } while (index.inc());
                der_res.set_domain(num_dom).std_r_base();
            }
            rad_term_eq = new Term_eq(num_dom, val_res, der_res);
        } else
            rad_term_eq = new Term_eq(num_dom, val_res);

        // Computation of der_rad_term_eq which is dr / dxi
        val_res.set_domain(num_dom) = (outer_radius - (*inner_radius_term_eq->val_t)()(num_dom)) / 2.;
        if (doder) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = -(*inner_radius_term_eq->der_t)()(num_dom) / 2.;
            der_rad_term_eq = new Term_eq(num_dom, val_res, der_res);
        } else
            der_rad_term_eq = new Term_eq(num_dom, val_res);

        // Computation of dt_rad_term_eq which is dr / d theta
        val_res.set_domain(num_dom) = (*rad_term_eq->val_t)()(num_dom).der_var(2);
        if (doder) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = (*rad_term_eq->der_t)()(num_dom).der_var(2);
            dt_rad_term_eq = new Term_eq(num_dom, val_res, der_res);
        } else
            dt_rad_term_eq = new Term_eq(num_dom, val_res);

        // Computation of dt_rad_term_eq which is dr / d phi
        val_res.set_domain(num_dom) = (*rad_term_eq->val_t)()(num_dom).der_var(3);
        if (doder) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = (*rad_term_eq->der_t)()(num_dom).der_var(3);
            dp_rad_term_eq = new Term_eq(num_dom, val_res, der_res);
        } else
            dp_rad_term_eq = new Term_eq(num_dom, val_res);

        do_normal_cart();
    }

    void Domain_shell_inner_adapted::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(outer_radius);
        inner_radius->save(sink);
    }

    // ostream& operator<< (ostream& o, const Domain_shell_inner_adapted& so) {
    ostream& Domain_shell_inner_adapted::print(ostream& o) const
    {
        o << "Adapted shell on the inside boundary" << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << "Outer radius " << outer_radius << endl;
        o << "Inner radius " << endl;
        o << *inner_radius << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_shell_inner_adapted::der_normal(const Val_domain&, int) const
    {
        KADATH_THROW("Domain_shell_inner_adapted::der_normal not implemeted");
    }

    // Computes the cartesian coordinates
    void Domain_shell_inner_adapted::do_absol() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {

            double rr = (outer_radius - (*inner_radius)(index)) / 2. * ((*coloc[0])(index(0))) +
                        (outer_radius + (*inner_radius)(index)) / 2.;
            absol[0]->set(index) = rr * sin((*coloc[1])(index(1))) * cos((*coloc[2])(index(2))) + center(1);
            absol[1]->set(index) = rr * sin((*coloc[1])(index(1))) * sin((*coloc[2])(index(2))) + center(2);
            absol[2]->set(index) = rr * cos((*coloc[1])(index(1))) + center(3);
        } while (index.inc());
        absol[0]->std_base();
        absol[1]->std_base();
        absol[2]->std_anti_base();
    }

    // Computes the cartesian coordinates
    void Domain_shell_inner_adapted::do_cart() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(cart[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            cart[i] = new Val_domain(this);
            cart[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {
            double rr = (outer_radius - (*inner_radius)(index)) / 2. * ((*coloc[0])(index(0))) +
                        (outer_radius + (*inner_radius)(index)) / 2.;
            cart[0]->set(index) = rr * sin((*coloc[1])(index(1))) * cos((*coloc[2])(index(2))) + center(1);
            cart[1]->set(index) = rr * sin((*coloc[1])(index(1))) * sin((*coloc[2])(index(2))) + center(2);
            cart[2]->set(index) = rr * cos((*coloc[1])(index(1))) + center(3);
        } while (index.inc());
        cart[0]->std_base();
        cart[1]->std_base();
        cart[2]->std_anti_base();
    }

    // Computes the cartesian coordinates over the radius
    void Domain_shell_inner_adapted::do_cart_surr() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 3; i++)
            assert(cart_surr[i] == nullptr);
        for (int i = 0; i < 3; i++) {
            cart_surr[i] = new Val_domain(this);
            cart_surr[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {
            cart_surr[0]->set(index) = sin((*coloc[1])(index(1))) * cos((*coloc[2])(index(2)));
            cart_surr[1]->set(index) = sin((*coloc[1])(index(1))) * sin((*coloc[2])(index(2)));
            cart_surr[2]->set(index) = cos((*coloc[1])(index(1)));
        } while (index.inc());
        cart_surr[0]->std_base();
        cart_surr[1]->std_base();
        cart_surr[2]->std_anti_base();
    }

    // Is a point inside this domain ?
    bool Domain_shell_inner_adapted::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 3);
        Point num(absol_to_num(xx));
        bool res = ((num(1) > -1 - prec) && (num(1) < 1 + prec)) ? true : false;
        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_shell_inner_adapted::absol_to_num(const Point& abs) const
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
        inner_radius->coef();
        double inner = inner_radius->get_base().summation(num, inner_radius->get_coef());
        num.set(1) = (2. / (outer_radius - inner)) * (air - (outer_radius + inner) / 2.);

        return num;
    }

    const Point Domain_shell_inner_adapted::absol_to_num_bound(const Point& abs, int bound) const
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
                KADATH_THROW("Unknown case in Domain_shell_inner_adapted::absol_to_num_bound");
        }
        return num;
    }

    double coloc_leg(int, int);
    void Domain_shell_inner_adapted::do_coloc()
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
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
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
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * 2. * k / nbr_points(2);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_shell_inner_adapted::do_coloc");
        }
    }

    // standard base for a symetric function in z, using Chebyshev
    void Domain_shell_inner_adapted::set_cheb_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_inner_adapted::set_cheb_r_base(Base_spectral& base) const
    {
        set_cheb_base(base);
    }

    void Domain_shell_inner_adapted::set_legendre_r_base(Base_spectral& base) const
    {
        set_legendre_base(base);
    }

    // standard base for a anti-symetric function in z, using Chebyshev
    void Domain_shell_inner_adapted::set_anti_cheb_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_inner_adapted::set_cheb_base_r_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_inner_adapted::set_cheb_base_t_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN_EVEN : COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_inner_adapted::set_cheb_base_p_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN_ODD : COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // standard base for a symetric function in z, using Legendre
    void Domain_shell_inner_adapted::set_legendre_base(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // standard base for a anti-symetric function in z, using Legendre
    void Domain_shell_inner_adapted::set_anti_legendre_base(Base_spectral& base) const
    {

        int m;
        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_inner_adapted::set_legendre_base_r_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_inner_adapted::set_legendre_base_t_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN_EVEN : COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_inner_adapted::set_legendre_base_p_spher(Base_spectral& base) const
    {

        int m;

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;

        Index index(base.bases_1d[0]->get_dimensions());

        for (int k = 0; k < nbr_coefs(2); k++) {
            m = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (m % 2 == 0) ? SIN_ODD : COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to XYZ as a function of the numerical ones.
    void Domain_shell_inner_adapted::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
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
    Base_spectral Domain_shell_inner_adapted::mult(const Base_spectral& a, const Base_spectral& b) const
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
                    case COS_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_EVEN : SIN_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_ODD : SIN_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_EVEN : COS_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_ODD : COS_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case COS_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_ODD : SIN_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_EVEN : SIN_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_ODD : COS_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_EVEN : COS_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_EVEN : COS_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_ODD : COS_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_EVEN : SIN_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_ODD : SIN_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_ODD : COS_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? SIN_EVEN : COS_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_ODD : SIN_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 < 2) ? COS_EVEN : SIN_ODD;
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

    int Domain_shell_inner_adapted::give_place_var(char* p) const
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

    void Domain_shell_inner_adapted::update_constante(const Val_domain& cor_inner_radius, const Scalar& old,
                                                      Scalar& res) const
    {
        update_variable(cor_inner_radius, old, res);
        return;

        Point MM(3);
        Index pos(nbr_points);
        res.set_domain(num_dom).allocate_conf();
        do {

            double rr = (outer_radius - cor_inner_radius(pos) - (*inner_radius)(pos)) * (*coloc[0])(pos(0)) / 2. +
                        (outer_radius + cor_inner_radius(pos) + (*inner_radius)(pos)) / 2.;
            double theta = (*coloc[1])(pos(1));
            double phi = (*coloc[2])(pos(2));

            MM.set(1) = rr * sin(theta) * cos(phi) + center(1);
            MM.set(2) = rr * sin(theta) * sin(phi) + center(2);
            MM.set(3) = rr * cos(theta) + center(3);

            res.set_domain(num_dom).set(pos) = old.val_point(MM, +1);

        } while (pos.inc());

        res.set_domain(num_dom).set_base() = old(num_dom).get_base();
    }
    double Domain_shell_inner_adapted::integ(const Val_domain& so, int bound) const
    {

        if (so.check_if_zero())
            return 0;
        else {
            Val_domain integrant(this);
            switch (bound) {
                case INNER_BC: {

                    Val_domain rad(get_radius()); // Radius
                    Val_domain sint2(get_cart_surr(1) * get_cart_surr(1) +
                                     get_cart_surr(2) *
                                         get_cart_surr(2)); // sin^2 theta (should be able to get that in other manner)
                    Val_domain drdt(inner_radius->der_var(1)); // partial r / partial theta
                    Val_domain drdp(inner_radius->der_var(2)); // partial r / partial phi
                    Val_domain detq(rad * rad * (rad * rad * sint2 + drdt * drdt + sint2 * drdp * drdp));
                    Val_domain sqrtdetq(sqrt(detq));

                    Val_domain one(this);
                    one = 1;
                    one.std_base();
                    Val_domain auxi(one.mult_sin_theta());

                    sqrtdetq.set_base() = auxi.get_base();

                    integrant = sqrtdetq * so;
                    break;
                }
                case OUTER_BC: {
                    Val_domain rad(get_radius()); // Radius
                    Val_domain sint2(get_cart_surr(1) * get_cart_surr(1) +
                                     get_cart_surr(2) *
                                         get_cart_surr(2)); // sin^2 theta (should be able to get that in other manner)
                    Val_domain detq(rad * rad * rad * rad * sint2);
                    Val_domain sqrtdetq(sqrt(detq));

                    Val_domain one(this);
                    one = 1;
                    one.std_base();
                    Val_domain auxi(one.mult_sin_theta());

                    sqrtdetq.set_base() = auxi.get_base();

                    integrant = sqrtdetq * so;
                    break;
                }
                default:
                    cerr << "Unknown type of boundary in Domain_shell_inner_adapted::integ" << endl;
            }

            double res = 0;
            int baset = (*integrant.get_base().bases_1d[1])(0);
            Index pcf(nbr_coefs);
            switch (baset) {
                case COS_ODD:
                    break;
                case SIN_EVEN:
                    break;
                case COS_EVEN: {
                    res += M_PI * val_boundary(bound, integrant, pcf);
                    break;
                }
                case SIN_ODD: {
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        pcf.set(1) = j;
                        res += 2. / (2 * double(j) + 1) * val_boundary(bound, integrant, pcf);
                    }
                    break;
                }
                default:
                    KADATH_THROW("Case not yet implemented in Domain_shell::integ");
            }

            res *= 2 * M_PI;
            return res;
        }
    }
} // namespace Kadath
