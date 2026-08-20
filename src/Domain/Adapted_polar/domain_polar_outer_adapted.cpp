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

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_polar_shell_outer_adapted::Domain_polar_shell_outer_adapted(const Space& sss, int num, int ttype, double rin,
                                                                       double rout, const Point& cr,
                                                                       const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), inner_radius(rin), center(cr)
    {
        assert(nbr.get_ndim() == 2);
        assert(cr.get_ndim() == 2);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        outer_radius = new Val_domain(this);
        *outer_radius = rout;
        outer_radius->std_base();
    }

    Domain_polar_shell_outer_adapted::Domain_polar_shell_outer_adapted(const Space& sss, int num, int ttype, double rin,
                                                                       const Val_domain& rout, const Point& cr,
                                                                       const Dim_array& nbr)
        : Domain(num, ttype, nbr), sp(sss), inner_radius(rin), center(cr)
    {
        assert(nbr.get_ndim() == 2);
        assert(cr.get_ndim() == 2);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        outer_radius = new Val_domain(rout);
    }

    // Constructor by copy
    Domain_polar_shell_outer_adapted::Domain_polar_shell_outer_adapted(const Domain_polar_shell_outer_adapted& so)
        : Domain(so), sp(so.sp), inner_radius(so.inner_radius), center(so.center)
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
        if (so.normal_spher != nullptr)
            normal_spher = new Term_eq(*so.normal_spher);
        if (so.normal_cart != nullptr)
            normal_cart = new Term_eq(*so.normal_cart);
    }

    // Constructor by copy
    Domain_polar_shell_outer_adapted::Domain_polar_shell_outer_adapted(const Space& sp,
                                                                       const Domain_polar_shell_outer_adapted& so)
        : Domain(so, true), sp(sp), inner_radius(so.inner_radius), center(so.center)
    {

        outer_radius = new Val_domain(this, *so.outer_radius);
        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
        outer_radius->coef();
    }

    Domain_polar_shell_outer_adapted::Domain_polar_shell_outer_adapted(const Space& sss, int num,
                                                                       BinarySource& source)
        : Domain(num, source), sp(sss), center(source)
    {
        inner_radius = source.read<double>();
        outer_radius = new Val_domain(this, source);

        outer_radius_term_eq = nullptr;
        rad_term_eq = nullptr;
        der_rad_term_eq = nullptr;
        dt_rad_term_eq = nullptr;
        normal_spher = nullptr;
        normal_cart = nullptr;

        do_coloc();
    }

    void Domain_polar_shell_outer_adapted::do_radius() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = ((*outer_radius)(index)-inner_radius) / 2. * ((*coloc[0])(index(0))) +
                                 ((*outer_radius)(index) + inner_radius) / 2.;
        while (index.inc());
        radius->std_r_base();
    }

    // Destructor
    Domain_polar_shell_outer_adapted::~Domain_polar_shell_outer_adapted()
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
    }

    void Domain_polar_shell_outer_adapted::del_deriv() const
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
    }

    int Domain_polar_shell_outer_adapted::nbr_unknowns_from_adapted() const
    {
        return nbr_coefs(1);
    }

    void Domain_polar_shell_outer_adapted::vars_to_terms() const
    {

        if (outer_radius_term_eq != nullptr)
            delete outer_radius_term_eq;
        Scalar val(sp);
        val.set_domain(num_dom) = *outer_radius;

        outer_radius_term_eq = new Term_eq(num_dom, val);
        update();
    }

    void Domain_polar_shell_outer_adapted::affecte_coef(int& conte, int cc, bool& found) const
    {

        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        found = false;

        for (int j = 0; j < nbr_coefs(1); j++) {
            if (conte == cc) {
                Index pos_cf(nbr_coefs);
                pos_cf.set(0) = 0;
                pos_cf.set(1) = j;
                auxi.cf->set(pos_cf) = 1;
                found = true;
            }
            conte++;
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

    void Domain_polar_shell_outer_adapted::xx_to_vars_from_adapted(Val_domain& new_outer_radius,
                                                                   const Array<double>& xx, int& pos) const
    {

        new_outer_radius.allocate_coef();
        *new_outer_radius.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        for (int j = 0; j < nbr_coefs(1); j++) {
            pos_cf.set(1) = j;
            new_outer_radius.cf->set(pos_cf) -= xx(pos);
            pos++;
        }

        new_outer_radius.set_base() = outer_radius->get_base();
    }

    void Domain_polar_shell_outer_adapted::update_mapping(const Val_domain& cor)
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

    void Domain_polar_shell_outer_adapted::set_mapping(const Val_domain& cor) const
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

    void Domain_polar_shell_outer_adapted::update_variable(const Val_domain& cor_outer_radius, const Scalar& old,
                                                           Scalar& res) const
    {

        Val_domain dr(old(num_dom).der_r());
        if (dr.check_if_zero())
            res.set_domain(num_dom) = 0;
        else {

            Index pos(nbr_points);
            res.set_domain(num_dom).allocate_conf();
            do {
                res.set_domain(num_dom).set(pos) = dr(pos) * (1 + (*coloc[0])(pos(0))) / 2.;
            } while (pos.inc());

            res.set_domain(num_dom) = cor_outer_radius * res(num_dom) + old(num_dom);
            res.set_domain(num_dom).set_base() = old(num_dom).get_base();
        }
    }

    void Domain_polar_shell_outer_adapted::xx_to_ders_from_adapted(const Array<double>& xx, int& pos) const
    {

        Val_domain auxi(this);
        auxi.std_base();
        auxi.set_in_coef();
        auxi.allocate_coef();
        *auxi.cf = 0;

        Index pos_cf(nbr_coefs);
        pos_cf.set(0) = 0;

        for (int j = 0; j < nbr_coefs(1); j++) {
            pos_cf.set(1) = j;
            auxi.cf->set(pos_cf) = xx(pos);
            pos++;
        }

        Scalar auxi_scal(sp);
        auxi_scal.set_domain(num_dom) = auxi;
        outer_radius_term_eq->set_der_t(auxi_scal);
        update();
    }

    void Domain_polar_shell_outer_adapted::update() const
    {

        if (rad_term_eq != nullptr)
            delete rad_term_eq;
        if (der_rad_term_eq != nullptr)
            delete der_rad_term_eq;
        if (dt_rad_term_eq != nullptr)
            delete dt_rad_term_eq;

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

        bool doder = (outer_radius_term_eq->der_t == nullptr) ? false : true;
        if (doder) {
            Scalar der_res(sp);
            if ((*outer_radius_term_eq->der_t)()(num_dom).check_if_zero()) {
                der_res.set_domain(num_dom).set_zero();
            } else {
                der_res.set_domain(num_dom).allocate_conf();
                index.set_start();
                do {
                    der_res.set_domain(num_dom).set(index) =
                        ((*outer_radius_term_eq->der_t)()(num_dom))(index) / 2. * ((*coloc[0])(index(0))) +
                        ((*outer_radius_term_eq->der_t)()(num_dom))(index) / 2.;
                } while (index.inc());
                der_res.set_domain(num_dom).std_r_base();
            }
            rad_term_eq = new Term_eq(num_dom, val_res, der_res);
        } else
            rad_term_eq = new Term_eq(num_dom, val_res);

        // Computation of der_rad_term_eq which is dr / dxi
        val_res.set_domain(num_dom) = ((*outer_radius_term_eq->val_t)()(num_dom)-inner_radius) / 2.;
        if (doder) {
            Scalar der_res(sp);
            der_res.set_domain(num_dom) = (*outer_radius_term_eq->der_t)()(num_dom) / 2.;
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

        do_normal_cart();
    }

    void Domain_polar_shell_outer_adapted::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(inner_radius);
        outer_radius->save(sink);
    }

    ostream& Domain_polar_shell_outer_adapted::print(ostream& o) const
    {
        o << "Adapted polar shell on the outside boundary" << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << "Inner radius " << inner_radius << endl;
        o << "Outer radius " << endl;
        o << *outer_radius << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_polar_shell_outer_adapted::der_normal(const Val_domain&, int) const
    {
        KADATH_THROW("Domain_polar_shell_outer_adapted::der_normal not implemeted");
    }

    // Computes the cartesian coordinates
    void Domain_polar_shell_outer_adapted::do_absol() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {

            double rr = ((*outer_radius)(index)-inner_radius) / 2. * ((*coloc[0])(index(0))) +
                        ((*outer_radius)(index) + inner_radius) / 2.;
            absol[0]->set(index) = rr * sin((*coloc[1])(index(1))) + center(1);
            absol[1]->set(index) = rr * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the cartesian coordinates
    void Domain_polar_shell_outer_adapted::do_cart() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(cart[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            cart[i] = new Val_domain(this);
            cart[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {
            double rr = ((*outer_radius)(index)-inner_radius) / 2. * ((*coloc[0])(index(0))) +
                        ((*outer_radius)(index) + inner_radius) / 2.;
            cart[0]->set(index) = rr * sin((*coloc[1])(index(1))) + center(1);
            cart[1]->set(index) = rr * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the cartesian coordinates over the radius
    void Domain_polar_shell_outer_adapted::do_cart_surr() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(cart_surr[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            cart_surr[i] = new Val_domain(this);
            cart_surr[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {
            cart_surr[0]->set(index) = sin((*coloc[1])(index(1)));
            cart_surr[1]->set(index) = cos((*coloc[1])(index(1)));
        } while (index.inc());
    }

    // Is a point inside this domain ?
    bool Domain_polar_shell_outer_adapted::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);
        Point num(absol_to_num(xx));
        bool res = ((num(1) > -1 - prec) && (num(1) < 1 + prec)) ? true : false;
        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_polar_shell_outer_adapted::absol_to_num(const Point& abs) const
    {
        Point num(2);

        double rho_loc = fabs(abs(1) - center(1));
        double z_loc = abs(2) - center(2);
        double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);

        if (rho_loc == 0) {
            // On the axis ?
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
        } else {
            num.set(2) = atan(rho_loc / z_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        // Get the boundary for those angles
        num.set(1) = 1;
        outer_radius->coef();
        double outer = outer_radius->get_base().summation(num, outer_radius->get_coef());
        num.set(1) = (2. / (outer - inner_radius)) * (air - (outer + inner_radius) / 2.);

        return num;
    }

    double coloc_leg(int, int);
    void Domain_polar_shell_outer_adapted::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = -cos(M_PI * i / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_polar_shell_outer_adapted::do_coloc");
        }
    }

    void Domain_polar_shell_outer_adapted::set_cheb_base(Base_spectral& base) const
    {
        set_cheb_base_with_m(base, 0);
    }

    // standard base for a anti-symetric function in z, using Chebyshev
    void Domain_polar_shell_outer_adapted::set_anti_cheb_base(Base_spectral& base) const
    {
        set_anti_cheb_base_with_m(base, 0);
    }

    // standard base for a symetric function in z, using Legendre
    void Domain_polar_shell_outer_adapted::set_legendre_base(Base_spectral& base) const
    {
        set_legendre_base_with_m(base, 0);
    }

    // standard base for a anti-symetric function in z, using Legendre
    void Domain_polar_shell_outer_adapted::set_anti_legendre_base(Base_spectral& base) const
    {
        set_anti_legendre_base_with_m(base, 0);
    }

    // standard base for a symetric function in z, using Chebyshev
    void Domain_polar_shell_outer_adapted::set_cheb_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }

    // standard base for a anti-symetric function in z, using Chebyshev
    void Domain_polar_shell_outer_adapted::set_anti_cheb_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }

    // standard base for a symetric function in z, using Legendre
    void Domain_polar_shell_outer_adapted::set_legendre_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }

    // standard base for a anti-symetric function in z, using Legendre
    void Domain_polar_shell_outer_adapted::set_anti_legendre_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }

    void Domain_polar_shell_outer_adapted::set_cheb_r_base(Base_spectral& base) const
    {
        set_cheb_base_with_m(base, 0);
    }

    void Domain_polar_shell_outer_adapted::set_legendre_r_base(Base_spectral& base) const
    {
        set_legendre_base_with_m(base, 0);
    }

    // Computes the derivatives with respect to XYZ as a function of the numerical ones.
    void Domain_polar_shell_outer_adapted::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {
        Val_domain rr(get_radius());
        Val_domain dr(*der_var[0] / (*der_rad_term_eq->val_t)()(num_dom));
        Val_domain dtsr((*der_var[1] - rr.der_var(2) * dr) / rr);
        dtsr.set_base() = der_var[1]->get_base();

        // d/dx :
        Val_domain sintdr(dr.mult_sin_theta());
        Val_domain costdtsr(dtsr.mult_cos_theta());

        der_abs[0] = new Val_domain(sintdr + costdtsr);

        // d/dz :
        der_abs[1] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_polar_shell_outer_adapted::mult(const Base_spectral& a, const Base_spectral& b) const
    {

        assert(a.ndim == 2);
        assert(b.ndim == 2);

        Base_spectral res(2);
        bool res_def = true;

        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;

        if (res_def) {

            // Bases in theta :
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            switch ((*a.bases_1d[1])(0)) {
                case COS_EVEN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case COS_ODD:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN_EVEN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN_ODD:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = COS_EVEN;
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

    void Domain_polar_shell_outer_adapted::update_constante(const Val_domain& cor_outer_radius, const Scalar& old,
                                                            Scalar& res) const
    {

        Point MM(2);
        Index pos(nbr_points);
        res.set_domain(num_dom).allocate_conf();
        do {

            double rr = ((*outer_radius)(pos) + cor_outer_radius(pos) - inner_radius) * (*coloc[0])(pos(0)) / 2. +
                        ((*outer_radius)(pos) + cor_outer_radius(pos) + inner_radius) / 2.;
            double theta = (*coloc[1])(pos(1));

            MM.set(1) = rr * sin(theta) + center(1);
            MM.set(2) = rr * cos(theta) + center(2);

            res.set_domain(num_dom).set(pos) = old.val_point(MM, -1);

        } while (pos.inc());

        res.set_domain(num_dom).set_base() = old(num_dom).get_base();
    }

    int Domain_polar_shell_outer_adapted::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "R ") == 0)
            res = 0;
        if (strcmp(p, "T ") == 0)
            res = 1;
        return res;
    }

    Val_domain Domain_polar_shell_outer_adapted::dt(const Val_domain& so) const
    {
        return (so.der_var(2));
    }
} // namespace Kadath
