/*
    Copyright 2025 Philippe Grandclement

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

#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "bispheric_nosym_der_abs_lane_batch.hpp"

namespace Kadath
{
    double eta_lim_chi(double, double, double, double);

    // Standard constructor
    Domain_bispheric_chi_first_nosym::Domain_bispheric_chi_first_nosym(int num, int ttype, double a, double etalim, double air,
                                                           double chim, const Dim_array& nbr)
        : Domain(num, ttype, nbr), aa(a), eta_lim(etalim), r_ext(air), origin_x(0.), chi_max(chim),
          bound_eta(nullptr), bound_eta_der(nullptr), p_eta(nullptr), p_chi(nullptr), p_phi(nullptr),
          p_detadx(nullptr), p_detady(nullptr), p_detadz(nullptr), p_dchidx(nullptr), p_dchidy(nullptr),
          p_dchidz(nullptr), p_dphidy(nullptr), p_dphidz(nullptr), p_dsint(nullptr)
    {

        assert(nbr.get_ndim() == 3);
        eta_c = log((1 + r_ext / aa) / (r_ext / aa - 1));
        do_coloc();
    }

    // Constructor by copy
    Domain_bispheric_chi_first_nosym::Domain_bispheric_chi_first_nosym(const Domain_bispheric_chi_first_nosym& so)
        : Domain(so), aa(so.aa), eta_lim(so.eta_lim), r_ext(so.r_ext), origin_x(so.origin_x), chi_max(so.chi_max)
    {

        bound_eta = (so.bound_eta != nullptr) ? new Val_domain(*so.bound_eta) : nullptr;
        bound_eta_der = (so.bound_eta_der != nullptr) ? new Val_domain(*so.bound_eta_der) : nullptr;
        p_eta = (so.p_eta != nullptr) ? new Val_domain(*so.p_eta) : nullptr;
        p_chi = (so.p_chi != nullptr) ? new Val_domain(*so.p_chi) : nullptr;
        p_phi = (so.p_phi != nullptr) ? new Val_domain(*so.p_phi) : nullptr;
        p_detadx = (so.p_detadx != nullptr) ? new Val_domain(*so.p_detadx) : nullptr;
        p_detady = (so.p_detady != nullptr) ? new Val_domain(*so.p_detady) : nullptr;
        p_detadz = (so.p_detadz != nullptr) ? new Val_domain(*so.p_detadz) : nullptr;
        p_dchidx = (so.p_dchidx != nullptr) ? new Val_domain(*so.p_dchidx) : nullptr;
        p_dchidy = (so.p_dchidy != nullptr) ? new Val_domain(*so.p_dchidy) : nullptr;
        p_dchidz = (so.p_dchidz != nullptr) ? new Val_domain(*so.p_dchidz) : nullptr;
        p_dphidy = (so.p_dphidy != nullptr) ? new Val_domain(*so.p_dphidy) : nullptr;
        p_dphidz = (so.p_dphidz != nullptr) ? new Val_domain(*so.p_dphidz) : nullptr;
        p_dsint = (so.p_dsint != nullptr) ? new Val_domain(*so.p_dsint) : nullptr;
    }

    Domain_bispheric_chi_first_nosym::Domain_bispheric_chi_first_nosym(int num, BinarySource& source)
        : Domain(num, source)
    {
        aa = source.read<double>();
        eta_lim = source.read<double>();
        r_ext = source.read<double>();
        origin_x = 0.;
        chi_max = source.read<double>();
        eta_c = source.read<double>();

        bound_eta = nullptr;
        bound_eta_der = nullptr;
        p_eta = nullptr;
        p_chi = nullptr;
        p_phi = nullptr;
        p_detadx = nullptr;
        p_detady = nullptr;
        p_detadz = nullptr;
        p_dchidx = nullptr;
        p_dchidy = nullptr;
        p_dchidz = nullptr;
        p_dphidy = nullptr;
        p_dphidz = nullptr;
        p_dsint = nullptr;
        do_coloc();
    }

    // Destructor
    Domain_bispheric_chi_first_nosym::~Domain_bispheric_chi_first_nosym()
    {
        del_deriv();
    }

    int Domain_bispheric_chi_first_nosym::phi_coefficient_parity(
        int k, int phi_basis) const
    {
        if (phi_basis != COSSIN)
            return 0;
        return (((k + 1) / 2) % 2) ? -1 : +1;
    }

    void Domain_bispheric_chi_first_nosym::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(aa);
        sink.write<double>(eta_lim);
        sink.write<double>(r_ext);
        sink.write<double>(chi_max);
        sink.write<double>(eta_c);
    }

    void Domain_bispheric_chi_first_nosym::set_origin_x(double origin)
    {
        if (origin_x == origin)
            return;
        origin_x = origin;
        for (int l = 0; l < ndim; l++) {
            safe_delete(absol[l]);
            safe_delete(cart[l]);
            safe_delete(cart_surr[l]);
        }
        safe_delete(radius);
        safe_delete(p_detadx);
        safe_delete(p_detady);
        safe_delete(p_detadz);
        safe_delete(p_dchidx);
        safe_delete(p_dchidy);
        safe_delete(p_dchidz);
        safe_delete(p_dphidy);
        safe_delete(p_dphidz);
        safe_delete(p_dsint);
    }

    Point Domain_bispheric_chi_first_nosym::get_center() const
    {
        Point center(3);
        center.set(1) = origin_x;
        center.set(2) = 0.;
        center.set(3) = 0.;
        return center;
    }

    // Deletes the derived members.
    void Domain_bispheric_chi_first_nosym::del_deriv() const
    {
        for (int l = 0; l < ndim; l++) {
            safe_delete(coloc[l]);
            safe_delete(cart[l]);
        }
        safe_delete(radius);
        safe_delete(bound_eta);
        safe_delete(bound_eta_der);
        safe_delete(p_eta);
        safe_delete(p_chi);
        safe_delete(p_phi);
        safe_delete(p_detadx);
        safe_delete(p_detady);
        safe_delete(p_detadz);
        safe_delete(p_dchidx);
        safe_delete(p_dchidy);
        safe_delete(p_dchidz);
        safe_delete(p_dphidy);
        safe_delete(p_dphidz);
        safe_delete(p_dsint);
    }

    // Display
    ostream& Domain_bispheric_chi_first_nosym::print(ostream& o) const
    {
        o << "Bispherical domain, eta fonction of chi" << endl;
        o << "aa      = " << aa << endl;
        o << "Radius   = " << r_ext << endl;
        o << " 0 < chi < " << chi_max << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    // Computes the function giving the bound for eta.
    void Domain_bispheric_chi_first_nosym::do_bound_eta() const
    {

        assert(p_chi != nullptr);
        assert(bound_eta == nullptr);
        assert(bound_eta_der == nullptr);
        bound_eta = new Val_domain(this);
        bound_eta->allocate_conf();
        Index index(nbr_points);
        int signe = (eta_lim < 0) ? -1 : 1;
        do {
            bound_eta->set(index) = signe * eta_lim_chi((*p_chi)(index), r_ext, aa, eta_c);
        } while (index.inc());

        bound_eta->std_base();
        bound_eta_der = new Val_domain(bound_eta->der_var(2));
    }

    // Computes eta from eta star
    void Domain_bispheric_chi_first_nosym::do_eta() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(p_eta == nullptr);
        assert(bound_eta != nullptr);
        p_eta = new Val_domain(this);
        p_eta->allocate_conf();
        Index index(nbr_points);
        do
            p_eta->set(index) =
                ((*bound_eta)(index)-eta_lim) / 2. * ((*coloc[0])(index(0))) + ((*bound_eta)(index) + eta_lim) / 2.;
        while (index.inc());
    }

    // Computes chi from chi star
    void Domain_bispheric_chi_first_nosym::do_chi() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(p_chi == nullptr);
        p_chi = new Val_domain(this);
        p_chi->allocate_conf();
        Index index(nbr_points);
        do
            p_chi->set(index) = chi_max * (*coloc[1])(index(1));
        while (index.inc());
    }

    // Computes phi from phi star
    void Domain_bispheric_chi_first_nosym::do_phi() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(p_phi == nullptr);
        p_phi = new Val_domain(this);
        p_phi->allocate_conf();
        Index index(nbr_points);
        do
            p_phi->set(index) = ((*coloc[2])(index(2)));
        while (index.inc());
    }

    const Val_domain& Domain_bispheric_chi_first_nosym::get_chi() const
    {
        if (p_chi == nullptr)
            do_chi();
        return *p_chi;
    }

    const Val_domain& Domain_bispheric_chi_first_nosym::get_eta() const
    {
        if (p_eta == nullptr)
            do_eta();
        return *p_eta;
    }

    // Computes the Cartesian coordinates
    void Domain_bispheric_chi_first_nosym::do_absol() const
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
        if (p_chi == nullptr)
            do_chi();
        if (p_phi == nullptr)
            do_phi();
        if (bound_eta == nullptr)
            do_bound_eta();
        if (p_eta == nullptr)
            do_eta();

        do {
            absol[0]->set(index) =
                origin_x + aa * sinh((*p_eta)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            absol[1]->set(index) =
                aa * sin((*p_chi)(index)) * cos((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            absol[2]->set(index) =
                aa * sin((*p_chi)(index)) * sin((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
        } while (index.inc());

        // Basis
        absol[0]->std_base();
        absol[1]->std_base();
        absol[2]->std_base();
    }

    // Computes the generalized radius
    void Domain_bispheric_chi_first_nosym::do_radius() const
    {

        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        Val_domain local_x(get_cart(1) - origin_x);
        radius = new Val_domain(sqrt(local_x * local_x + get_cart(2) * get_cart(2) + get_cart(3) * get_cart(3)));
    }

    // Computes the Cartesian coordinates
    void Domain_bispheric_chi_first_nosym::do_cart() const
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
        if (p_chi == nullptr)
            do_chi();
        if (p_phi == nullptr)
            do_phi();
        if (bound_eta == nullptr)
            do_bound_eta();
        if (p_eta == nullptr)
            do_eta();

        do {
            cart[0]->set(index) =
                origin_x + aa * sinh((*p_eta)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            cart[1]->set(index) =
                aa * sin((*p_chi)(index)) * cos((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            cart[2]->set(index) =
                aa * sin((*p_chi)(index)) * sin((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
        } while (index.inc());

        // Basis
        cart[0]->std_base();
        cart[1]->std_base();
        cart[2]->std_base();
    }

    void Domain_bispheric_chi_first_nosym::do_dsint() const
    {
        double rr = aa / sinh(fabs(eta_lim));
        double xc = aa * cosh(eta_lim) / sinh(eta_lim);

        Val_domain rho2(get_cart(2) * get_cart(2) + get_cart(3) * get_cart(3));
        Val_domain xx(this);
        Val_domain local_x(get_cart(1) - origin_x);
        xx = (xc < 0) ? xc - local_x : local_x - xc;

        assert(p_dsint == nullptr);
        p_dsint = new Val_domain((0.5 * rho2.der_var(2) * xx - xx.der_var(2) * rho2) / rr);
    }

    // Check if a point belongs to this domain
    bool Domain_bispheric_chi_first_nosym::is_in(const Point& abs, double prec) const
    {
        assert(abs.get_ndim() == 3);

        double xx = abs(1) - origin_x;
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        double chi;
        double rho = sqrt(yy * yy + zz * zz);

        double x_out = aa * cosh(eta_lim) / sinh(eta_lim);

        if (rho < prec)
            chi = (fabs(xx) < fabs(x_out)) ? M_PI : 0;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));

        bool res = true;

        if (chi - prec > chi_max)
            res = false;
        if (fabs(eta) - prec > fabs(eta_lim))
            res = false;
        if (res) {
            int signe = (eta_lim < 0) ? -1 : 1;
            double eta_bound = signe * eta_lim_chi(chi, r_ext, aa, eta_c);
            if (fabs(eta) < fabs(eta_bound) - prec)
                res = false;
            if (eta * eta_lim < 0)
                res = false;
        }
        return res;
    }

    // Converts the absolute coordinates to the numerical ones.
    const Point Domain_bispheric_chi_first_nosym::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs, 1e-12));
        Point num(3);

        double xx = abs(1) - origin_x;
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        num.set(3) = atan2(zz, yy);

        if (num(3) < 0)
            num.set(3) += 2 * M_PI;

        double chi;
        double rho = sqrt(yy * yy + zz * zz);

        if (rho < PRECISION)
            chi = 0;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));
        int signe = (eta < 0) ? -1 : 1;
        double eta_bound = signe * eta_lim_chi(chi, r_ext, aa, eta_c);

        num.set(1) = (eta - (eta_bound + eta_lim) / 2.) * 2 / (eta_bound - eta_lim);
        num.set(2) = chi / chi_max;

        return num;
    }

    // Converts the absolute coordinates to the numerical ones.
    const Point Domain_bispheric_chi_first_nosym::absol_to_num_bound(const Point& abs, int bound) const
    {

        assert(is_in(abs, 1e-3));
        Point num(3);

        double xx = abs(1) - origin_x;
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        num.set(3) = atan2(zz, yy);

        if (num(3) < 0)
            num.set(3) += 2 * M_PI;

        double chi;
        double rho = sqrt(yy * yy + zz * zz);

        if (rho < PRECISION)
            chi = 0;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));
        int signe = (eta < 0) ? -1 : 1;
        double eta_bound = signe * eta_lim_chi(chi, r_ext, aa, eta_c);

        num.set(1) = (eta - (eta_bound + eta_lim) / 2.) * 2 / (eta_bound - eta_lim);
        num.set(2) = chi / chi_max;

        switch (bound) {
            case INNER_BC:
                num.set(1) = -1;
                break;
            case OUTER_BC:
                num.set(1) = 1;
                break;
            case CHI_ONE_BC:
                num.set(2) = 1;
                break;
            default:
                KADATH_THROW("Unknown case in Domain_bispheric_chi_first_nosym::absol_to_num_bound");
        }

        return num;
    }

    double coloc_leg(int, int);
    double coloc_leg_parity(int, int);
    void Domain_bispheric_chi_first_nosym::do_coloc()
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
                    coloc[1]->set(j) = sin(M_PI * j / 2. / (nbr_points(1) - 1));
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = 2. * M_PI * k / nbr_points(2);
                do_phi();
                do_chi();
                do_bound_eta();
                do_eta();
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
                    coloc[1]->set(j) = coloc_leg_parity(j, nbr_points(1));
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = 2. * M_PI * k / nbr_points(2);
                do_phi();
                do_chi();
                do_bound_eta();
                do_eta();
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_bispheric_rect_nosym::do_coloc");
        }
    }

    // standard basis for Chebyshev
    void Domain_bispheric_chi_first_nosym::set_cheb_base(Base_spectral& base) const
    {
        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (mm % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // Standard basis for Legendre
    void Domain_bispheric_chi_first_nosym::set_legendre_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (mm % 2 == 0) ? LEG_EVEN : LEG_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Antisymetric basis for Chebyshev
    void Domain_bispheric_chi_first_nosym::set_anti_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (mm % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // Antisymetric basis for Legendre
    void Domain_bispheric_chi_first_nosym::set_anti_legendre_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COSSIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            int mm = (k % 2 == 0) ? k / 2 : (k - 1) / 2;
            base.bases_1d[1]->set(k) = (mm % 2 == 0) ? LEG_EVEN : LEG_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to the absolute coordinates with respect of the numerical ones.
    void Domain_bispheric_chi_first_nosym::do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const
    {
        if (p_detadx == nullptr)
            do_for_der();
        bispheric_nosym_detail::do_der_abs_from_der_var_lanes(
            batch, *p_detadx, *p_detady, *p_detadz, *p_dchidx, *p_dchidy, *p_dchidz,
            *p_dphidy, *p_dphidz, 0, 1);
    }

    void Domain_bispheric_chi_first_nosym::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {

        if (p_detadx == nullptr)
            do_for_der();

        // d/dx :
        der_abs[0] = new Val_domain((*der_var[0]) * (*p_detadx) + (*der_var[1]) * (*p_dchidx));

        // d/dy :
        Val_domain auchi_y((*der_var[2]) * (*p_dphidy));
        Val_domain part_y_phi(auchi_y.div_sin_chi());

        der_abs[1] = new Val_domain((*der_var[0]) * (*p_detady) + (*der_var[1]) * (*p_dchidy) + part_y_phi);
        // d/dz :
        Val_domain auchi_z((*der_var[2]) * (*p_dphidz));
        Val_domain part_z_phi(auchi_z.div_sin_chi());
        der_abs[2] = new Val_domain((*der_var[0]) * (*p_detadz) + (*der_var[1]) * (*p_dchidz) + part_z_phi);
    }

    // Computes the various partial derivatives of the numerical coordinates with respect to the Cartesian ones.
    void Domain_bispheric_chi_first_nosym::do_for_der() const
    {

        if (cart[0] == nullptr)
            do_cart();
        if (radius == nullptr)
            do_radius();

        Val_domain local_x(*cart[0] - origin_x);

        // Partial derivatives of chi
        Val_domain rho(sqrt((*cart[1]) * (*cart[1]) + (*cart[2]) * (*cart[2])));
        Val_domain denom_chi(chi_max * (((*radius) * (*radius) - aa * aa) * ((*radius) * (*radius) - aa * aa) +
                                        4. * aa * aa * rho * rho));
        p_dchidx = new Val_domain(-4. * aa * local_x * rho / denom_chi);

        Val_domain chiyz(2. * aa * ((*radius) * (*radius) - aa * aa - 2 * rho * rho) / denom_chi);

        // Partial derivatives of phi :
        Val_domain phiyz(((exp(*p_eta) + exp(-*p_eta)) / 2. - cos(*p_chi)) / aa);

        chiyz.std_base();
        phiyz.std_base();
        p_dchidx->base = chiyz.der_var(2).get_base();

        p_dchidy = new Val_domain(chiyz.mult_cos_phi());
        p_dchidz = new Val_domain(chiyz.mult_sin_phi());
        p_dphidy = new Val_domain(-phiyz.mult_sin_phi());
        p_dphidz = new Val_domain(phiyz.mult_cos_phi());

        // Derivative eta_bound :
        Val_domain dfdx((*bound_eta_der) * (*p_dchidx));
        Val_domain dfdy((*bound_eta_der) * (*p_dchidy));
        Val_domain dfdz((*bound_eta_der) * (*p_dchidz));

        // Partial derivatives of eta
        Val_domain denom_eta((aa * aa + (*radius) * (*radius)) * ((aa * aa + (*radius) * (*radius))) -
                             4. * aa * aa * local_x * local_x);
        Val_domain detadx((2. * aa * (aa * aa + (*radius) * (*radius) - 2. * local_x * local_x)) / denom_eta);
        Val_domain detady(-4. * aa * local_x * (*cart[1]) / denom_eta);
        Val_domain detadz(-4. * aa * local_x * (*cart[2]) / denom_eta);

        p_detadx =
            new Val_domain((detadx - 0.5 * dfdx) * (2. / ((*bound_eta) - eta_lim)) +
                           ((*p_eta) - ((*bound_eta) + eta_lim) / 2.) * (-2. * dfdx / pow((*bound_eta) - eta_lim, 2.)));
        p_detady =
            new Val_domain((detady - 0.5 * dfdy) * (2. / ((*bound_eta) - eta_lim)) +
                           ((*p_eta) - ((*bound_eta) + eta_lim) / 2.) * (-2. * dfdy / pow((*bound_eta) - eta_lim, 2.)));
        p_detadz =
            new Val_domain((detadz - 0.5 * dfdz) * (2. / ((*bound_eta) - eta_lim)) +
                           ((*p_eta) - ((*bound_eta) + eta_lim) / 2.) * (-2. * dfdz / pow((*bound_eta) - eta_lim, 2.)));

        // The basis :
        p_detadx->std_base();
        p_detady->std_base();
        p_detadz->std_base();
    }

    // Multiplication rules for the basis.
    Base_spectral Domain_bispheric_chi_first_nosym::mult(const Base_spectral& a, const Base_spectral& b) const
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
            res.bases_1d[2]->set(0) = COSSIN;

            // Bases in chi :
            // On check l'alternance :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                switch ((*a.bases_1d[1])(index_1)) {
                    case CHEB_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case CHEB_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? CHEB_EVEN : CHEB_ODD;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? CHEB_ODD : CHEB_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case CHEB_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case CHEB_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? CHEB_ODD : CHEB_EVEN;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? CHEB_EVEN : CHEB_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case LEG_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? LEG_EVEN : LEG_ODD;
                                break;
                            case LEG_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? LEG_ODD : LEG_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case LEG_EVEN:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? LEG_ODD : LEG_EVEN;
                                break;
                            case LEG_ODD:
                                res.bases_1d[1]->set(index_1) = (index_1(0) % 4 <= 1) ? LEG_EVEN : LEG_ODD;
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

            // Base in eta :
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

    Val_domain Domain_bispheric_chi_first_nosym::der_normal(const Val_domain& so, int bound) const
    {

        double rr, xc;
        switch (bound) {
            case OUTER_BC:
                // Outer sphere is centered on the (possibly translated) chart origin
                return Val_domain(so.der_abs(1) * (get_cart(1) - origin_x) / r_ext +
                                  so.der_abs(2) * get_cart(2) / r_ext + so.der_abs(3) * get_cart(3) / r_ext);
            case INNER_BC:
                xc = origin_x + aa * cosh(eta_lim) / sinh(eta_lim);
                rr = aa / sinh(fabs(eta_lim));

                return Val_domain(so.der_abs(1) * (get_cart(1) - xc) / rr + so.der_abs(2) * get_cart(2) / rr +
                                  so.der_abs(3) * get_cart(3) / rr);
            case CHI_ONE_BC: {
                Val_domain res(so.der_var(2) + so.der_var(1) * (*bound_eta_der) *
                                                   (-2. / (*bound_eta - eta_lim) / (*bound_eta - eta_lim) *
                                                        (get_eta() - (*bound_eta + eta_lim) / 2.) -
                                                    1. / (*bound_eta - eta_lim)));
                res /= chi_max;
                res.set_base() = so.der_var(2).get_base();
                return res;
            }
            default:
                KADATH_THROW("Unknown boundary case in Domain_bispheric_chi_first_nosym::der_normal");
        }
    }

    double Domain_bispheric_chi_first_nosym::integ(const Val_domain& so, int bound) const
    {

        if (bound != INNER_BC) {
            KADATH_THROW("Domain_bispheric_chi_first_nosym::integ only defined for inner boundary yet..");
        }

        double res = 0;

        int basep = (*so.base.bases_1d[2])(0);
        if (basep == SIN) {
            // Odd function
            return res;
        } else {
            // Multiply by the surface element :
            if (p_dsint == nullptr)
                do_dsint();

            Val_domain auxi(so * (*p_dsint));
            auxi.coef();

            Index pos(get_nbr_coefs());
            pos.set(2) = 0;

            if (type_base == CHEB_TYPE) {
                for (int j = 0; j < nbr_coefs(1); j++) {
                    pos.set(1) = j;
                    int base_chi = (*auxi.base.bases_1d[1])(0);
                    double val_cheb;
                    switch (base_chi) {
                        case CHEB_EVEN:
                            if (j == 0)
                                val_cheb = 1.;
                            else
                                val_cheb = double(2 * j) / double(4 * j * j - 1) - 1. / double(2 * j - 1);
                            break;
                        case CHEB_ODD:
                            if (j == 0)
                                val_cheb = 1. / 2.;
                            else if (j % 2 == 0)
                                val_cheb =
                                    2 * double(2 * j + 1) / double((2 * j + 1) * (2 * j + 1) - 1) - 1. / double(2 * j);
                            else
                                val_cheb = -1. / double(2 * j);
                            break;
                        default:
                            KADATH_THROW("Unknown basis in Domain_bispheric_chi_first_nosym::integ");
                    }

                    for (int i = 0; i < nbr_coefs(0); i++) {
                        pos.set(0) = i;
                        if (i % 2 == 0)
                            res += val_cheb * (*auxi.cf)(pos);
                        else
                            res -= val_cheb * (*auxi.cf)(pos);
                    }
                }
            }

            if (type_base == LEG_TYPE) {
                int base_chi = (*auxi.base.bases_1d[1])(0);

                double val_leg = 1.;
                double val_m1 = 1.;
                double val_m = -0.5;

                for (int j = 0; j < nbr_coefs(1); j++) {
                    pos.set(1) = j;

                    switch (base_chi) {
                        case LEG_EVEN:
                            val_leg = 0.;
                            break;
                        case LEG_ODD:
                            val_leg = (val_m1 - val_m) / double(4 * j + 3);
                            val_m *= -double(2 * j + 3) / double(2 * j + 4);
                            val_m1 *= -double(2 * j + 1) / double(2 * j + 2);
                            break;
                        default:
                            KADATH_THROW("Unknown basis in Domain_bispheric_chi_first_nosym::integ");
                    }

                    for (int i = 0; i < nbr_coefs(0); i++) {
                        pos.set(0) = i;
                        if (i % 2 == 0)
                            res += val_leg * (*auxi.cf)(pos);
                        else
                            res -= val_leg * (*auxi.cf)(pos);
                    }
                }
            }
            return res * M_PI * 2;
        }
    }

    int Domain_bispheric_chi_first_nosym::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "ETA ") == 0)
            res = 0;
        if (strcmp(p, "CHI ") == 0)
            res = 1;
        if (strcmp(p, "P ") == 0)
            res = 2;
        return res;
    }
} // namespace Kadath
