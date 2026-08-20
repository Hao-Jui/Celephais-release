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

#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    double chi_lim_eta(double, double, double, double);

    // Standard constructor
    Domain_bispheric_eta_first::Domain_bispheric_eta_first(int num, int ttype, double a, double air, double etamin,
                                                           double etamax, const Dim_array& nbr)
        : Domain(num, ttype, nbr), aa(a), r_ext(air), eta_min(etamin), eta_max(etamax), bound_chi(nullptr),
          bound_chi_der(nullptr), p_eta(nullptr), p_chi(nullptr), p_phi(nullptr), p_detadx(nullptr), p_detady(nullptr),
          p_detadz(nullptr), p_dchidx(nullptr), p_dchidy(nullptr), p_dchidz(nullptr), p_dphidy(nullptr),
          p_dphidz(nullptr)
    {

        assert(nbr.get_ndim() == 3);
        chi_c = 2 * atan(aa / r_ext);
        do_coloc();
    }

    // Constructor by copy
    Domain_bispheric_eta_first::Domain_bispheric_eta_first(const Domain_bispheric_eta_first& so)
        : Domain(so), aa(so.aa), r_ext(so.r_ext), eta_min(so.eta_min), eta_max(so.eta_max)
    {

        bound_chi = (so.bound_chi != nullptr) ? new Val_domain(*so.bound_chi) : nullptr;
        bound_chi_der = (so.bound_chi_der != nullptr) ? new Val_domain(*so.bound_chi_der) : nullptr;
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
    }

    Domain_bispheric_eta_first::Domain_bispheric_eta_first(int num, BinarySource& source)
        : Domain(num, source)
    {
        aa = source.read<double>();
        r_ext = source.read<double>();
        eta_min = source.read<double>();
        eta_max = source.read<double>();
        chi_c = source.read<double>();

        bound_chi = nullptr;
        bound_chi_der = nullptr;
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
        do_coloc();
    }

    // Destructor
    Domain_bispheric_eta_first::~Domain_bispheric_eta_first()
    {
        del_deriv();
    }

    int Domain_bispheric_eta_first::phi_coefficient_parity(
        int k, int phi_basis) const
    {
        const int harmonic_parity = (k % 2) ? -1 : +1;
        if (phi_basis == COS)
            return harmonic_parity;
        if (phi_basis == SIN)
            return -harmonic_parity;
        return 0;
    }

    void Domain_bispheric_eta_first::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(aa);
        sink.write<double>(r_ext);
        sink.write<double>(eta_min);
        sink.write<double>(eta_max);
        sink.write<double>(chi_c);
    }

    // Deletes the derived members
    void Domain_bispheric_eta_first::del_deriv() const
    {
        for (int l = 0; l < ndim; l++) {
            safe_delete(coloc[l]);
            safe_delete(cart[l]);
        }
        safe_delete(radius);
        safe_delete(bound_chi);
        safe_delete(bound_chi_der);
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
    }

    // Display
    ostream& Domain_bispheric_eta_first::print(ostream& o) const
    {

        o << "Bispherical domain, chi fonction of eta" << endl;
        o << "aa      = " << aa << endl;
        o << "Radius   = " << r_ext << endl;
        o << eta_min << " < eta < " << eta_max << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    // Comptes the lower bound for chi as a function of eta
    void Domain_bispheric_eta_first::do_bound_chi() const
    {

        assert(p_eta != nullptr);
        assert(bound_chi == nullptr);
        assert(bound_chi_der == nullptr);
        bound_chi = new Val_domain(this);
        bound_chi->allocate_conf();
        Index index(nbr_points);
        do {
            bound_chi->set(index) = chi_lim_eta(fabs((*p_eta)(index)), r_ext, aa, chi_c);
        } while (index.inc());
        bound_chi->std_base();

        bound_chi_der = new Val_domain(bound_chi->der_var(2));
    }

    // Comptes chi from chi star
    void Domain_bispheric_eta_first::do_chi() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(p_chi == nullptr);
        assert(bound_chi != nullptr);
        p_chi = new Val_domain(this);
        p_chi->allocate_conf();
        Index index(nbr_points);
        do
            p_chi->set(index) = ((*bound_chi)(index)-M_PI) * ((*coloc[0])(index(0))) + M_PI;
        while (index.inc());
    }

    // Computes eta from eta star
    void Domain_bispheric_eta_first::do_eta() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(p_eta == nullptr);
        p_eta = new Val_domain(this);
        p_eta->allocate_conf();
        Index index(nbr_points);
        do
            p_eta->set(index) = (eta_max - eta_min) / 2. * ((*coloc[1])(index(1))) + (eta_min + eta_max) / 2.;
        while (index.inc());
    }

    // Computes phi from phi star
    void Domain_bispheric_eta_first::do_phi() const
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

    const Val_domain& Domain_bispheric_eta_first::get_chi() const
    {
        if (p_chi == nullptr)
            do_chi();
        return *p_chi;
    }

    const Val_domain& Domain_bispheric_eta_first::get_eta() const
    {
        if (p_eta == nullptr)
            do_eta();
        return *p_eta;
    }

    void Domain_bispheric_eta_first::do_absol() const
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
        if (bound_chi == nullptr)
            do_bound_chi();
        if (p_eta == nullptr)
            do_eta();

        do {
            absol[0]->set(index) = aa * sinh((*p_eta)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            absol[1]->set(index) =
                aa * sin((*p_chi)(index)) * cos((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            absol[2]->set(index) =
                aa * sin((*p_chi)(index)) * sin((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
        } while (index.inc());

        // Basis
        absol[0]->std_base();
        absol[1]->std_base();
        absol[2]->std_anti_base();
    }

    // Computes the generalized radius.
    void Domain_bispheric_eta_first::do_radius() const
    {

        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius =
            new Val_domain(sqrt(get_cart(1) * get_cart(1) + get_cart(2) * get_cart(2) + get_cart(3) * get_cart(3)));
    }

    // Computes the Cartesian coordinates
    void Domain_bispheric_eta_first::do_cart() const
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
        if (bound_chi == nullptr)
            do_bound_chi();
        if (p_eta == nullptr)
            do_eta();

        do {
            cart[0]->set(index) = aa * sinh((*p_eta)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            cart[1]->set(index) =
                aa * sin((*p_chi)(index)) * cos((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
            cart[2]->set(index) =
                aa * sin((*p_chi)(index)) * sin((*p_phi)(index)) / (cosh((*p_eta)(index)) - cos((*p_chi)(index)));
        } while (index.inc());

        // Basis
        cart[0]->std_base();
        cart[1]->std_base();
        cart[2]->std_anti_base();
    }

    // Check if the point is inside this domain
    bool Domain_bispheric_eta_first::is_in(const Point& abs, double prec) const
    {
        assert(abs.get_ndim() == 3);

        double xx = abs(1);
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        double chi;
        double rho = sqrt(yy * yy + zz * zz);

        double x_out = aa * cosh(eta_max) / sinh(eta_max);

        if (rho < prec)
            chi = (fabs(abs(1)) < fabs(x_out)) ? M_PI : 0;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));

        bool res = true;
        if (eta > eta_max + prec)
            res = false;
        if (eta < eta_min - prec)
            res = false;

        if (res) {
            double chi_bound = chi_lim_eta(fabs(eta), r_ext, aa, chi_c);
            if (chi < chi_bound - prec)
                res = false;
        }
        return res;
    }

    // Converts the absolute coordinates to the numerical ones.
    const Point Domain_bispheric_eta_first::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs, 1e-12));
        Point num(3);

        double xx = abs(1);
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        num.set(3) = atan2(zz, yy);

        if (num(3) < 0)
            num.set(3) += 2 * M_PI;

        double chi;
        double rho = sqrt(yy * yy + zz * zz);
        if (rho < PRECISION)
            chi = M_PI;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));

        num.set(2) = (eta - (eta_max + eta_min) / 2.) * 2. / (eta_max - eta_min);
        double chi_bound = chi_lim_eta(fabs(eta), r_ext, aa, chi_c);
        num.set(1) = (chi - M_PI) / (chi_bound - M_PI);

        return num;
    }

    const Point Domain_bispheric_eta_first::absol_to_num_bound(const Point& abs, int bound) const
    {

        assert(is_in(abs, 1e-3));
        Point num(3);

        double xx = abs(1);
        double yy = abs(2);
        double zz = abs(3);
        double air = sqrt(xx * xx + yy * yy + zz * zz);

        num.set(3) = atan2(zz, yy);

        if (num(3) < 0)
            num.set(3) += 2 * M_PI;

        double chi;
        double rho = sqrt(yy * yy + zz * zz);
        if (rho < PRECISION)
            chi = M_PI;
        else
            chi = atan(2 * aa * rho / (air * air - aa * aa));

        if (chi < 0)
            chi += M_PI;

        double eta =
            0.5 * log((1 + (2 * aa * xx) / (aa * aa + air * air)) / (1 - (2 * aa * xx) / (aa * aa + air * air)));

        num.set(2) = (eta - (eta_max + eta_min) / 2.) * 2. / (eta_max - eta_min);
        double chi_bound = chi_lim_eta(fabs(eta), r_ext, aa, chi_c);
        num.set(1) = (chi - M_PI) / (chi_bound - M_PI);

        switch (bound) {
            case ETA_PLUS_BC:
                num.set(2) = 1;
                break;
            case ETA_MINUS_BC:
                num.set(2) = -1;
                break;
            case OUTER_BC:
                num.set(1) = 1;
                break;
            default:
                KADATH_THROW("Unknown case in Domain_bispheric_eta_first::absol_to_num_bound");
        }

        return num;
    }

    double coloc_leg(int, int);
    double coloc_leg_parity(int, int);
    void Domain_bispheric_eta_first::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = sin(M_PI * i / 2. / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = -cos(M_PI * j / (nbr_points(1) - 1));
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * k / (nbr_points(2) - 1);
                do_phi();
                do_eta();
                do_bound_chi();
                do_chi();
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg_parity(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = coloc_leg(j, nbr_points(1));
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * k / (nbr_points(2) - 1);
                do_phi();
                do_eta();
                do_bound_chi();
                do_chi();
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_bispheric_eta_first::do_coloc");
        }
    }
    // Standard basis for Chebyshev
    void Domain_bispheric_eta_first::set_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = CHEB;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = (k % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
            }
        }
    }

    // Standard basis for Legendre
    void Domain_bispheric_eta_first::set_legendre_base(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = LEG;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = (k % 2 == 0) ? LEG_EVEN : LEG_ODD;
            }
        }
    }

    // Antisymetric basis for Chebyshev
    void Domain_bispheric_eta_first::set_anti_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = CHEB;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = (k % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
            }
        }
    }

    // Antisymetric fo Legendre
    void Domain_bispheric_eta_first::set_anti_legendre_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = LEG;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = (k % 2 == 0) ? LEG_EVEN : LEG_ODD;
            }
        }
    }

    // Computes the derivatives of the numerical coordinates with respect to the Cartesian ones.
    void Domain_bispheric_eta_first::do_for_der() const
    {

        if (cart[0] == nullptr)
            do_cart();
        if (radius == nullptr)
            do_radius();
        // Partial derivatives of eta
        Val_domain denom_eta((eta_max - eta_min) / 2. *
                             ((aa * aa + (*radius) * (*radius)) * ((aa * aa + (*radius) * (*radius))) -
                              4. * aa * aa * (*cart[0]) * (*cart[0])));
        p_detadx =
            new Val_domain((2. * aa * (aa * aa + (*radius) * (*radius) - 2. * (*cart[0]) * (*cart[0]))) / denom_eta);
        p_detady = new Val_domain(-4. * aa * (*cart[0]) * (*cart[1]) / denom_eta);
        p_detadz = new Val_domain(-4. * aa * (*cart[0]) * (*cart[2]) / denom_eta);

        // Partial derivatives of phi :
        Val_domain phiyz(((exp(*p_eta) + exp(-*p_eta)) / 2. - cos(*p_chi)) / aa);

        // Derivative eta_bound :
        Val_domain dgdx((*bound_chi_der) * (*p_detadx));
        Val_domain dgdy((*bound_chi_der) * (*p_detady));
        Val_domain dgdz((*bound_chi_der) * (*p_detadz));

        // Partial derivatives of chi
        Val_domain rho(sqrt((*cart[1]) * (*cart[1]) + (*cart[2]) * (*cart[2])));
        Val_domain denom_chi(((*radius) * (*radius) - aa * aa) * ((*radius) * (*radius) - aa * aa) +
                             4. * aa * aa * rho * rho);

        Val_domain auchi_chi(2. * aa * ((*radius) * (*radius) - aa * aa - 2 * rho * rho) / denom_chi);
        // The basis :
        p_detadx->std_base();
        p_detady->std_base();
        p_detadz->std_anti_base();
        phiyz.std_base();
        auchi_chi.std_base();

        p_dphidy = new Val_domain(-phiyz.mult_sin_phi());
        p_dphidz = new Val_domain(phiyz.mult_cos_phi());

        Val_domain dchidx(-4. * aa * (*cart[0]) * rho / denom_chi);
        Val_domain dchidy(auchi_chi.mult_cos_phi());
        Val_domain dchidz(auchi_chi.mult_sin_phi());

        p_dchidx =
            new Val_domain((dchidx * ((*bound_chi) - M_PI) - dgdx * ((*p_chi) - M_PI)) / pow((*bound_chi) - M_PI, 2.));
        p_dchidy =
            new Val_domain((dchidy * ((*bound_chi) - M_PI) - dgdy * ((*p_chi) - M_PI)) / pow((*bound_chi) - M_PI, 2.));
        p_dchidz =
            new Val_domain((dchidz * ((*bound_chi) - M_PI) - dgdz * ((*p_chi) - M_PI)) / pow((*bound_chi) - M_PI, 2.));

        p_dchidx->base = auchi_chi.der_var(1).base;
        p_dchidy->base = auchi_chi.mult_cos_phi().base;
        p_dchidz->base = auchi_chi.mult_sin_phi().base;
    }

    // Computes the derivatives with respect to the Cartesian coordinates giving the ones with respect to the numerical
    // ones.
    void Domain_bispheric_eta_first::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {

        if (p_detadx == nullptr)
            do_for_der();

        // d/dx :
        der_abs[0] = new Val_domain((*der_var[1]) * (*p_detadx) + (*der_var[0]) * (*p_dchidx));

        // d/dy :
        Val_domain auchi_y((*der_var[2]) * (*p_dphidy));
        Val_domain part_y_phi(auchi_y.div_sin_chi());

        der_abs[1] = new Val_domain((*der_var[1]) * (*p_detady) + (*der_var[0]) * (*p_dchidy) + part_y_phi);

        // d/dz :
        Val_domain auchi_z((*der_var[2]) * (*p_dphidz));
        Val_domain part_z_phi(auchi_z.div_sin_chi());
        der_abs[2] = new Val_domain((*der_var[1]) * (*p_detadz) + (*der_var[0]) * (*p_dchidz) + part_z_phi);
    }

    // Multiplication rule for the basis.
    Base_spectral Domain_bispheric_eta_first::mult(const Base_spectral& a, const Base_spectral& b) const
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
                case COS:
                    switch ((*b.bases_1d[2])(0)) {
                        case COS:
                            res.bases_1d[2]->set(0) = COS;
                            break;
                        case SIN:
                            res.bases_1d[2]->set(0) = SIN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN:
                    switch ((*b.bases_1d[2])(0)) {
                        case COS:
                            res.bases_1d[2]->set(0) = SIN;
                            break;
                        case SIN:
                            res.bases_1d[2]->set(0) = COS;
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

            // Base in eta :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                switch ((*a.bases_1d[1])(index_1)) {
                    case CHEB:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case CHEB:
                                res.bases_1d[1]->set(index_1) = CHEB;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case LEG:
                                res.bases_1d[1]->set(index_1) = LEG;
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

            // Bases in chi :
            Index index_0(a.bases_1d[0]->get_dimensions());
            res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions());
            do {
                switch ((*a.bases_1d[0])(index_0)) {
                    case CHEB_EVEN:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? CHEB_ODD : CHEB_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case CHEB_ODD:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? CHEB_ODD : CHEB_EVEN;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_EVEN:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? LEG_EVEN : LEG_ODD;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? LEG_ODD : LEG_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_ODD:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? LEG_ODD : LEG_EVEN;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(index_0) = (index_0(1) % 2 == 0) ? LEG_EVEN : LEG_ODD;
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

    Val_domain Domain_bispheric_eta_first::der_normal(const Val_domain& so, int bound) const
    {

        switch (bound) {
            case OUTER_BC:
                return Val_domain(so.der_abs(1) * get_cart(1) / r_ext + so.der_abs(2) * get_cart(2) / r_ext +
                                  so.der_abs(3) * get_cart(3) / r_ext);
            case ETA_MINUS_BC: {
                Val_domain res(so.der_var(2) - (get_chi() - M_PI) / (*bound_chi - M_PI) / (*bound_chi - M_PI) *
                                                   (*bound_chi_der) * so.der_var(1));
                res *= 2. / (eta_max - eta_min);
                res.set_base() = so.der_var(2).get_base();
                return res;
            }
            case ETA_PLUS_BC: {
                Val_domain res(so.der_var(2) - (get_chi() - M_PI) / (*bound_chi - M_PI) / (*bound_chi - M_PI) *
                                                   (*bound_chi_der) * so.der_var(1));
                res *= 2. / (eta_max - eta_min);
                res.set_base() = so.der_var(2).get_base();
                return res;
            }
            default:
                KADATH_THROW("Unknown boundary case in Domain_bispheric_eta_first::der_normal");
        }
    }

    int Domain_bispheric_eta_first::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "CHI ") == 0)
            res = 0;
        if (strcmp(p, "ETA ") == 0)
            res = 1;
        if (strcmp(p, "P ") == 0)
            res = 2;
        return res;
    }
} // namespace Kadath
