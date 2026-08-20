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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    // Standard constructor
    Domain_polar_compact::Domain_polar_compact(int num, int ttype, double r, const Point& cr, const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha(-0.5 / r), center(cr)
    {

        assert(nbr.get_ndim() == 2);
        assert(cr.get_ndim() == 2);
        do_coloc();
    }

    // Copy constructor
    Domain_polar_compact::Domain_polar_compact(const Domain_polar_compact& so)
        : Domain(so), alpha(so.alpha), center(so.center)
    {
    }

    Domain_polar_compact::Domain_polar_compact(const Domain_polar_compact& so, bool import)
        : Domain(so, import), alpha(so.alpha), center(so.center)
    {
    }

    Domain_polar_compact::Domain_polar_compact(int num, BinarySource& source)
        : Domain(num, source), center(source)
    {
        alpha = source.read<double>();
        do_coloc();
    }

    // Destructor
    Domain_polar_compact::~Domain_polar_compact() {}

    ostream& Domain_polar_compact::print(ostream& o) const
    {
        o << "Compactified polar domain" << endl;
        o << "Rmin    = " << -0.5 / alpha << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    void Domain_polar_compact::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(alpha);
    }

    Val_domain Domain_polar_compact::der_normal(const Val_domain& so, int bound) const
    {

        Val_domain res(so.der_var(1));
        switch (bound) {
            case INNER_BC:
                res = res.mult_xm1();
                res = res.mult_xm1();
                res *= -alpha;
                break;
            case OUTER_BC:
                res = res.mult_xm1();
                res = res.mult_xm1();
                res *= -alpha;
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_polar_compact::der_normal");
        }
        return res;
    }

    double Domain_polar_compact::integ(const Val_domain& so, int bound) const
    {
        double res = 0;

        int baset = (*so.base.bases_1d[1])(0);
        if (baset != COS_EVEN) {
            // Odd function
            return res;
        } else {
            // For now only at infinity
            if (bound != OUTER_BC) {
                KADATH_THROW("Domain_polar_compact::integ only defined for outer boundary");
            }

            Val_domain auxi(so.div_xm1());
            auxi = auxi.div_xm1();
            auxi /= alpha * alpha;
            auxi.coef();

            // Loop on theta :
            Index pos(get_nbr_coefs());
            for (int j = 0; j < nbr_coefs(1); j++) {
                pos.set(1) = j;
                double fact_tet = 2. / double(1 - 4 * j * j);
                // Loop on r :
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos.set(0) = i;
                    res += fact_tet * (*auxi.cf)(pos);
                }
            }
            return res * 2 * M_PI;
        }
    }

    // Computes the Cartesian coordinates
    void Domain_polar_compact::do_absol() const
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
            absol[0]->set(index) = (1. / alpha / ((*coloc[0])(index(0)) - 1)) * sin((*coloc[1])(index(1))) + center(1);
            absol[1]->set(index) = (1. / alpha / ((*coloc[0])(index(0)) - 1)) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the radius
    void Domain_polar_compact::do_radius() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = 1. / alpha / ((*coloc[0])(index(0)) - 1);
        while (index.inc());
    }

    // Computes the Cartesian coordinates
    void Domain_polar_compact::do_cart() const
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
            cart[0]->set(index) = (1. / alpha / ((*coloc[0])(index(0)) - 1)) * sin((*coloc[1])(index(1))) + center(1);
            cart[1]->set(index) = (1. / alpha / ((*coloc[0])(index(0)) - 1)) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the Cartesian coordinates ovzr radius
    void Domain_polar_compact::do_cart_surr() const
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
            cart_surr[0]->set(index) = sin((*coloc[1])(index(1))) + center(1);
            cart_surr[1]->set(index) = cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Is inside ?
    bool Domain_polar_compact::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);

        double rho_loc = xx(1) - center(1);
        double z_loc = xx(2) - center(2);
        double air_loc = sqrt(rho_loc * rho_loc + z_loc * z_loc);

        bool res = (air_loc >= -0.5 / alpha - prec) ? true : false;
        return res;
    }

    // Converts the absolute coordinates to the numerical ones.
    const Point Domain_polar_compact::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(3);

        double rho_loc = fabs(abs(1) - center(1));
        double z_loc = abs(2) - center(2);
        double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);
        num.set(1) = 1. + 1. / air / alpha;

        if (rho_loc == 0) {
            // On the axis ?
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
        } else {
            num.set(2) = atan(rho_loc / z_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        return num;
    }

    double coloc_leg(int, int);
    void Domain_polar_compact::do_coloc()
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
                KADATH_THROW("Unknown type of basis in Domain_polar_compact::do_coloc");
        }
    }
    // standard base for a symetric function using Chebyshev
    void Domain_polar_compact::set_cheb_base(Base_spectral& base) const
    {
        set_cheb_base_with_m(base, 0);
    }

    // standard base for a anti-symetric function using Chebyshev
    void Domain_polar_compact::set_anti_cheb_base(Base_spectral& base) const
    {
        set_anti_cheb_base_with_m(base, 0);
    }
    // standard base for a symetric function using Legendre
    void Domain_polar_compact::set_legendre_base(Base_spectral& base) const
    {
        set_legendre_base_with_m(base, 0);
    }

    // standard base for a anti-symetric function using Legendre
    void Domain_polar_compact::set_anti_legendre_base(Base_spectral& base) const
    {
        set_anti_legendre_base_with_m(base, 0);
    }

    // standard base for a symetric function using Chebyshev
    void Domain_polar_compact::set_cheb_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }

    // standard base for a anti-symetric function using Chebyshev
    void Domain_polar_compact::set_anti_cheb_base_with_m(Base_spectral& base, int m) const
    {
        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }
    // standard base for a symetric function using Legendre
    void Domain_polar_compact::set_legendre_base_with_m(Base_spectral& base, int m) const
    {
        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }

    // standard base for a anti-symetric function using Legendre
    void Domain_polar_compact::set_anti_legendre_base_with_m(Base_spectral& base, int m) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);

        base.def = true;

        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }

    // sets the value at infinity
    void Domain_polar_compact::set_val_inf(Val_domain& so, double x) const
    {

        assert(so.get_domain() == this);

        so.coef_i();
        so.set_in_conf();
        Index inf(nbr_points);
        inf.set(0) = nbr_points(0) - 1;
        do
            so.set(inf) = x;
        while (inf.inc(1, 1));
    }

    // Computes the derivatives with respect to XYZ as function of the numerical ones.
    void Domain_polar_compact::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {
        Val_domain dr(-der_var[0]->mult_xm1());

        // d/dx :
        Val_domain auxi_x(dr.mult_sin_theta() + der_var[1]->mult_cos_theta());
        der_abs[0] = new Val_domain(alpha * auxi_x.mult_xm1());

        // d/dz :
        Val_domain auxi_z(dr.mult_cos_theta() - der_var[1]->mult_sin_theta());
        der_abs[1] = new Val_domain(alpha * auxi_z.mult_xm1());
    }

    // Rules for basis multiplication
    Base_spectral Domain_polar_compact::mult(const Base_spectral& a, const Base_spectral& b) const
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

    int Domain_polar_compact::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "R ") == 0)
            res = 0;
        if (strcmp(p, "T ") == 0)
            res = 1;
        return res;
    }
} // namespace Kadath
