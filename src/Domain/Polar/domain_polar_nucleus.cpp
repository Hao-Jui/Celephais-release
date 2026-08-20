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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_polar_nucleus::Domain_polar_nucleus(int num, int ttype, double r, const Point& cr, const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha(r), center(cr)
    {
        assert(nbr.get_ndim() == 2);
        assert(cr.get_ndim() == 2);
        do_coloc();
    }

    // Constructor by copy
    Domain_polar_nucleus::Domain_polar_nucleus(const Domain_polar_nucleus& so)
        : Domain(so), alpha(so.alpha), center(so.center)
    {
    }

    // Constructor by copy
    Domain_polar_nucleus::Domain_polar_nucleus(const Domain_polar_nucleus& so, bool import)
        : Domain(so, import), alpha(so.alpha), center(so.center)
    {
    }

    Domain_polar_nucleus::Domain_polar_nucleus(int num, BinarySource& source)
        : Domain(num, source), center(source)
    {
        alpha = source.read<double>();
        do_coloc();
    }

    // Destructor
    Domain_polar_nucleus::~Domain_polar_nucleus() {}

    void Domain_polar_nucleus::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(alpha);
    }

    ostream& Domain_polar_nucleus::print(ostream& o) const
    {
        o << "Polar nucleus" << endl;
        o << "Rmax    = " << alpha << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_polar_nucleus::der_normal(const Val_domain& so, int bound) const
    {

        Val_domain res(so.der_var(1));
        switch (bound) {
            case OUTER_BC:
                res /= alpha;
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_polar_nucleus::der_normal");
        }
        return res;
    }

    // Computes the cartesian coordinates
    void Domain_polar_nucleus::do_absol() const
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
            absol[0]->set(index) = alpha * ((*coloc[0])(index(0))) * sin((*coloc[1])(index(1))) + center(1);
            absol[1]->set(index) = alpha * ((*coloc[0])(index(0))) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the radius
    void Domain_polar_nucleus::do_radius() const
    {

        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = alpha * ((*coloc[0])(index(0)));
        while (index.inc());
    }

    // Computes the cartesian coordinates
    void Domain_polar_nucleus::do_cart() const
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
            cart[0]->set(index) = alpha * ((*coloc[0])(index(0))) * sin((*coloc[1])(index(1))) + center(1);
            cart[1]->set(index) = alpha * ((*coloc[0])(index(0))) * cos((*coloc[1])(index(1))) + center(2);
        } while (index.inc());
    }

    // Computes the cartesian coordinates over the radius
    void Domain_polar_nucleus::do_cart_surr() const
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

    // Is a point inside this domain ?
    bool Domain_polar_nucleus::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);

        double rho_loc = xx(1) - center(1);
        double z_loc = xx(2) - center(2);
        double air_loc = sqrt(rho_loc * rho_loc + z_loc * z_loc);

        bool res = (air_loc <= alpha + prec) ? true : false;
        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_polar_nucleus::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(2);

        double rho_loc = fabs(abs(1) - center(1));
        double z_loc = abs(2) - center(2);
        double air = sqrt(rho_loc * rho_loc + z_loc * z_loc);
        num.set(1) = air / alpha;

        if (rho_loc == 0) {
            // Sur l'axe
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
        } else {
            num.set(2) = atan(rho_loc / z_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        return num;
    }

    double coloc_leg_parity(int, int);
    void Domain_polar_nucleus::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = sin(M_PI / 2. * i / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg_parity(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_polar_nucleus::do_coloc");
        }
    }

    // Base for a function symetric in z, using Chebyshev
    void Domain_polar_nucleus::set_cheb_base(Base_spectral& base) const
    {
        set_cheb_base_with_m(base, 0);
    }

    // Base for a function anti-symetric in z, using Chebyshev
    void Domain_polar_nucleus::set_anti_cheb_base(Base_spectral& base) const
    {
        set_anti_cheb_base_with_m(base, 0);
    }

    // Base for a function symetric in z, using Legendre
    void Domain_polar_nucleus::set_legendre_base(Base_spectral& base) const
    {
        set_legendre_base_with_m(base, 0);
    }

    // Base for a function anti-symetric in z, using Legendre
    void Domain_polar_nucleus::set_anti_legendre_base(Base_spectral& base) const
    {
        set_anti_legendre_base_with_m(base, 0);
    }

    // Base for a function symetric in z, using Chebyshev
    void Domain_polar_nucleus::set_cheb_base_with_m(Base_spectral& base, int m) const
    {

        int l;

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++) {
            l = (m % 2 == 0) ? 2 * j : 2 * j + 1;
            base.bases_1d[0]->set(j) = (l % 2 == 0) ? CHEB_EVEN : CHEB_ODD;
        }
    }

    // Base for a function anti-symetric in z, using Chebyshev
    void Domain_polar_nucleus::set_anti_cheb_base_with_m(Base_spectral& base, int m) const
    {

        int l;

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++) {
            l = (m % 2 == 0) ? 2 * j + 1 : 2 * j;
            base.bases_1d[0]->set(j) = (l % 2 == 1) ? CHEB_ODD : CHEB_EVEN;
        }
    }

    // Base for a function symetric in z, using Legendre
    void Domain_polar_nucleus::set_legendre_base_with_m(Base_spectral& base, int m) const
    {

        int l;

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_EVEN : SIN_ODD;
        for (int j = 0; j < nbr_coefs(1); j++) {
            l = (m % 2 == 0) ? 2 * j : 2 * j + 1;
            base.bases_1d[0]->set(j) = (l % 2 == 0) ? LEG_EVEN : LEG_ODD;
        }
    }

    // Base for a function anti-symetric in z, using Legendre
    void Domain_polar_nucleus::set_anti_legendre_base_with_m(Base_spectral& base, int m) const
    {

        int l;

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[1]->set(0) = (m % 2 == 0) ? COS_ODD : SIN_EVEN;
        for (int j = 0; j < nbr_coefs(1); j++) {
            l = (m % 2 == 0) ? 2 * j + 1 : 2 * j;
            base.bases_1d[0]->set(j) = (l % 2 == 1) ? LEG_ODD : LEG_EVEN;
        }
    }

    // Computes the derivatives with respect to rho,Z as a function of the numerical ones.
    void Domain_polar_nucleus::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {
        Val_domain dr(*der_var[0] / alpha);
        Val_domain dtsr(der_var[1]->div_x() / alpha);

        // D / drho
        der_abs[0] = new Val_domain(dr.mult_sin_theta() + dtsr.mult_cos_theta());
        // d/dz :
        der_abs[1] = new Val_domain(dr.mult_cos_theta() - dtsr.mult_sin_theta());
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_polar_nucleus::mult(const Base_spectral& a, const Base_spectral& b) const
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
                    case CHEB_EVEN:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(index_0) = CHEB_EVEN;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(index_0) = CHEB_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case CHEB_ODD:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(index_0) = CHEB_ODD;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(index_0) = CHEB_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_EVEN:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(index_0) = LEG_EVEN;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(index_0) = LEG_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_ODD:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(index_0) = LEG_ODD;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(index_0) = LEG_EVEN;
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

    int Domain_polar_nucleus::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "R ") == 0)
            res = 0;
        if (strcmp(p, "T ") == 0)
            res = 1;
        return res;
    }
} // namespace Kadath
