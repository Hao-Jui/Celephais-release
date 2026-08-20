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
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    // Standard constructor
    Domain_shell_symphi::Domain_shell_symphi(int num, int ttype, double rint, double rext, const Point& cr,
                                             const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha((rext - rint) / 2.), beta((rext + rint) / 2.), center(cr)
    {

        assert(nbr.get_ndim() == 3);
        assert(cr.get_ndim() == 3); // Affectation de type_point :
        do_coloc();
    }

    // Constructor by copy
    Domain_shell_symphi::Domain_shell_symphi(const Domain_shell_symphi& so)
        : Domain(so), alpha(so.alpha), beta(so.beta), center(so.center)
    {
    }

    Domain_shell_symphi::Domain_shell_symphi(int num, BinarySource& source)
        : Domain(num, source), center(source)
    {
        alpha = source.read<double>();
        beta = source.read<double>();
        do_coloc();
    }

    // Destructor
    Domain_shell_symphi::~Domain_shell_symphi() {}

    void Domain_shell_symphi::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        center.save(sink);
        sink.write<double>(alpha);
        sink.write<double>(beta);
    }

    ostream& Domain_shell_symphi::print(ostream& o) const
    {
        o << "Shell symphi" << endl;
        o << beta - alpha << " < r < " << beta + alpha << endl;
        o << "Center  = " << center << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_shell_symphi::der_normal(const Val_domain& so, int bound) const
    {

        Val_domain res(so.der_var(1));
        switch (bound) {
            case OUTER_BC:
                res /= alpha;
                break;
            case INNER_BC:
                res /= alpha;
                break;

            default:
                KADATH_THROW("Unknown boundary case in Domain_shell_symphi::der_normal");
        }
        return res;
    }

    // Computes the Cartesian coordinates
    void Domain_shell_symphi::do_absol() const
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
            absol[0]->set(index) =
                (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) * cos((*coloc[2])(index(2))) +
                center(1);
            absol[1]->set(index) =
                (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) * sin((*coloc[2])(index(2))) +
                center(2);
            absol[2]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))) + center(3);
        } while (index.inc());
    }

    // Computes the radius
    void Domain_shell_symphi::do_radius() const
    {
        for (int i = 0; i < 3; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = alpha * ((*coloc[0])(index(0))) + beta;
        while (index.inc());

        switch (type_base) {
            case CHEB_TYPE:
                set_cheb_base_forr(radius->set_base());
                break;
            case LEG_TYPE:
                set_legendre_base_forr(radius->set_base());
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_nucleus_symphi::do_radius");
        }
    }

    // Computes the Cartesian coordinates
    void Domain_shell_symphi::do_cart() const
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
            cart[0]->set(index) =
                (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) * cos((*coloc[2])(index(2))) +
                center(1);
            cart[1]->set(index) =
                (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) * sin((*coloc[2])(index(2))) +
                center(2);
            cart[2]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))) + center(3);
        } while (index.inc());

        switch (type_base) {
            case CHEB_TYPE:
                set_cheb_base_forx_cart(cart[0]->set_base());
                set_cheb_base_fory_cart(cart[1]->set_base());
                set_cheb_base_forz_cart(cart[2]->set_base());
                break;
            case LEG_TYPE:
                set_legendre_base_forx_cart(cart[0]->set_base());
                set_legendre_base_fory_cart(cart[1]->set_base());
                set_legendre_base_forz_cart(cart[2]->set_base());
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_shell_symphi::do_cart");
        }
    }

    // Computes the Cartesian coordinates
    void Domain_shell_symphi::do_cart_surr() const
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

        switch (type_base) {
            case CHEB_TYPE:
                set_cheb_base_forx_cart(cart_surr[0]->set_base());
                set_cheb_base_fory_cart(cart_surr[1]->set_base());
                set_cheb_base_forz_cart(cart_surr[2]->set_base());
                break;
            case LEG_TYPE:
                set_legendre_base_forx_cart(cart_surr[0]->set_base());
                set_legendre_base_fory_cart(cart_surr[1]->set_base());
                set_legendre_base_forz_cart(cart_surr[2]->set_base());
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_shell_symphi::do_cart_surr");
        }
    }

    // Check if a point is inside this domain
    bool Domain_shell_symphi::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 3);

        double x_loc = xx(1) - center(1);
        double y_loc = xx(2) - center(2);
        double z_loc = xx(3) - center(3);
        double air_loc = sqrt(x_loc * x_loc + y_loc * y_loc + z_loc * z_loc);

        bool res = ((air_loc <= alpha + beta + prec) && (air_loc >= beta - alpha - prec)) ? true : false;
        return res;
    }

    // Computes the numerical coordinates, as a function of the absolute ones.
    const Point Domain_shell_symphi::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(3);

        double x_loc = abs(1) - center(1);
        double y_loc = abs(2) - center(2);
        double z_loc = abs(3) - center(3);
        double air = sqrt(x_loc * x_loc + y_loc * y_loc + z_loc * z_loc);
        num.set(1) = (air - beta) / alpha;
        double rho = sqrt(x_loc * x_loc + y_loc * y_loc);

        if (rho == 0) {
            // On the axis ?
            num.set(2) = (z_loc >= 0) ? 0 : M_PI;
            num.set(3) = 0;
        } else {
            num.set(2) = atan(rho / z_loc);
            num.set(3) = atan2(y_loc, x_loc);
        }

        if (num(2) < 0)
            num.set(2) = M_PI + num(2);

        return num;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_shell_symphi::absol_to_num_bound(const Point& abs, int bound) const
    {

        assert((bound == OUTER_BC) || (bound == INNER_BC));
        assert(is_in(abs, 1e-3));
        Point num(3);

        double x_loc = abs(1) - center(1);
        double y_loc = abs(2) - center(2);
        double z_loc = abs(3) - center(3);

        switch (bound) {
            case INNER_BC:
                num.set(1) = -1;
                break;
            case OUTER_BC:
                num.set(1) = 1;
                break;
            default:
                KADATH_THROW("unknown boundary in Domain_shell_symphi::absol_to_num");
        }

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

        return num;
    }
    double coloc_leg(int, int);
    void Domain_shell_symphi::do_coloc()
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
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI / 2. * k / (nbr_points(2) - 1);
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
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI / 2. * k / (nbr_points(2) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_shell_symphi::do_coloc");
        }
    }

    // Base for a function symetric in z, using Chebyshev
    void Domain_shell_symphi::set_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_r_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_t_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_p_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_rt_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_rp_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_tp_spher(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_x_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_y_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_z_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_xy_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_xz_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = COS_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_yz_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = SIN_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_forr(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);
        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_forx_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_fory_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    void Domain_shell_symphi::set_cheb_base_forz_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = CHEB;
            }
        }
    }

    // Base for a function symetric in z, using Legendre
    void Domain_shell_symphi::set_legendre_base(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_r_spher(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_t_spher(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_p_spher(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_x_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_y_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_z_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_forr(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_EVEN;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_forx_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_fory_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = SIN_ODD;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = SIN_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    void Domain_shell_symphi::set_legendre_base_forz_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        base.bases_1d[2]->set(0) = COS_EVEN;
        for (int k = 0; k < nbr_coefs(2); k++) {
            base.bases_1d[1]->set(k) = COS_ODD;
            for (int j = 0; j < nbr_coefs(1); j++) {
                index.set(0) = j;
                index.set(1) = k;
                base.bases_1d[0]->set(index) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to XYZ as a function of the numerical ones.
    void Domain_shell_symphi::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {

        // d/dx :
        Val_domain sintdr(der_var[0]->mult_sin_theta() / alpha);
        Val_domain dtsr(*der_var[1] / get_radius());
        dtsr.set_base() = der_var[1]->get_base();
        Val_domain dpsr(*der_var[2] / get_radius());
        dpsr.set_base() = der_var[2]->get_base();
        Val_domain costdtsr(dtsr.mult_cos_theta());

        Val_domain dpsrssint(dpsr.div_sin_theta());
        der_abs[0] = new Val_domain((sintdr + costdtsr).mult_cos_phi() - dpsrssint.mult_sin_phi());

        // d/dy :
        der_abs[1] = new Val_domain((sintdr + costdtsr).mult_sin_phi() + dpsrssint.mult_cos_phi());
        // d/dz :
        der_abs[2] = new Val_domain(der_var[0]->mult_cos_theta() / alpha - dtsr.mult_sin_theta());
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_shell_symphi::mult(Base_spectral const& a, Base_spectral const& b) const
    {
        assert(a.ndim == 3);
        assert(b.ndim == 3);
        Base_spectral res(3);
        bool res_def(true);
        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;
        if (res_def) {
            // Base in phi :
            // On check l'alternance :
            Index index_2(a.bases_1d[2]->get_dimensions());
            res.bases_1d[2] = std::make_unique<Array<int>>(a.bases_1d[2]->get_dimensions());
            do {
                int basea((*a.bases_1d[2])(index_2));
                int baseb((*b.bases_1d[2])(index_2));
                res.bases_1d[2]->set(index_2) = mult_base_angle_int(basea, baseb);
            } while (index_2.inc());

            // Bases in theta :
            // On check l'alternance :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                int basea((*a.bases_1d[1])(index_1));
                int baseb((*b.bases_1d[1])(index_1));
                res.bases_1d[1]->set(index_1) = mult_base_angle_int(basea, baseb);
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

    int Domain_shell_symphi::mult_base_angle_int(int basea, int baseb) const
    {
        int res(0);
        if (basea == COS_EVEN and baseb == COS_EVEN)
            res = COS_EVEN;
        if ((basea == COS_EVEN and baseb == COS_ODD) or (basea == COS_ODD and baseb == COS_EVEN))
            res = COS_ODD;
        if ((basea == COS_EVEN and baseb == SIN_EVEN) or (basea == SIN_EVEN and baseb == COS_EVEN))
            res = SIN_EVEN;
        if ((basea == COS_EVEN and baseb == SIN_ODD) or (basea == SIN_ODD and baseb == COS_EVEN))
            res = SIN_ODD;
        if (basea == COS_ODD and baseb == COS_ODD)
            res = COS_EVEN;
        if ((basea == COS_ODD and baseb == SIN_EVEN) or (basea == SIN_EVEN and baseb == COS_ODD))
            res = SIN_ODD;
        if ((basea == COS_ODD and baseb == SIN_ODD) or (basea == SIN_ODD and baseb == COS_ODD))
            res = SIN_EVEN;
        if (basea == SIN_EVEN and baseb == SIN_EVEN)
            res = COS_EVEN;
        if ((basea == SIN_EVEN and baseb == SIN_ODD) or (basea == SIN_ODD and baseb == SIN_EVEN))
            res = COS_ODD;
        if (basea == SIN_ODD and baseb == SIN_ODD)
            res = COS_EVEN;
        return res;
    }

    int Domain_shell_symphi::give_place_var(char* p) const
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

    double
    Domain_shell_symphi::integ(const Val_domain& so,
                               int bound) const // compute int (so sin(theta) dtheta dphi) in x = 1 (r = outer_bc)
    {
        if (bound != OUTER_BC) {
            KADATH_THROW("Domain_shell_symphi::integ only defined for r=rmax yet...");
        }
        double res(0.0);
        int baset((*so.base.bases_1d[1])(0));
        if (baset != COS_EVEN)
            return res;
        else {
            so.coef();
            // Loop on theta :
            Index pos(get_nbr_coefs());
            pos.set(2) = 0;
            for (int j(0); j < nbr_coefs(1); ++j) {
                pos.set(1) = j;
                double fact_tet(2.0 / (1.0 - 4.0 * j * j));
                // Loop on r :
                for (int i(0); i < nbr_coefs(0); ++i) {
                    pos.set(0) = i;
                    res += fact_tet * (*so.cf)(pos);
                }
            }
            return res * 2.0 * M_PI;
        }
    }
} // namespace Kadath
