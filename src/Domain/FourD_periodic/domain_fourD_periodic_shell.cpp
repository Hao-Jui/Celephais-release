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
#include "For_Kadath/Domain/fourD_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_fourD_periodic_shell::Domain_fourD_periodic_shell(int num, int ttype, double rin, double rout, double oome,
                                                             const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha((rout - rin) / 2.), beta((rout + rin) / 2.), ome(oome), type_time(TO_PI)
    {
        assert(nbr.get_ndim() == 4);
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
    }

    // Constructor by copy
    Domain_fourD_periodic_shell::Domain_fourD_periodic_shell(const Domain_fourD_periodic_shell& so)
        : Domain(so), alpha(so.alpha), beta(so.beta), ome(so.ome), type_time(so.type_time), maxt(so.maxt)
    {
    }

    Domain_fourD_periodic_shell::Domain_fourD_periodic_shell(int num, BinarySource& source)
        : Domain(num, source)
    {
        alpha = source.read<double>();
        beta = source.read<double>();
        ome = source.read<double>();
        type_time = source.read<int>();
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
    }

    // Destructor
    Domain_fourD_periodic_shell::~Domain_fourD_periodic_shell() {}

    void Domain_fourD_periodic_shell::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(alpha);
        sink.write<double>(beta);
        sink.write<double>(ome);
        sink.write<int>(type_time);
    }

    ostream& Domain_fourD_periodic_shell::print(ostream& o) const
    {
        o << "fourD_periodic shell" << endl;
        o << "time goes to " << maxt << endl;
        o << beta - alpha << " < r < " << beta + alpha << endl;
        o << "Omega   = " << ome << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    // Computes the cartesian coordinates
    void Domain_fourD_periodic_shell::do_absol() const
    {
        for (int i = 0; i < 4; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 4; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 4; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {

            absol[0]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) *
                                   cos((*coloc[2])(index(2))); // xx
            absol[1]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) *
                                   sin((*coloc[2])(index(2)));                                            // yy
            absol[2]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))); // zz
            absol[3]->set(index) = (*coloc[3])(index(3)) / ome;                                           // time
        } while (index.inc());
    }

    void Domain_fourD_periodic_shell::do_cart() const
    {
        for (int i = 0; i < 4; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 4; i++)
            assert(cart[i] == nullptr);
        for (int i = 0; i < 4; i++) {
            cart[i] = new Val_domain(this);
            cart[i]->allocate_conf();
        }
        Index index(nbr_points);
        do {

            cart[0]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) *
                                  cos((*coloc[2])(index(2))); // xx
            cart[1]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * sin((*coloc[1])(index(1))) *
                                  sin((*coloc[2])(index(2)));                                            // yy
            cart[2]->set(index) = (alpha * ((*coloc[0])(index(0))) + beta) * cos((*coloc[1])(index(1))); // zz
            cart[3]->set(index) = (*coloc[3])(index(3)) / ome;                                           // time
        } while (index.inc());

        // Put base for x and y
        switch (type_base) {
            case CHEB_TYPE:
                set_cheb_base_x_cart(cart[0]->set_base());
                set_cheb_base_y_cart(cart[1]->set_base());
                break;
            case LEG_TYPE:
                set_legendre_base_x_cart(cart[0]->set_base());
                set_legendre_base_y_cart(cart[1]->set_base());
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_fourD_periodic_shell::do_cart");
        }
    }

    // Computes the radius
    void Domain_fourD_periodic_shell::do_radius() const
    {

        for (int i = 0; i < 4; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = alpha * ((*coloc[0])(index(0))) + beta;
        while (index.inc());
    }

    // Is a point inside this domain ?
    bool Domain_fourD_periodic_shell::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 4);

        double x_loc = xx(1);
        double y_loc = xx(2);
        double z_loc = xx(3);
        double air_loc = sqrt(x_loc * x_loc + y_loc * y_loc + z_loc * z_loc);

        bool res = ((air_loc <= alpha + beta + prec) && (air_loc >= beta - alpha - prec)) ? true : false;

        if ((xx(4) < 0 - prec) || (xx(4) > maxt / ome + prec))
            res = false;

        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_fourD_periodic_shell::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(4);

        double x_loc = abs(1);
        double y_loc = abs(2);
        double z_loc = abs(3);
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

        num.set(4) = abs(4) * ome;

        return num;
    }

    double coloc_leg(int, int);
    void Domain_fourD_periodic_shell::do_coloc()
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
                for (int l = 0; l < nbr_points(3); l++)
                    coloc[3]->set(l) = maxt * l / (nbr_points(3) - 1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                nbr_coefs.set(2) += 2;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI / 2. * j / (nbr_points(1) - 1);
                for (int k = 0; k < nbr_points(2); k++)
                    coloc[2]->set(k) = M_PI * 2. * k / nbr_points(2);
                for (int l = 0; l < nbr_points(3); l++)
                    coloc[3]->set(l) = maxt * l / (nbr_points(3) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_fourD_periodic_shell::do_coloc");
        }
    }

    void Domain_fourD_periodic_shell::set_cheb_base_x_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        // Time
        base.bases_1d[3]->set(0) = COS;

        for (int l = 0; l < nbr_coefs(3); l++) {
            // Phi
            base.bases_1d[2]->set(l) = COSSIN;
            for (int k = 0; k < nbr_coefs(2); k++) {
                // Theta
                base.bases_1d[1]->set(k, l) = SIN_ODD;
                for (int j = 0; j < nbr_coefs(1); j++)
                    // r
                    base.bases_1d[0]->set(j, k, l) = CHEB;
            }
        }
    }

    void Domain_fourD_periodic_shell::set_cheb_base_y_cart(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        // Time
        base.bases_1d[3]->set(0) = COS;

        for (int l = 0; l < nbr_coefs(3); l++) {
            // Phi
            base.bases_1d[2]->set(l) = COSSIN;
            for (int k = 0; k < nbr_coefs(2); k++) {
                // Theta
                base.bases_1d[1]->set(k, l) = SIN_ODD;
                for (int j = 0; j < nbr_coefs(1); j++)
                    // r
                    base.bases_1d[0]->set(j, k, l) = CHEB;
            }
        }
    }

    void Domain_fourD_periodic_shell::set_legendre_base_x_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        // Time
        base.bases_1d[3]->set(0) = COS;

        for (int l = 0; l < nbr_coefs(3); l++) {
            // Phi
            base.bases_1d[2]->set(l) = COSSIN;
            for (int k = 0; k < nbr_coefs(2); k++) {
                // Theta
                base.bases_1d[1]->set(k, l) = SIN_ODD;
                for (int j = 0; j < nbr_coefs(1); j++)
                    // r
                    base.bases_1d[0]->set(j, k, l) = LEG;
            }
        }
    }

    void Domain_fourD_periodic_shell::set_legendre_base_y_cart(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        base.def = true;
        // Time
        base.bases_1d[3]->set(0) = COS;

        for (int l = 0; l < nbr_coefs(3); l++) {
            // Phi
            base.bases_1d[2]->set(l) = COSSIN;
            for (int k = 0; k < nbr_coefs(2); k++) {
                // Theta
                base.bases_1d[1]->set(k, l) = SIN_ODD;
                for (int j = 0; j < nbr_coefs(1); j++)
                    // r
                    base.bases_1d[0]->set(j, k, l) = LEG;
            }
        }
    }

    // Computes the derivatives with respect to rho,Z as a function of the numerical ones.
    void Domain_fourD_periodic_shell::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
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

        // d/dt :
        der_abs[3] = new Val_domain(*der_var[3] * ome);
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_fourD_periodic_shell::mult(const Base_spectral& a, const Base_spectral& b) const
    {

        assert(a.ndim == 4);
        assert(b.ndim == 4);

        Base_spectral res(4);
        bool res_def = true;

        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;

        if (res_def) {

            // Bases in time :
            res.bases_1d[3] = std::make_unique<Array<int>>(a.bases_1d[3]->get_dimensions());
            switch ((*a.bases_1d[3])(0)) {
                case COS:
                    switch ((*b.bases_1d[3])(0)) {
                        case COS:
                            res.bases_1d[3]->set(0) = COS;
                            break;
                        case SIN:
                            res.bases_1d[3]->set(0) = SIN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN:
                    switch ((*b.bases_1d[3])(0)) {
                        case COS:
                            res.bases_1d[3]->set(0) = SIN;
                            break;
                        case SIN:
                            res.bases_1d[3]->set(0) = COS;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
            }

            // Base in phi :
            Index index_2(a.bases_1d[2]->get_dimensions());
            res.bases_1d[2] = std::make_unique<Array<int>>(a.bases_1d[2]->get_dimensions());
            do {
                switch ((*a.bases_1d[2])(index_2)) {
                    case COSSIN:
                        switch ((*b.bases_1d[2])(index_2)) {
                            case COSSIN:
                                res.bases_1d[2]->set(index_2) = COSSIN;
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
            } while (index_2.inc());

            // Bases in theta :
            Index index_1(a.bases_1d[1]->get_dimensions());
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            do {
                switch ((*a.bases_1d[1])(index_1)) {
                    case COS_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case COS_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_EVEN:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case SIN_ODD:
                        switch ((*b.bases_1d[1])(index_1)) {
                            case COS_EVEN:
                                res.bases_1d[1]->set(index_1) = SIN_ODD;
                                break;
                            case COS_ODD:
                                res.bases_1d[1]->set(index_1) = SIN_EVEN;
                                break;
                            case SIN_EVEN:
                                res.bases_1d[1]->set(index_1) = COS_ODD;
                                break;
                            case SIN_ODD:
                                res.bases_1d[1]->set(index_1) = COS_EVEN;
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

    Val_domain Domain_fourD_periodic_shell::translate(const Val_domain& so) const
    {

        Val_domain res(this);

        // Construction bases
        Base_spectral base(4);
        base.allocate(get_nbr_coefs());

        // Time base
        base.bases_1d[3] = std::make_unique<Array<int>>(1);
        base.bases_1d[3]->set(0) = (*so.get_base().bases_1d[2])(0);

        // Phi base
        base.bases_1d[2] = std::make_unique<Array<int>>(get_nbr_coefs()(3));
        for (int i = 0; i < get_nbr_coefs()(3); i++)
            base.bases_1d[2]->set(i) = COSSIN;

        // Theta base
        base.bases_1d[1] = std::make_unique<Array<int>>(get_nbr_coefs()(2), get_nbr_coefs()(3));
        for (int l = 0; l < get_nbr_coefs()(3); l++)
            for (int k = 0; k < get_nbr_coefs()(2); k++) {
                base.bases_1d[1]->set(k, l) = (*so.get_base().bases_1d[1])(l);
            }

        // radial base
        base.bases_1d[0] = std::make_unique<Array<int>>(get_nbr_coefs()(1), get_nbr_coefs()(2), get_nbr_coefs()(3));
        for (int l = 0; l < get_nbr_coefs()(3); l++)
            for (int k = 0; k < get_nbr_coefs()(2); k++)
                for (int j = 0; j < get_nbr_coefs()(1); j++) {
                    base.bases_1d[0]->set(j, k, l) = (*so.get_base().bases_1d[0])(j, l);
                }
        base.def = true;
        res.set_base() = base;

        // Now affecte the coefficients
        so.coef();
        res.annule_hard();
        res.coef();
        Index pso(so.get_domain()->get_nbr_coefs());
        Index pcf(get_nbr_coefs());

        do {
            pcf.set(3) = pso(2); // t coef
            pcf.set(2) = 0.;     // phi coef
            pcf.set(1) = pso(1); // Theta coef
            pcf.set(0) = pso(0); // r coef
            res.set_coef(pcf) = so.get_coef(pso);
        } while (pso.inc());

        return res;
    }

} // namespace Kadath
